import os
import sys
import json
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

# Ensure project and sibling repos are importable when run directly.
_this_dir = os.path.dirname(__file__)
parent_path = os.path.abspath(os.path.join(_this_dir, os.pardir))
grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
llava_rp_path = os.path.join(grandparent_path, "llava_rp")
for _p in (parent_path, grandparent_path, llava_rp_path):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# Require llava_rp (GlimpsePrune) to ensure GP-specific builder/logic is used.
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava_rp.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava_rp.model.builder import load_pretrained_model


def _default_attn_impl() -> str:
    return "flash_attention_2"


class LlavaLlamaGP(lmms):
    """
    GlimpsePrune LLaVA wrapper for lmms-eval.

    Key point: generation is executed in GP tutorial style:
      - build prompt with (IM_START + IMAGE + IM_END) if mm_use_im_start_end
      - tokenizer_image_token(prompt)
      - image_tensor = process_images([image], ...)[0]
      - stack input_ids/images to [B, ...]
      - model.generate(**inputs, generation_config=..., do_selection=True)
    """

    def __init__(
        self,
        pretrained: str = "/mnt/nvme/fjh/llava-v1.5-7b",
        new_modules_dir: str = "/mnt/nvme/fjh/GlimpsePrune_LLaVA-1.5-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        conv_template: str = "vicuna_v1",
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reduce_layer: Optional[int] = None,
        min_remain_num: Optional[int] = None,
        max_remain_ratio: Optional[float] = None,
        base_score_pool_multiplier: Optional[float] = None,
        use_ref_masks: bool = False,
        use_zero_masks: bool = False,
        anchor_positions: Optional[Tuple[int, ...]] = None,
        adapter_dir: Optional[str] = None,
        adapter_merge: bool = False,
        use_cache: bool = True,
        token_scores_output_dir: Optional[str] = None,
        save_token_scores: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.token_scores_output_dir = token_scores_output_dir
        self.save_token_scores = save_token_scores

        if kwargs:
            eval_logger.warning(f"Unexpected kwargs in LlavaLlamaGP: {kwargs}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_name = get_model_name_from_path(pretrained)
        resolved_attn_impl = attn_implementation or _default_attn_impl()
        if resolved_attn_impl != "flash_attention_2":
            raise ValueError(
                f"LlavaLlamaGP only supports attn_implementation='flash_attention_2', got {resolved_attn_impl!r}."
            )
        builder_kwargs = {
            "model_path": pretrained,
            "model_base": None,
            "model_name": model_name,
            "device_map": self.device_map,
            "attn_implementation": resolved_attn_impl,
        }
        tokenizer, model, image_processor, context_len = load_pretrained_model(**builder_kwargs)

        self._model = model.eval()

        # Load GlimpsePrune modules / LoRA adapters if provided.
        if new_modules_dir:
            if hasattr(self._model, "load_new_modules"):
                self._model.load_new_modules(new_modules_dir)
                eval_logger.info(f"Loaded new modules from {new_modules_dir}")
            else:
                eval_logger.warning("new_modules_dir provided but model does not support load_new_modules; skipping.")

        if adapter_dir is not None:
            try:
                from peft import PeftModel
            except Exception as e:
                raise ImportError("peft is required when adapter_dir is specified") from e
            self._model = PeftModel.from_pretrained(self._model, adapter_dir, is_trainable=False)
            if adapter_merge:
                self._model = self._model.merge_and_unload()
            eval_logger.info(f"Loaded adapter from {adapter_dir}, merge: {adapter_merge}")

        # GP config knobs
        if reduce_layer is not None and hasattr(self._model.config, "reduce_layer"):
            self._model.config.reduce_layer = reduce_layer
        if min_remain_num is not None and hasattr(self._model.config, "min_remain_num"):
            self._model.config.min_remain_num = min_remain_num
        if max_remain_ratio is not None and hasattr(self._model.config, "max_remain_ratio"):
            self._model.config.max_remain_ratio = max_remain_ratio
        if base_score_pool_multiplier is not None and hasattr(self._model.config, "base_score_pool_multiplier"):
            self._model.config.base_score_pool_multiplier = float(base_score_pool_multiplier)
        if use_ref_masks and hasattr(self._model.config, "use_ref_masks"):
            self._model.config.use_ref_masks = True
        if use_zero_masks and hasattr(self._model.config, "use_zero_masks"):
            if use_ref_masks:
                eval_logger.warning("use_ref_masks=True while use_zero_masks=True; zero masks will take effect.")
            self._model.config.use_zero_masks = True
        if anchor_positions is not None and hasattr(self._model.config, "anchor_positions"):
            self._model.config.anchor_positions = anchor_positions

        self._tokenizer = tokenizer
        self.image_processor = image_processor
        self._max_length = context_len
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.conv_template = conv_template
        self.system_prompt = system_prompt

        # accelerate wrap
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            if self.device_map != "auto":
                self._model.to(self.device)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError(f"Loglikelihood is not implemented for {self.__class__.__name__}")

    def _prepare_images(self, visuals):
        """ChartQA / lmms-eval visuals may be str, PIL images, dict, or list[dict]."""
        images: List[Image.Image] = []
        for visual in visuals:
            if visual is None:
                continue

            # dict: {"image": "..."} / {"path": "..."} / {"img": "..."}
            if isinstance(visual, dict):
                visual = visual.get("image", visual.get("path", visual.get("img", None)))

            if isinstance(visual, str):
                if not os.path.exists(visual):
                    continue
                img = Image.open(visual).convert("RGB")
            elif isinstance(visual, Image.Image):
                img = visual.convert("RGB")
            else:
                continue

            images.append(img)
        return images

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=("cuda" if self.device_map == "auto" else self.device))

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk_idx, chunk in enumerate(chunks):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}

            eos_text = self.tokenizer.eos_token or ""
            until = [eos_text] if eos_text else []
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `until` to be str/list but got {type(until)}")

            # listify contexts
            contexts = list(contexts) if isinstance(contexts, tuple) else contexts

            # visuals
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            answers: List[str] = []

            for i, context in enumerate(contexts):
                visuals = visual_list[i]
                if visuals is None:
                    visuals = []
                if not isinstance(visuals, (list, tuple)):
                    visuals = [visuals]

                processed_images = self._prepare_images(visuals)

                query = context
                if "<image>" in query:
                    query = query.replace("<image>", "").strip()

                if processed_images:
                    if getattr(self.model.config, "mm_use_im_start_end", False):
                        query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
                    else:
                        query = DEFAULT_IMAGE_TOKEN + "\n" + query

                conv = conv_templates[self.conv_template].copy()
                if self.system_prompt:
                    if hasattr(conv, "system_message"):
                        conv.system_message = self.system_prompt
                    elif hasattr(conv, "system"):
                        conv.system = self.system_prompt
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_id = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )

                input_ids = torch.stack([input_id], dim=0)
                input_ids = self._to_device(input_ids)

                images = None
                image_sizes = None
                if processed_images:
                    img0 = processed_images[0]
                    img_tensor = process_images([img0], self.image_processor, self.model.config)[0]
                    target_dtype = getattr(self.model, "dtype", torch.float16)
                    images = torch.stack([img_tensor], dim=0).to(
                        device=("cuda" if self.device_map == "auto" else self.device),
                        dtype=target_dtype,
                    )
                    image_sizes = [img0.size]

                default_gen = {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                merged = {**default_gen, **gen_kwargs}
                temperature = float(merged.get("temperature") or 0.0)
                do_sample = True if temperature > 0 else False

                generation_config = GenerationConfig(
                    max_new_tokens=int(merged.get("max_new_tokens", 128)),
                    temperature=temperature,
                    top_p=(float(merged["top_p"]) if merged.get("top_p", None) is not None else None),
                    num_beams=int(merged.get("num_beams", 1)),
                    do_sample=do_sample,
                    pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                inputs = {
                    "input_ids": input_ids,
                    "images": images,
                    "image_sizes": image_sizes,
                    "attention_mask": torch.ones_like(input_ids),
                }

                # ===============================
                # Save soft token scores (GP)
                # ===============================
                if self.save_token_scores and self.token_scores_output_dir is not None:
                    os.makedirs(self.token_scores_output_dir, exist_ok=True)

                    with torch.inference_mode():
                        out = self.model(
                            **inputs,
                            do_selection=True,
                            return_dict=True,
                        )

                    if hasattr(out, "image_token_mask_logits") and out.image_token_mask_logits is not None:
                        soft_masks = out.image_token_mask_logits

                        # Support both tensors and list-like values.
                        if isinstance(soft_masks, torch.Tensor):
                            batch_iter = range(soft_masks.shape[0])
                            get_one = lambda b: soft_masks[b]
                        else:
                            batch_iter = range(len(soft_masks))
                            get_one = lambda b: soft_masks[b]

                        for b in batch_iter:
                            try:
                                one_image_masks = get_one(b)

                                # Some implementations return list[step]; use the final step.
                                if isinstance(one_image_masks, list):
                                    if len(one_image_masks) == 0:
                                        continue
                                    mask_tensor = one_image_masks[-1]
                                elif isinstance(one_image_masks, torch.Tensor):
                                    mask_tensor = one_image_masks
                                else:
                                    continue

                                mask_tensor = mask_tensor.squeeze()
                                soft_prob = mask_tensor.sigmoid().detach().cpu().float()
                                mask_flat = soft_prob.reshape(-1).tolist()

                                sample_id = int(doc_id[i])

                                out_path = os.path.join(
                                    self.token_scores_output_dir,
                                    f"{task}_{split}_{sample_id}.json"
                                )

                                data = {
                                    "task": task,
                                    "split": split,
                                    "doc_id": sample_id,
                                    "num_tokens": len(mask_flat),
                                    "mask_flat": mask_flat,
                                    "mask_type": "soft_sigmoid_from_image_token_mask_logits",
                                    "score_stats": {
                                        "min": float(min(mask_flat)) if mask_flat else 0.0,
                                        "max": float(max(mask_flat)) if mask_flat else 0.0,
                                        "mean": float(sum(mask_flat) / len(mask_flat)) if mask_flat else 0.0,
                                    },
                                }
                                with open(out_path, "w") as f:
                                    json.dump(data, f, indent=2)

                            except Exception as e:
                                eval_logger.error(
                                    f"[TokenScore] Error saving token scores for sample {int(doc_id[i])}: {e}"
                                )

                # ===============================
                # Original GP generation
                # ===============================
                with torch.inference_mode():
                    generate_ids = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        do_selection=True,
                        use_cache=self.use_cache,
                    )

                if generate_ids.ndim == 1:
                    generate_ids = generate_ids.unsqueeze(0)

                if generate_ids.shape[1] <= input_ids.shape[1]:
                    gen_part = generate_ids
                else:
                    gen_part = generate_ids[:, input_ids.shape[1]:]

                if gen_part.numel() == 0:
                    answer_text = ""
                else:
                    answer_text = self.tokenizer.batch_decode(
                        gen_part,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )[0].strip()

                for term in until:
                    if term:
                        answer_text = answer_text.split(term)[0]

                answers.append(answer_text.strip())

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


__all__ = [
    "LlavaLlamaGP",
]
