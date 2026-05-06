import json
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

_this_dir = os.path.dirname(__file__)
parent_path = os.path.abspath(os.path.join(_this_dir, os.pardir))
grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
for _p in (parent_path, grandparent_path):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

from llava_ov_rp import (  # noqa: E402
    LLaVAOneVision1_5_ForConditionalGeneration,
    LLaVAOneVision1_5_GP_Processor,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
    eval_logger.warning("Failed to import qwen_vl_utils; please install it via `pip install qwen-vl-utils`")


class LlavaOV15GP(lmms):
    """
    LLaVA-OneVision-1.5 GlimpsePrune wrapper for lmms-eval.

    This follows the upstream LLaVA-OV-1.5 chat-template/processor path and
    calls the GP model with do_selection=True during generation.
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        new_modules_dir: str = "",
        adapter_dir: Optional[str] = None,
        adapter_merge: bool = False,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_length: int = 2048,
        reduce_layer: Optional[int] = None,
        min_remain_num: Optional[int] = None,
        max_remain_ratio: Optional[float] = None,
        base_score_pool_multiplier: Optional[float] = None,
        use_ref_masks: bool = False,
        use_zero_masks: bool = False,
        anchor_positions: Optional[Tuple[int, ...]] = None,
        token_scores_output_dir: Optional[str] = None,
        save_token_scores: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Unexpected kwargs in LlavaOV15GP: {kwargs}")
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for LLaVA-OneVision-1.5 visual preprocessing")
        if max_image_size is not None and not use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.token_scores_output_dir = token_scores_output_dir
        self.save_token_scores = save_token_scores

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
            pretrained,
            **model_kwargs,
        ).eval()
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = 1.0
            self._model.generation_config.top_p = 1.0

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

        self._apply_gp_config(
            reduce_layer=reduce_layer,
            min_remain_num=min_remain_num,
            max_remain_ratio=max_remain_ratio,
            base_score_pool_multiplier=base_score_pool_multiplier,
            use_ref_masks=use_ref_masks,
            use_zero_masks=use_zero_masks,
            anchor_positions=anchor_positions,
        )

        processor_kwargs = {"trust_remote_code": True}
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        self.processor = LLaVAOneVision1_5_GP_Processor.from_pretrained(pretrained, **processor_kwargs)
        self._tokenizer = self.processor.tokenizer
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"

        self._config = self.model.config
        self._max_length = int(max_length)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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

    def _apply_gp_config(
        self,
        reduce_layer: Optional[int],
        min_remain_num: Optional[int],
        max_remain_ratio: Optional[float],
        base_score_pool_multiplier: Optional[float],
        use_ref_masks: bool,
        use_zero_masks: bool,
        anchor_positions: Optional[Tuple[int, ...]],
    ) -> None:
        config = self._model.config
        if reduce_layer is not None and hasattr(config, "reduce_layer"):
            config.reduce_layer = int(reduce_layer)
        if min_remain_num is not None and hasattr(config, "min_remain_num"):
            config.min_remain_num = int(min_remain_num)
        if max_remain_ratio is not None and hasattr(config, "max_remain_ratio"):
            config.max_remain_ratio = float(max_remain_ratio)
        if base_score_pool_multiplier is not None and hasattr(config, "base_score_pool_multiplier"):
            config.base_score_pool_multiplier = float(base_score_pool_multiplier)
        if use_ref_masks and hasattr(config, "use_ref_masks"):
            config.use_ref_masks = True
        if use_zero_masks and hasattr(config, "use_zero_masks"):
            if use_ref_masks:
                eval_logger.warning("use_ref_masks=True while use_zero_masks=True; zero masks will take effect.")
            config.use_zero_masks = True
        if anchor_positions is not None and hasattr(config, "anchor_positions"):
            config.anchor_positions = anchor_positions

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(f"Loglikelihood is not implemented for {self.__class__.__name__}")

    def _normalize_visuals(self, visuals):
        if visuals is None:
            return []
        if not isinstance(visuals, (list, tuple)):
            visuals = [visuals]
        return [visual for visual in visuals if visual is not None]

    def _build_visual_content(self, visuals):
        processed_visuals = []
        for visual in self._normalize_visuals(visuals):
            if isinstance(visual, dict):
                visual = visual.get("image", visual.get("path", visual.get("video", visual.get("img", None))))
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                if self.max_pixels is None and self.min_pixels is None:
                    processed_visuals.append({"type": "video", "video": visual})
                else:
                    processed_visuals.append(
                        {
                            "type": "video",
                            "video": visual,
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels,
                        }
                    )
            elif isinstance(visual, str):
                if os.path.exists(visual):
                    processed_visuals.append({"type": "image", "image": Image.open(visual).convert("RGB")})
            elif isinstance(visual, Image.Image):
                processed_visuals.append({"type": "image", "image": visual.convert("RGB")})
        return processed_visuals

    def _make_message(self, context: str, visuals):
        context = context.replace("<image>", "").strip()
        if self.reasoning_prompt:
            context = context + self.reasoning_prompt

        message = []
        if self.system_prompt:
            message.append({"role": "system", "content": self.system_prompt})

        processed_visuals = self._build_visual_content(visuals)
        if self.interleave_visuals:
            image_placeholders = re.findall(r"<image \d+>", context)
            content_parts = []
            text_parts = re.split(r"<image \d+>", context)
            if text_parts and text_parts[0]:
                content_parts.append({"type": "text", "text": text_parts[0]})
            for idx, placeholder in enumerate(image_placeholders):
                img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                if processed_visuals and image_idx < len(processed_visuals):
                    content_parts.append(processed_visuals[image_idx])
                if idx + 1 < len(text_parts) and text_parts[idx + 1]:
                    content_parts.append({"type": "text", "text": text_parts[idx + 1]})
        else:
            content_parts = processed_visuals + [{"type": "text", "text": context}]

        message.append({"role": "user", "content": content_parts})
        return message, context

    def _to_device(self, inputs):
        if self.device_map == "auto":
            return inputs.to("cuda")
        return inputs.to(self.device)

    def _save_token_scores(self, inputs, task, split, doc_id):
        if not self.save_token_scores or self.token_scores_output_dir is None:
            return
        os.makedirs(self.token_scores_output_dir, exist_ok=True)
        with torch.inference_mode():
            out = self.model(
                **inputs,
                do_selection=True,
                delay_selection=True,
                use_cache=False,
                return_dict=True,
            )
        soft_masks = getattr(out, "image_token_mask_logits", None)
        if soft_masks is None:
            return
        if isinstance(soft_masks, torch.Tensor):
            iterator = [soft_masks[i] for i in range(soft_masks.shape[0])]
        else:
            iterator = list(soft_masks)
        for i, one_mask in enumerate(iterator):
            try:
                if isinstance(one_mask, list):
                    if len(one_mask) == 0:
                        continue
                    one_mask = one_mask[-1]
                if not isinstance(one_mask, torch.Tensor):
                    continue
                mask_flat = one_mask.squeeze().sigmoid().detach().cpu().float().reshape(-1).tolist()
                sample_id = int(doc_id[min(i, len(doc_id) - 1)])
                out_path = os.path.join(self.token_scores_output_dir, f"{task}_{split}_{sample_id}.json")
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
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                eval_logger.error(f"[TokenScore] Error saving token scores for sample {i}: {e}")
        if hasattr(self.model, "reset_image_tokens_cache"):
            self.model.reset_image_tokens_cache()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            until = gen_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `until` to be str/list but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            contexts = list(contexts) if isinstance(contexts, tuple) else contexts

            batched_messages = []
            cleaned_contexts = []
            for context, visuals in zip(contexts, visual_list):
                message, clean_context = self._make_message(context, visuals)
                batched_messages.append(message)
                cleaned_contexts.append(clean_context)

            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batched_messages
            ]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                for idx, video in enumerate(video_inputs):
                    total_frames = video.shape[0]
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    video_inputs[idx] = video[indices]

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            temperature_value = current_gen_kwargs.get("temperature", None)
            temperature = float(temperature_value) if temperature_value is not None else 0.0
            do_sample = temperature > 0
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            if hasattr(self.model, "reset_image_tokens_cache"):
                self.model.reset_image_tokens_cache()
            self._save_token_scores(inputs, task, split, doc_id)

            gen_args = {
                **inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "do_sample": do_sample,
                "num_beams": int(current_gen_kwargs["num_beams"]),
                "max_new_tokens": int(current_gen_kwargs["max_new_tokens"]),
                "use_cache": self.use_cache,
                "do_selection": True,
                "delay_selection": False,
            }
            if do_sample:
                gen_args["temperature"] = temperature
                if current_gen_kwargs.get("top_p", None) is not None:
                    gen_args["top_p"] = float(current_gen_kwargs["top_p"])

            with torch.inference_mode():
                cont = self.model.generate(**gen_args)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[i] = ans.strip()

            if hasattr(self.model, "reset_image_tokens_cache"):
                self.model.reset_image_tokens_cache()

            for ans, context in zip(answers, cleaned_contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


__all__ = [
    "LlavaOV15GP",
]