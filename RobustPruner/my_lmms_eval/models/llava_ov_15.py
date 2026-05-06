import json
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

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


class LlavaOV15(lmms):
    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        max_length: int = 2048,
        timing_output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Unexpected kwargs in LlavaOV15: {kwargs}")
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for LLaVA-OneVision-1.5 visual preprocessing")

        valid_attn_impls = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_impls:
            raise ValueError(f"attn_implementation must be in {valid_attn_impls}, got {attn_implementation}")
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

        self.processor = LLaVAOneVision1_5_GP_Processor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"

        self._config = self.model.config
        self._max_length = int(max_length)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
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

        self.timing_log_path = None
        if timing_output_dir is not None:
            os.makedirs(timing_output_dir, exist_ok=True)
            self.timing_log_path = os.path.join(timing_output_dir, f"timing_records_rank{self.rank}.log")

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

    def _append_timing_record(self, record):
        if self.timing_log_path is None:
            return
        with open(self.timing_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")

    def _self_attn_flops(self, seq_len, hidden_size, num_heads, num_key_value_heads, head_dim):
        q_proj_flops = 2 * seq_len * hidden_size * hidden_size
        k_proj_flops = 2 * seq_len * hidden_size * num_key_value_heads * head_dim
        v_proj_flops = 2 * seq_len * hidden_size * num_key_value_heads * head_dim
        attn_scores_flops = 2 * seq_len * seq_len * hidden_size
        weight_sum_flops = 2 * seq_len * seq_len * hidden_size
        out_proj_flops = 2 * seq_len * hidden_size * hidden_size
        return (
            q_proj_flops
            + k_proj_flops
            + v_proj_flops
            + attn_scores_flops
            + weight_sum_flops
            + out_proj_flops
        )

    def _self_attn_flops_decoding(self, seq_len, hidden_size, num_key_value_heads, head_dim):
        q_proj_flops = 2 * hidden_size * hidden_size
        k_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim
        v_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim
        attn_scores_flops = 2 * seq_len * hidden_size
        weight_sum_flops = 2 * seq_len * hidden_size
        out_proj_flops = 2 * hidden_size * hidden_size
        return (
            q_proj_flops
            + k_proj_flops
            + v_proj_flops
            + attn_scores_flops
            + weight_sum_flops
            + out_proj_flops
        )

    def _mlp_flops(self, seq_len, hidden_size, intermediate_size):
        gate_proj_flops = 2 * seq_len * hidden_size * intermediate_size
        up_proj_flops = 2 * seq_len * hidden_size * intermediate_size
        down_proj_flops = 2 * seq_len * intermediate_size * hidden_size
        return gate_proj_flops + up_proj_flops + down_proj_flops

    def _mlp_flops_decoding(self, hidden_size, intermediate_size):
        gate_proj_flops = 2 * hidden_size * intermediate_size
        up_proj_flops = 2 * hidden_size * intermediate_size
        down_proj_flops = 2 * intermediate_size * hidden_size
        return gate_proj_flops + up_proj_flops + down_proj_flops

    def _prefill_flops(self, input_seq_len):
        config = self.model.config.text_config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size
        num_layers = config.num_hidden_layers
        attn_flops = self._self_attn_flops(
            input_seq_len,
            hidden_size,
            num_heads,
            num_key_value_heads,
            head_dim,
        )
        mlp_flops = self._mlp_flops(input_seq_len, hidden_size, intermediate_size)
        return num_layers * (attn_flops + mlp_flops)

    def _decode_flops(self, input_seq_len, generated_token_count):
        config = self.model.config.text_config
        hidden_size = config.hidden_size
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size
        num_layers = config.num_hidden_layers
        total = 0
        for i in range(generated_token_count):
            cache_len = input_seq_len + i
            attn_flops = self._self_attn_flops_decoding(
                cache_len,
                hidden_size,
                num_key_value_heads,
                head_dim,
            )
            mlp_flops = self._mlp_flops_decoding(hidden_size, intermediate_size)
            total += num_layers * (attn_flops + mlp_flops)
        return total

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for LlavaOV15")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        eval_time_enabled = os.environ.get("EVAL_TIME", "").lower() == "true"

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected until to be str or list, got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            contexts = list(contexts) if isinstance(contexts, tuple) else contexts
            cleaned_contexts = []
            batched_messages = []
            for idx, context in enumerate(contexts):
                context = context.replace("<image>", "")
                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt

                processed_visuals = []
                for visual in visual_list[idx]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append(
                            {"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append({"type": "image", "image": visual.convert("RGB")})

                if not self.interleave_visuals:
                    user_content = processed_visuals + [{"type": "text", "text": context}]
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    text_parts = re.split(r"<image \d+>", context)
                    user_content = []
                    if text_parts[0]:
                        user_content.append({"type": "text", "text": text_parts[0]})
                    for image_idx, placeholder in enumerate(image_placeholders):
                        visual_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        visual_idx = min(visual_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and visual_idx < len(processed_visuals):
                            user_content.append(processed_visuals[visual_idx])
                        if image_idx + 1 < len(text_parts) and text_parts[image_idx + 1]:
                            user_content.append({"type": "text", "text": text_parts[image_idx + 1]})

                message.append({"role": "user", "content": user_content})
                cleaned_contexts.append(context)
                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            input_token_count = int(inputs.input_ids.shape[1])
            image_token_id = getattr(self.model.config, "image_token_id", None)
            video_token_id = getattr(self.model.config, "video_token_id", None)
            visual_token_num = 0
            if image_token_id is not None:
                visual_token_num += int((inputs.input_ids == image_token_id).sum().item())
            if video_token_id is not None:
                visual_token_num += int((inputs.input_ids == video_token_id).sum().item())

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            do_sample = bool(current_gen_kwargs.get("temperature", 0) and current_gen_kwargs["temperature"] > 0)
            gen_args = {
                **inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "num_beams": int(current_gen_kwargs["num_beams"]),
                "max_new_tokens": int(current_gen_kwargs["max_new_tokens"]),
                "use_cache": self.use_cache,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_args["temperature"] = float(current_gen_kwargs.get("temperature", 1.0))
                if current_gen_kwargs.get("top_p", None) is not None:
                    gen_args["top_p"] = float(current_gen_kwargs["top_p"])

            with torch.inference_mode():
                cont = self.model.generate(**gen_args)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]

            if eval_time_enabled:
                generated_token_count = int(
                    sum(out_ids.shape[0] for out_ids in generated_ids_trimmed) / max(len(generated_ids_trimmed), 1)
                )
                reduced_input_token_count = input_token_count
                reduced_visual_token_num = visual_token_num
                kv_cache_length = reduced_input_token_count + generated_token_count
                dtype_bytes = next(self.model.parameters()).element_size()
                text_config = self.model.config.text_config
                kv_cache_mb = (
                    2
                    * text_config.num_hidden_layers
                    * text_config.num_key_value_heads
                    * text_config.head_dim
                    * kv_cache_length
                    * dtype_bytes
                    / (1024 ** 2)
                )
                max_memory_bytes = None
                if self.device.type == "cuda":
                    max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                self._append_timing_record(
                    {
                        "visual_token_num": visual_token_num,
                        "input_token_num": input_token_count,
                        "reduced_visual_token_num": reduced_visual_token_num,
                        "reduced_input_token_num": reduced_input_token_count,
                        "generated_token_num": generated_token_count,
                        "prefill_flops": self._prefill_flops(input_token_count),
                        "decode_flops": self._decode_flops(input_token_count, generated_token_count),
                        "kv_cache_length": kv_cache_length,
                        "kv_cache_size": kv_cache_mb,
                        "after_generation_memory": max_memory_bytes,
                    }
                )

            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for idx, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[idx] = ans

            for ans, context in zip(answers, cleaned_contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


__all__ = ["LlavaOV15"]