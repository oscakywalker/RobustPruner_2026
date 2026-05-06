import importlib.util
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
# from transformers import (
#     AutoProcessor,
#     AutoTokenizer,
#     Qwen2_5_VLForConditionalGeneration,
# )
from transformers import AutoProcessor, AutoTokenizer
# from qwen25vl import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from qwen25vl import Qwen2_5_VLForConditionalGeneration
from token_compression.visionzip_official import Qwen2_5_VLForConditionalGeneration_VisionZip
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
from token_compression.dynamic_model import Qwen2_5_VLForConditionalGeneration_Dynamic
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")
import argparse


@register_model("qwen2_5_vl_with_token_compression")
class Qwen2_5_VL_with_token_compression(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        # assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if kwargs != {}:
            defined_params = {
            "pretrained", "device", "batch_size", "attn_implementation",
            "use_flash_attention_2", "device_map", "max_pixels", "use_cache",
            "min_pixels", "max_num_frames", "method"
            }
            
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in defined_params}

            self.args = argparse.Namespace(**extra_kwargs)
        else:
            self.args = argparse.Namespace()
        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.method = method
        if self.method == 'visionzip_official':
            print('using visionzip official')
            self._model = Qwen2_5_VLForConditionalGeneration_VisionZip.from_pretrained(pretrained, **model_kwargs).eval() 
            self._model.budget = kwargs['budgets']
        elif self.method == 'selector':
            print('using selector')
            self._model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(pretrained, **model_kwargs).eval()
            self._model.visual.budgets = kwargs['budgets']
        elif self.method == 'dynamic':
            print('using dynamic')
            self._model = Qwen2_5_VLForConditionalGeneration_Dynamic.from_pretrained(pretrained, **model_kwargs).eval()
            self._model.model.budgets = kwargs['budgets']
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval() 
        
        
        
        # not change
        if self.method is not None:
            from token_compression.monkeypatch import replace_qwen25vl
            replace_qwen25vl(self.args,self._model,self.method.lower())

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.text_perturb = self._to_bool(kwargs.get("text_perturb", False))
        self.save_token_scores = self._to_bool(kwargs.get("save_token_scores", False))
        self.token_scores_output_dir = kwargs.get("token_scores_output_dir")
        self.eval_time = self._to_bool(kwargs.get("eval_time", False)) and self._device.type == "cuda"
        self.disable_tqdm = os.environ.get("LMMS_DISABLE_TQDM", "").lower() == "true"
        self.timing_output_dir = kwargs.get("timing_output_dir")
        self.timing_log_path = None

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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

        if self.timing_output_dir is not None:
            os.makedirs(self.timing_output_dir, exist_ok=True)
            self.timing_log_path = os.path.join(self.timing_output_dir, f"timing_records_rank{self.rank}.log")

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _maybe_to_list(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    def _resolve_token_scores_dir(self):
        if self.token_scores_output_dir:
            return self.token_scores_output_dir
        if hasattr(self.cache_hook, "log_samples_path") and self.cache_hook.log_samples_path:
            return os.path.join(os.path.dirname(self.cache_hook.log_samples_path), "token_scores")
        return None

    def _append_timing_record(self, record):
        if self.timing_log_path is None:
            return
        with open(self.timing_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")

    def _build_token_score_payload(self, task, split, doc_id):
        model = self.model
        visual = getattr(model, "visual", None)

        if self.method == "cdpruner":
            score_tensor = getattr(model, "cdpruner_last_mask_scores", None)
            if score_tensor is None and visual is not None:
                score_tensor = getattr(visual, "cdpruner_last_mask_scores", None)
            if score_tensor is None:
                return None

            selected_indices = getattr(model, "cdpruner_last_selected_indices", None)
            if selected_indices is None and visual is not None:
                selected_indices = getattr(visual, "cdpruner_last_selected_indices", None)
            num_tokens = getattr(model, "cdpruner_last_visual_token_num", None)
            if num_tokens is None and visual is not None:
                num_tokens = getattr(visual, "cdpruner_last_visual_token_num", None)
            keep_num = getattr(model, "cdpruner_last_keep_num", None)
            if keep_num is None and visual is not None:
                keep_num = getattr(visual, "cdpruner_last_keep_num", None)
            relevance_scores = getattr(model, "cdpruner_last_relevance_scores", None)
            if relevance_scores is None and visual is not None:
                relevance_scores = getattr(visual, "cdpruner_last_relevance_scores", None)

            if score_tensor is None or selected_indices is None or num_tokens is None:
                return None

            num_tokens = int(num_tokens)
            hard_mask = [0] * num_tokens
            selected_indices_list = self._maybe_to_list(selected_indices) or []
            for idx in selected_indices_list:
                if 0 <= int(idx) < num_tokens:
                    hard_mask[int(idx)] = 1

            mask_flat = self._maybe_to_list(score_tensor)
            grid_thw = getattr(model, "cdpruner_last_grid_thw", None)
            if grid_thw is None and visual is not None:
                grid_thw = getattr(visual, "cdpruner_last_grid_thw", None)

            payload = {
                "task": task,
                "split": split,
                "doc_id": doc_id,
                "num_tokens": num_tokens,
                "mask_flat": mask_flat,
                "hard_mask_flat": hard_mask,
            }
            return payload

        return None

    def _save_token_scores(self, task, split, doc_id):
        out_dir = self._resolve_token_scores_dir()
        if not out_dir:
            return
        payload = self._build_token_score_payload(task, split, doc_id)
        if payload is None:
            return

        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{task}_{split}_{doc_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        eval_logger.info("[TokenScores] Saved token score -> {}", save_path)

    def _get_task_doc_to_text_fn(self, task_name):
        if not hasattr(self, "_task_doc_to_text_cache"):
            self._task_doc_to_text_cache = {}
        if task_name in self._task_doc_to_text_cache:
            return self._task_doc_to_text_cache[task_name]

        tasks_root = Path(utils.__file__).resolve().parent / "tasks"
        task_dir = tasks_root / task_name
        candidates = [
            task_dir / f"{task_name}_utils.py",
            task_dir / "utils.py",
        ]

        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location(f"{task_name}_utils_dyn", str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            fn = getattr(mod, f"{task_name}_doc_to_text", None)
            if fn is None:
                fn = getattr(mod, "doc_to_text", None)
            if fn is None:
                raise AttributeError(f"No doc_to_text found in {path}")

            self._task_doc_to_text_cache[task_name] = fn
            return fn

        raise FileNotFoundError(f"Cannot find utils for task={task_name}. Tried: {candidates}")

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0 or self.disable_tqdm), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        # print(len(requests))
        cal_num = 0
        input_len = 0
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            if self.text_perturb:
                try:
                    task_doc_to_text = self._get_task_doc_to_text_fn(task)
                    contexts = [
                        task_doc_to_text(self.task_dict[task][split][doc_id[i]], {"text_perturb": True})
                        for i in range(len(contexts))
                    ]
                except Exception as e:
                    if self.rank == 0:
                        eval_logger.warning("[TextPerturb] failed for task={} split={}: {}", task, split, e)

            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                        processed_visuals.append({"type": "image", "image": visual.convert("RGB"), "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id
            input_token_count = int(inputs.input_ids.shape[1])
            visual_token_num = 0
            prefill_time_ms = None

            if self.eval_time:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                video_token_id = getattr(self.model.config, "video_token_id", None)
                if image_token_id is not None:
                    visual_token_num += int((inputs.input_ids == image_token_id).sum().item())
                if video_token_id is not None:
                    visual_token_num += int((inputs.input_ids == video_token_id).sum().item())

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            if self.eval_time:
                with torch.inference_mode():
                    prefill_start = torch.cuda.Event(enable_timing=True)
                    prefill_end = torch.cuda.Event(enable_timing=True)
                    prefill_start.record()
                    _ = self.model(**inputs, return_dict=True)
                    prefill_end.record()
                    torch.cuda.synchronize()
                prefill_time_ms = prefill_start.elapsed_time(prefill_end)

                torch.cuda.reset_peak_memory_stats(self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                end.record()
                torch.cuda.synchronize()
                latency_time_ms = start.elapsed_time(end)
                max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                latency_time_ms = None
                max_memory_bytes = None

            input_len += inputs.input_ids.shape[1]

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            for i, gen_ids in enumerate(generated_ids_trimmed):
                cal_num += len(gen_ids)
                # print(f"Sample {i}: Generated {len(gen_ids)} tokens")
            if self.eval_time:
                generated_token_count = int(
                    sum(out_ids.shape[0] for out_ids in generated_ids_trimmed) / max(len(generated_ids_trimmed), 1)
                )
                prefill_time_per_token_ms = None if input_token_count == 0 or prefill_time_ms is None else prefill_time_ms / input_token_count
                decode_time_per_token_ms = None if generated_token_count == 0 or latency_time_ms is None else latency_time_ms / generated_token_count
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                dtype_bytes = next(self.model.parameters()).element_size()
                kv_cache_mb = (
                    2
                    * self.model.config.num_hidden_layers
                    * self.model.config.num_key_value_heads
                    * head_dim
                    * (input_token_count + generated_token_count)
                    * dtype_bytes
                    / (1024 ** 2)
                )
                self._append_timing_record(
                    {
                        "visual_token_num": visual_token_num,
                        "input_token_num": input_token_count,
                        "generation_prefill_time": prefill_time_ms,
                        "prefill_time_per_token": prefill_time_per_token_ms,
                        "generated_token_num": generated_token_count,
                        "generation_latency": latency_time_ms,
                        "decode_time_per_token": decode_time_per_token_ms,
                        "kv_cache_size": kv_cache_mb,
                        "after_generation_memory": max_memory_bytes,
                    }
                )
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            # if task.lower().strip() != "mmvet" and task.lower().strip() != "llava_wilder_small" and task.lower().strip() != "mathvision_reason_testmini":
            #     for i, ans in enumerate(answers):
            #         for term in until:
            #             if len(term) > 0:
            #                 ans = ans.split(term)[0]
            #         answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            if self.save_token_scores:
                for one_doc_id in doc_id:
                    doc_id_val = one_doc_id[0] if isinstance(one_doc_id, tuple) else one_doc_id
                    try:
                        self._save_token_scores(task, split, doc_id_val)
                    except Exception as e:
                        eval_logger.error(
                            "Token score save failed for task={} split={} doc_id={}: {}",
                            task,
                            split,
                            doc_id_val,
                            e,
                        )
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        avg_input_token = input_len / len(requests)
        print(f"Generated Iutput Average {avg_input_token} tokens ===========")
        avg_output_token = cal_num / len(requests)
        print(f"Generated Output Average {avg_output_token} tokens")
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
