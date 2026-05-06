import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers_rp.models.qwen2_5_vl import (
    Qwen2_5_VL_GP_ForConditionalGeneration,
    Qwen2_5_VL_GP_Processor,
)
from peft import PeftModel


from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")




class Qwen2_5_VL_GP(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        new_modules_dir: str = "",
        adapter_dir: Optional[str] = None,
        adapter_merge: bool = False,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 2048 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        reduce_layer: Optional[int] = None,
        min_remain_num: Optional[int] = None,
        max_remain_ratio: Optional[float] = None,
        base_score_pool_multiplier: Optional[float] = None,
        token_scores_output_dir: Optional[str] = None,
        save_token_scores: bool = False,
        text_perturb: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.text_perturb = text_perturb
    
        if kwargs:
            eval_logger.warning(f"Unexpected kwargs in Qwen2_5_VL_GP: {kwargs}")
        
        assert not interleave_visuals, "Interleave visuals is not supported yet."

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        
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
        }

        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VL_GP_ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        
        self._model.load_new_modules(new_modules_dir)
        
        if adapter_dir is not None:
            self._model = PeftModel.from_pretrained(self._model, adapter_dir, is_trainable=False)
            if adapter_merge:
                self._model = self._model.merge_and_unload()
            print(f"Loaded adapter from {adapter_dir}, merge: {adapter_merge}")
        self._model.eval()
        
        if min_remain_num is not None:
            self._model.config.min_remain_num = min_remain_num
        if max_remain_ratio is not None:
            self._model.config.max_remain_ratio = max_remain_ratio
        if base_score_pool_multiplier is not None:
            self._model.config.base_score_pool_multiplier = float(base_score_pool_multiplier)
        
        if reduce_layer is not None:
            self._model.config.reduce_layer = reduce_layer

        self.max_num_frames = max_num_frames
        
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
            
        processor_kwargs = {
            "padding_side": "left",
        }
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels

        self.processor = Qwen2_5_VL_GP_Processor.from_pretrained(pretrained, **processor_kwargs)
        self._tokenizer = self.processor.tokenizer
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        
        self._config = self.model.config
        self._max_length = 2048
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
        raise NotImplementedError(f"Loglikelihood is not implemented for {self.__class__.__name__}")
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
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

            # Text perturbation (if enabled)
            if self.text_perturb:
                try:
                    import importlib.util
                    from pathlib import Path

                    # cache: avoid re-importing every batch
                    if not hasattr(self, "_task_doc_to_text_cache"):
                        self._task_doc_to_text_cache = {}

                    def _get_doc_to_text_fn(task_name: str):
                        """
                        Load doc_to_text function for the current task.
                        Priority:
                        1) lmms_eval/tasks/<task>/<task>_utils.py  -> <task>_doc_to_text
                        2) lmms_eval/tasks/<task>/utils.py        -> <task>_doc_to_text
                        3) lmms_eval/tasks/<task>/<task>_utils.py  -> doc_to_text (rare)
                        4) lmms_eval/tasks/<task>/utils.py        -> doc_to_text
                        """
                        if task_name in self._task_doc_to_text_cache:
                            return self._task_doc_to_text_cache[task_name]

                        tasks_root = Path("/home/fjh/VisionSelector/lmms-eval/lmms_eval/tasks")
                        task_dir = tasks_root / task_name

                        candidates = [
                            task_dir / f"{task_name}_utils.py",
                            task_dir / "utils.py",
                        ]

                        for p in candidates:
                            if p.exists():
                                spec = importlib.util.spec_from_file_location(f"{task_name}_utils_dyn", str(p))
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)

                                # common naming: gqa_doc_to_text / chartqa_doc_to_text / ...
                                fn = getattr(mod, f"{task_name}_doc_to_text", None)
                                if fn is None:
                                    fn = getattr(mod, "doc_to_text", None)

                                if fn is None:
                                    raise AttributeError(f"No doc_to_text found in {p}")

                                self._task_doc_to_text_cache[task_name] = fn
                                return fn

                        raise FileNotFoundError(f"Cannot find utils for task={task_name}. Tried: {candidates}")

                    task_doc_to_text = _get_doc_to_text_fn(task)

                    new_contexts = []
                    for i, original_context in enumerate(contexts):
                        doc = self.task_dict[task][split][doc_id[i]]
                        try:
                            # IMPORTANT: pass through existing kwargs if you have them
                            lmms_eval_specific_kwargs = {"text_perturb": True}
                            perturbed_context = task_doc_to_text(doc, lmms_eval_specific_kwargs)
                            new_contexts.append(perturbed_context)
                        except Exception:
                            new_contexts.append(original_context)

                    contexts = new_contexts

                except Exception as e:
                    # if you want to debug, uncomment:
                    # if self.rank == 0: eval_logger.warning(f"[TextPerturb] failed: {e}")
                    pass

            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

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
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
                        processed_visuals.append({"type": "video", "video": visual})
                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:
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
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id
            
            self.model.reset_image_tokens_cache()
            if isinstance(self.model, PeftModel):
                with self.model.disable_adapter():
                    output = self.model(**inputs, return_dict=True)
                    inputs["use_ref_masks"] = True
                    inputs["ref_token_masks"] = [
                        one_image_token_masks[-1:].sigmoid()
                        for one_image_token_masks in output.image_token_mask_logits
                    ]
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

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
    "Qwen2_5_VL_GP",
]