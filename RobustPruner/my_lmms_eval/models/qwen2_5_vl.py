import base64
import json
import os
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


class Qwen2_5_VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
        save_token_scores: bool = False,
        token_scores_output_dir: Optional[str] = None,
        timing_output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Unexpected kwargs in Qwen2_5_VL: {kwargs}")

        valid_attn_impls = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_impls:
            raise ValueError(f"attn_implementation must be in {valid_attn_impls}, got {attn_implementation}")
        self.attn_implementation = attn_implementation

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        self.save_token_scores = bool(save_token_scores)
        self.token_scores_output_dir = token_scores_output_dir
        self.disable_tqdm = os.environ.get("LMMS_DISABLE_TQDM", "").lower() == "true"
        self.timing_log_path = None

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {"torch_dtype": "auto", "device_map": self.device_map}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
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

        if timing_output_dir is not None:
            os.makedirs(timing_output_dir, exist_ok=True)
            self.timing_log_path = os.path.join(timing_output_dir, f"timing_records_rank{self.rank}.log")

    @property
    def config(self):
        return self._config

    def _append_timing_record(self, record):
        if self.timing_log_path is None:
            return
        with open(self.timing_log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")

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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_5_VL")

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
        config = self.model.config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = hidden_size // num_heads
        intermediate_size = config.intermediate_size
        num_layers = config.num_hidden_layers
        attn_flops = self._self_attn_flops(
            input_seq_len, hidden_size, num_heads, num_key_value_heads, head_dim
        )
        mlp_flops = self._mlp_flops(input_seq_len, hidden_size, intermediate_size)
        return num_layers * (attn_flops + mlp_flops)

    def _decode_flops(self, input_seq_len, generated_token_count):
        config = self.model.config
        hidden_size = config.hidden_size
        num_key_value_heads = config.num_key_value_heads
        head_dim = hidden_size // config.num_attention_heads
        intermediate_size = config.intermediate_size
        num_layers = config.num_hidden_layers
        all_flops = 0
        for i in range(generated_token_count):
            cache_len = input_seq_len + i
            attn_flops = self._self_attn_flops_decoding(
                cache_len, hidden_size, num_key_value_heads, head_dim
            )
            mlp_flops = self._mlp_flops_decoding(hidden_size, intermediate_size)
            all_flops += (attn_flops + mlp_flops) * num_layers
        return all_flops

    def _save_token_scores(
        self,
        inputs,
        cont,
        generated_ids_trimmed,
        doc_id,
        task,
        split,
    ):
        if not self.save_token_scores or not self.token_scores_output_dir:
            return
        os.makedirs(self.token_scores_output_dir, exist_ok=True)

        try:
            fwd_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            fwd_inputs["input_ids"] = cont
            fwd_inputs["attention_mask"] = torch.ones_like(cont, dtype=inputs["attention_mask"].dtype, device=cont.device)
            with torch.inference_mode():
                fwd = self.model(
                    **fwd_inputs,
                    return_dict=True,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                )
            llm_layers = self.model.model.layers
            num_layers = len(llm_layers)
            # From last decoding layer, step back by 2 layers each time.
            sampled_layer_indices = list(range(num_layers - 1, -1, -2))
            sampled_layer_indices_1based = [int(i + 1) for i in sampled_layer_indices]

            image_token_id = getattr(self.model.config, "image_token_id", None)
            video_token_id = getattr(self.model.config, "video_token_id", None)
            prompt_len = inputs["input_ids"].shape[1]

            for b_idx, sample_id in enumerate(doc_id):
                seq_input_ids = inputs["input_ids"][b_idx]
                valid_mask = inputs["attention_mask"][b_idx].bool()
                gen_token_ids = generated_ids_trimmed[b_idx].tolist()

                visual_mask = torch.zeros_like(seq_input_ids, dtype=torch.bool)
                if image_token_id is not None:
                    visual_mask |= seq_input_ids.eq(image_token_id)
                if video_token_id is not None:
                    visual_mask |= seq_input_ids.eq(video_token_id)
                visual_mask &= valid_mask
                visual_pos = torch.where(visual_mask)[0]
                visual_pos_list = [int(v) for v in visual_pos.tolist()]

                special_ids = set(self.tokenizer.all_special_ids or [])
                non_special_indices = [i for i, tid in enumerate(gen_token_ids) if int(tid) not in special_ids]
                if len(non_special_indices) == 0:
                    # Fallback to all generated tokens when everything is special.
                    selected_indices = list(range(len(gen_token_ids)))
                else:
                    selected_indices = non_special_indices

                selected_token_ids = [int(gen_token_ids[i]) for i in selected_indices]
                selected_token_texts = [
                    self.tokenizer.decode([gen_token_ids[i]], skip_special_tokens=False)
                    for i in selected_indices
                ]
                selected_seq_indices = [int(prompt_len + i) for i in selected_indices]

                per_layer_text_to_visual = {}
                for layer_idx in sampled_layer_indices:
                    layer = llm_layers[layer_idx]
                    hidden_before_layer = fwd.hidden_states[layer_idx]  # [B,S,D]
                    hidden_after_layer = fwd.hidden_states[layer_idx + 1]  # [B,S,D]

                    self_attn = layer.self_attn
                    proj_dtype = self_attn.q_proj.weight.dtype
                    x = layer.input_layernorm(hidden_before_layer).to(proj_dtype)
                    q_states = self_attn.q_proj(x)
                    k_states = self_attn.k_proj(x)

                    bsz, seqlen, _ = q_states.shape
                    num_heads = self_attn.num_heads
                    num_kv_heads = self_attn.num_key_value_heads
                    head_dim = self_attn.head_dim
                    num_kv_groups = self_attn.num_key_value_groups
                    q_states = q_states.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)  # [B,H,S,D]
                    k_states = k_states.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)  # [B,Hkv,S,D]
                    if num_kv_groups > 1:
                        k_states = (
                            k_states[:, :, None, :, :]
                            .expand(bsz, num_kv_heads, num_kv_groups, seqlen, head_dim)
                            .reshape(bsz, num_heads, seqlen, head_dim)
                        )

                    if len(visual_pos) == 0 or len(selected_seq_indices) == 0:
                        text_to_visual_attn_mean = []
                        text_to_visual_similarity_mean = []
                    else:
                        # Aggregate text->visual attention over selected generated tokens.
                        attn_vectors = []
                        for seq_idx in selected_seq_indices:
                            q_t = q_states[b_idx : b_idx + 1, :, seq_idx : seq_idx + 1, :]  # [1,H,1,D]
                            k_all = k_states[b_idx : b_idx + 1]  # [1,H,S,D]
                            attn_logits = torch.matmul(q_t, k_all.transpose(-1, -2)) / (head_dim**0.5)  # [1,H,1,S]
                            if seq_idx + 1 < seqlen:
                                attn_logits[..., seq_idx + 1 :] = torch.finfo(attn_logits.dtype).min
                            attn_prob = torch.softmax(attn_logits, dim=-1)  # [1,H,1,S]
                            attn_query = attn_prob[0, :, 0, :].mean(dim=0)  # [S]
                            attn_vectors.append(attn_query[visual_pos].float())  # [V]
                        text_to_visual_attn_mean = torch.stack(attn_vectors, dim=0).mean(dim=0).tolist()  # [V]
                        text_to_visual_attn_mean = [float(v) for v in text_to_visual_attn_mean]

                        # Aggregate text->visual cosine similarity over selected generated tokens.
                        vis_hidden = hidden_after_layer[b_idx, visual_pos, :].float()  # [V,D]
                        sim_vectors = []
                        for seq_idx in selected_seq_indices:
                            tok_vec = hidden_after_layer[b_idx, seq_idx, :].float().unsqueeze(0)  # [1,D]
                            sims = F.cosine_similarity(tok_vec, vis_hidden, dim=-1)  # [V]
                            sim_vectors.append(sims)
                        text_to_visual_similarity_mean = torch.stack(sim_vectors, dim=0).mean(dim=0).tolist()  # [V]
                        text_to_visual_similarity_mean = [float(v) for v in text_to_visual_similarity_mean]

                    layer_key = f"layer_{layer_idx + 1}"
                    per_layer_text_to_visual[layer_key] = {
                        "layer_index_0based": int(layer_idx),
                        "layer_index_1based": int(layer_idx + 1),
                        "text_to_visual_attention_mean": text_to_visual_attn_mean,
                        "text_to_visual_similarity_mean": text_to_visual_similarity_mean,
                    }

                # Keep last-layer fields for backward compatibility with existing readers.
                last_layer_key = f"layer_{num_layers}"
                last_layer_payload = per_layer_text_to_visual.get(
                    last_layer_key,
                    {"text_to_visual_attention_mean": [], "text_to_visual_similarity_mean": []},
                )
                text_to_visual_attn_mean = last_layer_payload["text_to_visual_attention_mean"]
                text_to_visual_similarity_mean = last_layer_payload["text_to_visual_similarity_mean"]

                out_path = os.path.join(self.token_scores_output_dir, f"{task}_{split}_{int(sample_id)}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": task,
                            "split": split,
                            "doc_id": int(sample_id),
                            "prompt_length": int(prompt_len),
                            "num_generated_tokens": len(generated_ids_trimmed[b_idx].tolist()),
                            "num_visual_tokens": int(len(visual_pos)),
                            "visual_token_positions": visual_pos_list,
                            "selected_generated_token_indices": selected_indices,
                            "selected_generated_token_seq_indices": selected_seq_indices,
                            "selected_generated_token_ids": selected_token_ids,
                            "selected_generated_token_texts": selected_token_texts,
                            "has_last_layer_attention": True,
                            "text_to_visual_attention_mean": text_to_visual_attn_mean,
                            "text_to_visual_similarity_mean": text_to_visual_similarity_mean,
                            "layer_sampling": "from_last_layer_step_2",
                            "sampled_layer_indices_1based": sampled_layer_indices_1based,
                            "per_layer_text_to_visual": per_layer_text_to_visual,
                            "score_type": "text_to_visual_attention_and_similarity_avg_non_special_tokens",
                            "attn_source": "reconstructed_from_qk_text_query_avg",
                            "attn_implementation": self.attn_implementation,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to save token scores in strict-attention mode: {e}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0 or self.disable_tqdm), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

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
                raise ValueError(f"Expected `gen_kwargs['until']` Union[str, list], but got {type(until)}")
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
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        _ = decord.VideoReader(visual)
                        processed_visuals.append(
                            {"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
                        )
                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

                if not self.interleave_visuals:
                    message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]})
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})
                    for j, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if j + 1 < len(text_parts) and text_parts[j + 1]:
                            content_parts.append({"type": "text", "text": text_parts[j + 1]})
                    message.append({"role": "user", "content": content_parts})

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
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            default_gen_kwargs = {"max_new_tokens": 128, "temperature": 0.0, "top_p": None, "num_beams": 1}
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            do_sample = current_gen_kwargs["temperature"] > 0
            pad_token_id = self.tokenizer.pad_token_id
            eval_time_enabled = os.environ.get("EVAL_TIME", "").lower() == "true"
            visual_token_num = 0
            input_token_count = int(inputs.input_ids.shape[1])
            if eval_time_enabled and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            if eval_time_enabled:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                video_token_id = getattr(self.model.config, "video_token_id", None)
                if image_token_id is not None:
                    visual_token_num += int((inputs.input_ids == image_token_id).sum().item())
                if video_token_id is not None:
                    visual_token_num += int((inputs.input_ids == video_token_id).sum().item())

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=do_sample,
                temperature=current_gen_kwargs["temperature"] if do_sample else None,
                top_p=current_gen_kwargs["top_p"] if do_sample else None,
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            if eval_time_enabled:
                generated_token_count = int(
                    sum(out_ids.shape[0] for out_ids in generated_ids_trimmed) / max(len(generated_ids_trimmed), 1)
                )
                max_memory_bytes = None
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                kv_cache_length = input_token_count + generated_token_count
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                dtype_bytes = next(self.model.parameters()).element_size()
                kv_cache_mb = (
                    2
                    * self.model.config.num_hidden_layers
                    * self.model.config.num_key_value_heads
                    * head_dim
                    * kv_cache_length
                    * dtype_bytes
                    / (1024 ** 2)
                )
                self._append_timing_record(
                    {
                        "after_generation_memory": max_memory_bytes,
                        "visual_token_num": visual_token_num,
                        "input_token_num": input_token_count,
                        "reduced_visual_token_num": visual_token_num,
                        "reduced_input_token_num": input_token_count,
                        "generated_token_num": generated_token_count,
                        "prefill_flops": self._prefill_flops(input_token_count),
                        "decode_flops": self._decode_flops(input_token_count, generated_token_count),
                        "kv_cache_length": kv_cache_length,
                        "kv_cache_size": kv_cache_mb,
                    }
                )
            self._save_token_scores(inputs, cont, generated_ids_trimmed, doc_id, task, split)

            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
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


__all__ = ["Qwen2_5_VL"]