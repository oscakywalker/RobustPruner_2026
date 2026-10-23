import base64
import re
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple, Union
import json
import argparse
import types
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from qwen25vl import Qwen2_5_VLForConditionalGeneration
import qwen25vl

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils")

# ============ Pruning constants ============
PRUNE_TARGET_LAYER = 20  # Zero-based index: layer 21.

def qwen25vl_model_forward_robustprune(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.cache_utils import DynamicCache
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            eval_logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        batch_size, seq_length = inputs_embeds.shape[:2]

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Process position_ids.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # ============ Prepare pruning before layer 21 ============
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ========== Apply RobustPrune at the target layer ==========
        # Prune only during prefill (seq_length > 1).
        if (layer_idx == PRUNE_TARGET_LAYER and 
            seq_length > 1 and 
            batch_size == 1):  # FastV also requires batch_size=1.
            
            # Get the RobustPrune mask and text_image_mask.
            robust_masks = getattr(self, '_robust_prune_masks', None)
            text_image_mask = getattr(self, '_text_image_mask', None)
            
            if robust_masks is not None and text_image_mask is not None:
                eval_logger.debug(f"[RobustPrune] Applying pruning at layer {layer_idx}")
                
                # 1. Locate the image region using text_image_mask, as in FastV.
                text_image_mask_batch = text_image_mask[0]  # [seq_length]
                image_positions = (text_image_mask_batch == False).nonzero(as_tuple=True)[0]
                
                if len(image_positions) > 0:
                    image_start = int(image_positions[0])
                    image_end = int(image_positions[-1])
                    image_length = image_end - image_start + 1
                    
                    eval_logger.debug(
                        f"[RobustPrune] Image region: [{image_start}, {image_end}], "
                        f"length={image_length}"
                    )
                    
                    # 2. Get the mask for the first sample (assuming batch_size=1).
                    token_mask = robust_masks[0]  # [total_image_tokens]
                    
                    if token_mask is not None and len(token_mask) == image_length:
                        # 3. Build keep_indices following FastV.
                        device = hidden_states.device
                        
                        # Relative indices of the selected image tokens.
                        selected_image_indices = token_mask.nonzero(as_tuple=True)[0]
                        # Convert to absolute indices.
                        selected_absolute_indices = selected_image_indices + image_start
                        
                        # Concatenate text before the image, selected image tokens, and text after the image.
                        keep_indices = torch.cat([
                            torch.arange(image_start, device=device),           # Text before the image.
                            selected_absolute_indices,                          # Selected image tokens.
                            torch.arange(image_end + 1, seq_length, device=device)  # Text after the image.
                        ])
                        keep_indices = keep_indices.sort().values
                        
                        kept_count = len(selected_absolute_indices)
                        pruned_count = image_length - kept_count
                        
                        eval_logger.info(
                            f"[RobustPrune] Layer {layer_idx}: "
                            f"Keeping {kept_count}/{image_length} image tokens "
                            f"({kept_count/image_length*100:.1f}%), "
                            f"pruning {pruned_count} tokens"
                        )
                        
                        # 4. Apply pruning.
                        hidden_states = hidden_states[:, keep_indices, :]
                        
                        # Update causal_mask.
                        if causal_mask is not None:
                            causal_mask = causal_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]
                        
                        # Update position_ids and position_embeddings.
                        position_ids = position_ids[:, :, keep_indices]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        
                        eval_logger.debug(
                            f"[RobustPrune] After pruning: hidden_states.shape={hidden_states.shape}"
                        )
                        
                        # Keep the pruned mask so decoding can use the pruned KV cache.
                        # Retain the mask to preserve the effect of pruning.
                        
                    else:
                        eval_logger.warning(
                            f"[RobustPrune] Mask length mismatch or None: "
                            f"mask_len={len(token_mask) if token_mask is not None else 'None'}, "
                            f"image_len={image_length}"
                        )
                else:
                    eval_logger.warning("[RobustPrune] No image tokens found in text_image_mask")

        # ========== Standard layer forward pass ==========
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

@register_model("RobustPrune")
class Qwen2_5_VL_RobustPrune(lmms):
    """
    RobustPrune selects stable visual tokens using statistics across
    multiple text perturbations.
    Prune tokens at layer 21 following FastV.
    """
    
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        base_ratio: str = "0.3",
        top_avg_ratio: float = 0.2,
        stable_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        
        if kwargs:
            eval_logger.debug(f"Extra kwargs: {kwargs}")
        
        self.base_ratio = base_ratio
        self.top_avg_ratio = top_avg_ratio
        self.stable_ratio = stable_ratio
        self.final_keep_ratio = top_avg_ratio * stable_ratio
        
        # Build the six input directory paths.
        ROOT = Path("/mnt/nvme/fjh/qwen-evaluation")
        ratio_str = f"{base_ratio:.1f}" if isinstance(base_ratio, float) else base_ratio
        base_path = ROOT / "visualization_text_perturb/chartqa/text_perturb" / ratio_str
        
        self.INPUT_DIRS = [
            base_path / f"qwen25vl_7b_glimpse_{base_ratio}_text_perturbations_{i}/token_scores"
            for i in range(0, 6)
        ]
        
        # Validate the paths.
        missing_dirs = [d for d in self.INPUT_DIRS if not d.exists()]
        if missing_dirs:
            eval_logger.warning(f"Missing {len(missing_dirs)} input directories")
        
        # Initialize the device.
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        
        # Load the model.
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, **model_kwargs
        ).eval()

        # ============ Monkey patch LLM model ============
        llm_model = self._model.model  # Qwen2_5_VLModel
        llm_model._orig_forward = llm_model.forward
        llm_model.forward = types.MethodType(qwen25vl_model_forward_robustprune, llm_model)
        eval_logger.info("[RobustPrune] Patched LLM model forward for layer-21 pruning")
        
        # Initialize the processor.
        self.processor = AutoProcessor.from_pretrained(
            pretrained, 
            max_pixels=max_pixels, 
            min_pixels=min_pixels
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        
        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        # Get the image token ID.
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        # Set up distributed execution.
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type"
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        
        if self._rank == 0:
            eval_logger.info("=" * 60)
            eval_logger.info(f"  Model: {pretrained}")
            eval_logger.info(f"  base_ratio        = {base_ratio}")
            eval_logger.info(f"  top_avg_ratio     = {top_avg_ratio} ({top_avg_ratio*100:.0f}%)")
            eval_logger.info(f"  stable_ratio      = {stable_ratio} ({stable_ratio*100:.0f}%)")
            eval_logger.info(f"  final_keep_ratio  = {self.final_keep_ratio} ({self.final_keep_ratio*100:.0f}%)")
            eval_logger.info(f"  Pruning at layer  = {PRUNE_TARGET_LAYER} (21th layer)")
            eval_logger.info(f"  Input dirs: {len(self.INPUT_DIRS)} directories")
            eval_logger.info(f"  Image token ID: {self.image_token_id}")
            eval_logger.info("=" * 60)
    
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
        raise NotImplementedError("Loglikelihood is not implemented")
    
    def population_variance(self, values):
        """Calculate the population variance."""
        if not values:
            return 0.0
        m = mean(values)
        return sum((x - m) ** 2 for x in values) / len(values)
    
    def extract_scores_from_mask_flat(self, data, fpath):
        """Extract scores from mask_flat."""
        if "mask_flat" not in data:
            raise ValueError(f"'mask_flat' missing in {fpath}")
        
        raw = data["mask_flat"]
        
        if isinstance(raw, dict):
            mask_dict = raw
        elif isinstance(raw, list):
            return list(raw)
        elif isinstance(raw, str):
            try:
                mask_dict = json.loads(raw)
            except Exception as e:
                raise ValueError(f"Failed to parse 'mask_flat' JSON in {fpath}: {e}")
        else:
            raise TypeError(f"Unexpected type for 'mask_flat' in {fpath}: {type(raw)}")
        
        try:
            items = sorted(mask_dict.items(), key=lambda kv: int(kv[0]))
        except Exception as e:
            raise ValueError(f"mask_flat keys are not numeric in {fpath}: {e}")
        
        scores = [v for _, v in items]
        
        num_tokens = data.get("num_tokens")
        if num_tokens is not None and len(scores) != num_tokens:
            raise ValueError(
                f"Length mismatch: mask_flat={len(scores)}, num_tokens={num_tokens} in {fpath}"
            )
        
        return scores
    
    def compute_token_stats(self, doc_id: int, task: str, split: str) -> Optional[dict]:
        filename = f"{task}_{split}_{doc_id}.json"
        token_score_lists = []
        
        for d in self.INPUT_DIRS:
            fpath = d / filename
            if not fpath.exists():
                eval_logger.warning(f"Missing: {fpath}")
                return None
            
            try:
                with fpath.open("r") as f:
                    data = json.load(f)
                scores = self.extract_scores_from_mask_flat(data, fpath)
                token_score_lists.append(scores)
            except Exception as e:
                eval_logger.error(f"Failed to load {fpath}: {e}")
                return None

        lengths = {len(lst) for lst in token_score_lists}
        if len(lengths) != 1:
            eval_logger.error(f"Length mismatch for {filename}: {lengths}")
            return None
        
        num_tokens = lengths.pop()
        token_stats = {}
        for i in range(num_tokens):
            values = [lst[i] for lst in token_score_lists]
            m = mean(values)
            v = self.population_variance(values)
            token_stats[str(i)] = {"avg": m, "var": v}
        
        return token_stats
    
    def select_stable_tokens(self, token_stats: dict, doc_id: int) -> List[int]:
        """Select stable tokens."""
        if not token_stats:
            return []
        
        token_list = []
        for token_id_str, stats in token_stats.items():
            token_list.append({
                "id": int(token_id_str),
                "avg": stats["avg"],
                "var": stats["var"]
            })
        
        total_tokens = len(token_list)

        token_list_sorted_by_avg = sorted(
            token_list, 
            key=lambda x: x["avg"], 
            reverse=True
        )
        top_avg_count = max(1, int(total_tokens * self.top_avg_ratio))
        top_avg_tokens = token_list_sorted_by_avg[:top_avg_count]

        top_avg_tokens_sorted_by_var = sorted(
            top_avg_tokens, 
            key=lambda x: x["var"]
        )
        final_count = max(1, int(len(top_avg_tokens) * self.stable_ratio))
        selected_tokens = top_avg_tokens_sorted_by_var[:final_count]
        
        selected_indices = sorted([t["id"] for t in selected_tokens])
        
        if self._rank == 0:
            eval_logger.debug(
                f"[doc_id={doc_id}] Selected {len(selected_indices)}/{total_tokens} "
                f"({len(selected_indices)/total_tokens*100:.1f}%)"
            )
        
        return selected_indices
    
    def create_token_mask(self, selected_indices: List[int], total_tokens: int) -> torch.Tensor:
        """Create a boolean mask."""
        mask = torch.zeros(total_tokens, dtype=torch.bool)
        if selected_indices:
            mask[selected_indices] = True
        return mask
    
    def create_text_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create text_image_mask following FastV.
        True denotes a text token; False denotes an image token.
        """
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask[input_ids == self.image_token_id] = False
        return mask
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        
        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        
        pbar = tqdm(
            total=len(requests), 
            disable=(self.rank != 0), 
            desc="RobustPrune Inference"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], 
            _collate, 
            grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) 
                for ids in doc_id
            ]
            
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            
            if isinstance(until, str):
                until = [until]
            until = [item for item in until if item != "\n\n"]
            
            contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": self.system_prompt}]
                
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context
                
                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append({
                            "type": "video", 
                            "video": visual,
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels
                        })
                    elif isinstance(visual, Image.Image):
                        buffer = BytesIO()
                        visual.convert("RGB").save(buffer, format="JPEG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        processed_visuals.append({
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_string}",
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels
                        })
                
                message.append({
                    "role": "user",
                    "content": processed_visuals + [{"type": "text", "text": context}],
                })
                batched_messages.append(message)
            
            # Process the inputs.
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                ) 
                for msg in batched_messages
            ]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            
            # ========= RobustPrune: calculate token masks =========
            eval_logger.info(f"[RobustPrune] Processing batch with {len(doc_id)} samples")
            
            token_masks = []
            for sample_id in doc_id:
                token_stats = self.compute_token_stats(int(sample_id), task, split)
                
                if token_stats:
                    selected_indices = self.select_stable_tokens(token_stats, int(sample_id))
                    total_tokens = len(token_stats)
                    token_mask = self.create_token_mask(selected_indices, total_tokens)
                    token_masks.append(token_mask.to(self.device))
                    
                    eval_logger.info(
                        f"[RobustPrune] doc_id={sample_id}: "
                        f"mask created with {token_mask.sum()}/{len(token_mask)} tokens"
                    )
                else:
                    eval_logger.warning(
                        f"[RobustPrune] doc_id={sample_id}: No stats found, will use full features"
                    )
                    token_masks.append(None)
            
            # Call the processor.
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to the device.
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            
            # ============ Create text_image_mask following FastV ============
            text_image_masks = []
            for i in range(inputs.input_ids.shape[0]):
                text_image_mask = self.create_text_image_mask(inputs.input_ids[i])
                text_image_masks.append(text_image_mask)
            text_image_masks = torch.stack(text_image_masks)
            
            # ============ Set masks on the language model ============
            self.model.model._robust_prune_masks = token_masks
            self.model.model._text_image_mask = text_image_masks.to(self.device)
            eval_logger.info(f"[RobustPrune] Masks prepared ({len(token_masks)} masks)")
            
            # Generation configuration.
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            
            if current_gen_kwargs["temperature"] > 0:
                do_sample = True
            else:
                do_sample = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
            
            # Generate the response.
            try:
                with torch.inference_mode():
                    cont = self.model.generate(
                        **inputs,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        temperature=current_gen_kwargs["temperature"],
                        top_p=current_gen_kwargs["top_p"],
                        num_beams=current_gen_kwargs["num_beams"],
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )
            except Exception as e:
                eval_logger.error(f"[RobustPrune] Generation failed: {e}")
                # Clear masks after an exception.
                if hasattr(self.model.model, '_robust_prune_masks'):
                    delattr(self.model.model, '_robust_prune_masks')
                if hasattr(self.model.model, '_text_image_mask'):
                    delattr(self.model.model, '_text_image_mask')
                raise
            
            # Clear masks after successful completion.
            if hasattr(self.model.model, '_robust_prune_masks'):
                delattr(self.model.model, '_robust_prune_masks')
            if hasattr(self.model.model, '_text_image_mask'):
                delattr(self.model.model, '_text_image_mask')
            eval_logger.debug("[RobustPrune] Masks cleared")
            
            # Decode the response.
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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