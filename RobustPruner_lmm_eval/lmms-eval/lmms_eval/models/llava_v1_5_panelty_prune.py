import base64
import re
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple, Union
import json
import argparse
import types
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
from lmms_eval.api.registry import register_model

# LLaVA imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except ImportError:
    eval_logger.warning("Failed to import llava modules")

# ============ Pruning constants ============
PRUNE_TARGET_LAYER = 20  # Zero-based index: layer 21.


def llava_model_forward_PaneltyPrune(
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
):
    """
    Run the LLaVA forward pass with PaneltyPrune at layer 21.
    Adapted for the LLaMA-2 architecture.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast
    
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

    # Get inputs_embeds.
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    # Process position_ids.
    if position_ids is None:
        device = inputs_embeds.device
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Process attention_mask.
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
        )
    
    # Create the causal mask.
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, 0
    )

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # ============ Prepare pruning before layer 21 ============
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ========== Apply PaneltyPrune at the target layer ==========
        # Prune only during prefill (seq_length > 1).
        if (layer_idx == PRUNE_TARGET_LAYER and 
            seq_length > 1 and 
            batch_size == 1):
            
            # Get the PaneltyPrune mask and image_token_mask.
            robust_masks = getattr(self, '_robust_prune_masks', None)
            image_token_mask = getattr(self, '_image_token_mask', None)
            
            if robust_masks is not None and image_token_mask is not None:
                eval_logger.debug(f"[PaneltyPrune] Applying pruning at layer {layer_idx} (Layer 21)")
                
                # 1. Locate the image region using image_token_mask.
                image_token_mask_batch = image_token_mask[0]  # [seq_length]
                image_positions = (image_token_mask_batch == True).nonzero(as_tuple=True)[0]
                
                if len(image_positions) > 0:
                    image_start = int(image_positions[0])
                    image_end = int(image_positions[-1])
                    image_length = image_end - image_start + 1
                    
                    eval_logger.debug(
                        f"[PaneltyPrune] Image region: [{image_start}, {image_end}], "
                        f"length={image_length}"
                    )
                    
                    # 2. Get the mask for the first sample.
                    token_mask = robust_masks[0]  # [total_image_tokens]
                    
                    if token_mask is not None and len(token_mask) == image_length:
                        # 3. Build keep_indices.
                        device = hidden_states.device
                        
                        # Relative indices of the selected image tokens.
                        selected_image_indices = token_mask.nonzero(as_tuple=True)[0]
                        # Convert to absolute indices.
                        selected_absolute_indices = selected_image_indices + image_start
                        
                        # Concatenate text before the image, selected image tokens, and text after the image.
                        keep_indices = torch.cat([
                            torch.arange(image_start, device=device),
                            selected_absolute_indices,
                            torch.arange(image_end + 1, seq_length, device=device)
                        ])
                        keep_indices = keep_indices.sort().values
                        
                        kept_count = len(selected_absolute_indices)
                        pruned_count = image_length - kept_count
                        
                        eval_logger.info(
                            f"[PaneltyPrune] Layer {layer_idx} (Layer 21): "
                            f"Keeping {kept_count}/{image_length} image tokens "
                            f"({kept_count/image_length*100:.1f}%), "
                            f"pruning {pruned_count} tokens"
                        )
                        
                        # 4. Apply pruning.
                        hidden_states = hidden_states[:, keep_indices, :]
                        
                        # Update attention_mask.
                        if attention_mask is not None:
                            attention_mask = attention_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]
                        
                        # Update position_ids.
                        position_ids = position_ids[:, keep_indices]
                        
                        # Update seq_length.
                        seq_length = hidden_states.shape[1]
                        
                        eval_logger.debug(
                            f"[PaneltyPrune] After pruning: hidden_states.shape={hidden_states.shape}"
                        )
                        
                    else:
                        eval_logger.warning(
                            f"[PaneltyPrune] Mask length mismatch or None: "
                            f"mask_len={len(token_mask) if token_mask is not None else 'None'}, "
                            f"image_len={image_length}"
                        )
                else:
                    eval_logger.warning("[PaneltyPrune] No image tokens found in image_token_mask")

        # ========== Standard layer forward pass ==========
        past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)
                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

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


@register_model("llava_v1_5_panelty_prune")
class LLaVA_v1_5_PaneltyPrune(lmms):
    """
    PaneltyPrune selects stable visual tokens using statistics across
    multiple text perturbations.
    Apply a variance penalty: score = avg - penalty * var.
    Prune tokens at layer 21.
    Adapted for LLaVA-v1.5-7B.
    """
    
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        base_ratio: str = "0.3",
        keep_ratio: float = 0.15,  # Final token retention ratio.
        variance_penalty: float = 10.0,  # Variance penalty coefficient.
        conv_template: str = "vicuna_v1",  # LLaVA conversation template
        **kwargs,
    ) -> None:
        super().__init__()
        
        if kwargs:
            eval_logger.debug(f"Extra kwargs: {kwargs}")
        
        self.base_ratio = base_ratio
        self.keep_ratio = keep_ratio
        self.variance_penalty = variance_penalty
        self.conv_template = conv_template
        
        # Build the input directory paths; adjust them for your environment.
        ROOT = Path("/mnt/nvme/fjh/llava-evaluation")
        ratio_str = f"{base_ratio:.1f}" if isinstance(base_ratio, float) else base_ratio
        base_path = ROOT / "visualization_text_perturb/gqa/text_perturb" / ratio_str
        
        self.INPUT_DIRS = [
            base_path / f"llava_v1_5_7b_glimpse_{base_ratio}_text_perturbations_{i}/token_scores"
            for i in range(0, 5)
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
        
        # Load the LLaVA model.
        eval_logger.info(f"Loading LLaVA model from {pretrained}")
        model_name = get_model_name_from_path(pretrained)
        
        self._tokenizer, self._model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=pretrained,
            model_base=None,
            model_name=model_name,
            device_map=self.device_map,
        )
        
        self._model.eval()

        # ============ Monkey patch LLM model ============
        # The LLaVA language model is model.model (LlamaModel).
        llm_model = self._model.model
        llm_model._orig_forward = llm_model.forward
        llm_model.forward = types.MethodType(llava_model_forward_PaneltyPrune, llm_model)
        eval_logger.info("[PaneltyPrune] Patched LLM model forward for Layer 21 pruning")
        
        self._config = self._model.config
        self._max_length = self.context_len
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        # Get the image token ID; LLaVA uses IMAGE_TOKEN_INDEX.
        self.image_token_id = IMAGE_TOKEN_INDEX
        
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
            eval_logger.info(f"  base_ratio         = {base_ratio}")
            eval_logger.info(f"  keep_ratio         = {keep_ratio} ({keep_ratio*100:.0f}%)")
            eval_logger.info(f"  variance_penalty   = {variance_penalty}")
            eval_logger.info(f"  Pruning at layer   = Layer 21 (0-based index: {PRUNE_TARGET_LAYER})")
            eval_logger.info(f"  Formula: score = avg - {variance_penalty} × var")
            eval_logger.info(f"  Input dirs: {len(self.INPUT_DIRS)} directories")
            eval_logger.info(f"  Image token ID: {self.image_token_id}")
            eval_logger.info(f"  Conv template: {conv_template}")
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
        """Calculate the mean score and variance for each token."""
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
            v = self.population_variance(values)
            token_stats[str(i)] = {"scores": values, "var": v}
        
        return token_stats
    
    def select_stable_tokens(self, token_stats: dict, doc_id: int) -> List[int]:
        """
        Select stable tokens using scores adjusted by a variance penalty.
        score = avg - variance_penalty * var
        """
        if not token_stats:
            return []
        
        # 1. Build the token list and calculate penalized scores.
        token_list = []
        for token_id_str, stats in token_stats.items():
            perturbation_0_score = stats["scores"][0]
            variance = stats["var"]
            # Combined score = original score - penalty coefficient * variance.
            penalized_score = perturbation_0_score - self.variance_penalty * variance
            
            token_list.append({
                "id": int(token_id_str),
                "avg": perturbation_0_score,
                "var": variance,
                "score": penalized_score
            })
        
        total_tokens = len(token_list)
        
        # 2. Sort by combined score in descending order.
        token_list_sorted = sorted(
            token_list, 
            key=lambda x: x["score"],
            reverse=True
        )
        
        # 3. Select the top keep_ratio fraction of tokens.
        final_count = max(1, int(total_tokens * self.keep_ratio))
        selected_tokens = token_list_sorted[:final_count]
        
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
    
    def create_image_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create image_token_mask.
        True denotes an image token; False denotes a text token.
        """
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[input_ids == self.image_token_id] = True
        return mask
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        
        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        
        pbar = tqdm(
            total=len(requests), 
            disable=(self.rank != 0), 
            desc="PaneltyPrune Inference (LLaVA)"
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
            
            # ========= PaneltyPrune: calculate token masks =========
            eval_logger.info(f"[PaneltyPrune] Processing batch with {len(doc_id)} samples")
            
            token_masks = []
            for sample_id in doc_id:
                token_stats = self.compute_token_stats(int(sample_id), task, split)
                
                if token_stats:
                    selected_indices = self.select_stable_tokens(token_stats, int(sample_id))
                    total_tokens = len(token_stats)
                    token_mask = self.create_token_mask(selected_indices, total_tokens)
                    token_masks.append(token_mask.to(self.device))
                    
                    eval_logger.info(
                        f"[PaneltyPrune] doc_id={sample_id}: "
                        f"mask created with {token_mask.sum()}/{len(token_mask)} tokens "
                        f"({token_mask.sum()/len(token_mask)*100:.1f}%)"
                    )
                else:
                    eval_logger.warning(
                        f"[PaneltyPrune] doc_id={sample_id}: No stats found, will use full features"
                    )
                    token_masks.append(None)
            
            # Process the LLaVA inputs.
            input_ids_list = []
            image_tensor_list = []
            
            for i, context in enumerate(contexts):
                # Process the images.
                image = visual_list[i][0] if visual_list[i] else None
                
                if image is not None:
                    if isinstance(image, str):
                        image = Image.open(image).convert('RGB')
                    
                    image_tensor = process_images([image], self.image_processor, self._config)
                    if isinstance(image_tensor, list):
                        image_tensor = image_tensor[0]
                    image_tensor_list.append(image_tensor.to(self.device))
                    
                    # Build the prompt using the LLaVA <image> placeholder.
                    if DEFAULT_IMAGE_TOKEN not in context:
                        context = DEFAULT_IMAGE_TOKEN + "\n" + context
                else:
                    image_tensor_list.append(None)
                
                # Use the conversation template.
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], context)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_image_token(
                    prompt, 
                    self.tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors='pt'
                ).unsqueeze(0).to(self.device)
                
                input_ids_list.append(input_ids)
            
            # ============ Create image_token_mask ============
            image_token_masks = []
            for input_ids in input_ids_list:
                image_token_mask = self.create_image_token_mask(input_ids[0])
                image_token_masks.append(image_token_mask)
            image_token_masks = torch.stack(image_token_masks)
            
            # ============ Set masks on the language model ============
            self.model.model._robust_prune_masks = token_masks
            self.model.model._image_token_mask = image_token_masks.to(self.device)
            eval_logger.info(f"[PaneltyPrune] Masks prepared ({len(token_masks)} masks)")
            
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
                    for idx, (input_ids, image_tensor) in enumerate(zip(input_ids_list, image_tensor_list)):
                        output_ids = self.model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0) if image_tensor is not None else None,
                            do_sample=do_sample,
                            temperature=current_gen_kwargs["temperature"] if do_sample else 1.0,
                            top_p=current_gen_kwargs["top_p"],
                            num_beams=current_gen_kwargs["num_beams"],
                            max_new_tokens=current_gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        # Decode the response.
                        output = self.tokenizer.batch_decode(
                            output_ids, 
                            skip_special_tokens=True
                        )[0].strip()
                        
                        # Process the until stop sequences.
                        for term in until:
                            if len(term) > 0:
                                output = output.split(term)[0]
                        
                        res.append(output)
                        self.cache_hook.add_partial("generate_until", (contexts[idx], gen_kwargs), output)
                        pbar.update(1)
                        
            except Exception as e:
                eval_logger.error(f"[PaneltyPrune] Generation failed: {e}")
                raise
            finally:
                # Clear the masks.
                if hasattr(self.model.model, '_robust_prune_masks'):
                    delattr(self.model.model, '_robust_prune_masks')
                if hasattr(self.model.model, '_image_token_mask'):
                    delattr(self.model.model, '_image_token_mask')
        
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")