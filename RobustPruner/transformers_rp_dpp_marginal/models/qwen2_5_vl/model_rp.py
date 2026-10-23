# RobustPruner with optional projected DPP marginal candidate selection.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache, Cache
from transformers.utils import logging
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2_5_VLFlashAttention2,
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLAttention,
    repeat_kv,
    apply_multimodal_rotary_pos_emb,
)
from transformers.activations import ACT2FN

from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
)

from .configuration import *
from transformers.modeling_outputs import ModelOutput
from utils.warppers import time_logger
from utils.download import download_model_from_hf

logger = logging.get_logger(__name__)

_PROJECTED_DPP_MATRICES = {}


def _use_projected_dpp_marginal_selector():
    return os.environ.get("GP_DPP_PROJECTED_MARGINAL", "").lower() == "true"


def print_rank0(message):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)


def convert_2d_to_4d_mask(mask_2d: torch.Tensor,
                          query_seq_len: int,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    batch_size, key_seq_len = mask_2d.shape
    mask_4d = mask_2d.unsqueeze(1).unsqueeze(2)
    mask_4d = mask_4d.expand(batch_size, 1, query_seq_len, key_seq_len)
    mask_4d = mask_4d.to(dtype)
    inverted_mask = 1 - mask_4d
    masked_value = -torch.inf
    inverted_mask = inverted_mask.masked_fill(inverted_mask == 1, masked_value)
    return inverted_mask


# ---------- Fusers ----------

class BaseAttnFuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        raise NotImplementedError("Subclasses should implement this method.")


ATTN_FUSER_REGISTRY = {}

def register_attn_fuser():
    def decorator(cls):
        name = cls.__name__
        if name in ATTN_FUSER_REGISTRY:
            raise ValueError(f"AttnFuser {name} already registered.")
        if not issubclass(cls, BaseAttnFuser):
            raise ValueError(f"AttnFuser {name} must be a subclass of BaseAttnFuser.")
        ATTN_FUSER_REGISTRY[name] = cls
        return cls
    return decorator


class PenaltyHead(nn.Module):
    def __init__(self, hidden_size, penalty_hidden_size, hidden_act):
        super().__init__()
        if penalty_hidden_size is None:
            penalty_hidden_size = hidden_size
        self.in_proj = nn.Linear(hidden_size, penalty_hidden_size)
        self.out_proj = nn.Linear(penalty_hidden_size, 1)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        penalty_hidden = self.act_fn(self.in_proj(hidden_states))
        return F.softplus(self.out_proj(penalty_hidden))


@register_attn_fuser()
class DPP_direct_fuser(BaseAttnFuser):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        attn_fuse_size = config.attn_fuse_size
        num_attn_layers = len(config.selected_layers)
        num_attn_heads = config.num_attention_heads
        attn_input_size = num_attn_layers * num_attn_heads
        vision_hidden_size = config.vision_config.hidden_size
        num_visual_layers = len(config.selected_visual_layers)

        self.attn_norm = nn.LayerNorm(attn_input_size)
        self.attn_proj = nn.Linear(attn_input_size, attn_fuse_size)
        self.visual_norms = nn.ModuleList(
            [nn.LayerNorm(vision_hidden_size) for _ in range(num_visual_layers)]
        )
        self.visual_projs = nn.ModuleList(
            [nn.Linear(vision_hidden_size, attn_fuse_size) for _ in range(num_visual_layers)]
        )
        self.act_fn = ACT2FN[config.attn_fuse_hidden_act]
        self.fusion_layers = nn.ModuleList(
            [nn.Linear(attn_fuse_size * 3, attn_fuse_size) for _ in range(num_visual_layers)]
        )
        self.fusion_out_proj = nn.Linear(attn_fuse_size, attn_fuse_size)
        self.quality_head = nn.Linear(attn_fuse_size, 1)
        if self.config.use_uncertainty_penalty_head:
            self.penalty_head = PenaltyHead(
                attn_fuse_size, self.config.penalty_head_hidden_size, config.attn_fuse_hidden_act
            )
        else:
            self.penalty_head = nn.Identity()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _ori_attn_supervision_outputs(self, attn_map, attn_grid_hw):
        attn_map_mean = attn_map.mean(dim=-1)
        st = 0
        ori_attns = []
        for h, w in attn_grid_hw:
            ed = st + h * w
            one_attn_map = attn_map_mean[st:ed]
            st = ed
            if self.config.use_attention_logits:
                one_attn_map = torch.softmax(one_attn_map, dim=-1)
            else:
                one_attn_map = torch.exp(one_attn_map)
            one_attn_map_min = one_attn_map.min()
            one_attn_map_max = one_attn_map.max()
            one_attn_map = (one_attn_map - one_attn_map_min) / (one_attn_map_max - one_attn_map_min + 1e-6)
            ori_attns.append(one_attn_map)
        return torch.cat(ori_attns, dim=0)

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        penalty_outs = []
        if self.config.ori_attn_supervision and not self.training:
            ori_attns = self._ori_attn_supervision_outputs(attn_map, attn_grid_hw)
            attn_outs.append(ori_attns)
            if self.config.use_uncertainty_penalty_head:
                penalty_outs.append(torch.zeros_like(ori_attns))

        if selected_image_embeds is None or len(selected_image_embeds) == 0:
            raise ValueError("DPP_direct_fuser requires selected_image_embeds.")
        if len(selected_image_embeds) != len(self.visual_projs):
            raise ValueError(
                "DPP_direct_fuser expected "
                f"{len(self.visual_projs)} selected visual layers, got {len(selected_image_embeds)}."
            )

        fused_hidden = self.act_fn(self.attn_proj(self.attn_norm(attn_map)))
        for visual_feat, visual_norm, visual_proj, fusion_layer in zip(
            selected_image_embeds, self.visual_norms, self.visual_projs, self.fusion_layers
        ):
            visual_hidden = self.act_fn(visual_proj(visual_norm(visual_feat)))
            fused_input = torch.cat(
                [fused_hidden, visual_hidden, fused_hidden * visual_hidden], dim=-1
            )
            fused_hidden = self.act_fn(fusion_layer(fused_input))
        fused_hidden = self.act_fn(self.fusion_out_proj(fused_hidden))

        quality_logits = self.quality_head(fused_hidden).squeeze(-1)
        attn_outs.append(quality_logits)
        if not isinstance(self.penalty_head, nn.Identity):
            penalties = self.penalty_head(fused_hidden).squeeze(-1)
            penalty_outs.append(penalties)

        attn_outs = torch.stack(attn_outs, dim=0)
        penalty_outs = torch.stack(penalty_outs, dim=0) if len(penalty_outs) > 0 else None
        return attn_outs, penalty_outs


@register_attn_fuser()
class DPP_direct_fuser_2(DPP_direct_fuser):
    def __init__(self, config):
        BaseAttnFuser.__init__(self, config)
        self.config = config
        attn_fuse_size = config.attn_fuse_size
        num_attn_layers = len(config.selected_layers)
        num_attn_heads = config.num_attention_heads
        attn_input_size = num_attn_layers * num_attn_heads
        vision_hidden_size = config.vision_config.hidden_size

        self.attn_norm = nn.LayerNorm(attn_input_size)
        self.attn_proj = nn.Linear(attn_input_size, attn_fuse_size)
        self.visual_norm = nn.LayerNorm(vision_hidden_size)
        self.visual_proj = nn.Linear(vision_hidden_size, attn_fuse_size)
        self.act_fn = ACT2FN[config.attn_fuse_hidden_act]
        self.fusion_proj = nn.Linear(attn_fuse_size * 3, attn_fuse_size * 2)
        self.fusion_out_proj = nn.Linear(attn_fuse_size * 2, attn_fuse_size)
        self.quality_head = nn.Linear(attn_fuse_size, 1)
        if self.config.use_uncertainty_penalty_head:
            self.penalty_head = PenaltyHead(
                attn_fuse_size, self.config.penalty_head_hidden_size, config.attn_fuse_hidden_act
            )
        else:
            self.penalty_head = nn.Identity()

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        penalty_outs = []
        if self.config.ori_attn_supervision and not self.training:
            ori_attns = self._ori_attn_supervision_outputs(attn_map, attn_grid_hw)
            attn_outs.append(ori_attns)
            if self.config.use_uncertainty_penalty_head:
                penalty_outs.append(torch.zeros_like(ori_attns))

        if selected_image_embeds is None or len(selected_image_embeds) == 0:
            raise ValueError("DPP_direct_fuser_2 requires selected_image_embeds.")

        visual_feats = torch.stack([one_embeds for one_embeds in selected_image_embeds], dim=0).mean(dim=0)
        attn_hidden = self.act_fn(self.attn_proj(self.attn_norm(attn_map)))
        visual_hidden = self.act_fn(self.visual_proj(self.visual_norm(visual_feats)))
        fused_hidden = torch.cat([attn_hidden, visual_hidden, attn_hidden * visual_hidden], dim=-1)
        fused_hidden = self.act_fn(self.fusion_proj(fused_hidden))
        fused_hidden = self.act_fn(self.fusion_out_proj(fused_hidden))

        #quality_logits = torch.sigmoid(self.quality_head(fused_hidden)).squeeze(-1)
        quality_logits = self.quality_head(fused_hidden).squeeze(-1)
        attn_outs.append(quality_logits)
        if not isinstance(self.penalty_head, nn.Identity):
            penalties = self.penalty_head(fused_hidden).squeeze(-1)
            penalty_outs.append(penalties)

        attn_outs = torch.stack(attn_outs, dim=0)
        penalty_outs = torch.stack(penalty_outs, dim=0) if len(penalty_outs) > 0 else None
        return attn_outs, penalty_outs


# ---------- VLM ----------


@dataclass
class Qwen2_5_VL_GP_CausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None
    le_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    input_ids: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    attn_grid: Optional[torch.LongTensor] = None
    image_token_mask_logits: Optional[torch.Tensor] = None
    image_token_penalties: Optional[torch.Tensor] = None
    # Cache the per-image token-token similarity matrices so delayed selection
    # and eval-time score export can reuse the same DPP inputs.
    image_token_similarity_matrices: Optional[List[torch.Tensor]] = None
    image_token_bool_masks: Optional[torch.Tensor] = None


def _resolve_dual_selection_counts(num_tokens, max_remain_ratio, base_score_pool_multiplier):
    if num_tokens <= 0 or max_remain_ratio is None:
        return None, None

    final_keep_num = max(int(max_remain_ratio * num_tokens), 1)
    if base_score_pool_multiplier is None:
        return final_keep_num, final_keep_num

    pool_multiplier = float(base_score_pool_multiplier)
    if pool_multiplier < 1.0:
        raise ValueError(
            f"base_score_pool_multiplier must be >= 1.0 for dual-rank selection, got {pool_multiplier}."
        )
    pool_keep_num = max(int(final_keep_num * pool_multiplier), 1)
    pool_keep_num = min(pool_keep_num, num_tokens)
    return pool_keep_num, final_keep_num


def _select_image_tokens_by_strategy(
    score_logits,
    penalties,
    similarity_matrix,
    max_remain_ratio,
    base_score_pool_multiplier,
    threshold,
    min_remain_num,
    selection_strategy="default",
):
    one_image_token_prob = score_logits.sigmoid()
    selection_meta = {
        "strategy": "threshold" if max_remain_ratio is None else "topk_base_only",
        "pool_keep_num": None,
        "final_keep_num": None,
        "projected_rank": None,
    }

    if max_remain_ratio is not None:
        pool_keep_num, final_keep_num = _resolve_dual_selection_counts(
            one_image_token_prob.numel(),
            max_remain_ratio,
            base_score_pool_multiplier,
        )
        selection_meta["pool_keep_num"] = int(pool_keep_num)
        selection_meta["final_keep_num"] = int(final_keep_num)
        if selection_strategy == "dpp":
            if similarity_matrix is None:
                raise ValueError("similarity_matrix is required when image_token_selection_strategy='dpp'.")
            use_projected = _use_projected_dpp_marginal_selector()
            dpp_selector = _select_with_projected_dpp_marginal if use_projected else _select_with_dpp
            if penalties is not None and base_score_pool_multiplier is not None:
                # Keep the original two-stage selection recipe, but replace the
                # first-stage top-k candidate pool with a DPP subset.
                selection_meta["strategy"] = (
                    "projected_dpp_marginal_pool_then_uncertainty"
                    if use_projected else "dpp_pool_then_uncertainty"
                )
                if use_projected:
                    selection_meta["projected_rank"] = min(pool_keep_num, similarity_matrix.shape[-1])
                base_selected_indices = dpp_selector(one_image_token_prob, similarity_matrix, pool_keep_num)
                candidate_penalties = penalties[base_selected_indices]
                rerank_keep_num = min(final_keep_num, candidate_penalties.numel())
                low_uncertainty_order = torch.topk(
                    candidate_penalties, rerank_keep_num, dim=-1, largest=False
                ).indices
                selected_indices = base_selected_indices[low_uncertainty_order]
            else:
                # Pure DPP mode: directly select the final keep set with
                # w_i = sigmoid(score_logits_i) and the provided S_ij.
                selection_meta["strategy"] = "projected_dpp_marginal" if use_projected else "dpp"
                if use_projected:
                    selection_meta["projected_rank"] = min(final_keep_num, similarity_matrix.shape[-1])
                selected_indices = dpp_selector(one_image_token_prob, similarity_matrix, final_keep_num)
        elif penalties is not None and base_score_pool_multiplier is not None:
            selection_meta["strategy"] = "dual_rank_base_then_uncertainty"
            base_selected_indices = torch.topk(score_logits, pool_keep_num, dim=-1).indices
            candidate_penalties = penalties[base_selected_indices]
            low_uncertainty_order = torch.topk(
                candidate_penalties, final_keep_num, dim=-1, largest=False
            ).indices
            selected_indices = base_selected_indices[low_uncertainty_order]
        else:
            selected_indices = torch.topk(score_logits, final_keep_num, dim=-1).indices

        one_image_token_bool_mask = torch.zeros_like(one_image_token_prob, dtype=torch.bool)
        one_image_token_bool_mask[selected_indices] = True
    else:
        one_image_token_bool_mask = one_image_token_prob > threshold

    if min_remain_num is not None:
        remain_num = one_image_token_bool_mask.sum().item()
        if remain_num < min_remain_num:
            selected_indices = torch.topk(score_logits, min_remain_num, dim=-1).indices
            one_image_token_bool_mask[selected_indices] = True

    return one_image_token_bool_mask, one_image_token_prob, selection_meta


def _build_similarity_matrices_from_visual_embeds(similarity_image_embeds, num_tokens_by_batch):
    if similarity_image_embeds is None:
        return None

    if _use_projected_dpp_marginal_selector():
        # Keep the legacy return-field name, but pass N x d features to the
        # marginal selector without constructing an N x N similarity matrix.
        return [
            F.normalize(features.float(), dim=-1).contiguous()
            for features in similarity_image_embeds.split(num_tokens_by_batch, dim=0)
        ]

    similarity_matrices = []
    for one_layer_embeds in similarity_image_embeds.split(num_tokens_by_batch, dim=0):
        one_layer_embeds = F.normalize(one_layer_embeds.float(), dim=-1)
        one_similarity = torch.matmul(one_layer_embeds, one_layer_embeds.transpose(0, 1))
        similarity_matrices.append(one_similarity)
    return similarity_matrices


def _get_projected_dpp_matrix(input_dim, rank, device, dtype):
    device_index = -1 if device.index is None else int(device.index)
    cache_key = (device.type, device_index, int(input_dim), int(rank), dtype)
    projection = _PROJECTED_DPP_MATRICES.get(cache_key)
    if projection is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        projection = torch.randn(
            (input_dim, rank),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        projection.mul_(1.0 / math.sqrt(rank))
        _PROJECTED_DPP_MATRICES[cache_key] = projection
    return projection


def _select_with_projected_dpp_marginal(
    one_image_token_prob,
    similarity_features,
    final_keep_num,
):
    num_tokens = one_image_token_prob.numel()
    if final_keep_num >= num_tokens:
        return torch.arange(num_tokens, device=one_image_token_prob.device)

    features = similarity_features.to(
        device=one_image_token_prob.device,
        dtype=torch.float32,
    ).contiguous()
    projection_rank = min(final_keep_num, num_tokens, features.shape[-1])
    projection = _get_projected_dpp_matrix(
        features.shape[-1],
        projection_rank,
        features.device,
        features.dtype,
    )

    with torch.autocast(device_type=features.device.type, enabled=False):
        projected_features = torch.mm(features, projection)
        projected_features = F.normalize(projected_features, dim=-1)
        weights = one_image_token_prob.to(device=features.device, dtype=features.dtype)
        weighted_features = projected_features.mul(weights.unsqueeze(-1))

        # For L ~= Z Z^T, the DPP marginal kernel is
        # K = Z (I + Z^T Z)^-1 Z^T. Its diagonal gives the one-shot
        # determinantal leverage score for every token.
        dual_kernel = torch.mm(weighted_features.t(), weighted_features)
        dual_kernel.diagonal().add_(1.0)
        cholesky_factor = torch.linalg.cholesky(dual_kernel)
        whitened_features = torch.linalg.solve_triangular(
            cholesky_factor,
            weighted_features.t(),
            upper=False,
        )
        marginal_scores = whitened_features.square().sum(dim=0).clamp_(min=0.0, max=1.0)
    return torch.topk(marginal_scores, final_keep_num, dim=-1).indices


def _select_with_dpp(one_image_token_prob, similarity_matrix, final_keep_num):
    num_tokens = one_image_token_prob.numel()
    if final_keep_num >= num_tokens:
        return torch.arange(num_tokens, device=one_image_token_prob.device)

    # Build the DPP kernel L = diag(w) S diag(w), where w_i is the VIP score
    # after sigmoid and S is the single-layer visual-token similarity matrix.
    kernel = one_image_token_prob.unsqueeze(1) * similarity_matrix.float() * one_image_token_prob.unsqueeze(0)
    cis = torch.zeros((final_keep_num, num_tokens), device=kernel.device, dtype=kernel.dtype)
    di2s = torch.diagonal(kernel).clone()
    selected_indices = []

    for step in range(final_keep_num):
        # Greedy MAP step: pick the token with the largest current marginal gain.
        selected_idx = torch.argmax(di2s).item()
        if not torch.isfinite(di2s[selected_idx]) or di2s[selected_idx] <= 0:
            # Numerical fallback: if the DPP gain degenerates, fill the rest by
            # the base importance weights instead of failing the selection.
            remaining_keep_num = final_keep_num - len(selected_indices)
            if remaining_keep_num > 0:
                remaining_scores = one_image_token_prob.clone()
                if selected_indices:
                    remaining_scores[torch.tensor(selected_indices, device=remaining_scores.device)] = -float("inf")
                remaining = torch.topk(remaining_scores, remaining_keep_num, dim=-1).indices
                selected_indices.extend(remaining.tolist())
            break

        selected_indices.append(selected_idx)
        if step == final_keep_num - 1:
            break

        if step == 0:
            eis = kernel[selected_idx] / torch.sqrt(di2s[selected_idx].clamp_min(1e-12))
        else:
            prev_cis = cis[:step, selected_idx]
            eis = (kernel[selected_idx] - torch.einsum("t,tj->j", prev_cis, cis[:step])) / torch.sqrt(
                di2s[selected_idx].clamp_min(1e-12)
            )
        cis[step] = eis
        # Update the remaining marginal gains after conditioning on the newly
        # selected token. This is the core greedy DPP MAP update.
        di2s = di2s - eis.square()
        di2s[selected_idx] = -float("inf")

    selected_tensor = torch.tensor(selected_indices, device=one_image_token_prob.device, dtype=torch.long)
    return selected_tensor


class Qwen2_5_VLAttention_GP(Qwen2_5_VLAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        q_indices=None,
        kv_mask=None,
        use_attention_logits=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        if output_attentions:
            attn_weights_out = attn_weights.clone()  # [bsz, nheads, q_len, k_len]
            if q_indices is not None:
                attn_weights_out = attn_weights_out[list(range(bsz)), :, q_indices, :].unsqueeze(2)  # [bsz, nheads, 1, k_len]
            if not use_attention_logits:
                attn_weights_out = nn.functional.log_softmax(attn_weights, dim=-1)
            if kv_mask is not None:
                attn_weights_out = attn_weights_out.squeeze(2)
                attn_weights_out = attn_weights_out.transpose(1, 2)
                selected_attn_weights = attn_weights_out[kv_mask]  # [N, nheads]
                kv_length = kv_mask.sum(dim=-1)
                attn_weights_out = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        else:
            attn_weights_out = None
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_out, past_key_value

class Qwen2_5_VLSdpaAttention_GP(Qwen2_5_VLSdpaAttention):

    def _cal_attn_weights(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor]=None,
                          q_indices: Optional[List[int]]=None,
                          kv_mask: Optional[torch.Tensor]=None,
                          use_attention_logits: bool = False
                          ):
        bsz, nheads, _, head_dim = query_states.size()
        selected_query_states = query_states[list(range(bsz)), :, q_indices, :].view(bsz, nheads, 1, head_dim)
        attn_weights = torch.matmul(selected_query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [bsz, nheads, 1, k_len]
        if not use_attention_logits:
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    min_dtype = torch.finfo(attn_weights.dtype).min
                    attention_mask_float = torch.full_like(attention_mask, min_dtype, dtype=attn_weights.dtype)
                    attention_mask_float.masked_fill_(attention_mask, 0.0)
                else:
                    attention_mask_float = attention_mask
                attn_weights += attention_mask_float
            attn_weights = torch.log_softmax(attn_weights, dim=-1)  # [bsz, nheads, 1, k_len]
        if kv_mask is not None:
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.transpose(1, 2)
            selected_attn_weights = attn_weights[kv_mask]  # [N, nheads]
            kv_length = kv_mask.sum(dim=-1)
            attn_weights = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        q_indices=None,
        kv_mask=None,
        use_attention_logits=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if output_attentions:
            attn_weights = self._cal_attn_weights(query_states, key_states, attention_mask,
                                                  q_indices=q_indices, kv_mask=kv_mask, use_attention_logits=use_attention_logits)
        else:
            attn_weights = None

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFlashAttention2_GP(Qwen2_5_VLFlashAttention2):
    # @time_logger
    def _cal_attn_weights(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor]=None,
                          q_indices: Optional[List[int]]=None,
                          kv_mask: Optional[torch.Tensor]=None,
                          use_attention_logits: bool = False
                          ):
        bsz, nheads, _, head_dim = query_states.size()
        selected_query_states = query_states[list(range(bsz)), :, q_indices, :].view(bsz, nheads, 1, head_dim)
        attn_weights = torch.matmul(selected_query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [bsz, nheads, 1, k_len]
        if not use_attention_logits:
            if attention_mask is not None:
                attention_mask_4d = convert_2d_to_4d_mask(attention_mask, 1, dtype=attn_weights.dtype)
                attn_weights = attn_weights + attention_mask_4d  # [bsz, nheads, 1, k_len]
            attn_weights = torch.log_softmax(attn_weights, dim=-1)  # [bsz, nheads, 1, k_len]
        if kv_mask is not None:
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.transpose(1, 2)  # [bsz, k_len, nheads]
            selected_attn_weights = attn_weights[kv_mask]  # [N, nheads]
            kv_length = kv_mask.sum(dim=-1)
            attn_weights = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        return attn_weights

    def forward(self,
                hidden_states,
                attention_mask = None,
                position_ids = None,
                past_key_value = None,
                output_attentions = False,
                use_cache = False,
                cache_position = None,
                position_embeddings = None,
                q_indices = None,
                kv_mask = None,
                use_attention_logits=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads, q_len, head_dim]
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads_kv, q_len, head_dim]
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads_kv, q_len, head_dim]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        if output_attentions:
            attn_weights = self._cal_attn_weights(query_states, key_states, attention_mask,
                                                  q_indices=q_indices, kv_mask=kv_mask, use_attention_logits=use_attention_logits)
        else:
            attn_weights = None

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)


        return attn_output, attn_weights, past_key_value


QWEN2_5_VL_GP_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention_GP,
    "sdpa": Qwen2_5_VLSdpaAttention_GP,
    "flash_attention_2": Qwen2_5_VLFlashAttention2_GP,
}


class Qwen2_5_VLDecoderLayer_GP(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        nn.Module.__init__(self)
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_VL_GP_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(self,
                hidden_states,
                attention_mask = None,
                position_ids = None,
                past_key_value = None,
                output_attentions = False,
                use_cache = False,
                cache_position = None,
                position_embeddings = None,
                q_indices = None,
                kv_mask = None,
                use_attention_logits=False,
                **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_indices=q_indices,
            kv_mask=kv_mask,
            use_attention_logits=use_attention_logits
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class Qwen2_5_VLModel_GP(Qwen2_5_VLModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer_GP(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2_5_VL_GP_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    # fix inference (for better speed?)

    config_class = Qwen2_5_VL_GPConfig
    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel_GP(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.reset_image_tokens_cache()
        self._init_new_modules(config)
        self.post_init()

    def _init_new_modules(self, config, re_init=False):
        self.config = config
        # NOTE: config of other sub-modules is not overwrited !


        selected_layers = self.config.selected_layers
        reduce_layer = self.config.reduce_layer  # do reduce after this layer
        le_layers = self.config.le_layers

        max_selected_layers = max(selected_layers) if len(selected_layers) > 0 else 0
        max_le_layers = max(le_layers) if len(le_layers) > 0 else 0

        assert max_le_layers < len(self.model.layers), f"max_le_layers {le_layers} must be less than the number of layers {len(self.model.layers)}"
        assert max_selected_layers <= reduce_layer, f"selected_layers {selected_layers} must be less than or equal to reduce_layer {reduce_layer}"


        if re_init:
            dtype = next(self.model.embed_tokens.parameters()).dtype
            new_module_to_device = {}
            for name, module in self.new_modules_to_be_loaded().items():
                if isinstance(module, nn.Parameter):
                    new_module_to_device[name] = module.device
                else:
                    new_module_to_device[name] = next(module.parameters()).device
            for name, module in self.new_modules_to_be_saved().items():
                if isinstance(module, nn.Parameter):
                    new_module_to_device[name] = module.device
                else:
                    new_module_to_device[name] = next(module.parameters()).device
        try:
            self.attn_fuser = ATTN_FUSER_REGISTRY[config.attn_fuse_type](config)
        except KeyError:
            raise ValueError(f"AttnFuser {config.attn_fuse_type} not found in registry. Available options: {list(ATTN_FUSER_REGISTRY.keys())}")

        if len(self.config.le_layers) > 0 and self.config.le_length > 0:
            self.register_parameter(
                "perturb_term",
                nn.Parameter(torch.empty(len(self.config.le_layers), self.config.le_length, self.config.hidden_size))
            )
            self.le_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            if self.config.le_norm_type == "rmsnorm":
                self.le_norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            elif self.config.le_norm_type == "layernorm":
                self.le_norm = nn.LayerNorm(self.config.hidden_size)
            else:
                raise ValueError(f"Unsupported le_norm_type: {self.config.le_norm_type}. Supported types: 'rmsnorm', 'layernorm'.")
            self.le_dropout = nn.Dropout(self.config.le_dropout_prob)
        else:
            if hasattr(self, "perturb_term"):
                del self.perturb_term
            if hasattr(self, "le_proj"):
                del self.le_proj
            if hasattr(self, "le_norm"):
                del self.le_norm
            if hasattr(self, "le_dropout"):
                del self.le_dropout

        # convert all new modules to the same device as attn_head_weights
        if re_init:
            for name, module in self.new_modules_to_be_loaded().items():
                device = new_module_to_device[name]
                if isinstance(module, nn.Parameter):
                    module.data = module.data.to(device=device, dtype=dtype)
                else:
                    module.to(device=device, dtype=dtype)
            for name, module in self.new_modules_to_be_saved().items():
                device = new_module_to_device[name]
                if isinstance(module, nn.Parameter):
                    module.data = module.data.to(device=device, dtype=dtype)
                else:
                    module.to(device=device, dtype=dtype)


    def peft_target_modules(self):
        target_modules = {}
        for name, module in self.model.named_modules():
            actual_name = "model." + name
            if "embed_tokens" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                target_modules[actual_name] = module
        return target_modules


    def new_modules_to_be_loaded(self):
        return {}


    def new_modules_to_be_saved(self):
        rtn = {
            "attn_fuser": self.attn_fuser,
        }
        if hasattr(self, "perturb_term"):
            rtn.update({
                "perturb_term": self.perturb_term,
                "le_proj": self.le_proj,
                "le_norm": self.le_norm,
            })
        return rtn

    def _init_weights(self, module):
        super()._init_weights(module)
        # init new modules and parameters
        if hasattr(self, "perturb_term"):
            nn.init.normal_(self.perturb_term, 0, 0.02)

        for name, module in self.new_modules_to_be_saved().items():
            if isinstance(module, nn.Module):
                if hasattr(module, "_init_weights"):
                    module._init_weights()
                else:
                    for p_name, param in module.named_parameters():
                        if 'norm' in name or 'norm' in p_name:
                            if 'weight' in p_name:
                                nn.init.ones_(param)
                            elif 'bias' in p_name:
                                nn.init.zeros_(param)
                        elif 'proj' in name or 'proj' in p_name:
                            if 'weight' in p_name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in p_name:
                                nn.init.zeros_(param)


    def save_new_modules(self, save_directory):
        #save config
        self.config.save_pretrained(save_directory)

        # save the new modules
        is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0
        if is_main_process:
            new_states = {}
            for name, module in self.new_modules_to_be_saved().items():
                if isinstance(module, nn.Parameter):
                    new_states[name] = module.data
                else:
                    new_states[name] = module.state_dict()
            for name, module in self.new_modules_to_be_loaded().items():
                if isinstance(module, nn.Parameter):
                    new_states[name] = module.data
                else:
                    new_states[name] = module.state_dict()
            torch.save(new_states, os.path.join(save_directory, "new_modules_rp.pt"))
            print(f"new_modules of {self.__class__.__name__} saved to {os.path.join(save_directory, 'new_modules_rp.pt')}")


    def load_new_modules(self, load_directory):
        if not os.path.isdir(load_directory):
            load_directory = download_model_from_hf(load_directory)
        # load config
        config_path = os.path.join(load_directory, "config.json")
        config = self.config_class.from_json_file(config_path)
        new_modules_path = os.path.join(load_directory, "new_modules_rp.pt")
        if not os.path.exists(new_modules_path):
            legacy_new_modules_path = os.path.join(load_directory, "new_modules_gp.pt")
            if os.path.exists(legacy_new_modules_path):
                new_modules_path = legacy_new_modules_path
        new_modules = None
        if os.path.exists(new_modules_path):
            new_modules = torch.load(new_modules_path, weights_only=True)
            if "perturb_term" not in new_modules and "learnable_embeddings" in new_modules:
                new_modules["perturb_term"] = new_modules.pop("learnable_embeddings")
            attn_fuser_state_dict = new_modules.get("attn_fuser")
            has_penalty_head_weights = isinstance(attn_fuser_state_dict, dict) and any(
                key.startswith("penalty_head.") or key.startswith("penalty_heads.")
                for key in attn_fuser_state_dict
            )
            if has_penalty_head_weights and not config.use_uncertainty_penalty_head:
                warnings.warn(
                    f"{new_modules_path} contains penalty head weights, but config disables them. "
                    "Enabling use_uncertainty_penalty_head to match the checkpoint."
                )
                config.use_uncertainty_penalty_head = True
            elif not has_penalty_head_weights and config.use_uncertainty_penalty_head:
                warnings.warn(
                    f"{new_modules_path} does not contain penalty head weights. "
                    "Disabling use_uncertainty_penalty_head so this checkpoint will not produce uncertainty scores."
                )
                config.use_uncertainty_penalty_head = False

        self._init_new_modules(config, re_init=True)
        # load
        if new_modules is not None:
            new_modules_in_ckpt = set(new_modules.keys())

            def _load_state_dict_with_compat(module_name, module, state_dict):
                try:
                    incompatible = module.load_state_dict(state_dict, strict=True)
                    return incompatible
                except RuntimeError as e:
                    err_msg = str(e)
                    if "Unexpected key(s) in state_dict" not in err_msg and "Missing key(s) in state_dict" not in err_msg:
                        raise e
                    incompatible = module.load_state_dict(state_dict, strict=False)
                    missing_keys = incompatible.missing_keys
                    unexpected_keys = incompatible.unexpected_keys
                    warnings.warn(
                        f"Loading {module_name} from {new_modules_path} with strict=False due to state_dict schema drift. "
                        f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
                    )
                    return incompatible

            for name, module in self.new_modules_to_be_loaded().items():
                try:
                    if isinstance(module, nn.Parameter):
                        module.data.copy_(new_modules[name])
                    else:
                        _load_state_dict_with_compat(name, module, new_modules[name])
                except Exception as e:
                    print(f"Failed to load new modules {name}: {e}")
                    raise e
                new_modules_in_ckpt.discard(name)
                print_rank0(f"new_modules of {self.__class__.__name__}.{name} loaded from {new_modules_path}")
            for name in new_modules_in_ckpt:
                module = getattr(self, name)
                try:
                    if isinstance(module, nn.Parameter):
                        module.data.copy_(new_modules[name])
                    else:
                        _load_state_dict_with_compat(name, module, new_modules[name])
                except Exception as e:
                    print(f"Failed to load new modules {name}: {e}")
                    raise e
                print_rank0(f"new_modules of {self.__class__.__name__}.{name} loaded from {new_modules_path}")
        else:
            warnings.warn(f"new_modules_rp.pt not found in {load_directory}.")


    def reset_image_tokens_cache(self):
        self.todo_selection = False
        self.glimpse_return_before_selection = None
        self.reduced_input_ids = None


    def _check_padding_side(self, attention_mask: torch.LongTensor, default_side='right') -> str:
        """
        Determines the padding side based on the attention mask.

        Args:
            attention_mask (torch.LongTensor): Attention mask [B, L]. 1 for non-padding, 0 for padding.
            default_side (str): Default padding side to return if unclear.

        Returns:
            str: 'right' or 'left'.
        """
        B, L = attention_mask.shape
        if L == 0:
            return default_side

        # Check if any sequence has actual content
        has_content = attention_mask.sum(dim=1) > 0

        # If no sequence has content, padding side doesn't matter much
        if not torch.any(has_content):
            return default_side

        # Check if all sequences *with content* start with 1 (suggests right padding)
        starts_with_one = attention_mask[has_content, 0].all()
        # Check if all sequences *with content* end with 1 (suggests left padding)
        ends_with_one = attention_mask[has_content, -1].all()

        if not starts_with_one and ends_with_one:
            return 'left'
        elif starts_with_one and not ends_with_one:
            raise NotImplementedError(f"Unsupported padding side: right")
        elif starts_with_one and ends_with_one:
            # Ambiguous: Could be right padding on sequences shorter than L,
            # or left padding on sequences shorter than L, or no padding needed.
            # Defaulting to 'right' is usually safer for truncation.
            # Check if any padding exists at all
            if torch.any(attention_mask[has_content] == 0):
                 warnings.warn(
                    "Could not definitively determine padding side from attention_mask; "
                    f"assuming '{default_side}'. If this is incorrect, results may be wrong.",
                    UserWarning
                 )
                 raise NotImplementedError(f"Unsupported padding side: uncontinuous")
            return default_side
        else:
            # Both start and end with 0 for some sequences, unusual padding.
            warnings.warn(
                "Attention mask suggests padding could be on both sides or is inconsistent. "
                f"Assuming '{default_side}' padding. If this is incorrect, results may be wrong.",
                 UserWarning
            )
            raise NotImplementedError(f"Unsupported padding side: both")

    def _get_query_summary_token(self, input_ids, inputs_embeds, attention_mask, labels=None):
        text_token_mask = attention_mask.bool()
        text_token_mask &= input_ids != self.config.image_token_id
        text_token_mask &= input_ids != self.config.video_token_id
        if labels is not None:
            text_token_mask &= labels.to(text_token_mask.device) == -100

        text_token_mask = text_token_mask.to(inputs_embeds.device)
        text_token_mask_float = text_token_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
        token_count = text_token_mask.sum(dim=1).clamp_min(1).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        query_summary_token = (inputs_embeds * text_token_mask_float).sum(dim=1) / token_count.unsqueeze(-1)
        return query_summary_token

    def _try_add_le(self, layer_id, hidden_states, q_indices, query_summary_token):
        """
        Add query-based tokens to hidden states if the layer ID matches the
        configured query-based token layer.
        """
        try:
            le_idx = self.config.le_layers.index(layer_id)
        except ValueError:
            return hidden_states
        perturb_term = self.perturb_term[le_idx]
        perturb_term = perturb_term.to(device=self.le_proj.weight.device)
        perturb_dtype = perturb_term.dtype
        perturb_term = self.le_dropout(self.le_norm(self.le_proj(perturb_term)))
        perturb_term = perturb_term.to(dtype=perturb_dtype)


        bsz, seq_len, hidden_size = hidden_states.shape
        le_len = self.config.le_length
        device = hidden_states.device

        query_summary_token = query_summary_token.to(device=device, dtype=perturb_term.dtype)
        query_based_l = perturb_term.view(1, le_len, hidden_size).expand(bsz, -1, -1)
        query_based_l = query_based_l.to(device=device)
        query_based_l = query_based_l + query_summary_token.unsqueeze(1)
        query_based_l = query_based_l.to(dtype=hidden_states.dtype)

        # `q_indices` is a list, convert it to a tensor of shape (bsz, 1)
        end_indices = torch.tensor(q_indices, device=device, dtype=torch.long).unsqueeze(1) + 1
        start_indices = end_indices - le_len

        # Create a range of offsets [0, 1, ..., le_len-1]
        le_range = torch.arange(le_len, device=device, dtype=torch.long).unsqueeze(0)

        # Calculate the target sequence index for each position in the le window for each batch item.
        # Shape: (bsz, le_len)
        target_seq_indices = start_indices + le_range

        # Create a mask to filter out-of-bounds additions
        # This mask is True only for `query_based_l` elements whose target position is within [0, seq_len-1]
        # This gracefully handles cases where q_indices[i] < le_len
        mask = (target_seq_indices >= 0) & (target_seq_indices < seq_len)

        # Select only the valid query-based token values and their target indices.
        # `query_based_l[mask]` flattens the selected values. Shape: (num_valid, hidden_size)
        valid_query_based_l_to_add = query_based_l[mask]

        # We also need the batch index for each valid value
        batch_idx_tensor = torch.arange(bsz, device=device).unsqueeze(1).expand_as(mask)
        valid_batch_indices = batch_idx_tensor[mask] # Shape: (num_valid,)
        valid_target_seq_indices = target_seq_indices[mask] # Shape: (num_valid,)

        # 6. Calculate the flat 1D index for `index_add_`
        # We are adding to a view of hidden_states reshaped to (bsz * seq_len, hidden_size)
        flat_destination_indices = valid_batch_indices * seq_len + valid_target_seq_indices

        # 7. Perform the addition using `index_add_`
        # This is an efficient, in-place operation that adds elements from `valid_query_based_l_to_add`
        # to the specified `flat_destination_indices` of the flattened hidden_states.
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        hidden_states_flat.index_add_(0, flat_destination_indices, valid_query_based_l_to_add)

        # The modification was in-place on a view, so hidden_states is already updated.

        # --- End of Modified Section ---

        return hidden_states



    def _append_le(self, input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position, query_summary_token):
        bsz, seq_len = input_ids.shape
        le_len = self.config.le_length
        device = inputs_embeds.device
        le_idx = self.config.le_layers.index(0)
        perturb_term = self.perturb_term[le_idx]
        perturb_term = perturb_term.to(device=self.le_proj.weight.device)
        perturb_dtype = perturb_term.dtype
        perturb_term = self.le_dropout(self.le_norm(self.le_proj(perturb_term)))
        perturb_term = perturb_term.to(dtype=perturb_dtype)
        query_summary_token = query_summary_token.to(device=inputs_embeds.device, dtype=perturb_term.dtype)
        query_based_l = perturb_term.view(1, le_len, self.config.hidden_size).expand(bsz, -1, -1)
        query_based_l = query_based_l.to(device=inputs_embeds.device)
        query_based_l = query_based_l + query_summary_token.unsqueeze(1)
        query_based_l = query_based_l.to(dtype=inputs_embeds.dtype)
        if labels is None:
            inputs_embeds = torch.cat([inputs_embeds, query_based_l], dim=1)
            input_ids = torch.cat([input_ids, torch.full((bsz, le_len), self.config.eos_token_id, device=input_ids.device)], dim=1)
        else:
            label_mask = labels != -100
            insert_pos = label_mask.int().argmax(dim=-1)

            new_len = seq_len + le_len
            split_idx = insert_pos.unsqueeze(1) # Shape: (bsz, 1)

            # 1. Create the master gather_indices tensor
            # This tensor tells us where to pick elements from a combined (original + le) source
            new_indices = torch.arange(new_len, device=device).unsqueeze(0) # Shape: (1, new_len)

            # Indices for part 1 (before insertion)
            indices_part1 = new_indices
            # Indices for the LE part (shifted to the source's LE block)
            indices_le_part = seq_len + (new_indices - split_idx)
            # Indices for part 2 (shifted back to the source's original part)
            indices_part2 = new_indices - le_len

            # Use masks to select which index rule to apply for each position
            mask_part1 = new_indices < split_idx
            mask_le_part = (new_indices >= split_idx) & (new_indices < split_idx + le_len)

            gather_indices = torch.where(mask_part1, indices_part1, torch.where(mask_le_part, indices_le_part, indices_part2))

            # 2. Apply gather to `inputs_embeds`
            source_embeds = torch.cat([inputs_embeds, query_based_l], dim=1)
            # Expand gather_indices to match the hidden_size dimension for gathering
            gather_indices_embeds = gather_indices.unsqueeze(2).expand(-1, -1, self.config.hidden_size)
            inputs_embeds = torch.gather(source_embeds, 1, gather_indices_embeds)

            # 3. Apply gather to `input_ids`, `labels`, `attention_mask`
            # Create source tensors by concatenating original data with LE placeholder data
            le_ids = torch.full((bsz, le_len), self.config.eos_token_id, device=input_ids.device, dtype=input_ids.dtype)
            source_ids = torch.cat([input_ids, le_ids], dim=1)
            input_ids = torch.gather(source_ids, 1, gather_indices)

            le_labels = torch.full((bsz, le_len), -100, device=labels.device, dtype=labels.dtype)
            source_labels = torch.cat([labels, le_labels], dim=1)
            labels = torch.gather(source_labels, 1, gather_indices)

        # It is simpler to deal with attention_mask, position_ids and cache_position
        attention_mask = torch.cat([attention_mask, torch.ones((bsz, le_len), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        le_pos_ids = []
        for b in range(bsz):
            one_last_pos_idx = position_ids[-1, b, -1].item()
            one_le_pos_ids = torch.arange(one_last_pos_idx + 1, one_last_pos_idx + 1 + le_len, device=position_ids.device)
            one_le_pos_ids = one_le_pos_ids.view(1, 1, -1).expand(3, -1, -1)
            le_pos_ids.append(one_le_pos_ids)
        le_pos_ids = torch.cat(le_pos_ids, dim=1)
        position_ids = torch.cat([position_ids, le_pos_ids], dim=2)
        last_cache_position = cache_position[-1].item()
        cache_position = torch.cat([cache_position, torch.arange(last_cache_position + 1, last_cache_position + 1 + le_len, device=cache_position.device)], dim=-1)


        return input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position


    # @time_logger
    def _decode_image_token_mask_logits(
        self,
        batched_attn_map,
        attn_grid,
        selected_image_embeds,
        similarity_image_embeds,
        window_index,
        cu_seqlens,
        cu_window_seqlens,
    ):
        """
        batched_attn_map: list(bsz) of [num_tokens, num_layers, num_heads]
        grid_size: [bsz, 2] (height, width)
        selected_image_embeds: list(num_selected_visual_layers) of [N, hidden_size]
        similarity_image_embeds: [N, hidden_size]
        """
        num_tokens_by_batch = [attn_map.shape[0] for attn_map in batched_attn_map]
        catted_attn_map = torch.cat(batched_attn_map, dim=0)  # [N, num_layers, num_heads]
        N = catted_attn_map.shape[0]
        catted_attn_map = catted_attn_map.view(N, -1)
        # DPP uses one S_ij per image, built from a single configured visual layer.
        image_token_similarity_matrices = _build_similarity_matrices_from_visual_embeds(
            similarity_image_embeds, num_tokens_by_batch
        )
        catted_attn_map, catted_penalties = self.attn_fuser(
            catted_attn_map,
            attn_grid,
            selected_image_embeds,
            window_index,
            cu_seqlens,
            cu_window_seqlens,
        )

        selected_masks = catted_attn_map.split(num_tokens_by_batch, dim=-1)  # list(bsz) of [num_layers, num_tokens]
        selected_penalties = catted_penalties.split(num_tokens_by_batch, dim=-1) if catted_penalties is not None else None
        return selected_masks, selected_penalties, image_token_similarity_matrices

    @time_logger
    def _glimpse_forward(
        self,
        input_ids,
        inputs_embeds,
        labels,
        position_ids,
        cache_position,
        attention_mask,
        past_key_values,
        use_cache,
        image_info,
        image_grid_thw,
        ref_token_masks,
        return_dict,
        delay_selection,
        use_ref_masks,
    ):
        # Step 1. Check inputs
        # Check padding left
        self._check_padding_side(attention_mask)

        # Other check
        if self.model.gradient_checkpointing and self.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Step 2. Build query-based tokens from the initial text embeddings.
        if not use_ref_masks and hasattr(self, "perturb_term"):
            query_summary_token = self._get_query_summary_token(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position = self._append_le(
                input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position, query_summary_token)
        else:
            query_summary_token = None

        # Step 3. Prepare other inputs for decoder
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions=False
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # q_indices, kv_mask decide the shape of extracted attentions
        bsz, seq_len = input_ids.shape
        if labels is None:
            q_indices = [seq_len - 1] * bsz
        else:
            label_mask = labels != -100
            q_indices = label_mask.int().argmax(dim=-1) - 1
            q_indices = q_indices.tolist()
        kv_mask = input_ids == self.config.image_token_id


        selected_layers = tuple(self.config.selected_layers)
        if len(selected_layers) == 0:
            batched_attns = None
        else:
            batched_attns = [[None] * len(selected_layers) for _ in range(bsz)]

        # Step 4. Extract attentions during the forward pass of LLM
        next_decoder_cache = None
        max_forward_layer = max(self.config.selected_layers) if len(self.config.selected_layers) > 0 else 0
        max_forward_layer = max(max_forward_layer, self.config.reduce_layer)
        if labels is not None:
            max_forward_layer = len(self.model.layers) - 1  # always forward all layers if labels are provided

        hidden_states_for_reduction = None
        kv_cache_for_reduction = None

        for layer_id, decoder_layer in enumerate(self.model.layers):
            if layer_id > 0 and query_summary_token is not None:  # layer 0 query-based token has been added
                hidden_states = self._try_add_le(layer_id, hidden_states, q_indices, query_summary_token)
            try:
                layer_pos = self.config.selected_layers.index(layer_id)
            except ValueError:
                layer_pos = None
            if layer_pos is not None and not use_ref_masks:
                _output_attentions = True
            else:
                _output_attentions = False

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    _output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    q_indices,
                    kv_mask,
                    self.config.use_attention_logits,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=_output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    q_indices=q_indices,
                    kv_mask=kv_mask,
                    use_attention_logits=self.config.use_attention_logits,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[-1]
            if _output_attentions:
                attn_weights = layer_outputs[1]
                for b in range(bsz):
                    batched_attns[b][layer_pos] = attn_weights[b]

            if hidden_states_for_reduction is None:
                if layer_id == self.config.reduce_layer and layer_id < len(self.model.layers) - 1:
                    if layer_id >= max_forward_layer:
                        hidden_states_for_reduction = hidden_states
                        kv_cache_for_reduction = next_decoder_cache if use_cache else None
                    else:
                        hidden_states_for_reduction = hidden_states.clone()
                        if use_cache:
                            kv_cache_for_reduction = DynamicCache()
                            for layer_idx, (key_cache, value_cache) in enumerate(next_decoder_cache):
                                kv_cache_for_reduction.update(key_cache.clone(), value_cache.clone(), layer_idx)
                        else:
                            kv_cache_for_reduction = None
            if layer_id >= max_forward_layer:
                break

        if max_forward_layer >= len(self.model.layers) - 1:
            hidden_states = self.model.norm(hidden_states)

        if hidden_states_for_reduction is None:
            hidden_states_for_reduction = hidden_states
            kv_cache_for_reduction = next_decoder_cache if use_cache else None

        if labels is not None:
            le_logits = self.lm_head(hidden_states).float()
            shift_le_logits = le_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_le_logits = shift_le_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_le_logits.device)
            le_loss = loss_fct(shift_le_logits, shift_labels)
            del le_logits, shift_le_logits, shift_labels
        else:
            le_loss = None

        del hidden_states
        del next_decoder_cache

        # Step 5. Decode image tokens
        if batched_attns is not None and not use_ref_masks:
            for i, one_attns in enumerate(batched_attns):
                batched_attns[i] = torch.stack(one_attns, dim=1)  # [num_tokens, num_layers, num_heads]
        attn_grid = image_grid_thw[:, 1:] // self.config.vision_config.spatial_merge_size

        if use_ref_masks:
            image_token_mask_logits = []
            for i in range(len(attn_grid)):
                image_token_mask_logits.append(torch.logit(ref_token_masks[i].float().to(device=hidden_states_for_reduction.device).view(1, -1)))
            image_token_penalties = None
            image_token_similarity_matrices = None
        elif self.config.use_zero_masks:
            image_token_mask_logits = []
            for i in range(len(attn_grid)):
                image_token_mask_logits.append(torch.logit(torch.zeros((1, attn_grid[i][0] * attn_grid[i][1]), device=hidden_states_for_reduction.device)))
            image_token_penalties = None
            image_token_similarity_matrices = None
        else:
            image_token_mask_logits, image_token_penalties, image_token_similarity_matrices = self._decode_image_token_mask_logits(
                batched_attns,
                attn_grid,
                **image_info,
            )

        # Step 6. Trim reduced image tokens and query-based tokens.
        if not use_ref_masks and hasattr(self, "perturb_term"):
            le_length = self.config.le_length
            input_ids = input_ids[:, :-le_length]
            inputs_embeds = inputs_embeds[:, :-le_length, :]
            hidden_states_for_reduction = hidden_states_for_reduction[:, :-le_length]
            # Assume past_key_values is DynamicCache
            if kv_cache_for_reduction is not None:
                # assert isinstance(next_cache, DynamicCache)
                kv_cache_for_reduction.crop(-le_length)
            position_ids = position_ids[:, :, :-le_length]
            attention_mask = attention_mask[:, :-le_length]

        if delay_selection:
            self.todo_selection = True
            logits = None
            rtn_dict = Qwen2_5_VL_GP_CausalLMOutputWithPast(
                    logits=logits,
                    le_loss=le_loss,
                    past_key_values=kv_cache_for_reduction,
                    hidden_states=hidden_states_for_reduction,
                    rope_deltas=self.rope_deltas,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    attn_grid=attn_grid,
                    image_token_mask_logits=image_token_mask_logits,
                    image_token_penalties=image_token_penalties,
                    # Keep the precomputed DPP similarities so delayed
                    # selection reuses the exact same subset-selection inputs.
                    image_token_similarity_matrices=image_token_similarity_matrices,
                )
            self.glimpse_return_before_selection = rtn_dict
            if return_dict:
                return rtn_dict
            else:
                return (
                    logits,
                    kv_cache_for_reduction,
                    hidden_states_for_reduction,
                    self.rope_deltas,
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    image_token_mask_logits,
                    image_token_penalties,
                    image_token_similarity_matrices,
                )
        else:
            reduced_info = self._reduce_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states_for_reduction,
                past_key_values=kv_cache_for_reduction,
                position_ids=position_ids,
                attention_mask=attention_mask,
                image_token_mask_logits=image_token_mask_logits,
                image_token_penalties=image_token_penalties,
                image_token_similarity_matrices=image_token_similarity_matrices,
                attn_grid=attn_grid,
            )
            return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)

    def _do_delayed_selection(self, override_image_token_mask_logits, return_dict, use_cache, do_clone=False):
        assert self.todo_selection, "No delayed selection to do."
        self.todo_selection = False
        if do_clone:
            self.todo_selection = True   # Can be called more than once
            cached_input_ids = self.glimpse_return_before_selection.input_ids.clone()
            cached_inputs_embeds = self.glimpse_return_before_selection.inputs_embeds.clone()
            cached_hidden_states = self.glimpse_return_before_selection.hidden_states.clone()
            cached_attention_mask = self.glimpse_return_before_selection.attention_mask.clone()
            cached_position_ids = self.glimpse_return_before_selection.position_ids.clone()
            cached_past_key_values = DynamicCache()
            for layer_idx, (key_cache, value_cache) in enumerate(self.glimpse_return_before_selection.past_key_values):
                cached_past_key_values.update(key_cache.clone(), value_cache.clone(), layer_idx)
            attn_grid = self.glimpse_return_before_selection.attn_grid
        else:
            cached_input_ids = self.glimpse_return_before_selection.input_ids
            cached_inputs_embeds = self.glimpse_return_before_selection.inputs_embeds
            cached_hidden_states = self.glimpse_return_before_selection.hidden_states
            cached_attention_mask = self.glimpse_return_before_selection.attention_mask
            cached_position_ids = self.glimpse_return_before_selection.position_ids
            cached_past_key_values = self.glimpse_return_before_selection.past_key_values
            attn_grid = self.glimpse_return_before_selection.attn_grid
            self.reset_image_tokens_cache()

        reduced_info = self._reduce_tokens(
            input_ids=cached_input_ids,
            inputs_embeds=cached_inputs_embeds,
            hidden_states=cached_hidden_states,
            past_key_values=cached_past_key_values,
            position_ids=cached_position_ids,
            attention_mask=cached_attention_mask,
            image_token_mask_logits=override_image_token_mask_logits,
            image_token_penalties=self.glimpse_return_before_selection.image_token_penalties,
            image_token_similarity_matrices=self.glimpse_return_before_selection.image_token_similarity_matrices,
            attn_grid=attn_grid,
            )
        return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)


    def _get_remain_masks(self, input_ids, attention_mask, image_token_mask_logits, image_token_penalties, image_token_similarity_matrices, attn_grid):
        threshold = self.config.reduce_threshold
        anchor_positions = list(self.config.anchor_positions) if self.config.anchor_positions is not None else []
        min_remain_num = self.config.min_remain_num
        max_remain_ratio = self.config.max_remain_ratio
        base_score_pool_multiplier = getattr(self.config, "base_score_pool_multiplier", None)
        selection_strategy = getattr(self.config, "image_token_selection_strategy", "default")

        # image_token_lengths = [logits.shape[1] for logits in image_token_mask_logits]
        image_token_bool_masks = []

        for b, one_logits in enumerate(image_token_mask_logits):
            one_penalties = None
            if image_token_penalties is not None:
                one_penalties = image_token_penalties[b][-1].to(one_logits.device)
            one_similarity_matrix = None
            if image_token_similarity_matrices is not None:
                one_similarity_matrix = image_token_similarity_matrices[b].to(one_logits.device)
            # Route each image through the configured selector. In DPP mode,
            # one_logits[-1] supplies w_i and one_similarity_matrix supplies S_ij.
            one_image_token_bool_mask, _, _ = _select_image_tokens_by_strategy(
                score_logits=one_logits[-1],
                penalties=one_penalties,
                similarity_matrix=one_similarity_matrix,
                max_remain_ratio=max_remain_ratio,
                base_score_pool_multiplier=base_score_pool_multiplier,
                threshold=threshold,
                min_remain_num=min_remain_num,
                selection_strategy=selection_strategy,
            )

            if anchor_positions:
                if attn_grid.shape[0] != len(image_token_mask_logits):
                    raise NotImplementedError("anchor positions are not supported when using multi-images input")
                one_attn_height, one_attn_width = attn_grid[b]
                for anchor_pos in anchor_positions:
                    if anchor_pos == 'tl':
                        one_image_token_bool_mask[0] = True
                    elif anchor_pos == 'tr':
                        anchor_pos_id = one_attn_width - 1
                        one_image_token_bool_mask[anchor_pos_id] = True
                    elif anchor_pos == 'bl':
                        anchor_pos_id = (one_attn_height - 1) * one_attn_width
                        one_image_token_bool_mask[anchor_pos_id] = True
                    elif anchor_pos == 'br':
                        anchor_pos_id = one_attn_height * one_attn_width - 1
                        one_image_token_bool_mask[anchor_pos_id] = True
                    else:
                        raise ValueError(f"Unknown anchor position: {anchor_pos}. Supported: tl, tr, bl, br.")

            image_token_bool_masks.append(one_image_token_bool_mask)

        is_image_mask = input_ids == self.config.image_token_id
        remain_masks = attention_mask.clone().bool()
        remain_masks[is_image_mask] = torch.cat(image_token_bool_masks, dim=0)
        remain_masks &= attention_mask.bool()  # Ensure remain_masks is still a valid attention mask, usually all image tokens are valid in attention_mask
        return remain_masks, image_token_bool_masks


    # @time_logger
    def _reduce_tokens(
        self,
        input_ids,
        inputs_embeds,
        hidden_states,
        past_key_values,
        position_ids,
        attention_mask,
        image_token_mask_logits,
        image_token_penalties,
        image_token_similarity_matrices,
        attn_grid,
    ):
        B = input_ids.shape[0]
        if past_key_values is not None:
            old_key_states = past_key_values.key_cache
            old_value_states = past_key_values.value_cache
        else:
            old_key_states = None
            old_value_states = None
            reduced_key_states = None

        remain_masks, image_token_bool_masks = self._get_remain_masks(
            input_ids,
            attention_mask,
            image_token_mask_logits,
            image_token_penalties,
            image_token_similarity_matrices,
            attn_grid,
        )

        lengths = remain_masks.sum(dim=1).cpu().tolist()
        max_remain_len = max(lengths)
        repad_masks = torch.zeros((B, max_remain_len), dtype=torch.bool, device=remain_masks.device)
        for b in range(B):
            repad_masks[b, -lengths[b]:] = True

        reduced_hidden_states = hidden_states[remain_masks]
        reduced_attention_mask = attention_mask[remain_masks]
        reduced_position_ids = position_ids[:, remain_masks]
        reduced_input_ids = input_ids[remain_masks]

        if self.training or past_key_values is None:
            reduced_input_embeds = inputs_embeds[remain_masks]
        else:
            reduced_input_embeds = None

        if past_key_values is not None:
            reduced_key_states = []
            reduced_value_states = []
            for layer_idx, (old_k, old_v) in enumerate(zip(old_key_states, old_value_states)):
                layer_device = old_k.device
                remain_masks_on_same_device = remain_masks.to(layer_device)
                remain_masks_on_same_device = remain_masks_on_same_device.unsqueeze(1).expand(-1, old_k.shape[1], -1)  # [B, num_heads, max_remain_len]
                reduced_key_states.append(old_k[remain_masks_on_same_device])
                reduced_value_states.append(old_v[remain_masks_on_same_device])

        max_remain_len = max(lengths)


        # Step 3: re-pad
        if reduced_input_embeds is not None:
            padded_input_embeds = torch.zeros((B, max_remain_len, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            padded_input_embeds[repad_masks] = reduced_input_embeds
        else:
            padded_input_embeds = None
        pad_token_id = getattr(self.config, "pad_token_id") or 0
        padded_input_ids = torch.full((B, max_remain_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_input_ids[repad_masks] = reduced_input_ids
        padded_hidden_states = torch.zeros((B, max_remain_len, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
        padded_hidden_states[repad_masks] = reduced_hidden_states
        padded_attention_mask = torch.zeros((B, max_remain_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_attention_mask[repad_masks] = reduced_attention_mask

        pos_pad_value = 1
        pos_padded_shape = list(position_ids.shape) # Get original dims structure
        pos_padded_shape[2] = max_remain_len # Set sequence length
        padded_position_ids = torch.full(tuple(pos_padded_shape), pos_pad_value, dtype=position_ids.dtype, device=position_ids.device)
        padded_position_ids[:, repad_masks] = reduced_position_ids

        if past_key_values is not None:
            padded_key_states = []
            padded_value_states = []
            for layer_idx, (old_k, old_v, reduced_k, reduced_v) in enumerate(zip(old_key_states, old_value_states, reduced_key_states, reduced_value_states)):
                k_shape = list(old_k.shape)
                k_shape[-2] = max_remain_len
                v_shape = list(old_v.shape)
                v_shape[-2] = max_remain_len
                repad_masks_on_same_device = repad_masks.to(old_k.device)
                repad_masks_on_same_device = repad_masks_on_same_device.unsqueeze(1).expand(-1, k_shape[1], -1)  # [B, num_heads, max_remain_len]
                new_k = torch.zeros(k_shape, dtype=old_k.dtype, device=old_k.device)
                new_k[repad_masks_on_same_device] = reduced_k
                new_v = torch.zeros(v_shape, dtype=old_v.dtype, device=old_v.device)
                new_v[repad_masks_on_same_device] = reduced_v
                padded_key_states.append(new_k)
                padded_value_states.append(new_v)


        if past_key_values is not None:
            # Because past_key_values is an object, we just overwrite the attribute of it directly.
            past_key_values._seen_tokens = max_remain_len
            past_key_values.key_cache = padded_key_states
            past_key_values.value_cache = padded_value_states

        self.reduced_input_ids = padded_input_ids
        return {
            "input_ids": padded_input_ids,
            "inputs_embeds": padded_input_embeds,
            "hidden_states": padded_hidden_states,
            "past_key_values": past_key_values,
            "position_ids": padded_position_ids,
            "attention_mask": padded_attention_mask,
            "image_token_mask_logits": image_token_mask_logits,
            "image_token_penalties": image_token_penalties,
            "image_token_similarity_matrices": image_token_similarity_matrices,
            "image_token_bool_masks": image_token_bool_masks,
        }



    def _glimpse_forward_after_reduction(
        self,
        input_ids,
        inputs_embeds,
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values,
        use_cache,
        image_token_mask_logits,
        image_token_penalties,
        image_token_similarity_matrices,
        image_token_bool_masks,
        return_dict,
    ):
        max_reduced_len = attention_mask.shape[1]
        # Step 4: If reduce_layer is not the last layer, we need to finish the decoding.
        if self.config.reduce_layer < len(self.model.layers) - 1:
            cache_position = torch.arange(0, max_reduced_len, device=attention_mask.device)
            position_embeddings = self.model.rotary_emb(
                hidden_states, position_ids
            )
            casual_mask = self.model._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions=False
            )

            for layer_id in range(self.config.reduce_layer + 1, len(self.model.layers)):
                decoder_layer = self.model.layers[layer_id]
                if self.model.gradient_checkpointing and self.model.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        casual_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=casual_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states = layer_outputs[0]
                if use_cache:
                    past_key_values = layer_outputs[-1]
            hidden_states = self.model.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        if return_dict:
            return Qwen2_5_VL_GP_CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                rope_deltas=self.rope_deltas,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_token_mask_logits=image_token_mask_logits,
                image_token_penalties=image_token_penalties,
                image_token_similarity_matrices=image_token_similarity_matrices,
                image_token_bool_masks=image_token_bool_masks,
            )

        return (
            logits,
            past_key_values,
            hidden_states,
            self.rope_deltas,
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            image_token_mask_logits,
            image_token_penalties,
            image_token_similarity_matrices,
        )



    def _scatter_image_embeds(self, input_ids, inputs_embeds, image_embeds):
        mask = input_ids == self.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    # @time_logger
    def _visual_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = self.visual.patch_embed(hidden_states)
        rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.visual.get_window_index(grid_thw)
        reverse_indices = torch.argsort(window_index)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        selected_visual_layers = tuple(self.config.selected_visual_layers)
        similarity_visual_layer = getattr(self.config, "dpp_similarity_visual_layer", None)
        if similarity_visual_layer is None:
            selected_layers = tuple(self.config.selected_layers)
            similarity_visual_layer = selected_layers[0] if len(selected_layers) == 1 else None
        selected_hidden_states = [None] * len(selected_visual_layers)
        similarity_hidden_states = None

        def restore_visual_token_order(states):
            states = states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
            states = states.mean(dim=1)
            return states[reverse_indices, :]

        for layer_num, blk in enumerate(self.visual.blocks):
            if layer_num in self.visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.visual.gradient_checkpointing and self.visual.training:
                hidden_states = self.visual._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

            try:
                layer_pos = selected_visual_layers.index(layer_num)
            except ValueError:
                layer_pos = None
            # save selected visual hidden states
            if layer_pos is not None:
                selected_hidden_states[layer_pos] = restore_visual_token_order(hidden_states)
            if similarity_visual_layer is not None and layer_num == similarity_visual_layer:
                similarity_hidden_states = restore_visual_token_order(hidden_states)


        hidden_states = self.visual.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]


        middle_info = {
            "selected_image_embeds": selected_hidden_states,
            "similarity_image_embeds": similarity_hidden_states,
            "window_index": window_index,
            "cu_window_seqlens": cu_window_seqlens,
            "cu_seqlens": cu_seqlens,
        }

        return hidden_states, middle_info

    def text_embed_forward(self, input_ids):
        return self.model.embed_tokens(input_ids)

    # @memory_logger
    @time_logger
    def llm_forward(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs


    @time_logger
    def llm_forward_prefilling(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_token_mask_logits: Optional[List[torch.FloatTensor]] = None,
        token_uncertainty: Optional[List[Any]] = None,
        ref_token_masks: Optional[List[torch.BoolTensor]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        do_selection: bool = True,
        delay_selection: bool = False,
        use_ref_masks: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5_VL_GP_CausalLMOutputWithPast]:
        # `token_uncertainty` is consumed by the trainer-side distillation loss only.
        # It stays in the public signature for batch compatibility, but is not used in model scoring.

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        actual_use_ref_masks = use_ref_masks if use_ref_masks is not None else self.config.use_ref_masks

        if inputs_embeds is None and not self.todo_selection:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, image_info = self._visual_forward(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                inputs_embeds = self._scatter_image_embeds(input_ids, inputs_embeds, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if not self.todo_selection:
            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if do_selection and pixel_values is not None:   # only support single/multi image inputs now
            if image_token_mask_logits is None:   # only do prune at the prefilling time
                assert not output_attentions
                return self._glimpse_forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    image_info=image_info,
                    image_grid_thw=image_grid_thw,
                    ref_token_masks=ref_token_masks,
                    return_dict=return_dict,
                    delay_selection=delay_selection,
                    use_ref_masks=actual_use_ref_masks,
                )
            elif self.todo_selection:
                return self._do_delayed_selection(
                    override_image_token_mask_logits=image_token_mask_logits,
                    return_dict=return_dict,
                    use_cache=use_cache,
                )

        if ((cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)):
            logits, outputs = self.llm_forward_prefilling(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            logits, outputs = self.llm_forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if not return_dict:
            return (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                self.rope_deltas,
                input_ids,
                attention_mask,
                image_token_mask_logits,
                None,
            )

        return Qwen2_5_VL_GP_CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            rope_deltas=self.rope_deltas,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_token_mask_logits=image_token_mask_logits,
            image_token_penalties=None,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        image_token_mask_logits=None,
        ref_token_masks=None,  # new added
        do_selection=True,   # new added
        delay_selection=False,   # new added
        use_ref_masks=None,   # new added
        total_num_new_tokens=None,  # new added
        **kwargs,
    ):
        if self.reduced_input_ids is not None:
            input_ids = torch.cat((self.reduced_input_ids, input_ids[:, -total_num_new_tokens:]), dim=-1)

        if inputs_embeds is not None:
            n = input_ids.shape[1] - inputs_embeds.shape[1]
            if n > 0:
                new_inputs_embeds = self.model.embed_tokens(input_ids[:, -n:])
                inputs_embeds = torch.cat((inputs_embeds, new_inputs_embeds), dim=1)

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        if self.reduced_input_ids is None:
            # Qwen2-5-VL position_ids are prepared with rope_deltas in forward
            model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        model_inputs.update(
            {
                "image_token_mask_logits": image_token_mask_logits,
                "ref_token_masks": ref_token_masks,
                "do_selection": do_selection,
                "delay_selection": delay_selection,
                "use_ref_masks": use_ref_masks,
            }
        )
        return model_inputs


    def _update_model_kwargs_for_generation(
        self,
        outputs: Qwen2_5_VL_GP_CausalLMOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # Update past_key_values keeping its naming used in model code
        model_kwargs["past_key_values"] = getattr(outputs, "past_key_values")

        # Calculate actual attention_mask and position_ids after reduction
        model_kwargs["attention_mask"] = torch.cat(
            [outputs.attention_mask, outputs.attention_mask.new_ones((outputs.attention_mask.shape[0], num_new_tokens))], dim=-1
        )

        new_postion_ids = torch.repeat_interleave(outputs.position_ids[:, :, -1:], num_new_tokens, dim=-1)
        add_position_ids = torch.arange(1, num_new_tokens+1, device=new_postion_ids.device, dtype=new_postion_ids.dtype).view(1, 1, -1)
        new_postion_ids = new_postion_ids + add_position_ids
        if not model_kwargs.get("use_cache", True):
            new_postion_ids = torch.cat((outputs.position_ids, new_postion_ids), dim=-1)
        model_kwargs["position_ids"] = new_postion_ids

        input_ids_x = outputs.input_ids
        if input_ids_x is not None:
            if input_ids_x.shape[1] < model_kwargs["cache_position"].shape[0]:
                # reduction happened, we need to update past_positions
                past_positions = torch.ones_like(input_ids_x[0, :], dtype=torch.int64).cumsum(0) - 1
            else:
                past_positions = model_kwargs["cache_position"]
        else:
            past_positions = model_kwargs["cache_position"]

        model_kwargs["total_num_new_tokens"] = model_kwargs.get("total_num_new_tokens", 0) + num_new_tokens

        if outputs.inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = outputs.inputs_embeds

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = past_positions[-1:] + num_new_tokens
        else:
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        # pass image_token_mask_logits to indicate no need to do reduction.
        model_kwargs["image_token_mask_logits"] = outputs.image_token_mask_logits
        return model_kwargs


__all__ = [
    "Qwen2_5_VL_GP_CausalLMOutputWithPast",
    "Qwen2_5_VL_GP_ForConditionalGeneration",
]
