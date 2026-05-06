from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


def _keep_count(total_tokens, budget):
    budget = float(budget)
    if budget <= 0:
        return 1
    if budget <= 1:
        return max(1, min(total_tokens, int(round(total_tokens * budget))))
    return max(1, min(total_tokens, int(round(budget))))


def cdpruner_select_indices(image_features, text_features, keep_num, eps=1e-6):
    if keep_num >= image_features.shape[0]:
        total = image_features.shape[0]
        selected = torch.arange(total, device=image_features.device)
        default_scores = torch.ones(total, device=image_features.device, dtype=torch.float32)
        return selected, default_scores, default_scores

    image_features = image_features.float()
    text_features = text_features.float()
    image_normalized = F.normalize(image_features, dim=-1, eps=eps)
    text_normalized = F.normalize(text_features, dim=-1, eps=eps)

    similarity = torch.matmul(image_normalized, image_normalized.transpose(0, 1))

    # Match CDPruner's conditional quality term: visual-token quality is conditioned
    # on the current text query, then normalized before DPP MAP inference.
    relevance = -torch.matmul(image_normalized, text_normalized.transpose(0, 1)).mean(dim=-1)
    relevance = (relevance - relevance.min() + eps) / (relevance.max() - relevance.min() + eps)
    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
    mask_scores = torch.diagonal(kernel).clone()

    cis = torch.zeros((keep_num, image_features.shape[0]), device=image_features.device, dtype=torch.float32)
    di2s = torch.diagonal(kernel).clone()
    select_idx = torch.empty(keep_num, dtype=torch.long, device=image_features.device)

    for i in range(keep_num):
        j = torch.argmax(di2s)
        select_idx[i] = j

        denom = torch.sqrt(torch.clamp(di2s[j], min=eps))
        if i == 0:
            eis = kernel[j] / denom
        else:
            correction = torch.einsum("t,tn->n", cis[:i, j], cis[:i])
            eis = (kernel[j] - correction) / denom

        cis[i] = eis
        di2s -= torch.square(eis)
        di2s[j] = -float("inf")

    return torch.sort(select_idx).values, mask_scores, relevance


def _query_text_features(inputs_embeds, input_ids, attention_mask, config):
    token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        token_mask &= attention_mask.bool()
    if getattr(config, "image_token_id", None) is not None:
        token_mask &= input_ids != config.image_token_id
    if getattr(config, "video_token_id", None) is not None:
        token_mask &= input_ids != config.video_token_id
    if getattr(config, "pad_token_id", None) is not None:
        token_mask &= input_ids != config.pad_token_id

    text_features = inputs_embeds[0, token_mask[0]]
    if text_features.numel() == 0:
        text_features = inputs_embeds[0, :1]
    return text_features


def _select_visual_embeds(model, visual_embeds, text_features, budget):
    keep_num = _keep_count(visual_embeds.shape[0], budget)
    selected_indices, mask_scores, relevance_scores = cdpruner_select_indices(
        visual_embeds, text_features, keep_num
    )
    model.cdpruner_last_selected_indices = selected_indices
    model.cdpruner_last_visual_token_num = int(visual_embeds.shape[0])
    model.cdpruner_last_mask_scores = mask_scores
    model.cdpruner_last_relevance_scores = relevance_scores
    model.cdpruner_last_keep_num = int(keep_num)
    if hasattr(model, "visual"):
        model.visual.cdpruner_last_selected_indices = selected_indices
        model.visual.cdpruner_last_visual_token_num = int(visual_embeds.shape[0])
        model.visual.cdpruner_last_mask_scores = mask_scores
        model.visual.cdpruner_last_relevance_scores = relevance_scores
        model.visual.cdpruner_last_keep_num = int(keep_num)
    return visual_embeds[selected_indices], selected_indices


def _compress_input_by_visual_indices(
    input_ids,
    inputs_embeds,
    attention_mask,
    token_id,
    selected_indices,
):
    visual_positions = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
    if visual_positions.numel() != selected_indices.numel() and selected_indices.numel() > visual_positions.numel():
        raise ValueError(
            f"Selected visual tokens exceed placeholders: selected={selected_indices.numel()}, "
            f"placeholders={visual_positions.numel()}"
        )

    keep_mask = torch.ones(input_ids.shape[-1], dtype=torch.bool, device=input_ids.device)
    keep_mask[visual_positions] = False
    keep_mask[visual_positions[selected_indices]] = True
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    input_ids = input_ids[:, keep_indices]
    inputs_embeds = inputs_embeds[:, keep_indices, :]
    if attention_mask is not None:
        attention_mask = attention_mask[:, keep_indices]
    return input_ids, inputs_embeds, attention_mask, keep_indices


def _replace_visual_embeds(inputs_embeds, input_ids, token_id, visual_embeds):
    visual_mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
    visual_embeds = visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(visual_mask.to(inputs_embeds.device), visual_embeds)


def qwen25vl_generation_forward_cdpruner(
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
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    original_input_ids = input_ids
    keep_indices = None

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        assert input_ids.shape[0] == 1, "cdpruner currently supports batch_size=1 for Qwen2.5-VL."
        text_features = _query_text_features(inputs_embeds, input_ids, attention_mask, self.config)

        has_image_tokens = (
            pixel_values is not None
            and input_ids is not None
            and (input_ids == self.config.image_token_id).any()
        )
        if has_image_tokens:
            pixel_values = pixel_values.type(self.visual.dtype)
            self.cdpruner_last_grid_thw = image_grid_thw
            self.cdpruner_last_modality = "image"
            if hasattr(self, "visual"):
                self.visual.cdpruner_last_grid_thw = image_grid_thw
                self.visual.cdpruner_last_modality = "image"
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                    f"features {image_embeds.shape[0]}"
                )

            budget = getattr(self, "cdpruner_keep_tokens", None)
            if budget is None:
                budget = getattr(self, "cdpruner_budget", 1.0)
            image_embeds, selected_indices = _select_visual_embeds(self, image_embeds, text_features, budget)
            input_ids, inputs_embeds, attention_mask, keep_indices = _compress_input_by_visual_indices(
                input_ids,
                inputs_embeds,
                attention_mask,
                self.config.image_token_id,
                selected_indices,
            )
            inputs_embeds = _replace_visual_embeds(inputs_embeds, input_ids, self.config.image_token_id, image_embeds)

        has_video_tokens = (
            pixel_values_videos is not None
            and input_ids is not None
            and (input_ids == self.config.video_token_id).any()
        )
        if has_video_tokens:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            self.cdpruner_last_grid_thw = video_grid_thw
            self.cdpruner_last_modality = "video"
            if hasattr(self, "visual"):
                self.visual.cdpruner_last_grid_thw = video_grid_thw
                self.visual.cdpruner_last_modality = "video"
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens != video_embeds.shape[0]:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                    f"features {video_embeds.shape[0]}"
                )

            budget = getattr(self, "cdpruner_keep_tokens", None)
            if budget is None:
                budget = getattr(self, "cdpruner_budget", 1.0)
            video_embeds, selected_indices = _select_visual_embeds(self, video_embeds, text_features, budget)
            input_ids, inputs_embeds, attention_mask, video_keep_indices = _compress_input_by_visual_indices(
                input_ids,
                inputs_embeds,
                attention_mask,
                self.config.video_token_id,
                selected_indices,
            )
            keep_indices = video_keep_indices if keep_indices is None else keep_indices[video_keep_indices]
            inputs_embeds = _replace_visual_embeds(inputs_embeds, input_ids, self.config.video_token_id, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                original_input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                None if keep_indices is not None else attention_mask,
            )
            self.rope_deltas = rope_deltas
            if keep_indices is not None:
                position_ids = position_ids[:, :, keep_indices]
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
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

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )