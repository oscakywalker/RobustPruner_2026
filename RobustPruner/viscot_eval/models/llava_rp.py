import importlib
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel

from .base import BaseInferModel


def _candidate_gp_impl_roots(explicit_root):
    if explicit_root:
        yield Path(explicit_root).expanduser()

    env_root = os.environ.get("GP_IMPL_ROOT")
    if env_root:
        yield Path(env_root).expanduser()

    yield Path("/home/fjh/GlimpsePrune_2")


def _resolve_gp_impl_root(explicit_root):
    for candidate in _candidate_gp_impl_roots(explicit_root):
        if candidate.is_dir():
            return candidate.resolve()
    tried = [str(path) for path in _candidate_gp_impl_roots(explicit_root)]
    raise FileNotFoundError(
        "Unable to locate a GlimpsePrune project root that provides the robustpruner llava_rp "
        f"implementation. Tried: {tried}"
    )


def _import_gp2_llava_modules(project_root):
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Force Python to re-resolve llava/llava_rp from the requested project root.
    for module_name in list(sys.modules):
        if module_name in {"llava", "llava_rp"} or module_name.startswith(("llava.", "llava_rp.")):
            del sys.modules[module_name]

    llava_constants = importlib.import_module("llava.constants")
    llava_conversation = importlib.import_module("llava.conversation")
    mm_utils = importlib.import_module("llava_rp.mm_utils")
    builder = importlib.import_module("llava_rp.model.builder")
    return llava_constants, llava_conversation, mm_utils, builder


class LLaVA_GP(BaseInferModel):
    def _init_model(
        self,
        new_modules_dir=None,
        gp_impl_root=None,
        use_ref_masks=False,
        use_zero_masks=False,
        reduce_layer=None,
        anchor_positions=None,
        min_remain_num=None,
        max_remain_ratio=None,
        base_score_pool_multiplier=None,
        adapter_dir=None,
        adapter_merge=False,
        **kwargs,
    ):
        project_root = _resolve_gp_impl_root(gp_impl_root)
        llava_constants, llava_conversation, mm_utils, builder = _import_gp2_llava_modules(project_root)
        self._default_image_token = llava_constants.DEFAULT_IMAGE_TOKEN
        self._image_token_index = llava_constants.IMAGE_TOKEN_INDEX
        self._default_im_start_token = llava_constants.DEFAULT_IM_START_TOKEN
        self._default_im_end_token = llava_constants.DEFAULT_IM_END_TOKEN
        self._conv_templates = llava_conversation.conv_templates
        self._get_model_name_from_path = mm_utils.get_model_name_from_path
        self._process_images = mm_utils.process_images
        self._process_bboxes = mm_utils.process_bboxes
        self._tokenizer_image_token = mm_utils.tokenizer_image_token
        self._gp_impl_root = project_root

        model_name = self._get_model_name_from_path(self._base_model)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": self._attn_implementation,
            "torch_dtype": self._torch_dtype,
        }
        try:
            self._tokenizer, self._model, self._image_processor, self._max_length = builder.load_pretrained_model(
                self._base_model,
                None,
                model_name,
                device_map=f"cuda:{local_rank}",
                **llava_model_args,
            )
        except TypeError:
            llava_model_args.pop("multimodal")
            self._tokenizer, self._model, self._image_processor, self._max_length = builder.load_pretrained_model(
                self._base_model,
                None,
                model_name,
                device_map=f"cuda:{local_rank}",
                **llava_model_args,
            )

        if new_modules_dir is not None:
            self._model.load_new_modules(new_modules_dir)
        if use_ref_masks:
            self._model.config.use_ref_masks = True
        if use_zero_masks:
            assert not use_ref_masks, "use_ref_masks should be False when use_zero_masks is True"
            self._model.config.use_zero_masks = True
        if reduce_layer is not None:
            self._model.config.reduce_layer = reduce_layer
        if anchor_positions is not None:
            self._model.config.anchor_positions = anchor_positions
        if min_remain_num is not None:
            self._model.config.min_remain_num = min_remain_num
        if max_remain_ratio is not None:
            self._model.config.max_remain_ratio = max_remain_ratio
        if base_score_pool_multiplier is not None:
            self._model.config.base_score_pool_multiplier = float(base_score_pool_multiplier)

        if adapter_dir is not None:
            self._model = PeftModel.from_pretrained(self._model, adapter_dir, is_trainable=False)
            if adapter_merge:
                self._model = self._model.merge_and_unload()
            print(f"Loaded adapter from {adapter_dir}, merge: {adapter_merge}")

        self._model.eval()
        self._conv_mode = "vicuna_v1"

    def _init_processor(self, **kwargs):
        pass

    def _do_generate(self, inputs, generation_config, do_selection):
        generate_ids = self._model.generate(
            **inputs,
            generation_config=generation_config,
            do_selection=do_selection,
        )
        return generate_ids

    def _do_glimpse(self, inputs, generation_config):
        if "attention_mask" not in inputs:
            attention_mask = torch.ones(inputs["input_ids"].shape, device=self._device, dtype=torch.long)
            inputs["attention_mask"] = attention_mask
        output = self._model(**inputs, return_dict=True)
        return output.image_token_bool_masks

    def batch_decode(self, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, **kwargs):
        return [
            self._tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )[0].strip()
        ]

    def prepare_batch_inputs(self, batched_querys, batched_img_paths, batched_bboxes):
        curr_batch_size = len(batched_querys)
        assert curr_batch_size == 1, "LLaVA_1.5 only supports batch size 1 for now."

        input_ids = []
        images = []
        image_sizes = []
        ref_token_masks = [] if batched_bboxes is not None else None

        grid_h = grid_w = self._model.get_vision_tower().num_patches_per_side

        for idx, (one_query, one_img_path) in enumerate(zip(batched_querys, batched_img_paths)):
            if self._model.config.mm_use_im_start_end:
                one_query = (
                    self._default_im_start_token + self._default_image_token + self._default_im_end_token + "\n" + one_query
                )
            else:
                one_query = self._default_image_token + "\n" + one_query
            conv = self._conv_templates[self._conv_mode].copy()
            conv.append_message(conv.roles[0], one_query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(one_img_path).convert("RGB")
            image_tensor = self._process_images([image], self._image_processor, self._model.config)[0]
            if ref_token_masks is not None:
                one_bboxes = batched_bboxes[idx]
                if one_bboxes is None:
                    ref_token_masks.append(None)
                else:
                    ref_token_masks.append(self._process_bboxes(one_bboxes, (grid_h, grid_w)))
            input_id = self._tokenizer_image_token(
                prompt, self._tokenizer, self._image_token_index, return_tensors="pt"
            )
            input_ids.append(input_id)
            images.append(image_tensor)
            image_sizes.append(image.size)

        input_ids = torch.stack(input_ids, dim=0).to(device=self._device)
        image_tensor = torch.stack(images, dim=0).to(device=self._device, dtype=self._torch_dtype)
        if ref_token_masks is not None:
            ref_token_masks = torch.stack(ref_token_masks, dim=0).to(device=self._device)

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "ref_token_masks": ref_token_masks,
        }