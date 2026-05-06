from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature


def get_ref_token_mask(normed_bboxes, grid_size):
    h, w = grid_size
    mask = torch.zeros((h, w), dtype=torch.bool)
    for bbox in normed_bboxes:
        x1, y1, x2, y2 = bbox
        x1_grid = int(x1 * w)
        y1_grid = int(y1 * h)
        x2_grid = min(int(x2 * w), w - 1)
        y2_grid = min(int(y2 * h), h - 1)
        mask[y1_grid:y2_grid + 1, x1_grid:x2_grid + 1] = True
    return mask


class LLaVAOneVision1_5_GP_Processor:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        max_pixels = kwargs.pop("max_pixels", None)
        min_pixels = kwargs.pop("min_pixels", None)
        kwargs.setdefault("trust_remote_code", True)
        processor = AutoProcessor.from_pretrained(*args, **kwargs)
        if max_pixels is not None and hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = max_pixels
        if min_pixels is not None and hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = min_pixels
        return cls(processor)

    def save_pretrained(self, *args, **kwargs):
        return self.processor.save_pretrained(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.processor.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(
        self,
        *args,
        normed_bboxes: Optional[List[Optional[List[List[float]]]]] = None,
        ref_image_masks: Optional[List[Optional[Image.Image]]] = None,
        **kwargs,
    ) -> BatchFeature:
        inputs = self.processor(*args, **kwargs)
        image_grid_thw = inputs.get("image_grid_thw", None)
        ref_token_masks = None
        if image_grid_thw is not None:
            merge_size = getattr(self.image_processor, "merge_size", None)
            if merge_size is None:
                merge_size = getattr(getattr(self.image_processor, "image_processor", None), "merge_size", None)
            if merge_size is None:
                merge_size = 2

            if normed_bboxes is not None:
                assert len(normed_bboxes) == image_grid_thw.shape[0]
                ref_token_masks = []
                for i, boxes in enumerate(normed_bboxes):
                    if boxes is None:
                        ref_token_masks.append(None)
                        continue
                    grid_hw = [
                        image_grid_thw[i][1].item() // merge_size,
                        image_grid_thw[i][2].item() // merge_size,
                    ]
                    ref_token_masks.append(get_ref_token_mask(boxes, grid_hw))
            elif ref_image_masks is not None:
                assert len(ref_image_masks) == image_grid_thw.shape[0]
                ref_token_masks = []
                for i, ref_mask in enumerate(ref_image_masks):
                    if ref_mask is None:
                        ref_token_masks.append(None)
                        continue
                    ref_h = image_grid_thw[i][1].item() // merge_size
                    ref_w = image_grid_thw[i][2].item() // merge_size
                    one_ref_token_mask = ref_mask.resize((ref_w, ref_h), Image.NEAREST)
                    one_ref_token_mask = torch.tensor(one_ref_token_mask > 127, dtype=torch.bool)
                    ref_token_masks.append(one_ref_token_mask)
        inputs["ref_token_masks"] = ref_token_masks
        return inputs


__all__ = ["LLaVAOneVision1_5_GP_Processor"]
