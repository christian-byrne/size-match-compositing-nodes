import torch
from typing import Tuple, Optional

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils.equalize_size import SizeMatcher
from .utils.tensor_utils import TensorImgUtils


class SizeMatchNode:
    CATEGORY = "image"
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (
                    [
                        "cover_crop_center",
                        "cover_crop",
                        "center_dont_resize",
                        "fill",
                        "fit_center",
                        "crop_larger_center",
                        "crop_larger_topleft",
                    ],
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
        }

    def main(
        self,
        method: str,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        mask_1: Optional[torch.Tensor] = None,
        mask_2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:

        inputs = [
            tensor
            for tensor in [image_1, image_2, mask_1, mask_2]
            if tensor is not None
        ]
        image_1, image_2 = inputs[:2]

        # Expand masks
        if image_1.dim() == 3:
            image_1 = image_1.unsqueeze(-1)
        if image_2.dim() == 3:
            image_2 = image_2.unsqueeze(-1)

        image_1 = TensorImgUtils.convert_to_type(image_1, "BCHW")
        image_2 = TensorImgUtils.convert_to_type(image_2, "BCHW")

        # Apply to all in batch
        if image_1.dim() == 4 or image_2.dim() == 4:
            out_1 = []
            out_2 = []
            for i in range(image_1.size(0)):
                matched_1, matched_2 = SizeMatcher().size_match_by_method_str(
                    image_1[i], image_2[i], method
                )
                out_1.append(matched_1)
                out_2.append(matched_2)
            result = (torch.cat(out_1, dim=0), torch.cat(out_2, dim=0))
        else:
            result = SizeMatcher().size_match_by_method_str(image_1, image_2, method)

        return (
            TensorImgUtils.convert_to_type(result[0], "BHWC"),
            TensorImgUtils.convert_to_type(result[1], "BHWC"),
        )