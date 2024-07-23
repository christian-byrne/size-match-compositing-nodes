import torch
from typing import Tuple, Optional

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils.equalize_size import SizeMatcher
from tensor_img_utils import TensorImgUtils


class SizeMatchNode:
    CATEGORY = "image"
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "MASK",
        "MASK",
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
        input_indices = [
            i for i, tensor in enumerate([image_1, image_2, mask_1, mask_2]) if tensor is not None
        ]
        image_1, image_2 = inputs[:2]

        input_types = []
        if image_1.dim() == 3:
            image_1 = image_1.unsqueeze(-1)
            input_types.append("MASK")
        else:
            input_types.append("IMAGE")
        if image_2.dim() == 3:
            image_2 = image_2.unsqueeze(-1)
            input_types.append("MASK")
        else:
            input_types.append("IMAGE")
        
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

        result= [
            TensorImgUtils.convert_to_type(result[0], "BHWC"),
            TensorImgUtils.convert_to_type(result[1], "BHWC"),
        ]
        if input_types[0] == "MASK":
            result[0] = result[0].squeeze(-1)
        if input_types[1] == "MASK":
            result[1] = result[1].squeeze(-1)

        res = []
        for i in range(4):
            if i in input_indices:
                res.append(result.pop(0))
            elif i <= 1:
                res.append(torch.rand(1, 64, 64, 3))
            else:
                res.append(torch.rand(1, 64, 64))
        
        return tuple(res)
