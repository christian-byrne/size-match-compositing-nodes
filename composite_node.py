import torch
from typing import Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tensor_img_utils import TensorImgUtils
from .utils.equalize_size import SizeMatcher
from .utils.chromakey import ChromaKey




class CompositeCutoutOnBaseNode:
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "cutout": ("IMAGE",),
                "cutout_alpha": ("MASK",),
                "size_matching_method": (
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
                "invert_cutout": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    def main(
        self,
        base_image: torch.Tensor,  # [B, H, W, 3]
        cutout: torch.Tensor,  # [B, H, W, 3]
        cutout_alpha: torch.Tensor,  # [B, H, W]
        size_matching_method: str,
        invert_cutout: bool,
    ) -> Tuple[torch.Tensor, ...]:

        if base_image.dim() == 4:
            out = []
            for i in range(base_image.size(0)):
                out.append(
                    self.main(
                        base_image[i],
                        cutout[i] if cutout.dim() == 4 else cutout,
                        cutout_alpha[i] if cutout_alpha.dim() == 3 else cutout_alpha,
                        size_matching_method,
                        invert_cutout,
                    )
                )
            return (torch.cat(out, dim=0),)

        base_image = TensorImgUtils.convert_to_type(base_image, "CHW")
        cutout = TensorImgUtils.convert_to_type(cutout, "CHW")

        if cutout_alpha.dim() == 2:
            cutout_alpha = cutout_alpha.unsqueeze(0)
        cutout_alpha = TensorImgUtils.convert_to_type(cutout_alpha, "CHW")

        # If base_image is rgba for some reason, remove alpha channel
        if base_image.size(0) == 4:
            base_image = base_image[:3, :, :]

        # Comfy creates a default 64x64 mask if rgb image was loaded, so we check for size mismatch to know the image was rgb and didn't have an alpha channel at load time
        if cutout_alpha.size(1) != cutout.size(1) or cutout_alpha.size(
            2
        ) != cutout.size(2):
            print(
                f"Cutout alpha size {cutout_alpha.size()} does not match cutout size {cutout.size()}. Inferring alpha channel automatically."
            )
            _, cutout_alpha, _ = ChromaKey().infer_bg_and_remove(cutout)

        if invert_cutout:
            cutout_alpha = 1 - cutout_alpha

        alpha_cutout = self.recombine_alpha(
            cutout, cutout_alpha
        )  # recombine just so resize is easier
        base_image, alpha_cutout = SizeMatcher().size_match_by_method_str(
            base_image, alpha_cutout, size_matching_method
        )

        return TensorImgUtils.convert_to_type(
            self.composite(base_image, alpha_cutout), "BHWC"
        )

    def composite(self, base_image: torch.Tensor, cutout: torch.Tensor) -> torch.Tensor:
        alpha_only = cutout[3, :, :]

        # All pixels that are not transparent should be from the cutout
        composite = cutout[:3, :, :] * alpha_only + base_image * (1 - alpha_only)

        return composite

    def recombine_alpha(self, image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Recombine the image and alpha channel into a single tensor.
        """
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        alpha = 1 - alpha
        return torch.cat((image, alpha), 0)
