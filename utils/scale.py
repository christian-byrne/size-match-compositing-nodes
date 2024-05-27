import torch
from torchvision import transforms
from ..utils.tensor_utils import TensorImgUtils


class ImageScaler:
    def __init__(self, pad_constant=0, weight=1):
        self.pad_constant = pad_constant
        self.weight = weight

    def by_side(self, image: torch.Tensor, target_size: int, axis: int) -> torch.Tensor:
        """
        Scales the given image tensor along the specified axis to the target size.

        Args:
            image (torch.Tensor): The input image tensor.
            target_size (int): The desired size of the scaled side.
            axis (int): The axis along which to scale the image.

        Returns:
            torch.Tensor: The scaled image tensor.
        """
        h, w = TensorImgUtils.height_width(image)
        h_axis, w_axis = TensorImgUtils.infer_hw_axis(image)
        if axis == h_axis:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_h = int(h * target_size / w)
            new_w = target_size

        return transforms.Resize((new_h, new_w))(image)
