import torch
from torchvision import transforms
from typing import Tuple

from ..utils.tensor_utils import TensorImgUtils
from ..utils.scale import ImageScaler


class SizeMatcher:
    def __init__(self):
        self.scale = ImageScaler()

    def size_match_by_method_str(
        self,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        size_matching_method: str,
    ) -> Tuple[torch.Tensor]:
        """
        Matches the size of two images based on the specified size matching method.

        Args:
            image_1: The first image to be resized.
            image_2: The second image to be resized.
            size_matching_method: The method used for size matching. Available options are:
                - "fit_center": Resizes the images to fit within the specified dimensions while maintaining the aspect ratio. The images are centered within the new dimensions.
                - "cover_crop_center": Resizes the images to cover the specified dimensions while maintaining the aspect ratio. The images are cropped from the center.
                - "cover_crop": Resizes the images to cover the specified dimensions while maintaining the aspect ratio. The images are cropped from the top-left corner.
                - "fill": Resizes the images to fill the specified dimensions while maintaining the aspect ratio. The images are distorted to fit.
                - "crop_larger_center": Crops the larger image to match the size of the smaller image while maintaining the aspect ratio. The images are cropped from the center.
                - "crop_larger_topleft": Crops the larger image to match the size of the smaller image while maintaining the aspect ratio. The images are cropped from the top-left corner.
                - "center_dont_resize": Pads the smaller image to match the size of the larger image. The images are centered within the new dimensions.

        Returns:
            A tuple containing the resized images.

        """
        match size_matching_method:
            case "fit_center":
                image_1, image_2 = self.fit_maintain_pad(image_1, image_2)
            case "cover_crop_center":
                image_1, image_2 = self.cover_maintain(image_1, image_2, center=True)
            case "cover_crop":
                image_1, image_2 = self.cover_maintain(image_1, image_2, center=False)
            case "fill":
                image_1, image_2 = self.cover_distort(image_1, image_2)
            case "crop_larger_center":
                image_1, image_2 = self.crop_larger_to_match(
                    image_1, image_2, center=True
                )
            case "crop_larger_topleft":
                image_1, image_2 = self.crop_larger_to_match(
                    image_1, image_2, center=False
                )
            case "center_dont_resize":
                image_1, image_2 = self.pad_smaller(image_1, image_2)

        return (image_1, image_2)

    def pad_smaller(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Pads the smaller image to match the dimensions of the larger image.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            top_pad = (h1 - h2) // 2
            bot_pad = h1 - h2 - top_pad
            left_pad = (w1 - w2) // 2
            right_pad = w1 - w2 - left_pad
            image_2 = transforms.Pad((left_pad, top_pad, right_pad, bot_pad))(image_2)
        else:
            top_pad = (h2 - h1) // 2
            bot_pad = h2 - h1 - top_pad
            left_pad = (w2 - w1) // 2
            right_pad = w2 - w1 - left_pad
            image_1 = transforms.Pad((left_pad, top_pad, right_pad, bot_pad))(image_1)

        return (image_1, image_2)

    def crop_to_match(
        self, image: torch.Tensor, target_dimensions, center=False
    ) -> torch.Tensor:
        """
        Crop the input image to match the specified target dimensions.

        Args:
            image (torch.Tensor): The input image tensor.
            target_dimensions (tuple): The target dimensions (height, width) to crop the image to.
            center (bool, optional): If True, the image will be cropped from the center. Defaults to False.

        Returns:
            torch.Tensor: The cropped image tensor.
        """
        h1, w1 = TensorImgUtils.height_width(image)
        diff_h = h1 - target_dimensions[0]
        diff_w = w1 - target_dimensions[1]

        # Calculate margins for center cropping
        top_margin = diff_h // 2
        bot_margin = top_margin + target_dimensions[0]
        left_margin = diff_w // 2
        right_margin = left_margin + target_dimensions[1]

        if diff_h < 0 and diff_w < 0:
            return image
        elif diff_h < 0 and diff_w >= 0:
            if center:
                return image[:, :, left_margin:right_margin]
            else:
                return image[:, :, : target_dimensions[1]]
        elif diff_w < 0 and diff_h >= 0:
            if center:
                return image[:, top_margin:bot_margin, :]
            else:
                return image[:, : target_dimensions[0], :]
        else:
            if center:
                return image[:, top_margin:bot_margin, left_margin:right_margin]
            else:
                return image[:, : target_dimensions[0], : target_dimensions[1]]

    def fit_maintain_pad(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Scale the smaller of the two images to fit inside dimensions of the larger
        image as much as possible while maintaining aspect ratio, then pad the
        smaller image to match the dimensions of the larger image so the tensors
        have the same shape.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        h_axis, w_axis = TensorImgUtils.infer_hw_axis(image_2)
        if h1 * w1 > h2 * w2:
            max_scale_factor = min(h1 / h2, w1 / w2)
            image_2 = self.scale.by_side(image_2, int(h2 * max_scale_factor), h_axis)

            margin_top = (h1 - image_2.shape[h_axis]) // 2
            margin_left = (w1 - image_2.shape[w_axis]) // 2
            margin_bot = h1 - image_2.shape[h_axis] - margin_top
            margin_right = w1 - image_2.shape[w_axis] - margin_left

            image_2 = transforms.Pad(
                (margin_left, margin_top, margin_right, margin_bot)
            )(image_2)
        else:
            max_scale_factor = min(h2 / h1, w2 / w1)
            image_1 = self.scale.by_side(image_1, int(h1 * max_scale_factor), h_axis)

            margin_top = (h2 - image_1.shape[h_axis]) // 2
            margin_left = (w2 - image_1.shape[w_axis]) // 2
            margin_bot = h2 - image_1.shape[h_axis] - margin_top
            margin_right = w2 - image_1.shape[w_axis] - margin_left

            image_1 = transforms.Pad(
                (margin_left, margin_top, margin_right, margin_bot)
            )(image_1)

        return (image_1, image_2)

    def cover_distort(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Resizes the smaller of the two images to cover the dimensions of the larger image.
        Scale and distort the smaller image to match the larger image's dimensions.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            image_2 = transforms.Resize((h1, w1))(image_2)
        else:
            image_1 = transforms.Resize((h2, w2))(image_1)

        return (image_1, image_2)

    def crop_larger_to_match(
        self, image_1: torch.Tensor, image_2: torch.Tensor, center=True
    ) -> Tuple[torch.Tensor]:
        """
        Crop the larger of the two images to match the dimensions of the smaller image.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        print(f"h1: {h1}, w1: {w1}")
        # If both sides are not larger, revert to cover_maintain
        if h1 * w1 > h2 * w2:
            if h1 < h2 or w1 < w2:
                print("h1 < h2 or w1 < w2")
                return self.cover_maintain(image_1, image_2)
            print("h1 >= h2 and w1 >= w2")
            image_1 = self.crop_to_match(image_1, (h2, w2), center=center)
        else:
            if h2 < h1 or w2 < w1:
                return self.cover_maintain(image_1, image_2)
            image_2 = self.crop_to_match(image_2, (h1, w1), center=center)

        return (image_1, image_2)

    def cover_maintain(
        self, image_1: torch.Tensor, image_2: torch.Tensor, center=True
    ) -> Tuple[torch.Tensor]:
        """
        Resizes the smaller of the two images to cover the dimensions of the larger image.
        Scale while maintaining aspect ratio, then crop to match the larger image's dimensions
        starting from the top-left corner.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        HEIGHT_AXIS, WIDTH_AXIS = TensorImgUtils.infer_hw_axis(image_1)
        other_axis = lambda axis: HEIGHT_AXIS if axis == WIDTH_AXIS else WIDTH_AXIS
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            if h1 - h2 > w1 - w2:
                target_axis = HEIGHT_AXIS
            else:
                target_axis = WIDTH_AXIS
            image_2_ret = self.scale.by_side(
                image_2, image_1.shape[target_axis], target_axis
            )
            image_2_ret = self.crop_to_match(image_2_ret, (h1, w1), center=center)
            if (
                image_2_ret.shape[HEIGHT_AXIS] != image_1.shape[HEIGHT_AXIS]
                or image_2_ret.shape[WIDTH_AXIS] != image_1.shape[WIDTH_AXIS]
            ):
                target_axis = other_axis(target_axis)
                image_2_ret = self.scale.by_side(
                    image_2, image_1.shape[target_axis], target_axis
                )
                image_2_ret = self.crop_to_match(image_2_ret, (h1, w1), center=center)

            return (image_1, image_2_ret)
        else:
            if h2 - h1 > w2 - w1:
                target_axis = HEIGHT_AXIS
            else:
                target_axis = WIDTH_AXIS

            image_1_ret = self.scale.by_side(
                image_1, image_2.shape[target_axis], target_axis
            )
            image_1_ret = self.crop_to_match(image_1_ret, (h2, w2), center=center)
            if (
                image_1_ret.shape[HEIGHT_AXIS] != image_2.shape[HEIGHT_AXIS]
                or image_1_ret.shape[WIDTH_AXIS] != image_2.shape[WIDTH_AXIS]
            ):
                target_axis = other_axis(target_axis)
                image_1_ret = self.scale.by_side(
                    image_1, image_2.shape[target_axis], target_axis
                )
                image_1_ret = self.crop_to_match(image_1_ret, (h2, w2), center=center)

            return (image_1_ret, image_2)
