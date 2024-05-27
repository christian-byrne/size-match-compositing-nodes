"""
Method signatures automatically generated

pyenv local 3.10.6
"""

import torch
from typing import Tuple


class ChromaKey:
    def __init__(self):
        self.closest = None
        self.__TRANSPARENT = 0.0
        self.__OPAQUE = 1.0

    def __check_inference(self, alpha: torch.Tensor) -> bool:
        # If remove method created and alpha with more than 50% and less than 85% transparency, consider hard success and return True
        if alpha.sum() > 0.5 * alpha.numel() and alpha.sum() < 0.85 * alpha.numel():
            return True  # Inference successful

        
        if self.closest is None or (
            abs(alpha.sum() - alpha.numel() * 0.5)
            < abs(self.closest.sum() - self.closest.numel() * 0.5)
        ):
            self.closest = alpha

        return False  # Inference failed

    def __validate_threshold(self, threshold: float) -> float:
        """Validates threshold value, which must be between 0 and 1."""
        if threshold < 0:
            return 0.01
        if threshold > 1:
            return 0.99
        return threshold

    def __package_return(
        self, image: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # When the remove method fails, return the original image with a fully transparent alpha channel
        if alpha.sum() == 0:
            # NOTE: use infer_bg_and_remove to get the closest alpha, dont do automatically if user isn't aware
            alpha = torch.zeros_like(image[0, ...])

        rgba = torch.cat((image, alpha.unsqueeze(0)), dim=0)
        mask = 1 - alpha
        return rgba, alpha, mask

    def remove_neutrals(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create alpha channel by making neutrals transparent (i.e. white, black, and gray).
        Return the image after being merged with the generated alpha channel, the isolated alpha channel, and the mask (inverse of the alpha channel).

        Args:
            image (torch.Tensor): The input image tensor.
            leniance (float, optional): The leniance value to adjust the threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """
        return self.__remove_by_diff(image, 0.02 + leniance, "less")

    def remove_non_neutrals(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create alpha channel by making non-neutrals transparent (i.e. not white, black, and gray). Return the image after being merged with the generated alpha channel, the isolated alpha channel, and the mask (inverse of the alpha channel).

        Args:
            image (torch.Tensor): 3HW input image tensor.
            leniance (float, optional): The leniance value to adjust the threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """
        return self.__remove_by_diff(image, 0.81 - leniance, "greater")

    def remove_white(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create alpha channel by making white transparent. Return the image after being merged with the generated alpha channel, the isolated alpha channel, and the mask (inverse of the alpha channel).

        Args:
            image (torch.Tensor): 3HW input image tensor.
            leniance (float, optional): The leniance value to adjust the threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """
        return self.__remove_by_threshold(image, 0.96 - leniance, "greater")

    def remove_black(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create alpha channel by making black transparent. Return the image after being merged with the generated alpha channel, the isolated alpha channel, and the mask (inverse of the alpha channel).

        Args:
            image (torch.Tensor): 3HW input image tensor.
            leniance (float, optional): The leniance value to adjust the threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """
        return self.__remove_by_threshold(image, 0.04 + leniance, "less")

    def infer_bg_and_remove(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Go through the other remove_*_bg methods in this class, retrying so long as the alpha generated is very small relative to the image size. If no matches, return the key that generated the most likely

        Args:
            image (torch.Tensor): 3HW input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """

        # Try remove white
        _, alpha, _ = self.remove_white(image)
        if self.__check_inference(alpha):
            return self.__package_return(image, alpha)

        # Try remove black
        _, alpha, _ = self.remove_black(image)
        if self.__check_inference(alpha):
            return self.__package_return(image, alpha)

        # Try remove neutrals
        _, alpha, _ = self.remove_neutrals(image)
        if self.__check_inference(alpha):
            return self.__package_return(image, alpha)

        # Try remove non-neutrals
        _, alpha, _ = self.remove_non_neutrals(image)
        if self.__check_inference(alpha):
            return self.__package_return(image, alpha)

        # Try common bg colors
        for color in [
            (255, 192, 203),
            (135, 206, 235),
            (173, 216, 230),
            (255, 228, 196),
            (0, 255, 0),
            (255, 255, 0),
            (255, 165, 0),
            (255, 0, 0),
        ]:
            _, alpha, _ = self.remove_specific_rgb(image, color)
            if self.__check_inference(alpha):
                return self.__package_return(image, alpha)

        return self.__package_return(image, alpha)

    def remove_specific_rgb(
        self, image: torch.Tensor, rgb: Tuple[int, int, int], leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Removes a specific RGB color from the image by creating an alpha channel. Return the image after being merged with the generated alpha channel, the isolated alpha channel, and the mask (inverse of the alpha channel).

        Args:
            image (torch.Tensor): 3HW input image tensor.
            rgb (Tuple[int, int, int]): The RGB color to remove.
            leniance (float, optional): The leniance value to adjust the threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 4HW merged image, 1HW alpha channel, 1HW mask
        """
        leniance = self.__validate_threshold(leniance)
        custom_rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
        custom_rgb_tensor = custom_rgb_tensor.view(3, 1, 1).expand(
            3, image.shape[1], image.shape[2]
        )

        threshold = self.__validate_threshold(0.45 + leniance)

        alpha = torch.where(
            # Total differences across all channels > threshold
            (torch.abs(image - custom_rgb_tensor) > threshold).sum(dim=0) > 0,
            torch.tensor(self.__OPAQUE),
            torch.tensor(self.__TRANSPARENT),
        )
        return self.__package_return(image, alpha)

    def __remove_by_diff(
        self, image: torch.Tensor, threshold: float, comparison_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if comparison_type == "greater":
            alpha = torch.where(
                (image.max(dim=0).values - image.min(dim=0).values)
                > self.__validate_threshold(threshold),
                torch.tensor(self.__TRANSPARENT),
                torch.tensor(self.__OPAQUE),
            )
        elif comparison_type == "less":
            alpha = torch.where(
                (image.max(dim=0).values - image.min(dim=0).values)
                < self.__validate_threshold(threshold),
                torch.tensor(self.__TRANSPARENT),
                torch.tensor(self.__OPAQUE),
            )
        return self.__package_return(image, alpha)

    def __remove_by_threshold(
        self, image: torch.Tensor, threshold: float, comparison_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels_mean = image.mean(dim=0)
        threshold = self.__validate_threshold(threshold)

        if comparison_type == "greater":
            alpha = torch.where(
                channels_mean > threshold,
                torch.tensor(self.__TRANSPARENT),
                torch.tensor(self.__OPAQUE),
            )
        elif comparison_type == "bounded":
            lower_bound = self.__validate_threshold(abs(threshold + 0.02))
            upper_bound = self.__validate_threshold(abs(1 - (threshold + 0.02)))
            alpha = torch.where(
                (channels_mean < lower_bound) & (channels_mean > upper_bound),
                torch.tensor(self.__TRANSPARENT),
                torch.tensor(self.__OPAQUE),
            )
        elif comparison_type == "less":
            alpha = torch.where(
                channels_mean < threshold,
                torch.tensor(self.__TRANSPARENT),
                torch.tensor(self.__OPAQUE),
            )

        return self.__package_return(image, alpha)
