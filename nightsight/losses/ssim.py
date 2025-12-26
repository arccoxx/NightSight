"""SSIM-based loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def create_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int
) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    # 1D Gaussian
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 2D Gaussian
    kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channels, 1, window_size, window_size)

    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    size_average: bool = True,
    K: tuple = (0.01, 0.03)
) -> torch.Tensor:
    """
    Compute SSIM between two images.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Sigma for Gaussian
        data_range: Range of pixel values
        size_average: Return mean SSIM
        K: Constants for stability

    Returns:
        SSIM value(s)
    """
    channels = pred.shape[1]

    # Create kernel
    kernel = create_gaussian_kernel(window_size, sigma, channels)
    kernel = kernel.to(pred.device, pred.dtype)

    # Constants
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    # Compute means
    mu_pred = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute variances
    sigma_pred_sq = F.conv2d(
        pred ** 2, kernel, padding=window_size // 2, groups=channels
    ) - mu_pred_sq
    sigma_target_sq = F.conv2d(
        target ** 2, kernel, padding=window_size // 2, groups=channels
    ) - mu_target_sq
    sigma_pred_target = F.conv2d(
        pred * target, kernel, padding=window_size // 2, groups=channels
    ) - mu_pred_target

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


class SSIMLoss(nn.Module):
    """
    SSIM loss for image quality.

    Loss = 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        size_average: bool = True
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM loss."""
        return 1 - ssim(
            pred, target,
            self.window_size, self.sigma,
            self.data_range, self.size_average
        )


class MS_SSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss.

    Computes SSIM at multiple scales for better perceptual quality.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: Optional[list] = None,
        levels: int = 5
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.levels = levels

        # Default weights from MS-SSIM paper
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = weights[:levels]
        self.weights = torch.tensor(self.weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MS-SSIM loss."""
        channels = pred.shape[1]
        kernel = create_gaussian_kernel(self.window_size, self.sigma, channels)
        kernel = kernel.to(pred.device, pred.dtype)

        weights = self.weights.to(pred.device)

        K = (0.01, 0.03)
        C1 = (K[0] * self.data_range) ** 2
        C2 = (K[1] * self.data_range) ** 2

        mcs_list = []
        ssim_val = None

        for i in range(self.levels):
            if pred.shape[2] < self.window_size or pred.shape[3] < self.window_size:
                break

            # Compute SSIM components
            mu_pred = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=channels)
            mu_target = F.conv2d(target, kernel, padding=self.window_size // 2, groups=channels)

            mu_pred_sq = mu_pred ** 2
            mu_target_sq = mu_target ** 2
            mu_pred_target = mu_pred * mu_target

            sigma_pred_sq = F.conv2d(
                pred ** 2, kernel, padding=self.window_size // 2, groups=channels
            ) - mu_pred_sq
            sigma_target_sq = F.conv2d(
                target ** 2, kernel, padding=self.window_size // 2, groups=channels
            ) - mu_target_sq
            sigma_pred_target = F.conv2d(
                pred * target, kernel, padding=self.window_size // 2, groups=channels
            ) - mu_pred_target

            # Contrast-structure
            cs = (2 * sigma_pred_target + C2) / (sigma_pred_sq + sigma_target_sq + C2)
            mcs_list.append(cs.mean())

            # Final level: also compute luminance
            if i == self.levels - 1:
                l = (2 * mu_pred_target + C1) / (mu_pred_sq + mu_target_sq + C1)
                ssim_val = l.mean()

            # Downsample
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)

        # Combine
        if ssim_val is None:
            ssim_val = torch.tensor(1.0, device=pred.device)

        mcs_prod = torch.ones_like(ssim_val)
        for i, mcs in enumerate(mcs_list[:-1]):
            mcs_prod = mcs_prod * mcs.pow(weights[i])

        ms_ssim = ssim_val.pow(weights[-1]) * mcs_prod

        return 1 - ms_ssim
