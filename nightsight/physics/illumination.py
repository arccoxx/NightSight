"""Physics-based illumination modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class IlluminationModel:
    """
    Physics-based illumination model.

    Based on Retinex theory: I = R * L
    Where I is observed image, R is reflectance, L is illumination.
    """

    def __init__(
        self,
        gamma: float = 2.2,
        ambient_light: float = 0.1
    ):
        """
        Initialize illumination model.

        Args:
            gamma: Camera gamma value
            ambient_light: Ambient light contribution
        """
        self.gamma = gamma
        self.ambient_light = ambient_light

    def apply_illumination(
        self,
        reflectance: torch.Tensor,
        illumination: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply illumination to reflectance.

        Args:
            reflectance: Albedo/reflectance map
            illumination: Illumination map

        Returns:
            Observed image
        """
        # Add ambient light
        total_light = illumination + self.ambient_light

        # Apply gamma
        image = reflectance * total_light
        image = torch.pow(torch.clamp(image, 0, 1), 1 / self.gamma)

        return image

    def decompose(
        self,
        image: torch.Tensor,
        method: str = "max_channel"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose image into reflectance and illumination.

        Args:
            image: Input image
            method: Decomposition method

        Returns:
            Tuple of (reflectance, illumination)
        """
        illumination = estimate_illumination(image, method)
        reflectance = image / (illumination + 1e-4)

        return reflectance, illumination


def estimate_illumination(
    image: torch.Tensor,
    method: str = "max_channel",
    **kwargs
) -> torch.Tensor:
    """
    Estimate illumination map from image.

    Args:
        image: Input image (B, C, H, W) or (C, H, W)
        method: Estimation method

    Returns:
        Illumination map (B, 1, H, W)
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape

    if method == "max_channel":
        # Maximum of RGB channels
        illum = torch.max(image, dim=1, keepdim=True)[0]

    elif method == "mean_channel":
        # Mean of RGB channels
        illum = torch.mean(image, dim=1, keepdim=True)

    elif method == "luminance":
        # Perceived luminance
        if C == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)
            illum = (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        else:
            illum = torch.mean(image, dim=1, keepdim=True)

    elif method == "smooth":
        # Smoothed version
        sigma = kwargs.get("sigma", 15)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel.view(1, 1, -1, 1) * kernel.view(1, 1, 1, -1)

        illum = torch.max(image, dim=1, keepdim=True)[0]
        illum = F.pad(illum, [kernel_size // 2] * 4, mode='reflect')
        illum = F.conv2d(illum, kernel_2d)

    elif method == "guided":
        # Guided filter for edge-preserving smoothing
        radius = kwargs.get("radius", 15)
        eps = kwargs.get("eps", 0.01)
        illum = _guided_filter_torch(image, image, radius, eps)
        illum = torch.max(illum, dim=1, keepdim=True)[0]

    else:
        raise ValueError(f"Unknown illumination estimation method: {method}")

    return illum


def _guided_filter_torch(
    guide: torch.Tensor,
    src: torch.Tensor,
    radius: int,
    eps: float
) -> torch.Tensor:
    """Guided filter implementation in PyTorch."""
    def box_filter(x, r):
        kernel_size = 2 * r + 1
        return F.avg_pool2d(
            F.pad(x, [r, r, r, r], mode='reflect'),
            kernel_size, stride=1
        )

    B, C, H, W = guide.shape

    mean_g = box_filter(guide, radius)
    mean_s = box_filter(src, radius)
    mean_gs = box_filter(guide * src, radius)
    mean_gg = box_filter(guide * guide, radius)

    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    return mean_a * guide + mean_b


def adjust_illumination(
    illumination: torch.Tensor,
    gamma: float = 0.5,
    target_mean: Optional[float] = None
) -> torch.Tensor:
    """
    Adjust illumination map for enhancement.

    Args:
        illumination: Input illumination map
        gamma: Gamma correction (< 1 brightens)
        target_mean: Target mean brightness

    Returns:
        Adjusted illumination
    """
    # Gamma correction
    adjusted = torch.pow(illumination + 1e-6, gamma)

    # Scale to target mean if specified
    if target_mean is not None:
        current_mean = adjusted.mean()
        if current_mean > 1e-6:
            scale = target_mean / current_mean
            adjusted = adjusted * scale

    return torch.clamp(adjusted, 0, 1)


class LearnableIlluminationModel(nn.Module):
    """
    Learnable illumination estimation and adjustment.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()

        # Illumination estimation
        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Learnable gamma
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # Adjustment network
        self.adjuster = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate and adjust illumination.

        Returns:
            Tuple of (illumination, adjusted_illumination, reflectance)
        """
        # Estimate illumination
        illum = self.estimator(image)

        # Compute reflectance
        reflectance = image / (illum + 1e-4)
        reflectance = torch.clamp(reflectance, 0, 1)

        # Adjust illumination
        gamma = torch.clamp(self.gamma, 0.1, 1.0)
        adjusted = torch.pow(illum + 1e-4, gamma)
        adjusted = self.adjuster(adjusted)

        return illum, adjusted, reflectance


class AtmosphericScatteringModel(nn.Module):
    """
    Atmospheric Scattering Model (ASM) for low-light enhancement.

    Based on haze removal principles adapted for low-light.
    I(x) = J(x)t(x) + A(1 - t(x))
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()

        # Transmission estimation
        self.transmission = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Atmospheric light estimation
        self.atm_light = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply ASM-based enhancement.

        Returns:
            Tuple of (enhanced, transmission, atmospheric_light)
        """
        # Estimate transmission
        t = self.transmission(image)
        t = torch.clamp(t, 0.1, 1.0)

        # Estimate atmospheric light
        A = self.atm_light(image)
        A = A.view(-1, 3, 1, 1)

        # Recover scene radiance
        # J = (I - A(1-t)) / t
        enhanced = (image - A * (1 - t)) / t
        enhanced = torch.clamp(enhanced, 0, 1)

        return enhanced, t, A
