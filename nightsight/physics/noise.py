"""Physics-based noise modeling for low-light images."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class NoiseModel:
    """
    Physics-based noise model for camera sensors.

    Models the heteroscedastic noise in low-light images, including:
    - Shot noise (Poisson distributed, signal-dependent)
    - Read noise (Gaussian, signal-independent)
    - Quantization noise
    - Dark current noise
    """

    def __init__(
        self,
        shot_noise_scale: float = 0.01,
        read_noise_std: float = 0.02,
        quantization_bits: int = 8,
        dark_current: float = 0.001
    ):
        """
        Initialize noise model.

        Args:
            shot_noise_scale: Scale for shot noise
            read_noise_std: Standard deviation of read noise
            quantization_bits: Bit depth for quantization
            dark_current: Dark current noise level
        """
        self.shot_noise_scale = shot_noise_scale
        self.read_noise_std = read_noise_std
        self.quantization_bits = quantization_bits
        self.dark_current = dark_current

    def add_noise(
        self,
        image: torch.Tensor,
        iso: float = 1600
    ) -> torch.Tensor:
        """
        Add realistic camera noise to clean image.

        Args:
            image: Clean image in [0, 1]
            iso: ISO sensitivity (higher = more noise)

        Returns:
            Noisy image
        """
        iso_factor = iso / 100.0

        # Shot noise (Poisson)
        shot_std = torch.sqrt(image * self.shot_noise_scale * iso_factor + 1e-8)
        shot_noise = torch.randn_like(image) * shot_std

        # Read noise (Gaussian)
        read_noise = torch.randn_like(image) * self.read_noise_std * np.sqrt(iso_factor)

        # Dark current
        dark_noise = torch.randn_like(image) * self.dark_current * iso_factor

        # Combined noise
        noisy = image + shot_noise + read_noise + dark_noise

        # Quantization
        if self.quantization_bits < 16:
            levels = 2 ** self.quantization_bits
            noisy = torch.round(noisy * levels) / levels

        return torch.clamp(noisy, 0, 1)

    def estimate_noise_level(self, image: torch.Tensor) -> float:
        """Estimate noise level in an image."""
        # Use Median Absolute Deviation (MAD) estimator
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # High-pass filter to isolate noise
        from scipy.ndimage import median_filter
        smooth = median_filter(image, size=3)
        noise = image - smooth

        # MAD estimator
        mad = np.median(np.abs(noise - np.median(noise)))
        sigma = mad / 0.6745

        return float(sigma)


class PoissonGaussianNoise(nn.Module):
    """
    Learnable Poisson-Gaussian noise model.

    Learns the noise parameters from data.
    """

    def __init__(self):
        super().__init__()

        # Learnable noise parameters
        self.log_shot_scale = nn.Parameter(torch.tensor(-4.0))
        self.log_read_std = nn.Parameter(torch.tensor(-3.5))

    def forward(
        self,
        image: torch.Tensor,
        return_params: bool = False
    ) -> torch.Tensor:
        """Add Poisson-Gaussian noise."""
        shot_scale = torch.exp(self.log_shot_scale)
        read_std = torch.exp(self.log_read_std)

        # Shot noise
        shot_std = torch.sqrt(image * shot_scale + 1e-8)
        shot_noise = torch.randn_like(image) * shot_std

        # Read noise
        read_noise = torch.randn_like(image) * read_std

        noisy = image + shot_noise + read_noise

        if return_params:
            return noisy, {'shot_scale': shot_scale, 'read_std': read_std}
        return noisy


def add_noise(
    image: torch.Tensor,
    noise_type: str = "gaussian",
    sigma: float = 0.05,
    **kwargs
) -> torch.Tensor:
    """
    Add noise to image.

    Args:
        image: Clean image
        noise_type: Type of noise ('gaussian', 'poisson', 'speckle', 'salt_pepper')
        sigma: Noise standard deviation
        **kwargs: Additional noise parameters

    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image + noise, 0, 1)

    elif noise_type == "poisson":
        lam = image / sigma if sigma > 0 else image * 100
        noisy = torch.poisson(lam) * sigma if sigma > 0 else torch.poisson(lam) / 100
        return torch.clamp(noisy, 0, 1)

    elif noise_type == "speckle":
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image + image * noise, 0, 1)

    elif noise_type == "salt_pepper":
        prob = kwargs.get("prob", 0.05)
        noisy = image.clone()
        salt = torch.rand_like(image) < prob / 2
        pepper = torch.rand_like(image) < prob / 2
        noisy[salt] = 1.0
        noisy[pepper] = 0.0
        return noisy

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def estimate_noise(
    image: torch.Tensor,
    method: str = "mad"
) -> torch.Tensor:
    """
    Estimate noise level in image.

    Args:
        image: Input image
        method: Estimation method ('mad', 'pca', 'wavelet')

    Returns:
        Estimated noise standard deviation
    """
    if method == "mad":
        # Compute high-frequency component using Laplacian
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape
        noise_estimates = []

        for c in range(C):
            channel = image[:, c:c+1]
            hf = F.conv2d(channel, laplacian_kernel, padding=1)
            # MAD estimator
            mad = torch.median(torch.abs(hf - torch.median(hf)))
            sigma = mad / 0.6745
            noise_estimates.append(sigma)

        return torch.stack(noise_estimates).mean()

    elif method == "wavelet":
        # Simple wavelet-based estimation using Haar
        # Approximate HH subband
        B, C, H, W = image.shape if len(image.shape) == 4 else (1, *image.shape)
        image = image.view(B, C, H, W)

        # Haar transform approximation
        hh = image[:, :, 1::2, 1::2] - image[:, :, 0::2, 1::2] - \
             image[:, :, 1::2, 0::2] + image[:, :, 0::2, 0::2]

        mad = torch.median(torch.abs(hh - torch.median(hh)))
        return mad / 0.6745

    else:
        raise ValueError(f"Unknown estimation method: {method}")


class AdaptiveNoiseModel(nn.Module):
    """
    Adaptive noise model that estimates noise parameters from input.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()

        # Noise parameter estimation network
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_channels * 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # shot_scale, read_std
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate noise parameters."""
        params = self.estimator(image)
        return {
            'shot_scale': torch.exp(params[:, 0]),
            'read_std': torch.exp(params[:, 1])
        }


# Import F for conv2d
import torch.nn.functional as F
