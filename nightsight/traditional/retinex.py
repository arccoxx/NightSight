"""
Retinex-based image enhancement methods.

Retinex theory models an image as the product of reflectance and illumination:
    I = R * L

Where:
    - I is the observed image
    - R is the reflectance (intrinsic scene properties)
    - L is the illumination

Enhancement involves estimating and adjusting the illumination component.
"""

import numpy as np
import cv2
from typing import Union, List, Tuple, Optional
import torch
from nightsight.core.base import BaseEnhancer


def single_scale_retinex(
    image: np.ndarray,
    sigma: float = 80.0
) -> np.ndarray:
    """
    Single Scale Retinex (SSR) algorithm.

    Estimates illumination using Gaussian blur and computes reflectance
    in the log domain.

    Args:
        image: Input image (float32, range [0, 1])
        sigma: Gaussian kernel standard deviation

    Returns:
        Enhanced image
    """
    # Avoid log(0)
    image = np.maximum(image, 1e-6)

    # Estimate illumination with Gaussian blur
    illumination = cv2.GaussianBlur(image, (0, 0), sigma)
    illumination = np.maximum(illumination, 1e-6)

    # Compute log reflectance: log(R) = log(I) - log(L)
    log_reflectance = np.log(image) - np.log(illumination)

    return log_reflectance


def multi_scale_retinex(
    image: np.ndarray,
    sigmas: List[float] = [15, 80, 250],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) algorithm.

    Combines SSR at multiple scales for better illumination estimation.

    Args:
        image: Input image (float32, range [0, 1])
        sigmas: List of Gaussian kernel standard deviations
        weights: Optional weights for each scale (default: equal)

    Returns:
        Enhanced image
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    assert len(sigmas) == len(weights), "sigmas and weights must have same length"

    # Compute weighted sum of SSR at different scales
    msr = np.zeros_like(image, dtype=np.float32)
    for sigma, weight in zip(sigmas, weights):
        msr += weight * single_scale_retinex(image, sigma)

    return msr


def normalize_output(
    image: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize the Retinex output to valid range.

    Args:
        image: Retinex output (can be any range)
        method: Normalization method ('minmax', 'sigmoid', 'tanh')

    Returns:
        Normalized image in [0, 1]
    """
    if method == "minmax":
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 1e-6:
            return (image - min_val) / (max_val - min_val)
        return np.clip(image, 0, 1)

    elif method == "sigmoid":
        return 1 / (1 + np.exp(-image))

    elif method == "tanh":
        return (np.tanh(image) + 1) / 2

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def msrcr(
    image: np.ndarray,
    sigmas: List[float] = [15, 80, 250],
    gain: float = 128,
    offset: float = 128,
    alpha: float = 125,
    beta: float = 46
) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).

    Adds color restoration to MSR to prevent color distortion.

    Args:
        image: Input RGB image (float32, range [0, 1])
        sigmas: Gaussian kernel standard deviations
        gain: Gain for output scaling
        offset: Offset for output
        alpha: Color restoration alpha
        beta: Color restoration beta

    Returns:
        Enhanced image in [0, 1]
    """
    # Apply MSR
    msr = multi_scale_retinex(image, sigmas)

    # Color Restoration Function (CRF)
    # C(i) = beta * [log(alpha * I(i)) - log(sum(I))]
    image_sum = np.sum(image, axis=-1, keepdims=True)
    image_sum = np.maximum(image_sum, 1e-6)

    crf = beta * (np.log(alpha * image + 1e-6) - np.log(image_sum))

    # Apply color restoration
    msrcr_output = gain * (msr * crf - offset)

    # Normalize to [0, 1]
    return normalize_output(msrcr_output, method="minmax")


def estimate_illumination(
    image: np.ndarray,
    method: str = "max_channel",
    **kwargs
) -> np.ndarray:
    """
    Estimate the illumination map from an image.

    Args:
        image: Input image
        method: Estimation method ('max_channel', 'average', 'gaussian', 'guided')
        **kwargs: Method-specific parameters

    Returns:
        Estimated illumination map
    """
    if method == "max_channel":
        # Maximum of RGB channels
        if len(image.shape) == 3:
            return np.max(image, axis=-1, keepdims=True)
        return image

    elif method == "average":
        if len(image.shape) == 3:
            return np.mean(image, axis=-1, keepdims=True)
        return image

    elif method == "gaussian":
        sigma = kwargs.get("sigma", 80)
        return cv2.GaussianBlur(image, (0, 0), sigma)

    elif method == "guided":
        from nightsight.traditional.filters import guided_filter
        radius = kwargs.get("radius", 15)
        eps = kwargs.get("eps", 1e-3)
        return guided_filter(image, image, radius, eps)

    else:
        raise ValueError(f"Unknown illumination estimation method: {method}")


def retinex_decomposition(
    image: np.ndarray,
    illumination_method: str = "gaussian",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose image into reflectance and illumination components.

    Args:
        image: Input image (float32, [0, 1])
        illumination_method: Method to estimate illumination
        **kwargs: Parameters for illumination estimation

    Returns:
        Tuple of (reflectance, illumination)
    """
    # Estimate illumination
    illumination = estimate_illumination(image, illumination_method, **kwargs)
    illumination = np.maximum(illumination, 1e-6)

    # Compute reflectance: R = I / L
    reflectance = image / illumination

    return reflectance, illumination


def adjust_illumination(
    illumination: np.ndarray,
    gamma: float = 0.5,
    target_brightness: float = 0.5
) -> np.ndarray:
    """
    Adjust illumination for enhancement.

    Args:
        illumination: Illumination map
        gamma: Gamma correction value (< 1 brightens)
        target_brightness: Target average brightness

    Returns:
        Adjusted illumination
    """
    # Gamma correction
    adjusted = np.power(illumination + 1e-6, gamma)

    # Scale to target brightness
    current_mean = adjusted.mean()
    if current_mean > 1e-6:
        adjusted = adjusted * (target_brightness / current_mean)

    return np.clip(adjusted, 0, 1)


class RetinexEnhancer(BaseEnhancer):
    """
    Retinex-based low-light image enhancer.

    Combines multiple Retinex techniques with denoising and
    color correction for robust enhancement.
    """

    def __init__(
        self,
        method: str = "msrcr",
        sigmas: List[float] = [15, 80, 250],
        gamma: float = 0.6,
        denoise: bool = True,
        color_correction: bool = True,
        device: str = "auto"
    ):
        """
        Initialize the Retinex enhancer.

        Args:
            method: Retinex variant ('ssr', 'msr', 'msrcr', 'decompose')
            sigmas: Gaussian kernel sigmas for multi-scale
            gamma: Gamma correction for illumination adjustment
            denoise: Whether to apply denoising
            color_correction: Whether to apply color correction
            device: Device for processing
        """
        super().__init__(device)
        self.method = method
        self.sigmas = sigmas
        self.gamma = gamma
        self.denoise = denoise
        self.color_correction = color_correction

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance a low-light image using Retinex.

        Args:
            image: Input image (H, W, C) in [0, 255] uint8 or [0, 1] float
            **kwargs: Additional parameters

        Returns:
            Enhanced image in same format as input
        """
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = self.tensor_to_numpy(image) / 255.0

        # Convert to float if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Override parameters from kwargs
        method = kwargs.get("method", self.method)
        sigmas = kwargs.get("sigmas", self.sigmas)
        gamma = kwargs.get("gamma", self.gamma)

        # Apply Retinex enhancement
        if method == "ssr":
            enhanced = single_scale_retinex(image, sigmas[0])
            enhanced = normalize_output(enhanced)

        elif method == "msr":
            enhanced = multi_scale_retinex(image, sigmas)
            enhanced = normalize_output(enhanced)

        elif method == "msrcr":
            enhanced = msrcr(image, sigmas)

        elif method == "decompose":
            # Decomposition-based enhancement
            reflectance, illumination = retinex_decomposition(
                image, illumination_method="guided"
            )
            adjusted_illum = adjust_illumination(illumination, gamma)
            enhanced = reflectance * adjusted_illum
            enhanced = np.clip(enhanced, 0, 1)

        else:
            raise ValueError(f"Unknown Retinex method: {method}")

        # Optional denoising
        if self.denoise:
            from nightsight.traditional.filters import bilateral_filter
            enhanced = bilateral_filter(enhanced, d=9, sigma_color=0.1, sigma_space=75)

        # Optional color correction
        if self.color_correction:
            enhanced = self._color_correction(enhanced, image)

        # Convert back to original format
        if is_tensor:
            enhanced = self.numpy_to_tensor(enhanced * 255).float()

        return enhanced

    def _color_correction(
        self,
        enhanced: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """Apply simple color correction to preserve color ratios."""
        if len(enhanced.shape) != 3 or enhanced.shape[2] != 3:
            return enhanced

        # Compute color ratios from original
        gray_orig = np.mean(original, axis=2, keepdims=True)
        gray_orig = np.maximum(gray_orig, 1e-6)

        gray_enh = np.mean(enhanced, axis=2, keepdims=True)
        gray_enh = np.maximum(gray_enh, 1e-6)

        # Preserve color ratios
        corrected = (original / gray_orig) * gray_enh

        return np.clip(corrected, 0, 1)

    def get_illumination_map(
        self,
        image: np.ndarray,
        method: str = "guided"
    ) -> np.ndarray:
        """
        Get the estimated illumination map for visualization.

        Args:
            image: Input image
            method: Estimation method

        Returns:
            Illumination map
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        return estimate_illumination(image, method)
