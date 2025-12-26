"""
Histogram-based image enhancement methods.

Includes histogram equalization, CLAHE, and adaptive gamma correction.
"""

import numpy as np
import cv2
from typing import Union, Optional, Tuple
import torch
from nightsight.core.base import BaseEnhancer


def histogram_equalization(
    image: np.ndarray,
    color_space: str = "hsv"
) -> np.ndarray:
    """
    Apply histogram equalization.

    Args:
        image: Input image (uint8, RGB)
        color_space: Color space for processing ('hsv', 'lab', 'yuv', 'rgb')

    Returns:
        Equalized image
    """
    if len(image.shape) == 2:
        # Grayscale
        return cv2.equalizeHist(image)

    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif color_space == "lab":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif color_space == "yuv":
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    elif color_space == "rgb":
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.equalizeHist(image[:, :, i])
        return result

    else:
        raise ValueError(f"Unknown color space: {color_space}")


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "lab"
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE divides the image into tiles and applies histogram equalization
    locally, with contrast limiting to reduce noise amplification.

    Args:
        image: Input image (uint8)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for tiles
        color_space: Color space for processing

    Returns:
        Enhanced image
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe_obj.apply(image)

    if color_space == "lab":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = clahe_obj.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif color_space == "yuv":
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = clahe_obj.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    else:
        raise ValueError(f"Unknown color space: {color_space}")


def gamma_correction(
    image: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Apply gamma correction.

    Args:
        image: Input image (float32, [0, 1])
        gamma: Gamma value (< 1 brightens, > 1 darkens)

    Returns:
        Gamma-corrected image
    """
    return np.power(np.clip(image, 1e-8, 1.0), gamma)


def adaptive_gamma(
    image: np.ndarray,
    method: str = "entropy",
    min_gamma: float = 0.3,
    max_gamma: float = 2.0
) -> np.ndarray:
    """
    Apply adaptive gamma correction based on image statistics.

    Args:
        image: Input image (float32, [0, 1])
        method: Method to compute adaptive gamma ('entropy', 'mean', 'median')
        min_gamma: Minimum gamma value
        max_gamma: Maximum gamma value

    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Use luminance channel
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        luminance = image

    # Compute adaptive gamma
    if method == "entropy":
        # Higher entropy -> lower gamma
        hist, _ = np.histogram(luminance.flatten(), bins=256, range=(0, 1))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        # Normalize entropy (max for 256 bins is 8)
        norm_entropy = entropy / 8.0
        gamma = min_gamma + (max_gamma - min_gamma) * (1 - norm_entropy)

    elif method == "mean":
        # Darker images get lower gamma
        mean_lum = luminance.mean()
        gamma = min_gamma + (max_gamma - min_gamma) * mean_lum

    elif method == "median":
        median_lum = np.median(luminance)
        gamma = min_gamma + (max_gamma - min_gamma) * median_lum

    else:
        raise ValueError(f"Unknown adaptive gamma method: {method}")

    return gamma_correction(image, gamma)


def exposure_fusion(
    image: np.ndarray,
    num_exposures: int = 3,
    exposure_stops: float = 2.0
) -> np.ndarray:
    """
    Create a pseudo-HDR image by fusing multiple simulated exposures.

    Args:
        image: Input low-light image (float32, [0, 1])
        num_exposures: Number of exposure levels
        exposure_stops: Range of exposure stops

    Returns:
        Fused image
    """
    # Generate exposure stack
    exposures = []
    gammas = np.linspace(1.0 / exposure_stops, exposure_stops, num_exposures)

    for gamma in gammas:
        exposed = gamma_correction(image, 1.0 / gamma)
        exposures.append(exposed)

    # Compute weights based on exposure quality
    weights = []
    for exp in exposures:
        # Favor well-exposed regions
        well_exposed = np.exp(-((exp - 0.5) ** 2) / (2 * 0.2 ** 2))
        if len(exp.shape) == 3:
            well_exposed = np.mean(well_exposed, axis=2)

        # Add small constant to avoid division by zero
        weights.append(well_exposed + 1e-6)

    # Normalize weights
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # Fuse exposures
    fused = np.zeros_like(image)
    for exp, w in zip(exposures, normalized_weights):
        if len(image.shape) == 3:
            fused += exp * w[:, :, np.newaxis]
        else:
            fused += exp * w

    return np.clip(fused, 0, 1)


def auto_brightness_and_contrast(
    image: np.ndarray,
    clip_hist_percent: float = 1.0
) -> Tuple[np.ndarray, float, float]:
    """
    Automatically adjust brightness and contrast.

    Args:
        image: Input image (uint8)
        clip_hist_percent: Percentage of histogram to clip

    Returns:
        Tuple of (adjusted image, alpha (contrast), beta (brightness))
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution
    accumulator = [hist[0]]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i - 1] + hist[i])

    # Locate cut-off points
    max_val = accumulator[-1]
    clip_hist_percent *= max_val / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut-off
    min_gray = 0
    while accumulator[min_gray] < clip_hist_percent:
        min_gray += 1

    # Locate right cut-off
    max_gray = hist_size - 1
    while accumulator[max_gray] >= (max_val - clip_hist_percent):
        max_gray -= 1

    # Calculate alpha (contrast) and beta (brightness)
    alpha = 255 / (max_gray - min_gray)
    beta = -min_gray * alpha

    # Apply adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted, alpha, beta


class CLAHEEnhancer(BaseEnhancer):
    """
    CLAHE-based low-light image enhancer with additional processing.
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        color_space: str = "lab",
        denoise: bool = True,
        gamma: Optional[float] = None,
        device: str = "auto"
    ):
        """
        Initialize CLAHE enhancer.

        Args:
            clip_limit: Contrast limiting threshold
            tile_grid_size: Grid size for local histogram equalization
            color_space: Color space for processing
            denoise: Whether to apply denoising
            gamma: Optional gamma correction (None for auto)
            device: Processing device
        """
        super().__init__(device)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.color_space = color_space
        self.denoise = denoise
        self.gamma = gamma

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance a low-light image using CLAHE.

        Args:
            image: Input image
            **kwargs: Additional parameters

        Returns:
            Enhanced image
        """
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = self.tensor_to_numpy(image)

        # Ensure uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Pre-processing: denoise if needed
        if self.denoise:
            image = cv2.fastNlMeansDenoisingColored(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                None, 10, 10, 7, 21
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE
        clip_limit = kwargs.get("clip_limit", self.clip_limit)
        enhanced = clahe(
            image,
            clip_limit=clip_limit,
            tile_grid_size=self.tile_grid_size,
            color_space=self.color_space
        )

        # Apply gamma correction
        gamma = kwargs.get("gamma", self.gamma)
        if gamma is not None:
            enhanced_float = enhanced.astype(np.float32) / 255.0
            enhanced_float = gamma_correction(enhanced_float, gamma)
            enhanced = (enhanced_float * 255).astype(np.uint8)

        if is_tensor:
            enhanced = self.numpy_to_tensor(enhanced).float()

        return enhanced


def compute_histogram(
    image: np.ndarray,
    bins: int = 256,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute histogram of an image.

    Args:
        image: Input image
        bins: Number of histogram bins
        normalized: Whether to normalize histogram

    Returns:
        Histogram array
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()

    if normalized:
        hist = hist / hist.sum()

    return hist


def histogram_specification(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Match the histogram of source image to target image.

    Args:
        source: Source image to modify
        target: Target image with desired histogram

    Returns:
        Source image with matched histogram
    """
    # Compute histograms
    source_hist = compute_histogram(source, normalized=True)
    target_hist = compute_histogram(target, normalized=True)

    # Compute CDFs
    source_cdf = source_hist.cumsum()
    target_cdf = target_hist.cumsum()

    # Create lookup table
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.abs(target_cdf - source_cdf[i]).argmin()
        lookup[i] = j

    # Apply to each channel
    if len(source.shape) == 3:
        result = np.zeros_like(source)
        for c in range(3):
            result[:, :, c] = cv2.LUT(source[:, :, c], lookup)
        return result
    else:
        return cv2.LUT(source, lookup)
