"""
Image filtering methods for denoising and enhancement.

Includes bilateral filtering, guided filtering, and non-local means.
"""

import numpy as np
import cv2
from typing import Union, Optional


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.

    The bilateral filter smooths images while preserving edges by
    combining domain and range filtering.

    Args:
        image: Input image (float32 [0,1] or uint8)
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Filtered image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if is_float:
        # Scale sigma for float images
        sigma_color_scaled = sigma_color / 255.0
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        sigma_color_scaled = sigma_color
        image_uint8 = image

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(
        image_uint8, d, sigma_color_scaled * 255, sigma_space
    )

    if is_float:
        return filtered.astype(np.float32) / 255.0

    return filtered


def guided_filter(
    image: np.ndarray,
    guide: Optional[np.ndarray] = None,
    radius: int = 8,
    eps: float = 0.01
) -> np.ndarray:
    """
    Apply guided filter for edge-preserving smoothing.

    The guided filter uses a guidance image to filter the input,
    preserving edges while smoothing homogeneous regions.

    Args:
        image: Input image to filter (float32 [0,1])
        guide: Guidance image (if None, uses image as guide)
        radius: Filter radius
        eps: Regularization parameter (controls smoothness)

    Returns:
        Filtered image
    """
    if guide is None:
        guide = image

    # Ensure float32
    image = image.astype(np.float32)
    guide = guide.astype(np.float32)

    # Box filter for mean computation
    def box_filter(x, r):
        return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))

    # Handle color images
    if len(guide.shape) == 3 and guide.shape[2] == 3:
        # Use grayscale guide for simplicity
        guide_gray = cv2.cvtColor(
            (guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
    else:
        guide_gray = guide if len(guide.shape) == 2 else guide[:, :, 0]

    # Process each channel
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _guided_filter_single(
                image[:, :, c], guide_gray, radius, eps
            )
        return result
    else:
        return _guided_filter_single(image, guide_gray, radius, eps)


def _guided_filter_single(
    p: np.ndarray,
    I: np.ndarray,
    r: int,
    eps: float
) -> np.ndarray:
    """Apply guided filter to a single channel."""
    # Box filter for mean computation
    def box_filter(x, r):
        return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b

    return q


def nlm_denoise(
    image: np.ndarray,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means denoising.

    NLM denoising exploits the self-similarity of images,
    averaging similar patches throughout the image.

    Args:
        image: Input image (uint8)
        h: Filter strength (higher removes more noise but also detail)
        template_window_size: Size of template patch (odd number)
        search_window_size: Size of search window (odd number)

    Returns:
        Denoised image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if is_float:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    if len(image_uint8.shape) == 3:
        # Color image
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            image_bgr, None, h, h,
            template_window_size, search_window_size
        )
        denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    else:
        denoised = cv2.fastNlMeansDenoising(
            image_uint8, None, h,
            template_window_size, search_window_size
        )

    if is_float:
        return denoised.astype(np.float32) / 255.0

    return denoised


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
    threshold: float = 0
) -> np.ndarray:
    """
    Apply unsharp masking for edge enhancement.

    Args:
        image: Input image
        sigma: Gaussian blur sigma
        strength: Sharpening strength
        threshold: Minimum difference for sharpening

    Returns:
        Sharpened image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if is_float:
        image_work = image.copy()
    else:
        image_work = image.astype(np.float32) / 255.0

    # Create blurred version
    blurred = cv2.GaussianBlur(image_work, (0, 0), sigma)

    # Compute difference
    diff = image_work - blurred

    # Apply threshold
    if threshold > 0:
        diff = np.where(np.abs(diff) > threshold, diff, 0)

    # Add sharpening
    sharpened = image_work + strength * diff

    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 1)

    if not is_float:
        sharpened = (sharpened * 255).astype(np.uint8)

    return sharpened


def median_filter(
    image: np.ndarray,
    ksize: int = 5
) -> np.ndarray:
    """
    Apply median filter for salt-and-pepper noise removal.

    Args:
        image: Input image
        ksize: Kernel size (must be odd)

    Returns:
        Filtered image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if is_float:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    filtered = cv2.medianBlur(image_uint8, ksize)

    if is_float:
        return filtered.astype(np.float32) / 255.0

    return filtered


def gaussian_filter(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter.

    Args:
        image: Input image
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Smoothed image
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)


def anisotropic_diffusion(
    image: np.ndarray,
    num_iterations: int = 10,
    kappa: float = 50,
    gamma: float = 0.1,
    option: int = 1
) -> np.ndarray:
    """
    Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.

    Args:
        image: Input image (float32, [0, 1])
        num_iterations: Number of diffusion iterations
        kappa: Conduction coefficient
        gamma: Max value of gradient (controls speed of diffusion)
        option: 1 for exp, 2 for 1/(1 + (grad/kappa)^2)

    Returns:
        Diffused image
    """
    image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    for _ in range(num_iterations):
        # Compute gradients
        nabla_n = np.roll(image, -1, axis=0) - image
        nabla_s = np.roll(image, 1, axis=0) - image
        nabla_e = np.roll(image, -1, axis=1) - image
        nabla_w = np.roll(image, 1, axis=1) - image

        # Compute diffusion coefficients
        if option == 1:
            c_n = np.exp(-(nabla_n / kappa) ** 2)
            c_s = np.exp(-(nabla_s / kappa) ** 2)
            c_e = np.exp(-(nabla_e / kappa) ** 2)
            c_w = np.exp(-(nabla_w / kappa) ** 2)
        else:
            c_n = 1 / (1 + (nabla_n / kappa) ** 2)
            c_s = 1 / (1 + (nabla_s / kappa) ** 2)
            c_e = 1 / (1 + (nabla_e / kappa) ** 2)
            c_w = 1 / (1 + (nabla_w / kappa) ** 2)

        # Update image
        image = image + gamma * (
            c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w
        )

    return np.squeeze(image)


def detail_enhancement_filter(
    image: np.ndarray,
    sigma_s: float = 60,
    sigma_r: float = 0.4
) -> np.ndarray:
    """
    Apply detail enhancement using edge-preserving filter.

    Args:
        image: Input image (uint8)
        sigma_s: Spatial sigma (larger = more smoothing)
        sigma_r: Range sigma (larger = edges less preserved)

    Returns:
        Enhanced image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if is_float:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    if len(image_uint8.shape) == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    enhanced_bgr = cv2.detailEnhance(image_bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    if is_float:
        return enhanced.astype(np.float32) / 255.0

    return enhanced
