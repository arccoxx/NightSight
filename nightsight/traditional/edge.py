"""
Edge detection and enhancement methods for low-light imagery.

Edges contain critical structural information that should be preserved
and potentially enhanced during low-light processing.
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional


def detect_edges(
    image: np.ndarray,
    method: str = "canny",
    **kwargs
) -> np.ndarray:
    """
    Detect edges in an image.

    Args:
        image: Input image (grayscale or color)
        method: Edge detection method ('canny', 'sobel', 'laplacian', 'scharr')
        **kwargs: Method-specific parameters

    Returns:
        Edge map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.dtype == np.float32 or image.dtype == np.float64:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        if image.dtype == np.float32 or image.dtype == np.float64:
            gray = (image * 255).astype(np.uint8)
        else:
            gray = image

    if method == "canny":
        low_threshold = kwargs.get("low_threshold", 50)
        high_threshold = kwargs.get("high_threshold", 150)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

    elif method == "sobel":
        ksize = kwargs.get("ksize", 3)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edges = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8)

    elif method == "laplacian":
        ksize = kwargs.get("ksize", 3)
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        edges = np.abs(edges)
        edges = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8)

    elif method == "scharr":
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        edges = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
        edges = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    return edges


def compute_gradients(
    image: np.ndarray,
    method: str = "sobel"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients.

    Args:
        image: Input image
        method: Gradient computation method ('sobel', 'scharr', 'central')

    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            (image * 255 if image.dtype == np.float32 else image).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1:
            gray = gray / 255.0

    if method == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    elif method == "scharr":
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    elif method == "central":
        # Central difference
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)
        grad_x[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2
        grad_y[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2

    else:
        raise ValueError(f"Unknown gradient method: {method}")

    return grad_x, grad_y


def gradient_magnitude(
    image: np.ndarray,
    method: str = "sobel"
) -> np.ndarray:
    """
    Compute gradient magnitude.

    Args:
        image: Input image
        method: Gradient computation method

    Returns:
        Gradient magnitude map
    """
    grad_x, grad_y = compute_gradients(image, method)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def structure_tensor(
    image: np.ndarray,
    sigma: float = 1.0,
    rho: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the structure tensor for edge/corner detection.

    The structure tensor captures local gradient statistics and can be
    used to detect edges, corners, and texture orientation.

    Args:
        image: Input image (grayscale)
        sigma: Derivative smoothing scale
        rho: Integration scale

    Returns:
        Tuple of (J11, J12, J22) structure tensor components
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            (image * 255 if image.dtype == np.float32 else image).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1:
            gray = gray / 255.0

    # Smooth for derivative
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)

    # Compute gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Compute products
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Integration (averaging over neighborhood)
    if rho > 0:
        J11 = cv2.GaussianBlur(Ixx, (0, 0), rho)
        J12 = cv2.GaussianBlur(Ixy, (0, 0), rho)
        J22 = cv2.GaussianBlur(Iyy, (0, 0), rho)
    else:
        J11, J12, J22 = Ixx, Ixy, Iyy

    return J11, J12, J22


def coherence_map(
    image: np.ndarray,
    sigma: float = 1.0,
    rho: float = 3.0
) -> np.ndarray:
    """
    Compute coherence map from structure tensor.

    High coherence indicates strong edge/structure, low coherence
    indicates noise or texture.

    Args:
        image: Input image
        sigma: Derivative smoothing scale
        rho: Integration scale

    Returns:
        Coherence map (0-1)
    """
    J11, J12, J22 = structure_tensor(image, sigma, rho)

    # Eigenvalue computation
    trace = J11 + J22
    det = J11 * J22 - J12 * J12
    disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

    lambda1 = (trace + disc) / 2
    lambda2 = (trace - disc) / 2

    # Coherence
    coherence = np.zeros_like(lambda1)
    mask = (lambda1 + lambda2) > 1e-10
    coherence[mask] = ((lambda1[mask] - lambda2[mask]) /
                       (lambda1[mask] + lambda2[mask])) ** 2

    return coherence


def enhance_edges(
    image: np.ndarray,
    strength: float = 1.0,
    method: str = "unsharp"
) -> np.ndarray:
    """
    Enhance edges in an image.

    Args:
        image: Input image
        strength: Enhancement strength
        method: Enhancement method ('unsharp', 'laplacian', 'gradient')

    Returns:
        Edge-enhanced image
    """
    is_float = image.dtype in [np.float32, np.float64]

    if not is_float:
        image = image.astype(np.float32) / 255.0

    if method == "unsharp":
        # Unsharp masking
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        enhanced = image + strength * (image - blurred)

    elif method == "laplacian":
        # Laplacian enhancement
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian = laplacian / (np.abs(laplacian).max() + 1e-10)

        if len(image.shape) == 3:
            enhanced = image - strength * laplacian[:, :, np.newaxis]
        else:
            enhanced = image - strength * laplacian

    elif method == "gradient":
        # Gradient-based enhancement
        magnitude = gradient_magnitude(image)
        magnitude = magnitude / (magnitude.max() + 1e-10)

        if len(image.shape) == 3:
            enhanced = image + strength * magnitude[:, :, np.newaxis] * image
        else:
            enhanced = image + strength * magnitude * image

    else:
        raise ValueError(f"Unknown edge enhancement method: {method}")

    enhanced = np.clip(enhanced, 0, 1)

    if not is_float:
        enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced


def extract_edge_features(
    image: np.ndarray
) -> dict:
    """
    Extract edge-based features for analysis.

    Args:
        image: Input image

    Returns:
        Dictionary of edge features
    """
    edges_canny = detect_edges(image, "canny")
    magnitude = gradient_magnitude(image)
    coherence = coherence_map(image)

    return {
        "edge_density": np.mean(edges_canny > 0),
        "mean_gradient_magnitude": np.mean(magnitude),
        "max_gradient_magnitude": np.max(magnitude),
        "mean_coherence": np.mean(coherence),
        "edge_pixels": np.sum(edges_canny > 0),
    }


def edge_aware_smooth(
    image: np.ndarray,
    sigma_space: float = 10,
    sigma_color: float = 0.1
) -> np.ndarray:
    """
    Edge-aware smoothing that preserves edges while denoising.

    Args:
        image: Input image
        sigma_space: Spatial smoothing sigma
        sigma_color: Color/intensity smoothing sigma

    Returns:
        Smoothed image with preserved edges
    """
    from nightsight.traditional.filters import bilateral_filter

    return bilateral_filter(
        image,
        d=int(sigma_space * 2 + 1),
        sigma_color=sigma_color * 255,
        sigma_space=sigma_space
    )


def detect_corners(
    image: np.ndarray,
    method: str = "harris",
    **kwargs
) -> np.ndarray:
    """
    Detect corners in an image.

    Args:
        image: Input image
        method: Corner detection method ('harris', 'shi_tomasi')
        **kwargs: Method-specific parameters

    Returns:
        Corner response map or corner points
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            (image * 255 if image.dtype == np.float32 else image).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)

    if method == "harris":
        block_size = kwargs.get("block_size", 2)
        ksize = kwargs.get("ksize", 3)
        k = kwargs.get("k", 0.04)
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        return corners

    elif method == "shi_tomasi":
        max_corners = kwargs.get("max_corners", 100)
        quality_level = kwargs.get("quality_level", 0.01)
        min_distance = kwargs.get("min_distance", 10)
        corners = cv2.goodFeaturesToTrack(
            gray, max_corners, quality_level, min_distance
        )
        return corners

    else:
        raise ValueError(f"Unknown corner detection method: {method}")
