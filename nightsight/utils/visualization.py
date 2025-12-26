"""Visualization utilities for NightSight."""

import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from pathlib import Path
import cv2


def normalize_for_display(
    image: Union[np.ndarray, torch.Tensor],
    percentile_clip: float = 0.0
) -> np.ndarray:
    """
    Normalize an image for display.

    Args:
        image: Input image
        percentile_clip: Percentile to clip (e.g., 1.0 for 1% and 99%)

    Returns:
        Normalized uint8 image
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    image = image.astype(np.float32)

    if percentile_clip > 0:
        low = np.percentile(image, percentile_clip)
        high = np.percentile(image, 100 - percentile_clip)
        image = np.clip(image, low, high)
        image = (image - low) / (high - low + 1e-8)
    else:
        if image.max() > 1.0:
            image = image / 255.0
        image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


def create_grid(
    images: List[Union[np.ndarray, torch.Tensor]],
    nrow: int = 4,
    padding: int = 2,
    pad_value: float = 1.0
) -> np.ndarray:
    """
    Create a grid of images.

    Args:
        images: List of images to arrange
        nrow: Number of images per row
        padding: Padding between images
        pad_value: Value for padding (0-1)

    Returns:
        Grid image as numpy array
    """
    if not images:
        raise ValueError("Empty image list")

    # Convert all to numpy
    processed = []
    for img in images:
        processed.append(normalize_for_display(img))

    # Get dimensions
    h, w = processed[0].shape[:2]
    c = 3 if len(processed[0].shape) == 3 else 1

    # Calculate grid size
    ncol = nrow
    nrow = (len(processed) + ncol - 1) // ncol

    # Create grid
    grid_h = h * nrow + padding * (nrow - 1)
    grid_w = w * ncol + padding * (ncol - 1)

    if c == 1:
        grid = np.full((grid_h, grid_w), int(pad_value * 255), dtype=np.uint8)
    else:
        grid = np.full((grid_h, grid_w, c), int(pad_value * 255), dtype=np.uint8)

    # Fill grid
    for idx, img in enumerate(processed):
        row = idx // ncol
        col = idx % ncol

        y = row * (h + padding)
        x = col * (w + padding)

        if c == 1 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif c == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        grid[y:y + h, x:x + w] = img

    return grid


def visualize_comparison(
    low_light: Union[np.ndarray, torch.Tensor],
    enhanced: Union[np.ndarray, torch.Tensor],
    ground_truth: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Create a side-by-side comparison visualization.

    Args:
        low_light: Low-light input image
        enhanced: Enhanced output image
        ground_truth: Optional ground truth image
        title: Optional title to add
        save_path: Optional path to save the visualization

    Returns:
        Comparison image
    """
    images = [low_light, enhanced]
    labels = ["Low-Light Input", "Enhanced"]

    if ground_truth is not None:
        images.append(ground_truth)
        labels.append("Ground Truth")

    # Create grid
    comparison = create_grid(images, nrow=len(images), padding=5)

    # Add labels
    h = images[0].shape[0] if isinstance(images[0], np.ndarray) else images[0].shape[-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for i, label in enumerate(labels):
        x = i * (comparison.shape[1] // len(labels)) + 10
        y = 20
        cv2.putText(comparison, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    if title:
        cv2.putText(comparison, title, (10, comparison.shape[0] - 10),
                    font, font_scale, (255, 255, 255), thickness)

    if save_path:
        from nightsight.utils.io import save_image
        save_image(comparison, save_path)

    return comparison


def visualize_enhancement(
    original: Union[np.ndarray, torch.Tensor],
    enhanced: Union[np.ndarray, torch.Tensor],
    intermediate: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Visualize the enhancement process with optional intermediate results.

    Args:
        original: Original low-light image
        enhanced: Final enhanced image
        intermediate: Optional list of intermediate results
        save_path: Optional path to save

    Returns:
        Visualization image
    """
    images = [original]

    if intermediate:
        images.extend(intermediate)

    images.append(enhanced)

    grid = create_grid(images, nrow=len(images), padding=3)

    if save_path:
        from nightsight.utils.io import save_image
        save_image(grid, save_path)

    return grid


def plot_histogram(
    image: Union[np.ndarray, torch.Tensor],
    channel: str = "all"
) -> np.ndarray:
    """
    Plot the histogram of an image.

    Args:
        image: Input image
        channel: Which channel to plot ('all', 'r', 'g', 'b', 'gray')

    Returns:
        Histogram plot as numpy array
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Create histogram image
    hist_h = 200
    hist_w = 256
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    if len(image.shape) == 2 or channel == "gray":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.max() * (hist_h - 10)

        for i in range(256):
            cv2.line(hist_img, (i, hist_h), (i, hist_h - int(hist[i])), (255, 255, 255), 1)

    else:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR for OpenCV
        for c, color in enumerate(colors):
            hist = cv2.calcHist([image], [c], None, [256], [0, 256])
            hist = hist / hist.max() * (hist_h - 10)

            for i in range(256):
                cv2.line(hist_img, (i, hist_h), (i, hist_h - int(hist[i])), color, 1)

    return hist_img


def create_heatmap(
    values: np.ndarray,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Create a heatmap visualization from a 2D array.

    Args:
        values: 2D array of values
        colormap: Colormap name ('jet', 'hot', 'viridis', etc.)

    Returns:
        Heatmap as RGB numpy array
    """
    # Normalize to 0-255
    normalized = ((values - values.min()) / (values.max() - values.min() + 1e-8) * 255).astype(np.uint8)

    # Apply colormap
    colormap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }

    cm = colormap_dict.get(colormap, cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(normalized, cm)

    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def visualize_illumination_map(
    image: Union[np.ndarray, torch.Tensor],
    illumination: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Visualize an illumination map alongside the original image.

    Args:
        image: Original image
        illumination: Estimated illumination map
        save_path: Optional path to save

    Returns:
        Visualization image
    """
    if isinstance(image, torch.Tensor):
        image = normalize_for_display(image)

    if isinstance(illumination, torch.Tensor):
        if illumination.dim() == 4:
            illumination = illumination.squeeze(0)
        if illumination.dim() == 3:
            illumination = illumination.mean(0)
        illumination = illumination.detach().cpu().numpy()

    # Create heatmap of illumination
    illum_heatmap = create_heatmap(illumination)

    # Resize if needed
    if illum_heatmap.shape[:2] != image.shape[:2]:
        illum_heatmap = cv2.resize(illum_heatmap, (image.shape[1], image.shape[0]))

    # Combine
    result = create_grid([image, illum_heatmap], nrow=2, padding=5)

    if save_path:
        from nightsight.utils.io import save_image
        save_image(result, save_path)

    return result
