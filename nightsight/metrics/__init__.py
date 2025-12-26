"""Evaluation metrics for image enhancement."""

import torch
import numpy as np
from typing import Union
import math


def psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image
        target: Target image
        data_range: Maximum value of pixels

    Returns:
        PSNR value in dB
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(data_range / math.sqrt(mse))


def ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        pred: Predicted image
        target: Target image
        data_range: Maximum value of pixels

    Returns:
        SSIM value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Flatten if needed
    if pred.ndim > 2:
        pred = pred.mean(axis=tuple(range(pred.ndim - 2)))
        target = target.mean(axis=tuple(range(target.ndim - 2)))

    mu_pred = pred.mean()
    mu_target = target.mean()

    sigma_pred = pred.std()
    sigma_target = target.std()
    sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()

    l = (2 * mu_pred * mu_target + C1) / (mu_pred ** 2 + mu_target ** 2 + C1)
    c = (2 * sigma_pred * sigma_target + C2) / (sigma_pred ** 2 + sigma_target ** 2 + C2)
    s = (sigma_pred_target + C2 / 2) / (sigma_pred * sigma_target + C2 / 2)

    return l * c * s


def mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> float:
    """Compute Mean Absolute Error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean(np.abs(pred - target))


def niqe(image: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Natural Image Quality Evaluator (simplified).

    No-reference metric based on natural scene statistics.
    Returns lower values for better quality.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)

    # Convert to grayscale
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image

    # Compute local statistics
    from scipy.ndimage import uniform_filter

    mu = uniform_filter(gray, size=7)
    mu_sq = uniform_filter(gray ** 2, size=7)
    sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0))

    # Normalize
    mscn = (gray - mu) / (sigma + 1e-7)

    # Compute statistics of MSCN
    mean_mscn = np.mean(mscn)
    var_mscn = np.var(mscn)

    # Simplified NIQE (real NIQE uses trained MVG model)
    niqe_score = np.abs(mean_mscn) + np.abs(var_mscn - 1)

    return float(niqe_score)


class MetricTracker:
    """Track metrics during training/evaluation."""

    def __init__(self, metrics: list = ['psnr', 'ssim', 'mae']):
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.values = {m: [] for m in self.metrics}

    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray]
    ):
        """Update metrics with new predictions."""
        for metric in self.metrics:
            if metric == 'psnr':
                self.values['psnr'].append(psnr(pred, target))
            elif metric == 'ssim':
                self.values['ssim'].append(ssim(pred, target))
            elif metric == 'mae':
                self.values['mae'].append(mae(pred, target))

    def get_averages(self) -> dict:
        """Get average values for all metrics."""
        return {m: np.mean(v) if v else 0.0 for m, v in self.values.items()}


__all__ = [
    "psnr",
    "ssim",
    "mae",
    "niqe",
    "MetricTracker",
]
