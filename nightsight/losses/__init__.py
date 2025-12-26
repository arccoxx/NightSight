"""Loss functions for training NightSight models."""

from nightsight.losses.perceptual import PerceptualLoss, VGGFeatures
from nightsight.losses.ssim import SSIMLoss, MS_SSIMLoss
from nightsight.losses.color import ColorLoss, HistogramLoss

__all__ = [
    "PerceptualLoss",
    "VGGFeatures",
    "SSIMLoss",
    "MS_SSIMLoss",
    "ColorLoss",
    "HistogramLoss",
]
