"""Color-based loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorLoss(nn.Module):
    """
    Color constancy loss.

    Encourages the mean of each color channel to be similar,
    promoting neutral color balance.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute color constancy loss.

        Args:
            pred: Predicted image (B, 3, H, W)

        Returns:
            Color loss value
        """
        mean_rgb = pred.mean(dim=(2, 3))

        if pred.shape[1] != 3:
            return torch.tensor(0.0, device=pred.device)

        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

        loss = ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()

        return self.weight * loss


class HistogramLoss(nn.Module):
    """
    Histogram matching loss.

    Encourages the histogram of the output to match a reference.
    """

    def __init__(self, bins: int = 256, sigma: float = 0.01):
        super().__init__()
        self.bins = bins
        self.sigma = sigma

    def soft_histogram(
        self,
        x: torch.Tensor,
        bins: int,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> torch.Tensor:
        """Compute differentiable soft histogram."""
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Bin centers
        bin_centers = torch.linspace(min_val, max_val, bins, device=x.device)

        # Compute soft assignments
        diff = x_flat.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)
        weights = torch.exp(-diff ** 2 / (2 * self.sigma ** 2))

        # Normalize
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Sum to get histogram
        hist = weights.sum(dim=1)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

        return hist

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute histogram matching loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            Histogram loss
        """
        pred_hist = self.soft_histogram(pred, self.bins)
        target_hist = self.soft_histogram(target, self.bins)

        # Earth Mover's Distance approximation
        pred_cdf = pred_hist.cumsum(dim=-1)
        target_cdf = target_hist.cumsum(dim=-1)

        return F.l1_loss(pred_cdf, target_cdf)


class GradientLoss(nn.Module):
    """
    Gradient preservation loss.

    Encourages similar gradients between predicted and target.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradient magnitude."""
        if x.shape[1] == 3:
            # Convert to grayscale
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient loss."""
        pred_grad = self.compute_gradient(pred)
        target_grad = self.compute_gradient(target)

        return F.l1_loss(pred_grad, target_grad)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 variant).

    More robust to outliers than MSE.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Charbonnier loss."""
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))


class CombinedLoss(nn.Module):
    """
    Combined loss for training.

    Combines multiple loss functions with weights.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.1,
        color_weight: float = 0.05,
        gradient_weight: float = 0.05
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.gradient_weight = gradient_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss() if ssim_weight > 0 else None
        self.color = ColorLoss() if color_weight > 0 else None
        self.gradient = GradientLoss() if gradient_weight > 0 else None

        # Perceptual loss (lazy init to avoid loading VGG if not needed)
        self._perceptual = None

    @property
    def perceptual(self):
        if self._perceptual is None and self.perceptual_weight > 0:
            from nightsight.losses.perceptual import PerceptualLoss
            self._perceptual = PerceptualLoss()
        return self._perceptual

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.

        Returns:
            Dictionary with total and component losses
        """
        losses = {}

        # L1 loss
        losses['l1'] = self.l1(pred, target) * self.l1_weight

        # SSIM loss
        if self.ssim is not None:
            losses['ssim'] = self.ssim(pred, target) * self.ssim_weight

        # Perceptual loss
        if self.perceptual is not None:
            self.perceptual.to(pred.device)
            losses['perceptual'] = self.perceptual(pred, target) * self.perceptual_weight

        # Color loss
        if self.color is not None:
            losses['color'] = self.color(pred) * self.color_weight

        # Gradient loss
        if self.gradient is not None:
            self.gradient.to(pred.device)
            losses['gradient'] = self.gradient(pred, target) * self.gradient_weight

        # Total
        losses['total'] = sum(losses.values())

        return losses


# Import for CombinedLoss
from nightsight.losses.ssim import SSIMLoss
