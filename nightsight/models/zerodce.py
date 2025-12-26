"""
Zero-Reference Deep Curve Estimation (Zero-DCE) for low-light enhancement.

Zero-DCE learns to estimate image-specific curves that map low-light
images to well-exposed versions without paired training data.

Reference: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
Li et al., CVPR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from nightsight.core.base import BaseModel, BaseEnhancer
from nightsight.core.registry import ModelRegistry


class DCENet(nn.Module):
    """Deep Curve Estimation Network for Zero-DCE."""

    def __init__(
        self,
        in_channels: int = 3,
        num_curves: int = 8,
        base_channels: int = 32
    ):
        """
        Initialize DCE-Net.

        Args:
            in_channels: Number of input channels
            num_curves: Number of curve iterations
            base_channels: Number of base feature channels
        """
        super().__init__()

        self.num_curves = num_curves

        # Simple CNN for curve estimation
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.conv6 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.conv7 = nn.Conv2d(base_channels * 2, in_channels * num_curves, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate enhancement curves.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Curve parameters (B, C * num_curves, H, W)
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))

        curves = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        return curves


@ModelRegistry.register("zerodce")
class ZeroDCE(BaseModel):
    """
    Zero-DCE model for low-light image enhancement.

    Uses iterative light enhancement curves learned without reference images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_curves: int = 8,
        base_channels: int = 32
    ):
        super().__init__()

        self.num_curves = num_curves
        self.in_channels = in_channels

        self.dce_net = DCENet(in_channels, num_curves, base_channels)

    def forward(
        self,
        x: torch.Tensor,
        return_curves: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhance low-light image.

        Args:
            x: Input image (B, C, H, W) in range [0, 1]
            return_curves: Whether to return estimated curves

        Returns:
            Enhanced image and optionally the curves
        """
        # Estimate curves
        curves = self.dce_net(x)

        # Apply iterative enhancement
        enhanced = x
        curve_list = torch.split(curves, self.in_channels, dim=1)

        for curve in curve_list:
            # LE(x) = x + alpha * x * (1 - x)
            enhanced = enhanced + curve * enhanced * (1 - enhanced)

        enhanced = torch.clamp(enhanced, 0, 1)

        if return_curves:
            return enhanced, curves
        return enhanced

    def get_losses(
        self,
        x: torch.Tensor,
        enhanced: torch.Tensor,
        curves: torch.Tensor
    ) -> dict:
        """
        Compute Zero-DCE losses.

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Spatial consistency loss
        losses["spatial"] = self._spatial_consistency_loss(x, enhanced)

        # Exposure control loss
        losses["exposure"] = self._exposure_control_loss(enhanced)

        # Color constancy loss
        losses["color"] = self._color_constancy_loss(enhanced)

        # Illumination smoothness loss
        losses["smooth"] = self._illumination_smoothness_loss(curves)

        return losses

    def _spatial_consistency_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_size: int = 4
    ) -> torch.Tensor:
        """Preserve spatial coherence between input and output."""
        # Average pool to get local means
        pool = nn.AvgPool2d(kernel_size)
        x_mean = pool(x)
        y_mean = pool(y)

        # Compute spatial differences
        d_x_h = x_mean[:, :, :, 1:] - x_mean[:, :, :, :-1]
        d_x_v = x_mean[:, :, 1:, :] - x_mean[:, :, :-1, :]
        d_y_h = y_mean[:, :, :, 1:] - y_mean[:, :, :, :-1]
        d_y_v = y_mean[:, :, 1:, :] - y_mean[:, :, :-1, :]

        loss_h = torch.mean((d_x_h - d_y_h) ** 2)
        loss_v = torch.mean((d_x_v - d_y_v) ** 2)

        return loss_h + loss_v

    def _exposure_control_loss(
        self,
        x: torch.Tensor,
        target_exposure: float = 0.6,
        patch_size: int = 16
    ) -> torch.Tensor:
        """Control exposure level of enhanced image."""
        pool = nn.AvgPool2d(patch_size)
        x_mean = pool(x)

        # Grayscale
        if x_mean.shape[1] == 3:
            x_gray = 0.299 * x_mean[:, 0] + 0.587 * x_mean[:, 1] + 0.114 * x_mean[:, 2]
        else:
            x_gray = x_mean.mean(dim=1)

        return torch.mean((x_gray - target_exposure) ** 2)

    def _color_constancy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve color balance."""
        if x.shape[1] != 3:
            return torch.tensor(0.0, device=x.device)

        mean_rgb = x.mean(dim=(2, 3))

        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

        loss = (
            (r - g) ** 2 +
            (r - b) ** 2 +
            (g - b) ** 2
        )

        return torch.mean(loss)

    def _illumination_smoothness_loss(self, curves: torch.Tensor) -> torch.Tensor:
        """Encourage smooth illumination maps."""
        # Gradient in x and y directions
        grad_x = curves[:, :, :, 1:] - curves[:, :, :, :-1]
        grad_y = curves[:, :, 1:, :] - curves[:, :, :-1, :]

        return torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)


class DCENetPP(nn.Module):
    """Lightweight DCE-Net for Zero-DCE++."""

    def __init__(
        self,
        in_channels: int = 3,
        num_curves: int = 8,
        base_channels: int = 32,
        scale_factor: int = 2
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.num_curves = num_curves
        self.in_channels = in_channels

        # Downsampled processing for efficiency
        self.down = nn.AvgPool2d(scale_factor)

        self.e_conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(base_channels * 2, in_channels * num_curves, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample for efficient processing
        x_down = self.down(x)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))

        curves = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # Upsample curves to original resolution
        curves = F.interpolate(
            curves, size=x.shape[2:],
            mode='bilinear', align_corners=True
        )

        return curves


@ModelRegistry.register("zerodce++")
class ZeroDCEPP(BaseModel):
    """
    Zero-DCE++ - Lightweight version of Zero-DCE.

    Processes at lower resolution for efficiency while maintaining quality.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_curves: int = 8,
        base_channels: int = 32,
        scale_factor: int = 2
    ):
        super().__init__()

        self.num_curves = num_curves
        self.in_channels = in_channels

        self.dce_net = DCENetPP(in_channels, num_curves, base_channels, scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curves = self.dce_net(x)

        enhanced = x
        curve_list = torch.split(curves, self.in_channels, dim=1)

        for curve in curve_list:
            enhanced = enhanced + curve * enhanced * (1 - enhanced)

        return torch.clamp(enhanced, 0, 1)


class ZeroDCEEnhancer(BaseEnhancer):
    """Wrapper for using Zero-DCE as an enhancer."""

    def __init__(
        self,
        model_type: str = "zerodce",
        checkpoint: Optional[str] = None,
        device: str = "auto",
        **model_kwargs
    ):
        super().__init__(device)

        if model_type == "zerodce":
            self.model = ZeroDCE(**model_kwargs)
        else:
            self.model = ZeroDCEPP(**model_kwargs)

        if checkpoint:
            self.model.load_pretrained(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def enhance(self, image, **kwargs):
        is_numpy = not isinstance(image, torch.Tensor)

        if is_numpy:
            tensor = self.numpy_to_tensor(image).to(self.device)
        else:
            tensor = image.to(self.device)

        with torch.no_grad():
            enhanced = self.model(tensor)

        if is_numpy:
            return self.tensor_to_numpy(enhanced)
        return enhanced
