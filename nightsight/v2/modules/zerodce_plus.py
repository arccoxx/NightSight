"""
Zero-DCE++ implementation for low-light enhancement.

Zero-Reference Deep Curve Estimation for adaptive image brightening
without paired training data. Enhanced version with better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class DCENet(nn.Module):
    """
    Deep Curve Estimation Network (DCE-Net++).

    Learns image-adaptive curves for low-light enhancement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_iterations: int = 8
    ):
        """
        Initialize DCE-Net++.

        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of feature channels
            num_iterations: Number of curve iterations
        """
        super().__init__()

        self.num_iterations = num_iterations

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)

        # Curve parameter estimation
        # Output: num_iterations * 3 channels (R, G, B curves)
        self.conv5 = nn.Conv2d(base_channels, num_iterations * 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate enhancement curves.

        Args:
            x: Input image (B, 3, H, W) in [0, 1]

        Returns:
            Curve parameters (B, num_iterations, 3, H, W)
        """
        # Feature extraction
        feat = self.relu1(self.conv1(x))
        feat = self.relu2(self.conv2(feat))
        feat = self.relu3(self.conv3(feat))
        feat = self.relu4(self.conv4(feat))

        # Curve parameters
        curves = self.conv5(feat)
        curves = self.tanh(curves)

        # Reshape to (B, num_iterations, 3, H, W)
        B, _, H, W = curves.shape
        curves = curves.view(B, self.num_iterations, 3, H, W)

        return curves

    def apply_curve(
        self,
        image: torch.Tensor,
        curve_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learned curve to image.

        The curve function is: LE(x) = x + α * x * (1 - x)

        Args:
            image: Image to enhance (B, 3, H, W)
            curve_params: Curve parameters (B, num_iterations, 3, H, W)

        Returns:
            Enhanced image (B, 3, H, W)
        """
        enhanced = image

        for i in range(self.num_iterations):
            alpha = curve_params[:, i:i+1, :, :, :]  # (B, 1, 3, H, W)
            alpha = alpha.squeeze(1)  # (B, 3, H, W)

            # Apply curve: LE(x) = x + α * x * (1 - x)
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)

        return enhanced


class ZeroDCEPlusPlus:
    """
    Zero-DCE++ enhancer for low-light images.

    Features:
    - Zero-reference learning (no paired data needed)
    - Adaptive curve estimation
    - Fast inference
    - Preserves image details
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        num_iterations: int = 8
    ):
        """
        Initialize Zero-DCE++.

        Args:
            model_path: Path to pretrained model
            device: Device to run on
            num_iterations: Number of enhancement iterations
        """
        self.device = self._get_device(device)
        self.num_iterations = num_iterations

        # Initialize model
        self.model = DCENet(num_iterations=num_iterations)

        # Load pretrained weights if available
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

        self.model.to(self.device)
        self.model.eval()

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance low-light image.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            return_numpy: Whether to return numpy array

        Returns:
            Enhanced image
        """
        is_numpy = isinstance(image, np.ndarray)

        # Convert to tensor if needed
        if is_numpy:
            if image.ndim == 3:
                # Single image
                tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
            else:
                # Batch
                tensor = torch.from_numpy(image.transpose(0, 3, 1, 2)).float()

            # Normalize to [0, 1]
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = image

        tensor = tensor.to(self.device)

        # Enhance
        with torch.no_grad():
            # Estimate curves
            curves = self.model(tensor)

            # Apply curves
            enhanced = self.model.apply_curve(tensor, curves)

            # Clip to valid range
            enhanced = torch.clamp(enhanced, 0, 1)

        # Convert back to numpy if needed
        if return_numpy and is_numpy:
            enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if image.max() > 1.0:
                enhanced = enhanced * 255.0

        return enhanced

    def get_curve_visualization(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Visualize the learned enhancement curves.

        Args:
            image: Input image

        Returns:
            Curve visualization (256, 256, 3)
        """
        is_numpy = isinstance(image, np.ndarray)

        # Convert to tensor if needed
        if is_numpy:
            if image.ndim == 3:
                tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
            else:
                tensor = torch.from_numpy(image.transpose(0, 3, 1, 2)).float()

            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = image

        tensor = tensor.to(self.device)

        # Get curves
        with torch.no_grad():
            curves = self.model(tensor)

        # Take spatial average of curve parameters
        curves_avg = curves.mean(dim=(3, 4))  # (B, num_iterations, 3)

        # Generate curve visualization
        x = np.linspace(0, 1, 256)
        y = np.zeros((3, 256))

        for i in range(3):  # For each channel
            y[i] = x.copy()
            for j in range(self.num_iterations):
                alpha = curves_avg[0, j, i].cpu().numpy()
                y[i] = y[i] + alpha * y[i] * (1 - y[i])

        # Create visualization image
        vis = np.ones((256, 256, 3)) * 255

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R, G, B
        for i in range(3):
            for j in range(255):
                x1, y1 = int(j), int((1 - y[i, j]) * 255)
                x2, y2 = int(j + 1), int((1 - y[i, j + 1]) * 255)

                # Draw line
                color = colors[i]
                cv2.line(vis, (x1, y1), (x2, y2), color, 2)

        return vis.astype(np.uint8)


# Loss functions for training Zero-DCE++
class SpatialConsistencyLoss(nn.Module):
    """Spatial consistency loss for smooth enhancement."""

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial consistency loss.

        Args:
            enhanced: Enhanced image (B, C, H, W)

        Returns:
            Loss value
        """
        # Compute differences with neighbors
        diff_h = enhanced[:, :, :-1, :] - enhanced[:, :, 1:, :]
        diff_w = enhanced[:, :, :, :-1] - enhanced[:, :, :, 1:]

        # Mean squared difference
        loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)

        return loss


class ExposureControlLoss(nn.Module):
    """Exposure control loss to avoid over/under exposure."""

    def __init__(self, target_level: float = 0.6, patch_size: int = 16):
        super().__init__()
        self.target = target_level
        self.patch_size = patch_size

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Compute exposure control loss.

        Args:
            enhanced: Enhanced image (B, C, H, W)

        Returns:
            Loss value
        """
        # Average pooling to get local exposures
        avg_pool = F.avg_pool2d(enhanced, self.patch_size, self.patch_size)

        # Loss is deviation from target
        loss = torch.mean((avg_pool - self.target) ** 2)

        return loss


class ColorConstancyLoss(nn.Module):
    """Color constancy loss to preserve natural colors."""

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Compute color constancy loss.

        Args:
            enhanced: Enhanced image (B, C, H, W)

        Returns:
            Loss value
        """
        # Compute pairwise channel differences
        mean_rgb = torch.mean(enhanced, dim=(2, 3), keepdim=True)

        diff_rg = (mean_rgb[:, 0] - mean_rgb[:, 1]) ** 2
        diff_rb = (mean_rgb[:, 0] - mean_rgb[:, 2]) ** 2
        diff_gb = (mean_rgb[:, 1] - mean_rgb[:, 2]) ** 2

        loss = torch.mean(diff_rg + diff_rb + diff_gb)

        return loss


class IlluminationSmoothnessLoss(nn.Module):
    """Illumination smoothness loss using total variation."""

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        """
        Compute illumination smoothness loss.

        Args:
            curves: Curve parameters (B, num_iterations, 3, H, W)

        Returns:
            Loss value
        """
        # Total variation on curve parameters
        diff_h = curves[:, :, :, :-1, :] - curves[:, :, :, 1:, :]
        diff_w = curves[:, :, :, :, :-1] - curves[:, :, :, :, 1:]

        loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

        return loss
