"""
Depth estimation module for object differentiation.

Uses lightweight depth estimation models to create depth maps
that help differentiate objects in low-light conditions, similar
to military night vision systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Union


class DepthEstimationNet(nn.Module):
    """
    Lightweight depth estimation network.

    Based on MobileNet-style architecture for fast inference.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            # Stage 1: 1/2
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            # Stage 2: 1/4
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            ),
            # Stage 3: 1/8
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            ),
            # Stage 4: 1/16
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            ),
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            # Upsample 1: 1/8
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            ),
            # Upsample 2: 1/4
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            ),
            # Upsample 3: 1/2
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            # Upsample 4: 1/1
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
        ])

        # Final depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth map from input image.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Depth map (B, 1, H, W) in [0, 1], where 0 is far, 1 is near
        """
        # Encoder with skip connections
        skip_connections = []
        feat = x
        for encoder_block in self.encoder:
            feat = encoder_block(feat)
            skip_connections.append(feat)

        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder):
            feat = decoder_block(feat)
            # Add skip connection (except for last layer)
            if i < len(self.decoder) - 1:
                skip = skip_connections[-(i + 2)]
                # Resize if needed
                if feat.shape[2:] != skip.shape[2:]:
                    feat = F.interpolate(feat, size=skip.shape[2:], mode='bilinear', align_corners=True)
                feat = torch.cat([feat, skip], dim=1)

        # Final depth prediction
        depth = self.depth_head(feat)

        # Resize to input size if needed
        if depth.shape[2:] != x.shape[2:]:
            depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear', align_corners=True)

        return depth


class DepthEstimator:
    """
    Depth estimator for object differentiation in night vision.

    Estimates depth maps from low-light images and provides utilities
    for depth-based object highlighting and differentiation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_pretrained: bool = True
    ):
        """
        Initialize depth estimator.

        Args:
            model_path: Path to pretrained model weights
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_pretrained: Whether to use pretrained weights
        """
        self.device = self._get_device(device)

        # Initialize model
        self.model = DepthEstimationNet()

        # Load pretrained weights if available
        if model_path and use_pretrained:
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

    def estimate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Estimate depth map from image.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            return_numpy: Whether to return numpy array

        Returns:
            Depth map (H, W) or (B, 1, H, W)
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

        # Estimate depth
        with torch.no_grad():
            depth = self.model(tensor)

        # Convert back to numpy if needed
        if return_numpy:
            depth = depth.squeeze(1).cpu().numpy()
            if depth.ndim == 2:
                return depth
            return depth

        return depth

    def create_depth_overlay(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Create colored depth overlay on image.

        Args:
            image: Original image (H, W, 3)
            depth: Depth map (H, W) or None to estimate
            colormap: OpenCV colormap to use
            alpha: Overlay transparency

        Returns:
            Image with depth overlay (H, W, 3)
        """
        # Estimate depth if not provided
        if depth is None:
            depth = self.estimate(image)

        # Normalize depth to [0, 255]
        depth_vis = (depth * 255).astype(np.uint8)

        # Apply colormap
        depth_color = cv2.applyColorMap(depth_vis, colormap)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image

        # Blend
        overlay = cv2.addWeighted(img_vis, 1 - alpha, depth_color, alpha, 0)

        return overlay

    def segment_by_depth(
        self,
        depth: np.ndarray,
        num_layers: int = 3
    ) -> Tuple[np.ndarray, list]:
        """
        Segment image into depth layers.

        Args:
            depth: Depth map (H, W)
            num_layers: Number of depth layers

        Returns:
            Layer mask (H, W) and list of layer depth ranges
        """
        # Quantize depth into layers
        depth_min, depth_max = depth.min(), depth.max()
        layer_width = (depth_max - depth_min) / num_layers

        layers = np.zeros_like(depth, dtype=np.uint8)
        layer_ranges = []

        for i in range(num_layers):
            min_d = depth_min + i * layer_width
            max_d = depth_min + (i + 1) * layer_width
            mask = (depth >= min_d) & (depth < max_d)
            layers[mask] = i
            layer_ranges.append((min_d, max_d))

        return layers, layer_ranges

    def highlight_depth_edges(
        self,
        depth: np.ndarray,
        threshold: float = 0.1,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Detect edges in depth map (depth discontinuities).

        Args:
            depth: Depth map (H, W)
            threshold: Edge detection threshold
            thickness: Edge thickness

        Returns:
            Binary edge map (H, W)
        """
        # Normalize depth
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)

        # Compute gradients
        grad_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag / grad_mag.max()

        # Threshold
        edges = (grad_mag > threshold).astype(np.uint8) * 255

        # Dilate to make edges thicker
        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            edges = cv2.dilate(edges, kernel, iterations=1)

        return edges
