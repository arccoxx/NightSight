"""
Edge detection and glowing outline overlay system.

Creates military night vision-style bright outlines around objects
and edges to improve visibility and object differentiation in low-light.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Union


class EdgeDetectionNet(nn.Module):
    """
    Deep edge detection network.

    Based on HED (Holistically-Nested Edge Detection) architecture.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Encoder blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Edge prediction heads
        self.edge1 = nn.Conv2d(64, 1, 1)
        self.edge2 = nn.Conv2d(128, 1, 1)
        self.edge3 = nn.Conv2d(256, 1, 1)
        self.edge4 = nn.Conv2d(512, 1, 1)

        # Fusion
        self.fusion = nn.Conv2d(4, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Detect edges at multiple scales.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Fused edge map and list of multi-scale edge maps
        """
        H, W = x.shape[2:]

        # Multi-scale features
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)

        # Edge predictions at each scale
        edge1 = self.edge1(feat1)
        edge2 = F.interpolate(self.edge2(feat2), size=(H, W), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(self.edge3(feat3), size=(H, W), mode='bilinear', align_corners=True)
        edge4 = F.interpolate(self.edge4(feat4), size=(H, W), mode='bilinear', align_corners=True)

        # Fuse edges
        edges_concat = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        fused = self.fusion(edges_concat)
        fused = torch.sigmoid(fused)

        return fused, [torch.sigmoid(e) for e in [edge1, edge2, edge3, edge4]]


class EdgeOutliner:
    """
    Edge detection and glowing outline overlay system.

    Creates military night vision-style bright outlines for enhanced
    object visibility in low-light conditions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_deep_edges: bool = True,
        use_traditional: bool = True
    ):
        """
        Initialize edge outliner.

        Args:
            model_path: Path to pretrained edge detection model
            device: Device to run on
            use_deep_edges: Use deep learning edge detection
            use_traditional: Use traditional edge detection (Canny)
        """
        self.device = self._get_device(device)
        self.use_deep_edges = use_deep_edges
        self.use_traditional = use_traditional

        if use_deep_edges:
            self.model = EdgeDetectionNet()

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

    def detect_edges_deep(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Detect edges using deep learning.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            return_numpy: Whether to return numpy array

        Returns:
            Edge map (H, W) or (B, 1, H, W)
        """
        if not self.use_deep_edges:
            raise ValueError("Deep edge detection not enabled")

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

        # Detect edges
        with torch.no_grad():
            edges, _ = self.model(tensor)

        # Convert back to numpy if needed
        if return_numpy:
            edges = edges.squeeze().cpu().numpy()
            return edges

        return edges

    def detect_edges_canny(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        aperture_size: int = 3
    ) -> np.ndarray:
        """
        Detect edges using Canny edge detector.

        Args:
            image: Input image (H, W, 3) or (H, W)
            low_threshold: Low threshold for Canny
            high_threshold: High threshold for Canny
            aperture_size: Aperture size for Sobel

        Returns:
            Binary edge map (H, W)
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # Apply Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)

        return edges.astype(np.float32) / 255.0

    def detect_edges_combined(
        self,
        image: np.ndarray,
        deep_weight: float = 0.6,
        canny_weight: float = 0.4
    ) -> np.ndarray:
        """
        Combine deep and traditional edge detection.

        Args:
            image: Input image (H, W, 3)
            deep_weight: Weight for deep edges
            canny_weight: Weight for Canny edges

        Returns:
            Combined edge map (H, W)
        """
        edges_combined = np.zeros(image.shape[:2], dtype=np.float32)

        if self.use_deep_edges:
            edges_deep = self.detect_edges_deep(image, return_numpy=True)
            edges_combined += deep_weight * edges_deep

        if self.use_traditional:
            edges_canny = self.detect_edges_canny(image)
            edges_combined += canny_weight * edges_canny

        # Normalize
        edges_combined = np.clip(edges_combined, 0, 1)

        return edges_combined

    def create_glowing_outline(
        self,
        edges: np.ndarray,
        glow_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        blur_radius: int = 5,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Create glowing outline effect from edge map.

        Args:
            edges: Edge map (H, W) in [0, 1]
            glow_color: RGB color for glow
            thickness: Edge thickness
            blur_radius: Glow blur radius
            intensity: Glow intensity

        Returns:
            Glowing outline (H, W, 3)
        """
        H, W = edges.shape

        # Convert edges to binary and thicken
        edges_binary = (edges > 0.3).astype(np.uint8) * 255

        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            edges_binary = cv2.dilate(edges_binary, kernel, iterations=1)

        # Create colored edge map
        outline = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(3):
            outline[:, :, i] = (edges_binary * glow_color[i] / 255.0).astype(np.uint8)

        # Add glow effect with Gaussian blur
        if blur_radius > 0:
            glow = cv2.GaussianBlur(outline, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
            outline = cv2.addWeighted(outline, 0.7, glow, 0.3 * intensity, 0)

        return outline

    def apply_outline_to_image(
        self,
        image: np.ndarray,
        edges: Optional[np.ndarray] = None,
        glow_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        blur_radius: int = 5,
        intensity: float = 0.8,
        blend_mode: str = "add"
    ) -> np.ndarray:
        """
        Apply glowing outlines to image (military night vision style).

        Args:
            image: Input image (H, W, 3)
            edges: Edge map or None to detect automatically
            glow_color: RGB color for glow (default green like night vision)
            thickness: Edge thickness
            blur_radius: Glow blur radius
            intensity: Glow intensity
            blend_mode: How to blend ('add', 'screen', 'overlay')

        Returns:
            Image with glowing outlines (H, W, 3)
        """
        # Detect edges if not provided
        if edges is None:
            if self.use_deep_edges and self.use_traditional:
                edges = self.detect_edges_combined(image)
            elif self.use_deep_edges:
                edges = self.detect_edges_deep(image, return_numpy=True)
            else:
                edges = self.detect_edges_canny(image)

        # Create glowing outline
        outline = self.create_glowing_outline(
            edges,
            glow_color=glow_color,
            thickness=thickness,
            blur_radius=blur_radius,
            intensity=intensity
        )

        # Ensure image is uint8
        input_is_float = image.dtype != np.uint8
        if input_is_float:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        # Blend outline with image
        if blend_mode == "add":
            result = cv2.add(img_vis, outline)
        elif blend_mode == "screen":
            # Screen blending: 1 - (1 - A) * (1 - B)
            img_norm = img_vis.astype(np.float32) / 255.0
            outline_norm = outline.astype(np.float32) / 255.0
            result = 1 - (1 - img_norm) * (1 - outline_norm)
            result = (result * 255).astype(np.uint8)
        elif blend_mode == "overlay":
            result = cv2.addWeighted(img_vis, 1.0, outline, intensity, 0)
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

        # Convert back to float32 [0, 1] if input was float
        if input_is_float:
            result = result.astype(np.float32) / 255.0

        return result

    def create_depth_aware_outlines(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        edges: Optional[np.ndarray] = None,
        near_color: Tuple[int, int, int] = (255, 0, 0),  # Red for near
        far_color: Tuple[int, int, int] = (0, 0, 255),   # Blue for far
        thickness: int = 2,
        blur_radius: int = 5
    ) -> np.ndarray:
        """
        Create depth-aware colored outlines (near objects different color than far).

        Args:
            image: Input image (H, W, 3)
            depth: Depth map (H, W)
            edges: Edge map or None to detect
            near_color: RGB color for near objects
            far_color: RGB color for far objects
            thickness: Edge thickness
            blur_radius: Glow blur radius

        Returns:
            Image with depth-aware outlines (H, W, 3)
        """
        # Detect edges if not provided
        if edges is None:
            edges = self.detect_edges_combined(image) if (self.use_deep_edges and self.use_traditional) else \
                    self.detect_edges_deep(image, return_numpy=True) if self.use_deep_edges else \
                    self.detect_edges_canny(image)

        # Ensure depth is 2D
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # Take first channel if 3D

        # Ensure depth matches edges dimensions
        H, W = edges.shape
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Create colored outlines based on depth
        edges_binary = (edges > 0.3).astype(np.uint8) * 255

        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            edges_binary = cv2.dilate(edges_binary, kernel, iterations=1)

        # Interpolate color based on depth
        outline = np.zeros((H, W, 3), dtype=np.float32)

        mask = edges_binary > 0
        for i in range(3):
            # Interpolate between far and near colors based on depth
            color_map = far_color[i] * (1 - depth_norm) + near_color[i] * depth_norm
            outline[:, :, i][mask] = color_map[mask]

        outline = outline.astype(np.uint8)

        # Add glow
        if blur_radius > 0:
            glow = cv2.GaussianBlur(outline, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
            outline = cv2.addWeighted(outline, 0.7, glow, 0.3, 0)

        # Blend with image
        input_is_float = image.dtype != np.uint8
        if input_is_float:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        result = cv2.add(img_vis, outline)

        # Convert back to float32 [0, 1] if input was float
        if input_is_float:
            result = result.astype(np.float32) / 255.0

        return result
