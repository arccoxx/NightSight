"""
Hybrid enhancement models combining multiple techniques.

NightSightNet combines Retinex decomposition, deep learning enhancement,
and temporal fusion for state-of-the-art night vision from standard cameras.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import numpy as np
from nightsight.core.base import BaseModel, BaseEnhancer, MultiFrameEnhancer
from nightsight.core.registry import ModelRegistry
from nightsight.models.backbones.common import ResidualBlock, CBAM, SEBlock


class IlluminationBranch(nn.Module):
    """Branch for illumination estimation and enhancement."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        illum = self.decoder(feat)
        # Ensure same size as input
        if illum.shape[2:] != x.shape[2:]:
            illum = F.interpolate(illum, size=x.shape[2:], mode='bilinear', align_corners=True)
        return illum


class ReflectanceBranch(nn.Module):
    """Branch for reflectance denoising and enhancement."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_blocks: int = 4):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_blocks)
        ])

        self.attention = CBAM(base_channels)

        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)

        for block in self.blocks:
            feat = block(feat)

        feat = self.attention(feat)
        out = self.conv_out(feat)

        return torch.sigmoid(x + out)


class ColorCorrectionModule(nn.Module):
    """Module for color correction and white balance."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        super().__init__()

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Color transformation network
        self.color_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels * in_channels)  # 3x3 color matrix
        )

        # Bias
        self.bias_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Global color statistics
        global_feat = self.global_pool(x).view(B, C)

        # Predict color transformation matrix
        color_matrix = self.color_net(global_feat).view(B, C, C)

        # Predict bias
        bias = self.bias_net(global_feat).view(B, C, 1, 1)

        # Apply transformation
        x_flat = x.view(B, C, -1)
        x_transformed = torch.bmm(color_matrix, x_flat).view(B, C, H, W)
        x_corrected = x_transformed + bias

        return torch.clamp(x_corrected, 0, 1)


class DetailEnhancer(nn.Module):
    """Module for enhancing fine details and edges."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, base_channels, 5, 1, 2)
        self.conv3 = nn.Conv2d(in_channels, base_channels, 7, 1, 3)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))

        fused = torch.cat([f1, f2, f3], dim=1)
        detail = self.fusion(fused)

        return x + 0.1 * detail


@ModelRegistry.register("nightsight")
class NightSightNet(BaseModel):
    """
    NightSight: Advanced night vision enhancement network.

    Combines:
    - Retinex-based illumination/reflectance decomposition
    - Deep denoising for reflectance
    - Illumination enhancement with gamma learning
    - Color correction
    - Detail enhancement
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_res_blocks: int = 4,
        use_color_correction: bool = True,
        use_detail_enhancement: bool = True
    ):
        super().__init__()

        self.use_color_correction = use_color_correction
        self.use_detail_enhancement = use_detail_enhancement

        # Illumination estimation and enhancement
        self.illum_branch = IlluminationBranch(in_channels, base_channels)

        # Reflectance denoising
        self.ref_branch = ReflectanceBranch(in_channels, base_channels, num_res_blocks)

        # Illumination adjustment (learnable gamma)
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # Color correction
        if use_color_correction:
            self.color_correction = ColorCorrectionModule(in_channels)

        # Detail enhancement
        if use_detail_enhancement:
            self.detail_enhancer = DetailEnhancer(in_channels, base_channels)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Enhance low-light image.

        Args:
            x: Input image (B, C, H, W) in [0, 1]
            return_components: Whether to return intermediate results

        Returns:
            Enhanced image, optionally with component dictionary
        """
        # Estimate illumination
        illum = self.illum_branch(x)

        # Compute reflectance
        reflectance = x / (illum + 1e-4)
        reflectance = torch.clamp(reflectance, 0, 1)

        # Denoise reflectance
        denoised_ref = self.ref_branch(reflectance)

        # Enhance illumination with learned gamma
        gamma = torch.clamp(self.gamma, 0.1, 1.0)
        enhanced_illum = torch.pow(illum + 1e-4, gamma)

        # Reconstruct
        enhanced = denoised_ref * enhanced_illum

        # Color correction
        if self.use_color_correction:
            enhanced = self.color_correction(enhanced)

        # Detail enhancement
        if self.use_detail_enhancement:
            enhanced = self.detail_enhancer(enhanced)

        enhanced = torch.clamp(enhanced, 0, 1)

        if return_components:
            components = {
                'illumination': illum,
                'reflectance': reflectance,
                'denoised_reflectance': denoised_ref,
                'enhanced_illumination': enhanced_illum,
                'gamma': gamma.item()
            }
            return enhanced, components

        return enhanced


@ModelRegistry.register("hybrid_enhancer")
class HybridEnhancer(BaseModel):
    """
    Hybrid enhancer combining traditional and deep learning methods.

    Uses traditional Retinex for initialization and deep learning for refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4
    ):
        super().__init__()

        # Deep refinement network
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, 1, 1),  # Input + Retinex init
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_channels) for _ in range(num_blocks)],
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def retinex_init(self, x: torch.Tensor) -> torch.Tensor:
        """Simple Retinex initialization."""
        # Estimate illumination as max channel
        illum = torch.max(x, dim=1, keepdim=True)[0]
        illum = F.avg_pool2d(illum, 31, 1, 15)  # Smooth

        # Compute reflectance
        reflectance = x / (illum + 1e-4)

        # Adjust illumination
        enhanced_illum = torch.pow(illum + 1e-4, 0.5)

        # Initial enhancement
        init_enhanced = reflectance * enhanced_illum
        return torch.clamp(init_enhanced, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get Retinex initialization
        init_enhanced = self.retinex_init(x)

        # Concatenate input and initialization
        concat = torch.cat([x, init_enhanced], dim=1)

        # Refine with deep network
        refined = self.refine(concat)

        return refined


class TemporalFusionModule(nn.Module):
    """Module for fusing multiple frames with attention."""

    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()

        self.num_frames = num_frames

        # Frame attention
        self.frame_attention = nn.Sequential(
            nn.Conv3d(channels, channels // 2, (3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, 1, (3, 1, 1), padding=(1, 0, 0)),
            nn.Softmax(dim=2)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple frames.

        Args:
            frames: (B, T, C, H, W) tensor of frames

        Returns:
            Fused frame (B, C, H, W)
        """
        B, T, C, H, W = frames.shape

        # Compute frame attention weights
        frames_3d = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        attention = self.frame_attention(frames_3d)  # (B, 1, T, H, W)

        # Weighted sum
        weighted = frames_3d * attention
        fused = weighted.sum(dim=2)  # (B, C, H, W)

        # Refine
        fused = self.fusion(fused) + fused

        return fused


@ModelRegistry.register("temporal_nightsight")
class TemporalNightSightNet(BaseModel):
    """
    NightSight with temporal fusion for video enhancement.

    Processes multiple frames and fuses them for improved quality
    and temporal consistency.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_frames: int = 5,
        num_res_blocks: int = 4
    ):
        super().__init__()

        self.num_frames = num_frames

        # Per-frame feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Temporal fusion
        self.temporal_fusion = TemporalFusionModule(base_channels, num_frames)

        # Enhancement network
        self.enhance = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)],
            CBAM(base_channels),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Enhance multiple frames.

        Args:
            frames: (B, T, C, H, W) tensor or list of (B, C, H, W) tensors

        Returns:
            Enhanced center frame (B, C, H, W)
        """
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)

        B, T, C, H, W = frames.shape

        # Extract features for each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(frames[:, t])
            features.append(feat)
        features = torch.stack(features, dim=1)  # (B, T, C', H, W)

        # Temporal fusion
        fused = self.temporal_fusion(features)

        # Enhancement
        enhanced = self.enhance(fused)

        return enhanced


class NightSightEnhancer(MultiFrameEnhancer):
    """Complete NightSight enhancer with multi-frame support."""

    def __init__(
        self,
        model_type: str = "nightsight",
        checkpoint: Optional[str] = None,
        num_frames: int = 5,
        device: str = "auto",
        **model_kwargs
    ):
        super().__init__(num_frames, device)

        if model_type == "nightsight":
            self.model = NightSightNet(**model_kwargs)
        elif model_type == "temporal":
            self.model = TemporalNightSightNet(num_frames=num_frames, **model_kwargs)
        elif model_type == "hybrid":
            self.model = HybridEnhancer(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if checkpoint:
            self.model.load_pretrained(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.model_type = model_type

    def enhance_temporal(
        self,
        frames: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Enhance using multiple frames."""
        if self.model_type == "temporal":
            stacked = torch.stack(frames, dim=1).to(self.device)
            with torch.no_grad():
                return self.model(stacked)
        else:
            # For non-temporal models, enhance center frame
            center_idx = len(frames) // 2
            with torch.no_grad():
                return self.model(frames[center_idx].to(self.device))

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """Enhance a single image."""
        is_numpy = isinstance(image, np.ndarray)

        if is_numpy:
            tensor = self.numpy_to_tensor(image).to(self.device)
        else:
            tensor = image.to(self.device)

        with torch.no_grad():
            enhanced = self.model(tensor)

        if is_numpy:
            return self.tensor_to_numpy(enhanced)
        return enhanced
