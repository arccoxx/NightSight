"""
Retinexformer: Retinex-based Transformer for Low-Light Image Enhancement.

Combines Retinex theory with Transformer architecture for effective
illumination estimation and reflectance restoration.

Reference: "Retinexformer: One-stage Retinex-based Transformer for Low-light
Image Enhancement", Cai et al., ICCV 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from nightsight.core.base import BaseModel, BaseEnhancer
from nightsight.core.registry import ModelRegistry


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x + \
               self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)


class IlluminationEstimator(nn.Module):
    """
    Illumination Estimator for Retinexformer.

    Estimates the illumination map from the input image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels, 1, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate illumination map."""
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        feat = self.relu(self.conv3(feat))
        illum = self.sigmoid(self.conv4(feat))
        return illum


class IlluminationGuidedAttention(nn.Module):
    """
    Illumination-Guided Multi-head Self-Attention (IG-MSA).

    Uses illumination map to guide attention computation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Illumination embedding
        self.illum_embed = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Apply illumination-guided attention.

        Args:
            x: Input features (B, N, C)
            illum: Illumination map (B, 1, H, W)
            H, W: Spatial dimensions
        """
        B, N, C = x.shape

        # Get illumination embedding
        illum_feat = self.illum_embed(illum)  # (B, C, H, W)
        illum_feat = illum_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        # Add illumination to features
        x = x + illum_feat

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class IGTransformerBlock(nn.Module):
    """Illumination-Guided Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = IlluminationGuidedAttention(
            dim, num_heads, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), illum, H, W)
        x = x + self.ffn(self.norm2(x))
        return x


class RetinexTransformer(nn.Module):
    """Transformer for Retinex-based enhancement."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            IGTransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                drop=drop, attn_drop=attn_drop
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Embed patches
        x = self.patch_embed(x)  # (B, embed_dim, H, W)

        # Flatten for transformer
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, illum, H, W)

        x = self.norm(x)

        # Reshape back
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Output projection
        x = self.output(x)

        return x


@ModelRegistry.register("retinexformer")
class Retinexformer(BaseModel):
    """
    Retinexformer model for low-light image enhancement.

    Combines illumination estimation with illumination-guided
    transformer for reflectance enhancement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        illum_channels: int = 32
    ):
        super().__init__()

        # Illumination estimator
        self.illum_estimator = IlluminationEstimator(in_channels, illum_channels)

        # Retinex transformer for reflectance enhancement
        self.transformer = RetinexTransformer(
            in_channels, embed_dim, depth, num_heads, mlp_ratio
        )

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Enhance low-light image.

        Args:
            x: Input image (B, C, H, W) in [0, 1]
            return_components: Return illumination and reflectance

        Returns:
            Enhanced image and optionally (illumination, reflectance)
        """
        # Estimate illumination
        illum = self.illum_estimator(x)

        # Compute initial reflectance: R = I / L
        reflectance = x / (illum + 1e-4)

        # Enhance reflectance with transformer
        enhanced_ref = self.transformer(reflectance, illum)
        enhanced_ref = enhanced_ref + reflectance  # Residual

        # Adjust illumination (gamma correction)
        enhanced_illum = torch.pow(illum + 1e-4, 0.5)

        # Final output: enhanced = R * L
        enhanced = enhanced_ref * enhanced_illum
        enhanced = torch.clamp(enhanced, 0, 1)

        if return_components:
            return enhanced, (illum, enhanced_ref)
        return enhanced


@ModelRegistry.register("retinexnet")
class RetinexNet(BaseModel):
    """
    Simplified RetinexNet for decomposition and enhancement.

    Decomposes image into reflectance and illumination, then
    enhances each component separately.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32
    ):
        super().__init__()

        # Decomposition network
        self.decomp = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels + 1, 3, 1, 1)  # R + L
        )

        # Illumination adjustment network
        self.illum_adjust = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Reflectance denoising network
        self.denoise = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decompose into R and L
        decomp_out = self.decomp(x)
        reflectance = torch.sigmoid(decomp_out[:, :3])
        illumination = torch.sigmoid(decomp_out[:, 3:4])

        # Denoise reflectance
        denoised_ref = self.denoise(reflectance) + reflectance
        denoised_ref = torch.clamp(denoised_ref, 0, 1)

        # Adjust illumination
        adjusted_illum = self.illum_adjust(illumination)

        # Reconstruct
        enhanced = denoised_ref * adjusted_illum
        return torch.clamp(enhanced, 0, 1)


class RetinexformerEnhancer(BaseEnhancer):
    """Wrapper for using Retinexformer as an enhancer."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "auto",
        **model_kwargs
    ):
        super().__init__(device)

        self.model = Retinexformer(**model_kwargs)

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
