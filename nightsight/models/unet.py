"""
U-Net and variants for low-light image enhancement.

U-Net provides a strong baseline architecture with encoder-decoder
structure and skip connections for preserving spatial information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from nightsight.core.base import BaseModel
from nightsight.core.registry import ModelRegistry


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm: str = "batch"
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels) if norm != "none" else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels) if norm != "none" else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with bilinear/transposed conv and double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        norm: str = "batch"
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@ModelRegistry.register("unet")
class UNet(BaseModel):
    """
    U-Net for low-light image enhancement.

    Standard encoder-decoder architecture with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        norm: str = "batch"
    ):
        """
        Initialize U-Net.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_channels: Base number of feature channels
            depth: Depth of the U-Net
            bilinear: Use bilinear upsampling vs transposed convolution
            norm: Normalization type ('batch', 'instance', 'none')
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Input layer
        self.inc = DoubleConv(in_channels, base_channels, norm=norm)

        # Encoder
        self.downs = nn.ModuleList()
        factor = 2 if bilinear else 1

        for i in range(depth):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            if i == depth - 1:
                out_ch = out_ch // factor
            self.downs.append(Down(in_ch, out_ch, norm=norm))

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            if i == depth - 1:
                in_ch = in_ch // factor
            out_ch = out_ch // factor if i > 0 else out_ch
            self.ups.append(Up(in_ch, out_ch, bilinear, norm=norm))

        # Output layer
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))

        # Decoder
        x = features[-1]
        for i, up in enumerate(self.ups):
            x = up(x, features[-(i + 2)])

        return torch.sigmoid(self.outc(x))


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Handle size mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpAttention(nn.Module):
    """Upsampling with attention gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2,
                kernel_size=2, stride=2
            )

        self.attention = AttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=in_channels // 4
        )
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Apply attention
        x2 = self.attention(x1, x2)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@ModelRegistry.register("attention_unet")
class AttentionUNet(BaseModel):
    """
    Attention U-Net for low-light image enhancement.

    U-Net with attention gates for better feature selection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Input
        self.inc = DoubleConv(in_channels, base_channels)

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(depth):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.downs.append(Down(in_ch, out_ch))

        # Decoder with attention
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.ups.append(UpAttention(in_ch, out_ch, bilinear))

        # Output
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))

        # Decoder with attention
        x = features[-1]
        for i, up in enumerate(self.ups):
            x = up(x, features[-(i + 2)])

        return torch.sigmoid(self.outc(x))


@ModelRegistry.register("residual_unet")
class ResidualUNet(BaseModel):
    """U-Net with global residual learning for image enhancement."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True
    ):
        super().__init__()

        self.unet = UNet(
            in_channels, out_channels, base_channels, depth, bilinear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Learn residual
        residual = self.unet(x)
        # Add to input
        return torch.clamp(x + residual, 0, 1)
