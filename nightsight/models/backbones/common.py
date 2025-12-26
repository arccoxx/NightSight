"""Common building blocks for neural network architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Standard convolution block with optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        norm: Optional[str] = None,
        activation: str = "relu"
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias
            )
        ]

        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm == "layer":
            layers.append(nn.GroupNorm(1, out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif activation == "prelu":
            layers.append(nn.PReLU(out_channels))
        elif activation is not None and activation != "none":
            raise ValueError(f"Unknown activation: {activation}")

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: Optional[str] = "batch",
        activation: str = "relu"
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            channels, channels, kernel_size,
            padding=kernel_size // 2, norm=norm, activation=activation
        )
        self.conv2 = ConvBlock(
            channels, channels, kernel_size,
            padding=kernel_size // 2, norm=norm, activation=None
        )

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class ChannelAttention(nn.Module):
    """Channel attention module (squeeze and excitation style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DenseBlock(nn.Module):
    """Dense block with skip connections to all previous layers."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        num_layers: int = 4
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ConvBlock(
                in_channels + i * growth_rate,
                growth_rate,
                kernel_size=3, padding=1,
                norm="batch", activation="relu"
            )
            self.layers.append(layer)

        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class PixelShuffle(nn.Module):
    """Pixel shuffle upsampling."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels * (scale_factor ** 2),
            kernel_size=3, padding=1
        )
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.shuffle(x)


class CoordConv(nn.Module):
    """Convolution with coordinate channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        with_r: bool = False
    ):
        super().__init__()

        extra_channels = 3 if with_r else 2
        self.with_r = with_r

        self.conv = nn.Conv2d(
            in_channels + extra_channels, out_channels,
            kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Create coordinate channels
        xx = torch.linspace(-1, 1, w, device=x.device)
        yy = torch.linspace(-1, 1, h, device=x.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')

        xx = xx.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)

        if self.with_r:
            rr = torch.sqrt(xx ** 2 + yy ** 2)
            coord = torch.cat([xx, yy, rr], dim=1)
        else:
            coord = torch.cat([xx, yy], dim=1)

        x = torch.cat([x, coord], dim=1)
        return self.conv(x)
