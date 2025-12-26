"""
SwinIR-based architecture for low-light image enhancement.

Uses Swin Transformer blocks for capturing long-range dependencies
while maintaining computational efficiency through window-based attention.

Reference: "SwinIR: Image Restoration Using Swin Transformer"
Liang et al., ICCV 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from nightsight.core.base import BaseModel
from nightsight.core.registry import ModelRegistry


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Create position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, (window_size, window_size), num_heads,
            qkv_bias, attn_drop, drop
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad for window partition
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._create_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x

    def _create_mask(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        """Create attention mask for shifted window."""
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop
            )
            for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        shortcut = x
        for block in self.blocks:
            x = block(x, x_size)

        # Reshape for conv
        B, L, C = x.shape
        H, W = x_size
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)

        return x + shortcut


@ModelRegistry.register("swinir")
class SwinIR(BaseModel):
    """
    SwinIR for low-light image enhancement.

    Uses Swin Transformer blocks for capturing global context
    while being efficient through window-based attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.layers = nn.ModuleList([
            RSTB(
                embed_dim, depths[i], num_heads[i], window_size,
                mlp_ratio, qkv_bias, drop, attn_drop
            )
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_size = (H, W)

        # Shallow features
        x_first = self.conv_first(x)

        # Deep features
        x_deep = x_first.permute(0, 2, 3, 1).view(B, H * W, self.embed_dim)

        for layer in self.layers:
            x_deep = layer(x_deep, x_size)

        x_deep = self.norm(x_deep)
        x_deep = x_deep.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        x_deep = self.conv_after(x_deep)

        # Residual + reconstruction
        out = self.conv_last(x_first + x_deep)

        return torch.sigmoid(x + out)  # Residual learning


@ModelRegistry.register("swinir_light")
class SwinIRLight(BaseModel):
    """Lightweight SwinIR for faster inference."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depths: Tuple[int, ...] = (4, 4),
        num_heads: Tuple[int, ...] = (4, 4),
        window_size: int = 8,
        mlp_ratio: float = 2.0
    ):
        super().__init__()

        self.model = SwinIR(
            in_channels, embed_dim, depths, num_heads,
            window_size, mlp_ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
