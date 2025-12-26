"""Frame alignment modules for temporal processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


def compute_flow(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    method: str = "learned"
) -> torch.Tensor:
    """
    Compute optical flow between two frames.

    Args:
        frame1: First frame (B, C, H, W)
        frame2: Second frame (B, C, H, W)
        method: Flow estimation method

    Returns:
        Flow field (B, 2, H, W)
    """
    if method == "correlation":
        return _correlation_flow(frame1, frame2)
    else:
        # Simple learned flow
        return _simple_flow(frame1, frame2)


def _correlation_flow(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    max_displacement: int = 4
) -> torch.Tensor:
    """Compute flow using correlation."""
    B, C, H, W = frame1.shape

    # Build correlation volume
    pad = max_displacement
    frame2_padded = F.pad(frame2, [pad] * 4, mode='replicate')

    correlations = []
    for dy in range(-max_displacement, max_displacement + 1):
        for dx in range(-max_displacement, max_displacement + 1):
            shifted = frame2_padded[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            corr = (frame1 * shifted).sum(dim=1, keepdim=True)
            correlations.append(corr)

    # Stack and find best match
    corr_volume = torch.cat(correlations, dim=1)  # (B, (2*d+1)^2, H, W)

    # Argmax to get flow
    best_idx = corr_volume.argmax(dim=1)
    grid_size = 2 * max_displacement + 1
    flow_y = (best_idx // grid_size - max_displacement).float()
    flow_x = (best_idx % grid_size - max_displacement).float()

    flow = torch.stack([flow_x, flow_y], dim=1)
    return flow


def _simple_flow(
    frame1: torch.Tensor,
    frame2: torch.Tensor
) -> torch.Tensor:
    """Simple CNN-based flow estimation."""
    B, C, H, W = frame1.shape

    # Very simple: difference-based flow approximation
    diff = frame2 - frame1
    grad_x = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    grad_y = diff[:, :, 1:, :] - diff[:, :, :-1, :]

    # Pad to original size
    grad_x = F.pad(grad_x, [0, 1, 0, 0])
    grad_y = F.pad(grad_y, [0, 0, 0, 1])

    flow_x = grad_x.mean(dim=1, keepdim=True)
    flow_y = grad_y.mean(dim=1, keepdim=True)

    flow = torch.cat([flow_x, flow_y], dim=1)
    return flow * 10  # Scale factor


def warp_frame(
    frame: torch.Tensor,
    flow: torch.Tensor
) -> torch.Tensor:
    """
    Warp frame using optical flow.

    Args:
        frame: Frame to warp (B, C, H, W)
        flow: Flow field (B, 2, H, W)

    Returns:
        Warped frame (B, C, H, W)
    """
    B, C, H, W = frame.shape

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=frame.device),
        torch.arange(W, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).float()
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Add flow
    new_grid = grid + flow

    # Normalize to [-1, 1]
    new_grid[:, 0] = 2 * new_grid[:, 0] / (W - 1) - 1
    new_grid[:, 1] = 2 * new_grid[:, 1] / (H - 1) - 1

    # Rearrange for grid_sample
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Warp
    warped = F.grid_sample(
        frame, new_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return warped


class FlowNet(nn.Module):
    """Simple flow estimation network."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 5, 2, 2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 2, 4, 2, 1),
        )

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frame1, frame2], dim=1)
        feat = self.encoder(x)
        flow = self.decoder(feat)

        # Ensure same size as input
        if flow.shape[2:] != frame1.shape[2:]:
            flow = F.interpolate(flow, size=frame1.shape[2:], mode='bilinear', align_corners=True)

        return flow


class FlowBasedAlignment(nn.Module):
    """Alignment module using optical flow."""

    def __init__(self, in_channels: int = 3, learned: bool = True):
        super().__init__()

        self.learned = learned
        if learned:
            self.flow_net = FlowNet(in_channels * 2)

    def forward(
        self,
        ref_frame: torch.Tensor,
        other_frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align other_frame to ref_frame.

        Returns:
            Tuple of (aligned_frame, flow)
        """
        if self.learned:
            flow = self.flow_net(ref_frame, other_frame)
        else:
            flow = compute_flow(ref_frame, other_frame, method="correlation")

        aligned = warp_frame(other_frame, flow)

        return aligned, flow


class DeformableConv2d(nn.Module):
    """Deformable convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size, stride, padding
        )

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Initialize
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict offsets
        offset = self.offset_conv(x)

        # Use regular conv as fallback (actual deformable conv needs torchvision)
        # This is a simplified version
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class DeformableAlignment(nn.Module):
    """Alignment using deformable convolutions."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_groups: int = 8
    ):
        super().__init__()

        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Offset prediction from concatenated features
        self.offset_conv1 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(base_channels, 18, 3, 1, 1)  # 2 * 3 * 3

        # Deformable conv
        self.deform_conv = DeformableConv2d(base_channels, base_channels)

        # Output
        self.output = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(
        self,
        ref_frame: torch.Tensor,
        other_frame: torch.Tensor
    ) -> torch.Tensor:
        """Align other_frame to ref_frame using deformable convolutions."""
        # Extract features
        ref_feat = self.feat_extract(ref_frame)
        other_feat = self.feat_extract(other_frame)

        # Concatenate and predict offsets
        concat_feat = torch.cat([ref_feat, other_feat], dim=1)
        offset = self.lrelu(self.offset_conv1(concat_feat))
        offset = self.lrelu(self.offset_conv2(offset))
        offset = self.offset_conv3(offset)

        # Apply deformable conv
        aligned_feat = self.deform_conv(other_feat)

        # Output
        aligned = self.output(aligned_feat)

        return aligned
