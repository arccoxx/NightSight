"""Temporal fusion modules for multi-frame enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from nightsight.temporal.alignment import FlowBasedAlignment, warp_frame, compute_flow


class TemporalFusion(nn.Module):
    """
    Basic temporal fusion module.

    Aligns and fuses multiple frames for enhanced quality.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        base_channels: int = 64
    ):
        super().__init__()

        self.num_frames = num_frames
        self.alignment = FlowBasedAlignment(in_channels, learned=True)

        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
        )

        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(base_channels * num_frames, base_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, num_frames, 1),
            nn.Softmax(dim=1)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
        )

    def forward(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple frames.

        Args:
            frames: List of frames, each (B, C, H, W)

        Returns:
            Fused frame (B, C, H, W)
        """
        B = frames[0].shape[0]
        ref_idx = len(frames) // 2
        ref_frame = frames[ref_idx]

        # Align all frames to reference
        aligned_frames = []
        for i, frame in enumerate(frames):
            if i == ref_idx:
                aligned_frames.append(frame)
            else:
                aligned, _ = self.alignment(ref_frame, frame)
                aligned_frames.append(aligned)

        # Extract features
        features = [self.feat_extract(f) for f in aligned_frames]
        features_cat = torch.cat(features, dim=1)

        # Compute attention weights
        attention = self.temporal_attn(features_cat)

        # Weighted fusion
        fused_feat = sum(
            features[i] * attention[:, i:i+1]
            for i in range(len(features))
        )

        # Output
        output = self.fusion(fused_feat) + ref_frame

        return torch.clamp(output, 0, 1)


class BurstFusion(nn.Module):
    """
    Burst image fusion for noise reduction.

    Implements pseudo-burst processing for low-light enhancement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 8,
        base_channels: int = 64
    ):
        super().__init__()

        self.num_frames = num_frames

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Per-frame processing
        self.frame_net = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
        )

        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Fuse burst of frames.

        Args:
            frames: (B, T, C, H, W) tensor

        Returns:
            Fused frame (B, C, H, W)
        """
        B, T, C, H, W = frames.shape

        # Process each frame
        all_features = []
        all_confidences = []

        for t in range(T):
            feat = self.encoder(frames[:, t])
            feat = self.frame_net(feat)
            conf = self.confidence(feat)

            all_features.append(feat)
            all_confidences.append(conf)

        # Stack and normalize confidences
        confidences = torch.stack(all_confidences, dim=1)  # (B, T, 1, H, W)
        confidences = confidences / (confidences.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted fusion
        features = torch.stack(all_features, dim=1)  # (B, T, C, H, W)
        fused = (features * confidences).sum(dim=1)

        # Final output
        output = self.fusion_net(fused)

        return output


class AttentionFusion(nn.Module):
    """
    Attention-based temporal fusion.

    Uses self-attention across frames for adaptive fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        embed_dim: int = 64,
        num_heads: int = 4
    ):
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Feature embedding
        self.embed = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Positional encoding for frames
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim, 1, 1))

        # Cross-frame attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse frames using attention.

        Args:
            frames: List of frames

        Returns:
            Fused frame
        """
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)

        B, T, C, H, W = frames.shape

        # Embed each frame
        embedded = []
        for t in range(T):
            emb = self.embed(frames[:, t])
            embedded.append(emb)
        embedded = torch.stack(embedded, dim=1)  # (B, T, D, H, W)

        # Add positional encoding
        pos = self.pos_embed[:, :T].expand(B, -1, -1, H, W)
        embedded = embedded + pos

        # Reshape for attention: (B*H*W, T, D)
        embedded = embedded.permute(0, 3, 4, 1, 2)  # (B, H, W, T, D)
        embedded = embedded.reshape(B * H * W, T, self.embed_dim)

        # Self-attention across frames
        attended, _ = self.cross_attn(embedded, embedded, embedded)

        # Take center frame
        center_idx = T // 2
        center_attended = attended[:, center_idx]  # (B*H*W, D)

        # Reshape back
        output = center_attended.reshape(B, H, W, self.embed_dim)
        output = output.permute(0, 3, 1, 2)  # (B, D, H, W)

        # Project to output
        output = self.output(output)

        return output


class RecurrentFusion(nn.Module):
    """
    Recurrent fusion for streaming video enhancement.

    Processes frames sequentially with hidden state.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Feature extraction
        self.feat_extract = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)

        # ConvLSTM cell
        self.conv_ih = nn.Conv2d(hidden_channels, hidden_channels * 4, 3, 1, 1)
        self.conv_hh = nn.Conv2d(hidden_channels, hidden_channels * 4, 3, 1, 1)

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        """Initialize hidden state."""
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process single frame."""
        h, c = hidden

        feat = self.feat_extract(x)

        gates_i = self.conv_ih(feat)
        gates_h = self.conv_hh(h)
        gates = gates_i + gates_h

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        output = self.output(h_new)

        return output, (h_new, c_new)

    def forward(
        self,
        frames: List[torch.Tensor],
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Process sequence of frames.

        Args:
            frames: List of frames
            return_all: Return all outputs or just last

        Returns:
            Enhanced frame(s)
        """
        B, C, H, W = frames[0].shape
        device = frames[0].device

        hidden = self.init_hidden(B, H, W, device)
        outputs = []

        for frame in frames:
            output, hidden = self.forward_step(frame, hidden)
            outputs.append(output)

        if return_all:
            return torch.stack(outputs, dim=1)
        return outputs[-1]
