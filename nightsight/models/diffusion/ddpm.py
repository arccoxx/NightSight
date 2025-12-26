"""
Denoising Diffusion Probabilistic Models (DDPM) for low-light enhancement.

Diffusion models learn to denoise images through a gradual denoising process,
which can be adapted for image enhancement by treating low-light images as
noisy versions of well-lit images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from nightsight.core.base import BaseModel, BaseEnhancer
from nightsight.core.registry import ModelRegistry


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()

        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * scale
        attn = attn.softmax(dim=-1)

        h = torch.einsum('bhij,bhcj->bhci', attn, v)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch_in, out_channels, time_emb_dim, dropout))
            if use_attention:
                self.attn_blocks.append(AttentionBlock(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, t_emb)
            x = attn(x)
            skips.append(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    """Upsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = in_channels + out_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch_in, out_channels, time_emb_dim, dropout))
            if use_attention:
                self.attn_blocks.append(AttentionBlock(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        x = self.upsample(x)

        for i, (res, attn) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            if i < len(skips):
                skip = skips[-(i + 1)]
                # Handle size mismatch
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb)
            x = attn(x)
        return x


class UNetDiffusion(nn.Module):
    """U-Net architecture for diffusion model."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 4),
        time_emb_dim: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            use_attn = (i + 1) in attention_resolutions
            self.down_blocks.append(
                DownBlock(ch, ch_out, time_emb_dim, num_res_blocks, use_attn, dropout)
            )
            ch = ch_out

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            ch_out = base_channels * mult
            use_attn = (i + 1) in attention_resolutions
            self.up_blocks.append(
                UpBlock(ch, ch_out, time_emb_dim, num_res_blocks, use_attn, dropout)
            )
            ch = ch_out

        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of diffusion U-Net.

        Args:
            x: Noisy image (B, C, H, W)
            t: Timesteps (B,)
            condition: Optional conditioning image

        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Concatenate condition if provided
        if condition is not None:
            x = torch.cat([x, condition], dim=1)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        all_skips = []
        for block in self.down_blocks:
            h, skips = block(h, t_emb)
            all_skips.extend(skips)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        skip_idx = len(all_skips)
        for block in self.up_blocks:
            n_skips = len(block.res_blocks)
            skips = all_skips[skip_idx - n_skips:skip_idx]
            skip_idx -= n_skips
            h = block(h, t_emb, skips)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


@ModelRegistry.register("ddpm")
class DDPM(BaseModel):
    """
    DDPM for low-light image enhancement.

    Learns to denoise images conditioned on low-light inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        conditional: bool = True
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.conditional = conditional

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # Model
        model_in_channels = in_channels * 2 if conditional else in_channels
        self.model = UNetDiffusion(
            model_in_channels, base_channels, channel_mults
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict noise."""
        return self.model(x, t, condition)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add noise to images (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single denoising step."""
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

        # Predict noise
        pred_noise = self.forward(x, t_tensor, condition)

        # Get schedule parameters
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Compute x_{t-1}
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / self.sqrt_one_minus_alphas_cumprod[t]

        mean = coef1 * (x - coef2 * pred_noise)

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean

        return x

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Generate enhanced image from low-light input."""
        if num_steps is None:
            num_steps = self.num_timesteps

        # Start from noise
        x = torch.randn_like(condition)

        # Use subset of timesteps for faster sampling
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for t in timesteps:
            x = self.p_sample(x, t, condition)

        return torch.clamp(x, 0, 1)


class DiffusionEnhancer(BaseEnhancer):
    """Wrapper for using diffusion models as enhancers."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        num_inference_steps: int = 50,
        device: str = "auto",
        **model_kwargs
    ):
        super().__init__(device)

        self.model = DDPM(**model_kwargs)
        self.num_inference_steps = num_inference_steps

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

        num_steps = kwargs.get("num_steps", self.num_inference_steps)

        with torch.no_grad():
            enhanced = self.model.sample(tensor, num_steps)

        if is_numpy:
            return self.tensor_to_numpy(enhanced)
        return enhanced
