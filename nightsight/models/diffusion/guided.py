"""
Guided diffusion models for low-light enhancement.

Implements physics-guided and classifier-free guidance approaches
for more controlled and accurate enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math
from nightsight.core.base import BaseModel
from nightsight.core.registry import ModelRegistry
from nightsight.models.diffusion.ddpm import DDPM, UNetDiffusion, get_timestep_embedding


class ConditionalUNet(nn.Module):
    """U-Net with additional conditioning mechanisms."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        cond_channels: int = 3,
        time_emb_dim: int = 256
    ):
        super().__init__()

        self.base_channels = base_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        )

        # Main U-Net (simplified)
        self.conv_in = nn.Conv2d(in_channels + base_channels, base_channels, 3, 1, 1)

        # Encoder
        self.down1 = self._make_down_block(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4, time_emb_dim)

        # Middle
        self.mid = self._make_res_block(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder
        self.up1 = self._make_up_block(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up2 = self._make_up_block(base_channels * 2, base_channels, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        )

    def _make_res_block(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'norm1': nn.GroupNorm(8, in_ch),
            'conv1': nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            'time_mlp': nn.Linear(time_dim, out_ch),
            'norm2': nn.GroupNorm(8, out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            'skip': nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        })

    def _make_down_block(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'res': self._make_res_block(in_ch, out_ch, time_dim),
            'down': nn.Conv2d(out_ch, out_ch, 3, 2, 1)
        })

    def _make_up_block(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'up': nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            'res': self._make_res_block(out_ch * 2, out_ch, time_dim)
        })

    def _apply_res_block(self, block, x, t_emb):
        h = F.silu(block['norm1'](x))
        h = block['conv1'](h)
        h = h + block['time_mlp'](F.silu(t_emb))[:, :, None, None]
        h = F.silu(block['norm2'](h))
        h = block['conv2'](h)
        return h + block['skip'](x)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        # Time embedding
        t_emb = get_timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        # Encode condition
        cond_feat = self.cond_encoder(condition)

        # Concatenate with input
        h = self.conv_in(torch.cat([x, cond_feat], dim=1))

        # Encoder
        h1 = self._apply_res_block(self.down1['res'], h, t_emb)
        h1_down = self.down1['down'](h1)

        h2 = self._apply_res_block(self.down2['res'], h1_down, t_emb)
        h2_down = self.down2['down'](h2)

        # Middle
        h_mid = self._apply_res_block(self.mid, h2_down, t_emb)

        # Decoder
        h_up1 = self.up1['up'](h_mid)
        h_up1 = torch.cat([h_up1, h2], dim=1)
        h_up1 = self._apply_res_block(self.up1['res'], h_up1, t_emb)

        h_up2 = self.up2['up'](h_up1)
        h_up2 = torch.cat([h_up2, h1], dim=1)
        h_up2 = self._apply_res_block(self.up2['res'], h_up2, t_emb)

        return self.conv_out(h_up2)


@ModelRegistry.register("guided_diffusion")
class GuidedDiffusion(BaseModel):
    """
    Guided diffusion model with classifier-free guidance.

    Supports both conditional and unconditional generation for
    classifier-free guidance during sampling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_timesteps: int = 1000,
        guidance_scale: float = 3.0,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        unconditional_prob: float = 0.1
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        self.unconditional_prob = unconditional_prob

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
        self.model = ConditionalUNet(in_channels, base_channels)

        # Null condition for unconditional generation
        self.register_buffer('null_cond', torch.zeros(1, in_channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        use_guidance: bool = True
    ) -> torch.Tensor:
        """Predict noise with optional classifier-free guidance."""
        if use_guidance and self.guidance_scale > 1.0:
            # Conditional prediction
            noise_cond = self.model(x, t, condition)

            # Unconditional prediction
            null_cond = self.null_cond.expand(x.shape[0], -1, x.shape[2], x.shape[3])
            noise_uncond = self.model(x, t, null_cond)

            # Classifier-free guidance
            noise = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
        else:
            noise = self.model(x, t, condition)

        return noise

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM sampling for faster inference."""
        # Start from noise
        x = torch.randn_like(condition)

        # Create timestep schedule
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # Predict noise
            pred_noise = self.forward(x, t_tensor, condition, use_guidance=True)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise

            if sigma_t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return torch.clamp(x, 0, 1)


@ModelRegistry.register("physics_guided_diffusion")
class PhysicsGuidedDiffusion(BaseModel):
    """
    Physics-guided diffusion model for low-light enhancement.

    Incorporates Retinex-based physics constraints to guide
    the diffusion process for more realistic enhancement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_timesteps: int = 1000,
        retinex_weight: float = 0.1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.retinex_weight = retinex_weight

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # Main denoising model
        self.model = ConditionalUNet(in_channels, base_channels)

        # Illumination estimator for physics guidance
        self.illum_estimator = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def estimate_illumination(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate illumination map from image."""
        return self.illum_estimator(x)

    def compute_retinex_guidance(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Retinex-based guidance gradient.

        Encourages the enhanced image to follow Retinex decomposition
        constraints relative to the input.
        """
        # Estimate illumination for both
        illum_x = self.estimate_illumination(x)
        illum_cond = self.estimate_illumination(condition)

        # Reflectance should be similar
        ref_x = x / (illum_x + 1e-4)
        ref_cond = condition / (illum_cond + 1e-4)

        # Gradient towards consistent reflectance
        guidance = ref_cond - ref_x

        return guidance * self.retinex_weight

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise with physics guidance."""
        # Standard noise prediction
        noise = self.model(x, t, condition)

        # Add physics guidance (only during training)
        if self.training:
            # Compute Retinex guidance
            guidance = self.compute_retinex_guidance(x, condition)
            noise = noise - guidance

        return noise

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """Sample with physics-guided denoising."""
        x = torch.randn_like(condition)

        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for t in timesteps:
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # Predict noise
            pred_noise = self.model(x, t_tensor, condition)

            # Apply physics guidance during sampling
            guidance = self.compute_retinex_guidance(x, condition)
            pred_noise = pred_noise - guidance

            # Denoising step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            coef1 = 1 / torch.sqrt(alpha)
            coef2 = beta / self.sqrt_one_minus_alphas_cumprod[t]

            mean = coef1 * (x - coef2 * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        return torch.clamp(x, 0, 1)
