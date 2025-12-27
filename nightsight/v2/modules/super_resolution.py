"""
Super-resolution and denoising module for enhanced detail.

Upscales and denoises low-light images to restore sharpness
and clarity lost in darkness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Union


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for feature extraction."""

    def __init__(self, channels: int, growth_channels: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))

        return x5 * 0.2 + x


class SuperResolutionNet(nn.Module):
    """
    Lightweight super-resolution network.

    Based on ESRGAN architecture but optimized for speed.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        scale: int = 2
    ):
        """
        Initialize super-resolution network.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_features: Number of feature channels
            num_blocks: Number of residual dense blocks
            scale: Upscaling factor (2 or 4)
        """
        super().__init__()

        self.scale = scale

        # First convolution
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # Residual dense blocks
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(num_features)
            for _ in range(num_blocks)
        ])

        # Fusion
        self.conv_fusion = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Upsampling
        if scale == 2:
            self.upconv = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif scale == 4:
            self.upconv = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.upconv = nn.Identity()

        # Final convolution
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Super-resolve input image.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Super-resolved image (B, C, H*scale, W*scale)
        """
        # First conv
        feat = self.conv_first(x)
        trunk = feat

        # RDB blocks
        for block in self.blocks:
            feat = block(feat)

        # Fusion with skip connection
        feat = self.conv_fusion(feat) + trunk

        # Upsample
        feat = self.upconv(feat)

        # Final conv
        out = self.conv_last(feat)

        return out


class DenoiserNet(nn.Module):
    """
    Lightweight denoising network for low-light images.
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64, num_blocks: int = 6):
        super().__init__()

        # Encoder
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # Processing blocks
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(num_features)
            for _ in range(num_blocks)
        ])

        # Decoder
        self.conv_last = nn.Conv2d(num_features, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise input image.

        Args:
            x: Noisy image (B, C, H, W)

        Returns:
            Denoised image (B, C, H, W)
        """
        feat = self.conv_first(x)

        for block in self.blocks:
            feat = block(feat)

        out = self.conv_last(feat)

        # Residual connection
        return x + out


class SuperResolution:
    """
    Super-resolution and denoising module.

    Provides upscaling and noise reduction for low-light images.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        scale: int = 2,
        use_denoiser: bool = True
    ):
        """
        Initialize super-resolution module.

        Args:
            model_path: Path to pretrained model
            device: Device to run on
            scale: Upscaling factor
            use_denoiser: Also use denoiser
        """
        self.device = self._get_device(device)
        self.scale = scale
        self.use_denoiser = use_denoiser

        # Initialize SR model
        self.sr_model = SuperResolutionNet(scale=scale)

        # Initialize denoiser if requested
        if use_denoiser:
            self.denoiser = DenoiserNet()

        # Load pretrained weights if available
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.sr_model.load_state_dict(state_dict.get('sr_model', state_dict))
                if use_denoiser and 'denoiser' in state_dict:
                    self.denoiser.load_state_dict(state_dict['denoiser'])
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

        self.sr_model.to(self.device)
        self.sr_model.eval()

        if use_denoiser:
            self.denoiser.to(self.device)
            self.denoiser.eval()

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def denoise(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denoise image.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            return_numpy: Whether to return numpy array

        Returns:
            Denoised image
        """
        if not self.use_denoiser:
            return image

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

        # Denoise
        with torch.no_grad():
            denoised = self.denoiser(tensor)
            denoised = torch.clamp(denoised, 0, 1)

        # Convert back to numpy if needed
        if return_numpy and is_numpy:
            denoised = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if image.max() > 1.0:
                denoised = denoised * 255.0

        return denoised

    def upscale(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Upscale image using super-resolution.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            return_numpy: Whether to return numpy array

        Returns:
            Upscaled image (H*scale, W*scale, 3)
        """
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

        # Upscale
        with torch.no_grad():
            upscaled = self.sr_model(tensor)
            upscaled = torch.clamp(upscaled, 0, 1)

        # Convert back to numpy if needed
        if return_numpy and is_numpy:
            upscaled = upscaled.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if image.max() > 1.0:
                upscaled = upscaled * 255.0

        return upscaled

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        denoise_first: bool = True,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance image with denoising and super-resolution.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)
            denoise_first: Denoise before upscaling
            return_numpy: Whether to return numpy array

        Returns:
            Enhanced image
        """
        if self.use_denoiser and denoise_first:
            image = self.denoise(image, return_numpy=False)

        if self.scale > 1:
            image = self.upscale(image, return_numpy=False)

        if self.use_denoiser and not denoise_first:
            image = self.denoise(image, return_numpy=False)

        # Convert to numpy if needed
        if return_numpy and isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        return image

    def denoise_traditional(
        self,
        image: np.ndarray,
        method: str = "bilateral"
    ) -> np.ndarray:
        """
        Denoise using traditional methods (fast fallback).

        Args:
            image: Input image (H, W, 3)
            method: Denoising method ('bilateral', 'nlm', 'bm3d')

        Returns:
            Denoised image (H, W, 3)
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img = image.copy()

        if method == "bilateral":
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
        elif method == "nlm":
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            # BM3D requires external library, use bilateral as fallback
            denoised = cv2.bilateralFilter(img, 9, 75, 75)

        # Convert back to original dtype
        if image.dtype != np.uint8:
            denoised = denoised.astype(np.float32) / 255.0

        return denoised
