"""Perceptual loss using VGG features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torchvision import models


class VGGFeatures(nn.Module):
    """
    Extract features from VGG network for perceptual loss.
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ):
        """
        Initialize VGG feature extractor.

        Args:
            layers: Layer indices to extract features from
            weights: Optional weights for each layer
            normalize: Whether to normalize input
        """
        super().__init__()

        self.layers = layers
        self.weights = weights if weights else [1.0] * len(layers)
        self.normalize = normalize

        # Load pretrained VGG
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())

        # Create slices
        self.slices = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*features[prev:layer + 1]))
            prev = layer + 1

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from multiple layers."""
        if self.normalize:
            x = (x - self.mean) / self.std

        features = []
        for slice_module in self.slices:
            x = slice_module(x)
            features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Measures perceptual similarity between images using
    deep feature representations.
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],
        weights: Optional[List[float]] = None,
        criterion: str = "l1"
    ):
        """
        Initialize perceptual loss.

        Args:
            layers: VGG layers to use
            weights: Weights for each layer
            criterion: Distance metric ('l1', 'l2', 'huber')
        """
        super().__init__()

        self.vgg = VGGFeatures(layers, weights)
        self.weights = weights if weights else [1.0] * len(layers)

        if criterion == "l1":
            self.criterion = F.l1_loss
        elif criterion == "l2":
            self.criterion = F.mse_loss
        elif criterion == "huber":
            self.criterion = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            Perceptual loss value
        """
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            loss += self.weights[i] * self.criterion(pf, tf)

        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices.
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],
        weights: Optional[List[float]] = None
    ):
        super().__init__()

        self.vgg = VGGFeatures(layers, weights)
        self.weights = weights if weights else [1.0] * len(layers)

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute style loss."""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            pred_gram = self.gram_matrix(pf)
            target_gram = self.gram_matrix(tf)
            loss += self.weights[i] * F.mse_loss(pred_gram, target_gram)

        return loss


class LPIPSLoss(nn.Module):
    """
    LPIPS-style learned perceptual loss.

    Simplified version without the full LPIPS training.
    """

    def __init__(self):
        super().__init__()

        self.vgg = VGGFeatures(layers=[3, 8, 15, 22, 29])

        # Learnable weights (initialized to 1)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(5)
        ])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute LPIPS-like loss."""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            # Normalize features
            pf_norm = pf / (pf.norm(dim=1, keepdim=True) + 1e-8)
            tf_norm = tf / (tf.norm(dim=1, keepdim=True) + 1e-8)

            # Compute distance
            diff = (pf_norm - tf_norm).pow(2)
            loss += self.weights[i] * diff.mean()

        return loss
