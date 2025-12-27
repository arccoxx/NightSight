"""
Scene classification for adaptive processing.

Classifies lighting conditions and scene types to automatically
adjust enhancement parameters for optimal results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union


class SceneClassifierNet(nn.Module):
    """
    Lightweight scene classifier network.

    Classifies scenes into lighting categories for adaptive processing.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        """
        Initialize scene classifier.

        Args:
            in_channels: Number of input channels
            num_classes: Number of scene classes
                - 0: Very dark (night)
                - 1: Dark (indoor low-light)
                - 2: Dim (twilight)
                - 3: Normal (well-lit indoor)
                - 4: Bright (outdoor daylight)
        """
        super().__init__()

        # Lightweight MobileNet-style architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify scene.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)

        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)

        logits = self.fc(feat)

        return logits


class SceneClassifier:
    """
    Scene classifier for adaptive enhancement.

    Analyzes image statistics and uses ML to determine optimal
    enhancement parameters for the current scene.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        use_ml: bool = True
    ):
        """
        Initialize scene classifier.

        Args:
            model_path: Path to pretrained classifier model
            device: Device to run on
            use_ml: Use ML classifier (otherwise use statistical methods)
        """
        self.device = self._get_device(device)
        self.use_ml = use_ml

        if use_ml:
            self.model = SceneClassifierNet()

            # Load pretrained weights if available
            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load pretrained weights: {e}")

            self.model.to(self.device)
            self.model.eval()

        # Scene class names
        self.class_names = [
            'very_dark',     # 0
            'dark',          # 1
            'dim',           # 2
            'normal',        # 3
            'bright'         # 4
        ]

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def classify(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Dict:
        """
        Classify scene lighting condition.

        Args:
            image: Input image (H, W, 3) or (B, 3, H, W)

        Returns:
            Dictionary with:
                - class_id: Scene class ID
                - class_name: Scene class name
                - confidence: Classification confidence
                - brightness: Mean brightness [0, 1]
                - parameters: Recommended enhancement parameters
        """
        # Get image statistics
        stats = self._compute_statistics(image)

        if self.use_ml:
            # ML-based classification
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

            # Classify
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)

                class_id = probs.argmax(dim=1).item()
                confidence = probs[0, class_id].item()

        else:
            # Statistical classification
            brightness = stats['brightness']

            if brightness < 0.1:
                class_id = 0  # very_dark
            elif brightness < 0.25:
                class_id = 1  # dark
            elif brightness < 0.4:
                class_id = 2  # dim
            elif brightness < 0.7:
                class_id = 3  # normal
            else:
                class_id = 4  # bright

            confidence = 1.0  # Statistical method doesn't provide confidence

        # Get recommended parameters
        parameters = self._get_enhancement_parameters(class_id, stats)

        return {
            'class_id': class_id,
            'class_name': self.class_names[class_id],
            'confidence': confidence,
            'brightness': stats['brightness'],
            'contrast': stats['contrast'],
            'parameters': parameters,
            'statistics': stats
        }

    def _compute_statistics(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Dict:
        """
        Compute image statistics for classification.

        Args:
            image: Input image

        Returns:
            Dictionary of statistics
        """
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            img = image.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            img = img.transpose(1, 2, 0)
        else:
            img = image

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Compute statistics
        brightness = np.mean(img)
        contrast = np.std(img)

        # Histogram stats
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
        hist = hist / hist.sum()

        # Dark pixel ratio
        dark_ratio = np.sum(img < 0.1) / img.size

        # Bright pixel ratio
        bright_ratio = np.sum(img > 0.9) / img.size

        # Color statistics
        if img.shape[2] == 3:
            color_std = np.std(img, axis=2).mean()
        else:
            color_std = 0

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'dark_ratio': float(dark_ratio),
            'bright_ratio': float(bright_ratio),
            'color_std': float(color_std)
        }

    def _get_enhancement_parameters(
        self,
        class_id: int,
        stats: Dict
    ) -> Dict:
        """
        Get recommended enhancement parameters for scene class.

        Args:
            class_id: Scene class ID
            stats: Image statistics

        Returns:
            Dictionary of enhancement parameters
        """
        # Base parameters for each class
        params = {
            0: {  # very_dark
                'enhancement_strength': 1.0,
                'denoise_strength': 0.8,
                'edge_intensity': 1.0,
                'color_boost': 1.2,
                'gamma': 0.4,
                'use_depth_outlines': True,
                'outline_color': (0, 255, 0),  # Green (night vision)
                'outline_thickness': 3
            },
            1: {  # dark
                'enhancement_strength': 0.9,
                'denoise_strength': 0.7,
                'edge_intensity': 0.8,
                'color_boost': 1.1,
                'gamma': 0.5,
                'use_depth_outlines': True,
                'outline_color': (0, 200, 100),
                'outline_thickness': 2
            },
            2: {  # dim
                'enhancement_strength': 0.7,
                'denoise_strength': 0.5,
                'edge_intensity': 0.6,
                'color_boost': 1.0,
                'gamma': 0.6,
                'use_depth_outlines': True,
                'outline_color': (0, 150, 150),
                'outline_thickness': 2
            },
            3: {  # normal
                'enhancement_strength': 0.4,
                'denoise_strength': 0.3,
                'edge_intensity': 0.3,
                'color_boost': 1.0,
                'gamma': 0.7,
                'use_depth_outlines': False,
                'outline_color': (0, 100, 200),
                'outline_thickness': 1
            },
            4: {  # bright
                'enhancement_strength': 0.1,
                'denoise_strength': 0.1,
                'edge_intensity': 0.1,
                'color_boost': 1.0,
                'gamma': 0.9,
                'use_depth_outlines': False,
                'outline_color': (0, 50, 255),
                'outline_thickness': 1
            }
        }

        base_params = params[class_id].copy()

        # Adjust based on statistics
        if stats['dark_ratio'] > 0.7:
            # Very dark image, boost enhancement
            base_params['enhancement_strength'] *= 1.2
            base_params['gamma'] *= 0.9

        if stats['contrast'] < 0.1:
            # Low contrast, boost edges
            base_params['edge_intensity'] *= 1.3

        return base_params

    def get_adaptive_config(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Dict:
        """
        Get complete adaptive configuration for image.

        Args:
            image: Input image

        Returns:
            Complete configuration dictionary for NightSightV2 pipeline
        """
        # Classify scene
        scene_info = self.classify(image)

        # Build configuration
        config = {
            'scene': scene_info,
            'modules': {
                'zerodce': {
                    'enabled': scene_info['brightness'] < 0.5,
                    'strength': scene_info['parameters']['enhancement_strength']
                },
                'denoiser': {
                    'enabled': scene_info['brightness'] < 0.6,
                    'strength': scene_info['parameters']['denoise_strength']
                },
                'edge_outliner': {
                    'enabled': scene_info['parameters']['use_depth_outlines'],
                    'intensity': scene_info['parameters']['edge_intensity'],
                    'color': scene_info['parameters']['outline_color'],
                    'thickness': scene_info['parameters']['outline_thickness']
                },
                'depth_estimator': {
                    'enabled': scene_info['parameters']['use_depth_outlines'],
                },
                'super_resolution': {
                    'enabled': scene_info['brightness'] < 0.4,
                    'scale': 2 if scene_info['brightness'] < 0.2 else 1
                },
                'object_detector': {
                    'enabled': True,
                    'conf_threshold': 0.3 if scene_info['brightness'] < 0.3 else 0.25
                },
                'tracker': {
                    'enabled': True
                },
                'color_correction': {
                    'enabled': True,
                    'boost': scene_info['parameters']['color_boost']
                }
            }
        }

        return config
