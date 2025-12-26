"""Single image enhancement pipeline."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from nightsight.core.base import BaseEnhancer
from nightsight.utils.io import load_image, save_image


class SingleImagePipeline:
    """
    Complete pipeline for single image enhancement.

    Combines preprocessing, enhancement, and postprocessing.
    """

    def __init__(
        self,
        model: Optional[Union[str, BaseEnhancer]] = None,
        device: str = "auto",
        use_traditional: bool = True,
        use_denoise: bool = True,
        use_color_correction: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            model: Model name, checkpoint path, or BaseEnhancer instance
            device: Device to use
            use_traditional: Apply traditional preprocessing
            use_denoise: Apply denoising
            use_color_correction: Apply color correction
        """
        self.device = self._get_device(device)
        self.use_traditional = use_traditional
        self.use_denoise = use_denoise
        self.use_color_correction = use_color_correction

        # Load or create model
        if model is None:
            # Use default NightSight model
            from nightsight.models.hybrid import NightSightNet
            self.model = NightSightNet()
        elif isinstance(model, str):
            self.model = self._load_model(model)
        else:
            self.model = model

        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        from nightsight.models.hybrid import NightSightNet
        from nightsight.core.registry import ModelRegistry

        path = Path(model_path)
        if path.exists():
            # Load from checkpoint
            model = NightSightNet()
            model.load_pretrained(model_path)
            return model
        else:
            # Try to create from registry
            try:
                return ModelRegistry.create(model_path)
            except ValueError:
                raise ValueError(f"Unknown model: {model_path}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before enhancement."""
        # Ensure float32 in [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        # Optional traditional preprocessing
        if self.use_traditional:
            from nightsight.traditional.histogram import adaptive_gamma

            # Light gamma correction to improve visibility
            image = adaptive_gamma(image, method="mean", min_gamma=0.6, max_gamma=1.0)

        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """Postprocess enhanced image."""
        # Denoise
        if self.use_denoise:
            from nightsight.traditional.filters import bilateral_filter
            image = bilateral_filter(image, d=5, sigma_color=0.1, sigma_space=10)

        # Color correction
        if self.use_color_correction:
            # Simple white balance
            for c in range(3):
                channel = image[:, :, c]
                p_low, p_high = np.percentile(channel, [1, 99])
                image[:, :, c] = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)

        return np.clip(image, 0, 1)

    def enhance(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Enhance a single image.

        Args:
            image: Image path or numpy array
            output_path: Optional path to save result

        Returns:
            Enhanced image as numpy array
        """
        # Load if path
        if isinstance(image, (str, Path)):
            image = load_image(str(image), dtype="float32")

        # Preprocess
        image = self.preprocess(image)

        # Convert to tensor
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        # Enhance
        with torch.no_grad():
            enhanced = self.model(tensor)

        # Convert back to numpy
        enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Postprocess
        enhanced = self.postprocess(enhanced)

        # Save if requested
        if output_path:
            save_image((enhanced * 255).astype(np.uint8), output_path)

        return enhanced

    def enhance_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[np.ndarray]:
        """
        Enhance multiple images.

        Args:
            images: List of images or paths
            output_dir: Optional directory to save results

        Returns:
            List of enhanced images
        """
        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            if isinstance(img, (str, Path)):
                output_path = output_dir / Path(img).name if output_dir else None
            else:
                output_path = output_dir / f"enhanced_{i:04d}.png" if output_dir else None

            enhanced = self.enhance(img, output_path)
            results.append(enhanced)

        return results


class QuickEnhance:
    """
    Quick enhancement without full pipeline setup.

    Provides simple functions for common use cases.
    """

    @staticmethod
    def enhance_image(
        image: Union[str, Path, np.ndarray],
        output_path: Optional[str] = None,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Quickly enhance an image.

        Args:
            image: Image to enhance
            output_path: Optional save path
            method: Enhancement method ('auto', 'retinex', 'clahe', 'deep')

        Returns:
            Enhanced image
        """
        # Load if needed
        if isinstance(image, (str, Path)):
            image = load_image(str(image), dtype="float32")
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if method == "auto" or method == "deep":
            pipeline = SingleImagePipeline()
            result = pipeline.enhance(image)
        elif method == "retinex":
            from nightsight.traditional.retinex import RetinexEnhancer
            enhancer = RetinexEnhancer()
            result = enhancer.enhance(image)
        elif method == "clahe":
            from nightsight.traditional.histogram import CLAHEEnhancer
            enhancer = CLAHEEnhancer()
            result = enhancer.enhance((image * 255).astype(np.uint8))
            result = result.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown method: {method}")

        if output_path:
            save_image((result * 255).astype(np.uint8), output_path)

        return result
