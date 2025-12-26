"""Base classes for all enhancers and models in NightSight."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class BaseEnhancer(ABC):
    """
    Abstract base class for all image enhancers.

    This provides a unified interface for both traditional image processing
    methods and deep learning-based approaches.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the enhancer.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance a low-light image.

        Args:
            image: Input image (H, W, C) numpy array or (B, C, H, W) tensor
            **kwargs: Additional enhancement parameters

        Returns:
            Enhanced image in the same format as input
        """
        pass

    def enhance_batch(
        self,
        images: Union[List[np.ndarray], torch.Tensor],
        **kwargs
    ) -> Union[List[np.ndarray], torch.Tensor]:
        """
        Enhance a batch of images.

        Args:
            images: List of images or batched tensor
            **kwargs: Additional enhancement parameters

        Returns:
            Enhanced images
        """
        if isinstance(images, list):
            return [self.enhance(img, **kwargs) for img in images]
        else:
            return self.enhance(images, **kwargs)

    @staticmethod
    def numpy_to_tensor(
        image: np.ndarray,
        normalize: bool = True
    ) -> torch.Tensor:
        """Convert numpy image (H, W, C) to tensor (1, C, H, W)."""
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # Handle different dtypes
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        # HWC -> CHW -> BCHW
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)

        if normalize:
            tensor = tensor.float()

        return tensor

    @staticmethod
    def tensor_to_numpy(
        tensor: torch.Tensor,
        denormalize: bool = True
    ) -> np.ndarray:
        """Convert tensor (B, C, H, W) or (C, H, W) to numpy (H, W, C)."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # CHW -> HWC
        image = tensor.detach().cpu().numpy().transpose(1, 2, 0)

        if denormalize:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        return image


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models.

    Provides common functionality for loading/saving weights,
    inference modes, and device management.
    """

    def __init__(self):
        super().__init__()
        self._is_training = True

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def load_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True
    ) -> None:
        """
        Load pretrained weights from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce matching keys
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        self.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **extra_info
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            **extra_info: Additional info to save in checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
        }

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        checkpoint.update(extra_info)

        torch.save(checkpoint, path)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def set_inference_mode(self) -> None:
        """Set model to inference mode with optimizations."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def inference(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference with no gradient computation."""
        return self.forward(x, **kwargs)


class MultiFrameEnhancer(BaseEnhancer):
    """
    Base class for multi-frame/video enhancement methods.

    Handles frame buffering, temporal alignment, and fusion.
    """

    def __init__(
        self,
        num_frames: int = 5,
        device: str = "auto"
    ):
        super().__init__(device)
        self.num_frames = num_frames
        self.frame_buffer: List[torch.Tensor] = []

    def add_frame(self, frame: Union[np.ndarray, torch.Tensor]) -> None:
        """Add a frame to the buffer."""
        if isinstance(frame, np.ndarray):
            frame = self.numpy_to_tensor(frame)

        self.frame_buffer.append(frame)

        # Keep only the last num_frames
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)

    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self.frame_buffer = []

    @abstractmethod
    def enhance_temporal(
        self,
        frames: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Enhance using multiple frames.

        Args:
            frames: List of frames (each is B, C, H, W tensor)
            **kwargs: Additional parameters

        Returns:
            Enhanced center/reference frame
        """
        pass

    def enhance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhance using the current frame buffer.

        Automatically adds the input to the buffer.
        """
        is_numpy = isinstance(image, np.ndarray)
        self.add_frame(image)

        if len(self.frame_buffer) < self.num_frames:
            # Not enough frames, pad with current frame
            frames = self.frame_buffer + [
                self.frame_buffer[-1]
            ] * (self.num_frames - len(self.frame_buffer))
        else:
            frames = self.frame_buffer

        result = self.enhance_temporal(frames, **kwargs)

        if is_numpy:
            return self.tensor_to_numpy(result)
        return result
