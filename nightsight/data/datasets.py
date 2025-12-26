"""Dataset classes for low-light image enhancement."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict
import cv2
import random


class LowLightDataset(Dataset):
    """
    Dataset for unpaired low-light images.

    Used for self-supervised or zero-reference training.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".png", ".bmp", ".jpeg")
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory containing images
            transform: Optional transforms to apply
            extensions: Valid image extensions
        """
        self.root = Path(root)
        self.transform = transform

        # Find all images
        self.images = []
        for ext in extensions:
            self.images.extend(self.root.glob(f"**/*{ext}"))
            self.images.extend(self.root.glob(f"**/*{ext.upper()}"))

        self.images = sorted(set(self.images))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.images[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image


class PairedDataset(Dataset):
    """
    Dataset for paired low-light and normal-light images.

    Used for supervised training.
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        transform: Optional[Callable] = None,
        paired_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".png", ".bmp")
    ):
        """
        Initialize paired dataset.

        Args:
            low_dir: Directory with low-light images
            high_dir: Directory with corresponding normal-light images
            transform: Per-image transforms
            paired_transform: Transforms applied to both images together
            extensions: Valid extensions
        """
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.transform = transform
        self.paired_transform = paired_transform

        # Find pairs
        self.pairs = []
        for ext in extensions:
            for low_path in self.low_dir.glob(f"*{ext}"):
                high_path = self.high_dir / low_path.name
                if high_path.exists():
                    self.pairs.append((low_path, high_path))

        self.pairs = sorted(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path, high_path = self.pairs[idx]

        # Load images
        low = cv2.imread(str(low_path))
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        low = low.astype(np.float32) / 255.0

        high = cv2.imread(str(high_path))
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
        high = high.astype(np.float32) / 255.0

        # Apply paired transform
        if self.paired_transform:
            low, high = self.paired_transform(low, high)

        # Apply individual transforms
        if self.transform:
            low = self.transform(low)
            high = self.transform(high)

        # Convert to tensors
        if isinstance(low, np.ndarray):
            low = torch.from_numpy(low.transpose(2, 0, 1))
            high = torch.from_numpy(high.transpose(2, 0, 1))

        return low, high


class VideoDataset(Dataset):
    """
    Dataset for video frames with temporal context.
    """

    def __init__(
        self,
        video_paths: List[str],
        num_frames: int = 5,
        transform: Optional[Callable] = None,
        max_frames_per_video: int = 100
    ):
        """
        Initialize video dataset.

        Args:
            video_paths: List of video file paths
            num_frames: Number of consecutive frames to return
            transform: Transforms to apply
            max_frames_per_video: Maximum frames to sample from each video
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.num_frames = num_frames
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video

        # Index all valid frame sequences
        self.sequences = []
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Sample sequences
            max_start = min(total_frames - num_frames, max_frames_per_video)
            for start in range(0, max_start, num_frames):
                self.sequences.append((video_path, start))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        video_path, start_frame = self.sequences[idx]

        # Read frames
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # Apply transforms
        if self.transform:
            frames = [self.transform(f) for f in frames]

        # Stack
        if isinstance(frames[0], np.ndarray):
            frames = [torch.from_numpy(f.transpose(2, 0, 1)) for f in frames]

        return torch.stack(frames)


class LOLDataset(Dataset):
    """
    LOL (Low-Light) dataset.

    Standard benchmark for low-light enhancement.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        version: str = "v1"
    ):
        """
        Initialize LOL dataset.

        Args:
            root: Root directory of LOL dataset
            split: 'train' or 'test'
            transform: Transforms to apply
            version: Dataset version ('v1' or 'v2')
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Set up paths based on version
        if version == "v1":
            self.low_dir = self.root / "our485" / "low" if split == "train" else self.root / "eval15" / "low"
            self.high_dir = self.root / "our485" / "high" if split == "train" else self.root / "eval15" / "high"
        else:
            self.low_dir = self.root / split / "Low"
            self.high_dir = self.root / split / "Normal"

        # Find pairs
        self.pairs = []
        if self.low_dir.exists() and self.high_dir.exists():
            for low_path in sorted(self.low_dir.glob("*.png")):
                high_path = self.high_dir / low_path.name
                if high_path.exists():
                    self.pairs.append((low_path, high_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path, high_path = self.pairs[idx]

        low = cv2.imread(str(low_path))
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        low = low.astype(np.float32) / 255.0

        high = cv2.imread(str(high_path))
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
        high = high.astype(np.float32) / 255.0

        if self.transform:
            # Apply same transform to both
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            low = self.transform(low)
            random.seed(seed)
            torch.manual_seed(seed)
            high = self.transform(high)

        if isinstance(low, np.ndarray):
            low = torch.from_numpy(low.transpose(2, 0, 1))
            high = torch.from_numpy(high.transpose(2, 0, 1))

        return low, high


class SyntheticDataset(Dataset):
    """
    Synthetic low-light dataset created from normal images.

    Applies realistic degradation to create training pairs.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        gamma_range: Tuple[float, float] = (2.0, 4.0),
        noise_range: Tuple[float, float] = (0.01, 0.05)
    ):
        """
        Initialize synthetic dataset.

        Args:
            root: Directory with normal-light images
            transform: Transforms to apply
            gamma_range: Range for gamma darkening
            noise_range: Range for noise addition
        """
        self.root = Path(root)
        self.transform = transform
        self.gamma_range = gamma_range
        self.noise_range = noise_range

        # Find images
        self.images = []
        for ext in [".jpg", ".png", ".jpeg"]:
            self.images.extend(self.root.glob(f"**/*{ext}"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]

        # Load clean image
        clean = cv2.imread(str(img_path))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        clean = clean.astype(np.float32) / 255.0

        # Create degraded version
        gamma = random.uniform(*self.gamma_range)
        noise_std = random.uniform(*self.noise_range)

        # Apply gamma (darken)
        low = np.power(clean, gamma)

        # Add noise
        noise = np.random.randn(*low.shape).astype(np.float32) * noise_std
        low = np.clip(low + noise, 0, 1)

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            low = self.transform(low)
            random.seed(seed)
            clean = self.transform(clean)

        if isinstance(low, np.ndarray):
            low = torch.from_numpy(low.transpose(2, 0, 1))
            clean = torch.from_numpy(clean.transpose(2, 0, 1))

        return low, clean


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
