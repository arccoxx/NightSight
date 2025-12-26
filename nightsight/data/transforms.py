"""Data augmentation transforms for NightSight."""

import numpy as np
import cv2
import random
from typing import Tuple, Optional, Callable, List, Union
import torch


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image


class RandomCrop:
    """Random crop augmentation."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.size

        if h < th or w < tw:
            # Resize if too small
            scale = max(th / h, tw / w)
            new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
            image = cv2.resize(image, (new_w, new_h))
            h, w = new_h, new_w

        y = random.randint(0, h - th)
        x = random.randint(0, w - tw)

        return image[y:y + th, x:x + tw]


class CenterCrop:
    """Center crop."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.size

        y = (h - th) // 2
        x = (w - tw) // 2

        return image[y:y + th, x:x + tw]


class RandomFlip:
    """Random horizontal and/or vertical flip."""

    def __init__(self, horizontal: bool = True, vertical: bool = False, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.horizontal and random.random() < self.p:
            image = image[:, ::-1].copy()
        if self.vertical and random.random() < self.p:
            image = image[::-1, :].copy()
        return image


class RandomRotation:
    """Random rotation by 90 degree increments."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
        return image


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Brightness
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = image * factor

        # Contrast
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation (convert to HSV)
        if self.saturation > 0 and image.shape[-1] == 3:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * factor
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255

        return np.clip(image, 0, 1)


class Resize:
    """Resize image."""

    def __init__(self, size: Union[int, Tuple[int, int]], keep_aspect: bool = False):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.size

        if self.keep_aspect:
            scale = min(th / h, tw / w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

            # Pad to target size
            pad_h = th - new_h
            pad_w = tw - new_w
            image = np.pad(
                image,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2),
                 (0, 0)),
                mode='constant'
            )
        else:
            image = cv2.resize(image, (tw, th))

        return image


class Normalize:
    """Normalize image with mean and std."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return (image - self.mean) / self.std


class ToTensor:
    """Convert numpy array to PyTorch tensor."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image

        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float()


class PairedRandomCrop:
    """Random crop applied to paired images."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image1.shape[:2]
        th, tw = self.size

        if h < th or w < tw:
            scale = max(th / h, tw / w)
            new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
            image1 = cv2.resize(image1, (new_w, new_h))
            image2 = cv2.resize(image2, (new_w, new_h))
            h, w = new_h, new_w

        y = random.randint(0, h - th)
        x = random.randint(0, w - tw)

        return image1[y:y + th, x:x + tw], image2[y:y + th, x:x + tw]


class PairedRandomFlip:
    """Random flip applied to paired images."""

    def __init__(self, horizontal: bool = True, vertical: bool = False, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.horizontal and random.random() < self.p:
            image1 = image1[:, ::-1].copy()
            image2 = image2[:, ::-1].copy()
        if self.vertical and random.random() < self.p:
            image1 = image1[::-1, :].copy()
            image2 = image2[::-1, :].copy()
        return image1, image2


def get_train_transforms(
    size: int = 256,
    augment: bool = True
) -> Compose:
    """Get standard training transforms."""
    transforms = [RandomCrop(size)]

    if augment:
        transforms.extend([
            RandomFlip(horizontal=True, vertical=False),
            RandomRotation(p=0.5),
        ])

    transforms.append(ToTensor())

    return Compose(transforms)


def get_val_transforms(size: int = 256) -> Compose:
    """Get validation transforms."""
    return Compose([
        CenterCrop(size),
        ToTensor(),
    ])


def get_test_transforms() -> Compose:
    """Get test transforms (minimal processing)."""
    return Compose([ToTensor()])
