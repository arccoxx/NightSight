"""I/O utilities for loading and saving images and videos."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple, Generator
import torch


def load_image(
    path: Union[str, Path],
    color_space: str = "rgb",
    dtype: str = "uint8"
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to image file
        color_space: Output color space ('rgb', 'bgr', 'gray')
        dtype: Output data type ('uint8', 'float32', 'float64')

    Returns:
        Image as numpy array (H, W, C) or (H, W) for grayscale
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Read image
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    # Handle color space conversion
    if len(image.shape) == 3:
        if color_space == "rgb":
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == "rgb" and len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Handle dtype conversion
    if dtype == "float32":
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
    elif dtype == "float64":
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float64) / 65535.0

    return image


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array or torch tensor
        path: Output path
        quality: JPEG quality (0-100) if applicable
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Normalize to uint8 if needed
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    # Set quality for JPEG
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif suffix == ".png":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(str(path), image)


def load_video(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    color_space: str = "rgb"
) -> Generator[np.ndarray, None, None]:
    """
    Load a video file and yield frames.

    Args:
        path: Path to video file
        max_frames: Maximum number of frames to load
        color_space: Output color space ('rgb', 'bgr')

    Yields:
        Video frames as numpy arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if color_space == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()


def get_video_info(path: Union[str, Path]) -> dict:
    """
    Get video metadata.

    Args:
        path: Path to video file

    Returns:
        Dictionary with video info (fps, frame_count, width, height)
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


class VideoWriter:
    """Context manager for writing video frames."""

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 30.0,
        size: Optional[Tuple[int, int]] = None,
        codec: str = "mp4v"
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.codec = codec
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.writer is not None:
            self.writer.release()

    def write(self, frame: Union[np.ndarray, torch.Tensor]) -> None:
        """Write a frame to the video."""
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 4:
                frame = frame.squeeze(0)
            if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
                frame = frame.permute(1, 2, 0)
            frame = frame.detach().cpu().numpy()

        if frame.dtype in [np.float32, np.float64]:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        # Initialize writer on first frame if size not specified
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, self.size
            )

        # Convert RGB to BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.writer.write(frame)


def save_video(
    frames: List[np.ndarray],
    path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v"
) -> None:
    """
    Save a list of frames as a video.

    Args:
        frames: List of frame arrays
        path: Output path
        fps: Frames per second
        codec: Video codec
    """
    with VideoWriter(path, fps=fps, codec=codec) as writer:
        for frame in frames:
            writer.write(frame)


def load_raw(
    path: Union[str, Path],
    pattern: str = "rggb",
    bit_depth: int = 12,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    """
    Load a RAW image file.

    Args:
        path: Path to RAW file
        pattern: Bayer pattern ('rggb', 'bggr', 'gbrg', 'grbg')
        bit_depth: Bit depth of the sensor
        width: Image width (required for headerless RAW)
        height: Image height (required for headerless RAW)

    Returns:
        RAW image as numpy array
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Try to use rawpy for common formats
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            # Return the raw data
            return raw.raw_image.astype(np.float32) / (2 ** bit_depth - 1)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: read as binary for headerless RAW
    if width is None or height is None:
        raise ValueError("Width and height required for headerless RAW files")

    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint16)

    # Reshape and normalize
    image = data.reshape(height, width)
    image = image.astype(np.float32) / (2 ** bit_depth - 1)

    return image


def demosaic_raw(
    raw_image: np.ndarray,
    pattern: str = "rggb"
) -> np.ndarray:
    """
    Demosaic a RAW Bayer image to RGB.

    Args:
        raw_image: RAW Bayer image
        pattern: Bayer pattern

    Returns:
        Demosaiced RGB image
    """
    # Map pattern to OpenCV code
    pattern_map = {
        "rggb": cv2.COLOR_BAYER_RG2RGB,
        "bggr": cv2.COLOR_BAYER_BG2RGB,
        "gbrg": cv2.COLOR_BAYER_GB2RGB,
        "grbg": cv2.COLOR_BAYER_GR2RGB,
    }

    if pattern.lower() not in pattern_map:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    # Convert to uint16 for OpenCV
    if raw_image.dtype == np.float32:
        raw_uint16 = (raw_image * 65535).astype(np.uint16)
    else:
        raw_uint16 = raw_image.astype(np.uint16)

    # Demosaic
    rgb = cv2.cvtColor(raw_uint16, pattern_map[pattern.lower()])

    # Normalize back to float
    return rgb.astype(np.float32) / 65535
