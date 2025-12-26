"""
Motion detection and optical flow methods for multi-frame enhancement.

Motion information from multiple frames can be used to improve
low-light enhancement by exploiting temporal redundancy.
"""

import numpy as np
import cv2
from typing import Union, Tuple, List, Optional


def optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    method: str = "farneback"
) -> np.ndarray:
    """
    Compute optical flow between two frames.

    Args:
        prev_frame: Previous frame (grayscale or color)
        curr_frame: Current frame
        method: Flow computation method ('farneback', 'lucas_kanade', 'dis')

    Returns:
        Flow field of shape (H, W, 2) containing (dx, dy) per pixel
    """
    # Convert to grayscale if needed
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame

    # Ensure uint8
    if prev_gray.dtype != np.uint8:
        prev_gray = (np.clip(prev_gray, 0, 1) * 255).astype(np.uint8)
        curr_gray = (np.clip(curr_gray, 0, 1) * 255).astype(np.uint8)

    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    elif method == "lucas_kanade":
        # Sparse flow - we'll compute it densely using a grid
        h, w = prev_gray.shape
        grid_size = 10
        p0 = np.array([
            [x, y] for y in range(0, h, grid_size) for x in range(0, w, grid_size)
        ], dtype=np.float32).reshape(-1, 1, 2)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None,
            winSize=(15, 15),
            maxLevel=2
        )

        # Interpolate to dense flow
        flow = np.zeros((h, w, 2), dtype=np.float32)
        for i, (pt0, pt1, st) in enumerate(zip(p0, p1, status)):
            if st[0]:
                y, x = int(pt0[0, 1]), int(pt0[0, 0])
                flow[max(0, y - grid_size // 2):min(h, y + grid_size // 2),
                     max(0, x - grid_size // 2):min(w, x + grid_size // 2)] = pt1[0] - pt0[0]

    elif method == "dis":
        # DIS (Dense Inverse Search) optical flow
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(prev_gray, curr_gray, None)

    else:
        raise ValueError(f"Unknown optical flow method: {method}")

    return flow


def motion_detection(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 25,
    min_area: int = 500
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Detect motion between two frames.

    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        threshold: Difference threshold for motion
        min_area: Minimum contour area for motion region

    Returns:
        Tuple of (motion mask, list of motion bounding boxes)
    """
    # Convert to grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame

    # Ensure uint8
    if prev_gray.dtype != np.uint8:
        prev_gray = (np.clip(prev_gray, 0, 1) * 255).astype(np.uint8)
        curr_gray = (np.clip(curr_gray, 0, 1) * 255).astype(np.uint8)

    # Compute absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get bounding boxes
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))

    return motion_mask, bboxes


def frame_difference(
    frames: List[np.ndarray],
    method: str = "absolute"
) -> np.ndarray:
    """
    Compute difference between consecutive frames.

    Args:
        frames: List of frames
        method: Difference method ('absolute', 'squared', 'temporal_gradient')

    Returns:
        Frame difference map
    """
    if len(frames) < 2:
        return np.zeros_like(frames[0])

    diffs = []
    for i in range(len(frames) - 1):
        f1 = frames[i].astype(np.float32)
        f2 = frames[i + 1].astype(np.float32)

        if f1.max() > 1:
            f1 = f1 / 255.0
            f2 = f2 / 255.0

        if method == "absolute":
            diff = np.abs(f2 - f1)
        elif method == "squared":
            diff = (f2 - f1) ** 2
        elif method == "temporal_gradient":
            diff = f2 - f1
        else:
            raise ValueError(f"Unknown difference method: {method}")

        diffs.append(diff)

    # Average differences
    return np.mean(diffs, axis=0)


def motion_compensate(
    reference: np.ndarray,
    frame: np.ndarray,
    flow: np.ndarray
) -> np.ndarray:
    """
    Warp a frame to align with reference using optical flow.

    Args:
        reference: Reference frame (for size)
        frame: Frame to warp
        flow: Optical flow from reference to frame

    Returns:
        Warped frame aligned to reference
    """
    h, w = reference.shape[:2]

    # Create sampling grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[:, :, 0]).astype(np.float32)
    map_y = (y + flow[:, :, 1]).astype(np.float32)

    # Warp frame
    warped = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


def temporal_median(
    frames: List[np.ndarray],
    align: bool = True
) -> np.ndarray:
    """
    Compute temporal median for noise reduction.

    Args:
        frames: List of frames
        align: Whether to align frames before computing median

    Returns:
        Temporal median frame
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    if len(frames) == 1:
        return frames[0]

    if align:
        # Use middle frame as reference
        ref_idx = len(frames) // 2
        reference = frames[ref_idx]
        aligned_frames = [reference]

        for i, frame in enumerate(frames):
            if i == ref_idx:
                continue
            flow = optical_flow(reference, frame, method="dis")
            aligned = motion_compensate(reference, frame, flow)
            aligned_frames.append(aligned)

        frames = aligned_frames

    # Stack and compute median
    stacked = np.stack(frames, axis=0)
    return np.median(stacked, axis=0).astype(frames[0].dtype)


def temporal_average(
    frames: List[np.ndarray],
    weights: Optional[List[float]] = None,
    align: bool = True
) -> np.ndarray:
    """
    Compute weighted temporal average.

    Args:
        frames: List of frames
        weights: Optional weights for each frame
        align: Whether to align frames before averaging

    Returns:
        Temporal average frame
    """
    if len(frames) == 0:
        raise ValueError("Empty frame list")

    if len(frames) == 1:
        return frames[0]

    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    if align:
        # Align to middle frame
        ref_idx = len(frames) // 2
        reference = frames[ref_idx]
        aligned_frames = []

        for i, frame in enumerate(frames):
            if i == ref_idx:
                aligned_frames.append(frame)
            else:
                flow = optical_flow(reference, frame, method="dis")
                aligned = motion_compensate(reference, frame, flow)
                aligned_frames.append(aligned)

        frames = aligned_frames

    # Weighted average
    result = np.zeros_like(frames[0], dtype=np.float32)
    for frame, weight in zip(frames, weights):
        result += weight * frame.astype(np.float32)

    return result.astype(frames[0].dtype)


def compute_motion_confidence(
    flow: np.ndarray,
    prev_frame: np.ndarray,
    curr_frame: np.ndarray
) -> np.ndarray:
    """
    Compute confidence map for optical flow.

    Args:
        flow: Optical flow field
        prev_frame: Previous frame
        curr_frame: Current frame

    Returns:
        Confidence map (0-1)
    """
    h, w = prev_frame.shape[:2]

    # Warp previous frame to current
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[:, :, 0]).astype(np.float32)
    map_y = (y + flow[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        prev_frame.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Compute reconstruction error
    error = np.abs(warped - curr_frame.astype(np.float32))
    if len(error.shape) == 3:
        error = np.mean(error, axis=2)

    # Convert error to confidence
    max_error = 50 if prev_frame.dtype == np.uint8 else 0.2
    confidence = 1 - np.clip(error / max_error, 0, 1)

    return confidence


def motion_blur_estimation(
    image: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate motion blur in an image.

    Args:
        image: Input image

    Returns:
        Tuple of (blur_magnitude, blur_direction_degrees)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0

    # Compute FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Analyze spectrum for blur characteristics
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Sample along different angles
    num_angles = 36
    profiles = []

    for angle in range(num_angles):
        theta = np.radians(angle * 180 / num_angles)
        profile = []
        for r in range(min(cx, cy)):
            x = int(cx + r * np.cos(theta))
            y = int(cy + r * np.sin(theta))
            if 0 <= x < w and 0 <= y < h:
                profile.append(magnitude_spectrum[y, x])
        profiles.append(np.array(profile))

    # Find angle with minimum energy (blur direction)
    energies = [np.sum(p) for p in profiles]
    blur_angle_idx = np.argmin(energies)
    blur_direction = blur_angle_idx * 180 / num_angles

    # Estimate blur magnitude from profile decay
    blur_profile = profiles[blur_angle_idx]
    if len(blur_profile) > 0:
        # Fit exponential decay
        log_profile = np.log(blur_profile + 1e-10)
        decay_rate = np.abs(np.polyfit(np.arange(len(log_profile)), log_profile, 1)[0])
        blur_magnitude = 1.0 / (decay_rate + 1e-10)
    else:
        blur_magnitude = 0

    return blur_magnitude, blur_direction


class BackgroundSubtractor:
    """
    Background subtraction for motion detection in video.
    """

    def __init__(
        self,
        method: str = "mog2",
        history: int = 500,
        var_threshold: float = 16
    ):
        """
        Initialize background subtractor.

        Args:
            method: Subtraction method ('mog2', 'knn')
            history: Number of frames for background model
            var_threshold: Variance threshold for foreground
        """
        if method == "mog2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=True
            )
        elif method == "knn":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=var_threshold * var_threshold,
                detectShadows=True
            )
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")

    def apply(
        self,
        frame: np.ndarray,
        learning_rate: float = -1
    ) -> np.ndarray:
        """
        Apply background subtraction to a frame.

        Args:
            frame: Input frame
            learning_rate: Learning rate (-1 for automatic)

        Returns:
            Foreground mask
        """
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return self.bg_subtractor.apply(frame, learningRate=learning_rate)

    def get_background(self) -> np.ndarray:
        """Get the current background model."""
        return self.bg_subtractor.getBackgroundImage()
