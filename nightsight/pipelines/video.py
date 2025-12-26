"""Video enhancement pipeline with temporal consistency."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Generator, Callable
from nightsight.core.base import BaseEnhancer
from nightsight.utils.io import load_video, get_video_info, VideoWriter


class VideoPipeline:
    """
    Complete pipeline for video enhancement.

    Supports temporal models for improved consistency.
    """

    def __init__(
        self,
        model: Optional[Union[str, BaseEnhancer]] = None,
        device: str = "auto",
        temporal_frames: int = 5,
        use_temporal: bool = True
    ):
        """
        Initialize video pipeline.

        Args:
            model: Model to use for enhancement
            device: Device for processing
            temporal_frames: Number of frames for temporal model
            use_temporal: Use temporal model if available
        """
        self.device = self._get_device(device)
        self.temporal_frames = temporal_frames
        self.use_temporal = use_temporal

        # Load model
        if model is None:
            if use_temporal:
                from nightsight.models.hybrid import TemporalNightSightNet
                self.model = TemporalNightSightNet(num_frames=temporal_frames)
            else:
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

        # Frame buffer for temporal processing
        self.frame_buffer = []

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, model_path: str):
        """Load model from checkpoint or registry."""
        from nightsight.models.hybrid import NightSightNet
        from nightsight.core.registry import ModelRegistry

        path = Path(model_path)
        if path.exists():
            model = NightSightNet()
            model.load_pretrained(model_path)
            return model
        else:
            return ModelRegistry.create(model_path)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess tensor to frame."""
        frame = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        frame = np.clip(frame, 0, 1)
        return (frame * 255).astype(np.uint8)

    def enhance_frame(
        self,
        frame: np.ndarray,
        use_buffer: bool = True
    ) -> np.ndarray:
        """
        Enhance a single frame.

        Args:
            frame: Input frame
            use_buffer: Use temporal buffer for context

        Returns:
            Enhanced frame
        """
        tensor = self._preprocess_frame(frame)

        if self.use_temporal and use_buffer:
            # Add to buffer
            self.frame_buffer.append(tensor)
            if len(self.frame_buffer) > self.temporal_frames:
                self.frame_buffer.pop(0)

            # Pad buffer if needed
            while len(self.frame_buffer) < self.temporal_frames:
                self.frame_buffer.insert(0, self.frame_buffer[0])

            # Stack frames
            stacked = torch.stack([f.squeeze(0) for f in self.frame_buffer], dim=0)
            stacked = stacked.unsqueeze(0)  # (1, T, C, H, W)

            with torch.no_grad():
                enhanced = self.model(stacked)
        else:
            with torch.no_grad():
                enhanced = self.model(tensor)

        return self._postprocess_frame(enhanced)

    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Enhance a video file.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            progress_callback: Optional callback for progress updates
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Get video info
        info = get_video_info(input_path)
        fps = info['fps']
        total_frames = info['frame_count']

        # Reset buffer
        self.frame_buffer = []

        with VideoWriter(output_path, fps=fps) as writer:
            for i, frame in enumerate(load_video(input_path)):
                # Enhance frame
                enhanced = self.enhance_frame(frame)

                # Write
                writer.write(enhanced)

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total_frames)

    def enhance_video_generator(
        self,
        input_path: Union[str, Path]
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields enhanced frames.

        Args:
            input_path: Path to input video

        Yields:
            Enhanced frames
        """
        self.frame_buffer = []

        for frame in load_video(input_path):
            enhanced = self.enhance_frame(frame)
            yield enhanced

    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self.frame_buffer = []


class RealtimePipeline:
    """
    Real-time enhancement pipeline for live video.

    Optimized for low latency.
    """

    def __init__(
        self,
        model: Optional[Union[str, BaseEnhancer]] = None,
        device: str = "auto",
        target_fps: float = 30.0
    ):
        """
        Initialize realtime pipeline.

        Args:
            model: Model to use
            device: Device for processing
            target_fps: Target frame rate
        """
        self.device = self._get_device(device)
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps

        # Use lightweight model for real-time
        if model is None:
            from nightsight.models.zerodce import ZeroDCEPP
            self.model = ZeroDCEPP(scale_factor=2)
        elif isinstance(model, str):
            from nightsight.core.registry import ModelRegistry
            self.model = ModelRegistry.create(model)
        else:
            self.model = model

        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

        # Compile model if supported
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception:
                pass

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame in real-time.

        Args:
            frame: Input frame (BGR uint8)

        Returns:
            Enhanced frame (BGR uint8)
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To tensor
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Enhance
        with torch.no_grad():
            enhanced = self.model(tensor)

        # To numpy
        enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    def run_webcam(
        self,
        camera_id: int = 0,
        window_name: str = "NightSight"
    ) -> None:
        """
        Run enhancement on webcam feed.

        Args:
            camera_id: Camera device ID
            window_name: Window name for display
        """
        import cv2

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        print(f"Press 'q' to quit, 's' to save frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance
            enhanced = self.process_frame(frame)

            # Show side by side
            combined = np.hstack([frame, enhanced])
            cv2.imshow(window_name, combined)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("enhanced_frame.png", enhanced)
                print("Saved enhanced_frame.png")

        cap.release()
        cv2.destroyAllWindows()
