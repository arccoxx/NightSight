#!/usr/bin/env python3
"""
Real-time Night Vision Enhancement Demo

Demonstrates real-time low-light enhancement using webcam or video file.

Usage:
    # Webcam mode
    python scripts/realtime_demo.py --camera 0

    # Video file mode
    python scripts/realtime_demo.py --video path/to/video.mp4

    # With trained checkpoint
    python scripts/realtime_demo.py --checkpoint outputs/nightsight/checkpoints/best_model.pth
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.models.hybrid import NightSightNet
from nightsight.models.zerodce import ZeroDCE
from nightsight.utils.checkpoint import load_checkpoint


class RealtimeEnhancer:
    """Real-time low-light enhancement."""

    def __init__(
        self,
        checkpoint_path=None,
        device="auto",
        model_type="zerodce",
        target_size=(640, 480)
    ):
        """
        Initialize enhancer.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (auto/cuda/cpu)
            model_type: Model type (zerodce/nightsight)
            target_size: Target frame size for processing
        """
        self.device = self._get_device(device)
        self.target_size = target_size
        self.model_type = model_type

        # Load model
        if model_type == "zerodce":
            self.model = ZeroDCE()
        else:
            self.model = NightSightNet()

        # Load checkpoint if provided
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model, info = load_checkpoint(
                checkpoint_path,
                self.model,
                device=str(self.device)
            )
            print(f"Loaded model from epoch {info['epoch']}")
            print(f"Metrics: PSNR={info['metrics'].get('psnr', 'N/A'):.2f}, "
                  f"SSIM={info['metrics'].get('ssim', 'N/A'):.4f}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Performance tracking
        self.frame_times = []
        self.max_history = 30

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR uint8)

        Returns:
            (enhanced_frame, processing_time_ms)
        """
        start_time = time.time()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        original_size = rgb.shape[:2]
        if self.target_size:
            rgb = cv2.resize(rgb, self.target_size[::-1])

        # To tensor
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Enhance
        with torch.no_grad():
            enhanced = self.model(tensor)

        # To numpy
        enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Resize back if needed
        if self.target_size:
            enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))

        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000

        # Track performance
        self.frame_times.append(process_time)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)

        return enhanced_bgr, process_time

    def get_fps_stats(self) -> dict:
        """Get FPS statistics."""
        if not self.frame_times:
            return {"fps": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}

        avg_time = np.mean(self.frame_times)
        return {
            "fps": 1000.0 / avg_time if avg_time > 0 else 0,
            "avg_ms": avg_time,
            "min_ms": np.min(self.frame_times),
            "max_ms": np.max(self.frame_times)
        }


def run_webcam(enhancer: RealtimeEnhancer, camera_id: int = 0):
    """Run enhancement on webcam feed."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    print("\n" + "="*60)
    print("Real-time Night Vision Enhancement - Webcam Mode")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  't' - Toggle side-by-side view")
    print("="*60 + "\n")

    show_both = True
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance
        enhanced, process_time = enhancer.process_frame(frame)

        # Get stats
        stats = enhancer.get_fps_stats()

        # Add stats overlay
        overlay_frame = enhanced.copy()
        cv2.putText(overlay_frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay_frame, f"Processing: {process_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay_frame, f"Device: {enhancer.device}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show
        if show_both:
            combined = np.hstack([frame, overlay_frame])
            cv2.imshow("NightSight - Input | Enhanced", combined)
        else:
            cv2.imshow("NightSight - Enhanced", overlay_frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"enhanced_frame_{frame_count:04d}.png"
            cv2.imwrite(filename, enhanced)
            print(f"Saved {filename}")
        elif key == ord('t'):
            show_both = not show_both
            cv2.destroyAllWindows()

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print(f"Average FPS: {stats['fps']:.2f}")


def run_video(enhancer: RealtimeEnhancer, video_path: str, output_path: str = None):
    """Run enhancement on video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    # Setup writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance
        enhanced, process_time = enhancer.process_frame(frame)

        # Write if output specified
        if writer:
            writer.write(enhanced)

        # Show progress
        frame_count += 1
        if frame_count % 30 == 0:
            stats = enhancer.get_fps_stats()
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                  f"FPS: {stats['fps']:.1f}, Avg time: {stats['avg_ms']:.1f}ms")

        # Display (optional)
        cv2.imshow("Processing...", enhanced)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stats = enhancer.get_fps_stats()
    print(f"\nCompleted! Processed {frame_count} frames")
    print(f"Average FPS: {stats['fps']:.2f}")
    if output_path:
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Night Vision Demo")

    parser.add_argument("--camera", type=int, default=None,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path (for video mode)")
    parser.add_argument("--checkpoint", type=str,
                       default="outputs/nightsight/checkpoints/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="nightsight",
                       choices=["nightsight", "zerodce"],
                       help="Model type to use")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--size", type=str, default="640x480",
                       help="Processing size (e.g., 640x480)")

    args = parser.parse_args()

    # Parse size
    if args.size:
        w, h = map(int, args.size.split('x'))
        target_size = (w, h)
    else:
        target_size = None

    # Check if checkpoint exists
    checkpoint_path = None
    if Path(args.checkpoint).exists():
        checkpoint_path = args.checkpoint
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using untrained model")

    # Create enhancer
    print("Initializing enhancer...")
    enhancer = RealtimeEnhancer(
        checkpoint_path=checkpoint_path,
        device=args.device,
        model_type=args.model,
        target_size=target_size
    )
    print(f"Using device: {enhancer.device}")
    print(f"Processing size: {target_size}")

    # Run
    if args.video:
        run_video(enhancer, args.video, args.output)
    else:
        camera_id = args.camera if args.camera is not None else 0
        run_webcam(enhancer, camera_id)


if __name__ == "__main__":
    main()
