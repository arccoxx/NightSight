#!/usr/bin/env python3
"""
Test real-time pipeline with validation images.

Usage:
    python scripts/test_realtime.py
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.models.hybrid import NightSightNet
from nightsight.utils.checkpoint import load_checkpoint


def test_realtime_performance(checkpoint_path, num_iterations=100):
    """Test real-time performance on various image sizes."""
    import torch

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = NightSightNet()
    model, info = load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {info['epoch']}\n")

    # Test different resolutions
    resolutions = [
        (320, 240, "QVGA"),
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD")
    ]

    print("Performance Testing")
    print("=" * 70)

    for width, height, name in resolutions:
        # Create dummy input
        dummy_input = torch.randn(1, 3, height, width).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / avg_time

        print(f"{name:10} ({width}x{height:4}): {avg_time:6.2f}ms +/- {std_time:5.2f}ms  "
              f"=>  {fps:5.1f} FPS")

    print("=" * 70)


def process_validation_images(checkpoint_path, output_dir="outputs/realtime_test"):
    """Process validation images to demonstrate real-time capability."""
    import torch
    from PIL import Image

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model...")
    model = NightSightNet()
    model, _ = load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    model.eval()

    # Process LOL validation images
    data_dir = Path("data/LOL/eval15/low")
    if not data_dir.exists():
        print(f"Validation images not found at {data_dir}")
        return

    print(f"\nProcessing validation images from {data_dir}...")
    print("-" * 70)

    images = sorted(data_dir.glob("*.png"))[:5]  # Process first 5

    for img_path in images:
        # Load image
        img = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        start_time = time.time()
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Enhance
        with torch.no_grad():
            enhanced = model(tensor)

        # Convert back
        enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        process_time = (time.time() - start_time) * 1000

        # Save
        output_path = output_dir / f"enhanced_{img_path.name}"
        cv2.imwrite(str(output_path), enhanced_bgr)

        print(f"{img_path.name:20} => {output_path.name:20} ({process_time:6.1f}ms)")

    print("-" * 70)
    print(f"\nEnhanced images saved to: {output_dir}")


def create_demo_video(checkpoint_path, output_path="outputs/realtime_demo.mp4"):
    """Create a demo video showing enhancement in action."""
    import torch
    from PIL import Image

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nCreating demo video...")
    model = NightSightNet()
    model, _ = load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    model.eval()

    # Get validation images
    data_dir = Path("data/LOL/eval15/low")
    if not data_dir.exists():
        print(f"Validation images not found at {data_dir}")
        return

    images = sorted(data_dir.glob("*.png"))

    # Video writer setup
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5  # Slow for demo
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

    print(f"Processing {len(images)} images...")

    for img_path in images:
        # Load and enhance
        img = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced = model(tensor)

        enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # Create side-by-side
        combined = np.hstack([img, enhanced_bgr])

        # Add text
        cv2.putText(combined, "Input", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Enhanced", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(combined)

    writer.release()
    print(f"Demo video saved to: {output_path}")


def main():
    checkpoint_path = "outputs/nightsight/checkpoints/best_model.pth"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or specify a valid checkpoint path")
        return

    print("=" * 70)
    print("NightSight Real-time Pipeline Test")
    print("=" * 70)

    # Test 1: Performance benchmarking
    print("\n[1] Performance Benchmarking")
    test_realtime_performance(checkpoint_path, num_iterations=100)

    # Test 2: Process validation images
    print("\n[2] Processing Validation Images")
    process_validation_images(checkpoint_path)

    # Test 3: Create demo video
    print("\n[3] Creating Demo Video")
    create_demo_video(checkpoint_path)

    print("\n" + "=" * 70)
    print("Real-time pipeline test complete!")
    print("=" * 70)
    print("\nTo test with webcam, run:")
    print("  python scripts/realtime_demo.py --camera 0")


if __name__ == "__main__":
    main()
