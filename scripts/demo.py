#!/usr/bin/env python3
"""
NightSight Demo Script

Interactive demo for testing enhancement methods.

Usage:
    python scripts/demo.py --webcam          # Live webcam demo
    python scripts/demo.py --image test.jpg  # Single image demo
    python scripts/demo.py --compare         # Compare multiple methods
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="NightSight Demo")

    parser.add_argument("--webcam", action="store_true",
                        help="Run live webcam demo")
    parser.add_argument("--image", type=str,
                        help="Single image to enhance")
    parser.add_argument("--compare", type=str,
                        help="Compare multiple methods on an image")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera ID for webcam")

    return parser.parse_args()


def demo_webcam(args):
    """Run live webcam demo."""
    print("Starting webcam demo...")
    print("Press 'q' to quit, 's' to save frame")

    from nightsight.pipelines.video import RealtimePipeline

    pipeline = RealtimePipeline(device=args.device)
    pipeline.run_webcam(camera_id=args.camera_id)


def demo_single_image(args):
    """Enhance and display a single image."""
    import numpy as np

    from nightsight.pipelines.single_image import SingleImagePipeline
    from nightsight.utils.io import load_image, save_image
    from nightsight.utils.visualization import visualize_comparison

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"Loading: {image_path}")
    image = load_image(str(image_path), dtype="float32")

    print("Enhancing...")
    pipeline = SingleImagePipeline(device=args.device)
    enhanced = pipeline.enhance(image)

    # Save comparison
    output_path = image_path.stem + "_enhanced" + image_path.suffix
    comparison = visualize_comparison(image, enhanced)
    save_image(comparison, output_path)
    print(f"Saved: {output_path}")

    # Try to display
    try:
        import cv2
        cv2.imshow("NightSight Demo", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("Could not display image (no display available)")


def demo_compare(args):
    """Compare multiple enhancement methods."""
    import numpy as np

    from nightsight.utils.io import load_image, save_image
    from nightsight.utils.visualization import create_grid

    image_path = Path(args.compare)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"Loading: {image_path}")
    image = load_image(str(image_path), dtype="float32")
    image_uint8 = (image * 255).astype(np.uint8)

    results = [("Original", image)]

    # CLAHE
    print("Testing CLAHE...")
    from nightsight.traditional.histogram import CLAHEEnhancer
    clahe = CLAHEEnhancer()
    results.append(("CLAHE", clahe.enhance(image_uint8) / 255.0))

    # Retinex
    print("Testing Retinex...")
    from nightsight.traditional.retinex import RetinexEnhancer
    retinex = RetinexEnhancer()
    results.append(("Retinex", retinex.enhance(image)))

    # Zero-DCE
    print("Testing Zero-DCE...")
    try:
        from nightsight.models.zerodce import ZeroDCE
        import torch
        model = ZeroDCE()
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            enhanced = model(tensor)
            enhanced = enhanced.squeeze(0).numpy().transpose(1, 2, 0)
        results.append(("Zero-DCE", enhanced))
    except Exception as e:
        print(f"Zero-DCE failed: {e}")

    # NightSight
    print("Testing NightSight...")
    try:
        from nightsight.pipelines.single_image import SingleImagePipeline
        pipeline = SingleImagePipeline(device="cpu")
        results.append(("NightSight", pipeline.enhance(image)))
    except Exception as e:
        print(f"NightSight failed: {e}")

    # Create grid
    print("Creating comparison grid...")
    images = [r[1] for r in results]
    labels = [r[0] for r in results]

    grid = create_grid(images, nrow=len(images), padding=5)

    # Add labels
    import cv2
    grid_with_labels = grid.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        x = i * (grid.shape[1] // len(labels)) + 10
        cv2.putText(grid_with_labels, label, (x, 25), font, 0.6, (255, 255, 255), 1)

    # Save
    output_path = image_path.stem + "_comparison.png"
    save_image(grid_with_labels, output_path)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    if args.webcam:
        demo_webcam(args)
    elif args.image:
        demo_single_image(args)
    elif args.compare:
        demo_compare(args)
    else:
        print("Please specify --webcam, --image, or --compare")
        print("Run with --help for more information")


if __name__ == "__main__":
    main()
