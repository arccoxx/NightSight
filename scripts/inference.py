#!/usr/bin/env python3
"""
NightSight Inference Script

Enhance images or videos using trained models.

Usage:
    python scripts/inference.py --input image.jpg --output enhanced.jpg
    python scripts/inference.py --input video.mp4 --output enhanced.mp4
    python scripts/inference.py --input folder/ --output results/
"""

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.pipelines.single_image import SingleImagePipeline, QuickEnhance
from nightsight.pipelines.video import VideoPipeline
from nightsight.utils.io import load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="NightSight Inference")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image, video, or directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output path or directory")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--model", "-m", type=str, default="nightsight",
                        choices=["nightsight", "zerodce", "retinexformer", "clahe", "retinex"],
                        help="Model to use")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Disable post-processing denoising")
    parser.add_argument("--no-color-correct", action="store_true",
                        help="Disable color correction")
    parser.add_argument("--compare", action="store_true",
                        help="Save side-by-side comparison")

    return parser.parse_args()


def is_image(path: Path) -> bool:
    """Check if path is an image file."""
    return path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def is_video(path: Path) -> bool:
    """Check if path is a video file."""
    return path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def enhance_image(args, input_path: Path, output_path: Path):
    """Enhance a single image."""
    print(f"Enhancing: {input_path}")

    start_time = time.time()

    # Use traditional methods if specified
    if args.model in ["clahe", "retinex"]:
        result = QuickEnhance.enhance_image(
            str(input_path),
            method=args.model
        )
    else:
        # Create pipeline
        pipeline = SingleImagePipeline(
            model=args.checkpoint if args.checkpoint else args.model,
            device=args.device,
            use_denoise=not args.no_denoise,
            use_color_correction=not args.no_color_correct
        )

        result = pipeline.enhance(str(input_path))

    elapsed = time.time() - start_time

    # Save result
    if args.compare:
        import numpy as np
        original = load_image(str(input_path), dtype="float32")

        # Resize if needed
        if original.shape != result.shape:
            import cv2
            result = cv2.resize(result, (original.shape[1], original.shape[0]))

        comparison = np.hstack([original, result])
        save_image((comparison * 255).astype(np.uint8), output_path)
    else:
        save_image((result * 255).astype(__import__("numpy").uint8), output_path)

    print(f"Saved: {output_path} ({elapsed:.2f}s)")


def enhance_video(args, input_path: Path, output_path: Path):
    """Enhance a video."""
    print(f"Enhancing video: {input_path}")

    pipeline = VideoPipeline(
        model=args.checkpoint if args.checkpoint else args.model,
        device=args.device,
        use_temporal=True
    )

    def progress_callback(current, total):
        print(f"\rFrame {current}/{total} ({100*current/total:.1f}%)", end="", flush=True)

    start_time = time.time()
    pipeline.enhance_video(str(input_path), str(output_path), progress_callback)
    elapsed = time.time() - start_time

    print(f"\nSaved: {output_path} ({elapsed:.1f}s)")


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Handle different input types
    if input_path.is_file():
        # Single file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if is_image(input_path):
            if output_path.is_dir():
                output_path = output_path / input_path.name
            enhance_image(args, input_path, output_path)
        elif is_video(input_path):
            if output_path.is_dir():
                output_path = output_path / input_path.name
            enhance_video(args, input_path, output_path)
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            sys.exit(1)

    elif input_path.is_dir():
        # Directory of files
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        images = [p for p in input_path.iterdir() if is_image(p)]
        videos = [p for p in input_path.iterdir() if is_video(p)]

        print(f"Found {len(images)} images and {len(videos)} videos")

        for img in images:
            out = output_path / img.name
            enhance_image(args, img, out)

        for vid in videos:
            out = output_path / vid.name
            enhance_video(args, vid, out)

    else:
        print(f"Error: Invalid input: {input_path}")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
