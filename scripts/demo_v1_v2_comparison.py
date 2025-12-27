"""
Quick demo comparing NightSight v1 and v2.

Shows both side-by-side on a test image.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch

from nightsight.models.hybrid import NightSightNet
from nightsight.v2 import NightSightV2Pipeline
from nightsight.utils.io import load_image, save_image


def compare_v1_v2(image_path, output_path=None):
    """
    Compare NightSight v1 and v2 on an image.

    Args:
        image_path: Input image path
        output_path: Optional output path for comparison
    """
    print("=" * 60)
    print("NightSight v1 vs v2 Comparison")
    print("=" * 60)
    print()

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path, dtype='float32')

    # Process with v1
    print("Processing with NightSight v1...")
    v1_model = NightSightNet()
    v1_model.eval()

    with torch.no_grad():
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        v1_enhanced = v1_model(image_tensor).squeeze(0).numpy().transpose(1, 2, 0)

    print("  ✓ v1 complete")

    # Process with v2
    print("Processing with NightSight v2...")
    v2_pipeline = NightSightV2Pipeline(
        device='auto',
        use_all_features=True
    )

    v2_enhanced, components = v2_pipeline.enhance_image(
        image,
        return_components=True
    )

    print("  ✓ v2 complete")
    print()
    print("v2 modules used:")
    if 'scene_config' in components:
        scene = components['scene_config']['scene']
        print(f"  - Scene: {scene['class_name']} (brightness: {scene['brightness']:.2f})")

    print(f"  - Depth estimation: {'✓' if 'depth_map' in components else '✗'}")
    print(f"  - Edge outlines: {'✓' if 'edges' in components else '✗'}")
    print(f"  - Object detection: {'✓' if 'detections' in components else '✗'}")
    print(f"  - Enhancement: {'✓' if 'zerodce_enhanced' in components else '✗'}")

    # Create comparison
    print()
    print("Creating comparison image...")

    h, w = image.shape[:2]

    # Create side-by-side comparison
    comparison = np.hstack([
        image,
        v1_enhanced,
        v2_enhanced
    ])

    # Convert to uint8 for visualization
    comparison_vis = (comparison * 255).astype(np.uint8)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w / 400, 2.0))  # Adaptive font size
    thickness = max(1, int(font_scale * 2))

    cv2.putText(
        comparison_vis,
        "Original",
        (10, 40),
        font, font_scale, (255, 255, 255), thickness
    )

    cv2.putText(
        comparison_vis,
        "NightSight v1",
        (w + 10, 40),
        font, font_scale, (255, 255, 255), thickness
    )

    cv2.putText(
        comparison_vis,
        "NightSight v2",
        (w * 2 + 10, 40),
        font, font_scale, (0, 255, 0), thickness
    )

    # Add dividers
    comparison_vis[:, w-1:w+1] = 255
    comparison_vis[:, w*2-1:w*2+1] = 255

    # Save if requested
    if output_path:
        save_image(comparison_vis, output_path)
        print(f"✓ Saved comparison to: {output_path}")
    else:
        # Display
        print("Displaying comparison (press any key to close)...")
        cv2.imshow('NightSight v1 vs v2', cv2.cvtColor(comparison_vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print()
    print("Comparison complete!")

    return comparison_vis


def main():
    parser = argparse.ArgumentParser(
        description='Compare NightSight v1 and v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display comparison
  python scripts/demo_v1_v2_comparison.py -i dark_image.jpg

  # Save comparison
  python scripts/demo_v1_v2_comparison.py -i dark_image.jpg -o comparison.jpg

  # Use specific device
  python scripts/demo_v1_v2_comparison.py -i dark_image.jpg --device cuda
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path (if not provided, will display)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use (default: auto)')

    args = parser.parse_args()

    # Check input exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run comparison
    try:
        compare_v1_v2(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
