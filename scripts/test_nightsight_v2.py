"""
Test script for NightSight v2.

Tests the complete v2 pipeline with all features.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from nightsight.v2 import NightSightV2Pipeline
from nightsight.utils.io import load_image, save_image


def test_single_image(
    pipeline: NightSightV2Pipeline,
    image_path: str,
    output_dir: Path
):
    """Test on a single image with full component breakdown."""
    print(f"\n{'='*60}")
    print(f"Testing on: {image_path}")
    print(f"{'='*60}\n")

    # Load image
    image = load_image(image_path, dtype='float32')
    print(f"Image size: {image.shape}")

    # Process with all components
    start_time = time.time()
    enhanced, components = pipeline.enhance_image(
        image,
        return_components=True
    )
    processing_time = time.time() - start_time

    print(f"\nProcessing time: {processing_time:.2f}s")
    print(f"FPS: {1/processing_time:.2f}")

    # Save main result
    output_path = output_dir / f"{Path(image_path).stem}_enhanced_v2.png"
    save_image((enhanced * 255).astype(np.uint8), output_path)
    print(f"\n[OK] Saved enhanced image: {output_path}")

    # Save components
    print("\nSaving components:")
    for name, component in components.items():
        if isinstance(component, np.ndarray) and component.ndim in [2, 3]:
            comp_path = output_dir / f"{Path(image_path).stem}_{name}.png"

            if component.ndim == 2:
                # Grayscale
                comp_vis = (component * 255).astype(np.uint8)
            else:
                # RGB
                comp_vis = (component * 255).astype(np.uint8)

            save_image(comp_vis, comp_path)
            print(f"  [OK] {name}")

    # Print scene info
    if 'scene_config' in components:
        scene = components['scene_config']['scene']
        print(f"\nScene Analysis:")
        print(f"  Class: {scene['class_name']}")
        print(f"  Brightness: {scene['brightness']:.3f}")
        print(f"  Contrast: {scene['contrast']:.3f}")

    # Print detection info
    if 'detections' in components:
        detections = components['detections']
        print(f"\nDetections: {len(detections)} objects")
        for i, det in enumerate(detections[:5]):  # Show first 5
            print(f"  {i+1}. {det['class']} (conf: {det['conf']:.2f})")

    return enhanced, components


def test_feature_comparison(
    image_path: str,
    output_dir: Path,
    device: str = 'auto'
):
    """Test with different feature combinations."""
    print(f"\n{'='*60}")
    print("Testing Different Feature Combinations")
    print(f"{'='*60}\n")

    image = load_image(image_path, dtype='float32')

    configs = [
        {
            'name': 'minimal',
            'desc': 'Zero-DCE only',
            'kwargs': {
                'use_depth': False,
                'use_zerodce': True,
                'use_edges': False,
                'use_detection': False,
                'use_tracking': False,
                'use_superres': False,
                'use_adaptive': False
            }
        },
        {
            'name': 'edges_only',
            'desc': 'Edges + Enhancement',
            'kwargs': {
                'use_depth': False,
                'use_zerodce': True,
                'use_edges': True,
                'use_detection': False,
                'use_tracking': False,
                'use_superres': False,
                'use_adaptive': False
            }
        },
        {
            'name': 'depth_aware',
            'desc': 'Depth-aware outlines',
            'kwargs': {
                'use_depth': True,
                'use_zerodce': True,
                'use_edges': True,
                'use_detection': False,
                'use_tracking': False,
                'use_superres': False,
                'use_adaptive': False
            }
        },
        {
            'name': 'full',
            'desc': 'All features',
            'kwargs': {
                'use_depth': True,
                'use_zerodce': True,
                'use_edges': True,
                'use_detection': True,
                'use_tracking': True,
                'use_superres': False,
                'use_adaptive': True
            }
        }
    ]

    results = {}

    for config in configs:
        print(f"\nTesting: {config['desc']}")

        # Create pipeline
        pipeline = NightSightV2Pipeline(
            device=device,
            use_all_features=False,
            **config['kwargs']
        )

        # Process
        start_time = time.time()
        enhanced = pipeline.enhance_image(image)
        processing_time = time.time() - start_time

        print(f"  Time: {processing_time:.2f}s ({1/processing_time:.1f} FPS)")

        # Save
        output_path = output_dir / f"{Path(image_path).stem}_{config['name']}.png"
        save_image((enhanced * 255).astype(np.uint8), output_path)
        print(f"  [OK] Saved: {output_path.name}")

        results[config['name']] = {
            'enhanced': enhanced,
            'time': processing_time
        }

    # Create comparison grid
    print("\nCreating comparison grid...")
    comparison = np.hstack([
        image,
        results['minimal']['enhanced'],
        results['edges_only']['enhanced'],
        results['depth_aware']['enhanced'],
        results['full']['enhanced']
    ])

    comparison_vis = (comparison * 255).astype(np.uint8)

    # Add labels
    h, w = image.shape[:2]
    labels = ['Original', 'Minimal', 'Edges', 'Depth-Aware', 'Full']

    for i, label in enumerate(labels):
        cv2.putText(
            comparison_vis,
            label,
            (i * w + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    comparison_path = output_dir / f"{Path(image_path).stem}_comparison.png"
    save_image(comparison_vis, comparison_path)
    print(f"[OK] Saved comparison: {comparison_path}")

    return results


def test_batch_images(
    pipeline: NightSightV2Pipeline,
    image_dir: str,
    output_dir: Path
):
    """Test on multiple images."""
    print(f"\n{'='*60}")
    print("Batch Image Processing")
    print(f"{'='*60}\n")

    image_paths = list(Path(image_dir).glob('*.png')) + \
                  list(Path(image_dir).glob('*.jpg'))

    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images\n")

    total_time = 0
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing {img_path.name}...", end=' ')

        image = load_image(str(img_path), dtype='float32')

        start_time = time.time()
        enhanced = pipeline.enhance_image(image)
        processing_time = time.time() - start_time
        total_time += processing_time

        # Save
        output_path = output_dir / f"{img_path.stem}_v2.png"
        save_image((enhanced * 255).astype(np.uint8), output_path)

        print(f"Done ({processing_time:.2f}s)")

    avg_time = total_time / len(image_paths)
    print(f"\nAverage time: {avg_time:.2f}s ({1/avg_time:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(
        description='Test NightSight v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python scripts/test_nightsight_v2.py -i dark.jpg -o test_output/

  # Test with feature comparison
  python scripts/test_nightsight_v2.py -i dark.jpg -o test_output/ --compare-features

  # Test batch processing
  python scripts/test_nightsight_v2.py --batch-dir data/LOL/eval15/low -o test_output/

  # Use specific Zero-DCE model
  python scripts/test_nightsight_v2.py -i dark.jpg -o test_output/ --zerodce-model outputs/zerodce_v2/zerodce_best.pth
        """
    )

    parser.add_argument('-i', '--input', type=str,
                        help='Input image path (for single image test)')
    parser.add_argument('--batch-dir', type=str,
                        help='Directory of images (for batch test)')
    parser.add_argument('-o', '--output-dir', type=str, default='test_outputs_v2',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')

    # Feature comparison
    parser.add_argument('--compare-features', action='store_true',
                        help='Compare different feature combinations')

    # Model paths
    parser.add_argument('--zerodce-model', type=str, default=None,
                        help='Path to trained Zero-DCE++ model')
    parser.add_argument('--depth-model', type=str, default=None,
                        help='Path to depth model')
    parser.add_argument('--edge-model', type=str, default=None,
                        help='Path to edge detector')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NightSight v2 Testing Suite")
    print("=" * 60)

    if args.compare_features and args.input:
        # Feature comparison test
        test_feature_comparison(args.input, output_dir, args.device)

    elif args.input:
        # Single image test
        pipeline = NightSightV2Pipeline(
            device=args.device,
            use_all_features=True,
            zerodce_model_path=args.zerodce_model,
            depth_model_path=args.depth_model,
            edge_model_path=args.edge_model
        )

        test_single_image(pipeline, args.input, output_dir)

    elif args.batch_dir:
        # Batch test
        pipeline = NightSightV2Pipeline(
            device=args.device,
            use_all_features=True,
            zerodce_model_path=args.zerodce_model,
            depth_model_path=args.depth_model,
            edge_model_path=args.edge_model
        )

        test_batch_images(pipeline, args.batch_dir, output_dir)

    else:
        parser.print_help()
        print("\nError: Provide either --input or --batch-dir")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
