"""
NightSight v2 inference script.

Process images and videos with military night vision-style enhancement.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.v2.pipeline import NightSightV2Pipeline


def main():
    parser = argparse.ArgumentParser(description='NightSight v2 Inference')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image or video path')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')

    # Module toggles
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth estimation')
    parser.add_argument('--no-zerodce', action='store_true',
                        help='Disable Zero-DCE++ enhancement')
    parser.add_argument('--no-edges', action='store_true',
                        help='Disable edge outlines')
    parser.add_argument('--no-detection', action='store_true',
                        help='Disable object detection')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable object tracking')
    parser.add_argument('--use-superres', action='store_true',
                        help='Enable super-resolution (slower)')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive processing')

    # Model paths
    parser.add_argument('--depth-model', type=str, default=None,
                        help='Path to depth model weights')
    parser.add_argument('--zerodce-model', type=str, default=None,
                        help='Path to Zero-DCE++ weights')
    parser.add_argument('--edge-model', type=str, default=None,
                        help='Path to edge detector weights')
    parser.add_argument('--detector-model', type=str, default=None,
                        help='Path to YOLO weights (default: yolov8n.pt)')
    parser.add_argument('--sr-model', type=str, default=None,
                        help='Path to super-resolution weights')

    # Additional options
    parser.add_argument('--show-components', action='store_true',
                        help='Save intermediate processing steps')
    parser.add_argument('--compare-v1', action='store_true',
                        help='Compare with NightSight v1')

    args = parser.parse_args()

    # Create pipeline
    print("Initializing NightSight v2...")
    pipeline = NightSightV2Pipeline(
        device=args.device,
        use_all_features=False,
        use_depth=not args.no_depth,
        use_zerodce=not args.no_zerodce,
        use_edges=not args.no_edges,
        use_detection=not args.no_detection,
        use_tracking=not args.no_tracking,
        use_superres=args.use_superres,
        use_adaptive=not args.no_adaptive,
        depth_model_path=args.depth_model,
        zerodce_model_path=args.zerodce_model,
        edge_model_path=args.edge_model,
        detector_model_path=args.detector_model,
        sr_model_path=args.sr_model
    )

    # Determine input type
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Process
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video processing
        print(f"Processing video: {input_path}")
        pipeline.enhance_video(input_path, output_path, show_progress=True)
        print(f"Saved enhanced video: {output_path}")

    else:
        # Image processing
        print(f"Processing image: {input_path}")

        if args.compare_v1:
            # Compare with v1
            comparison = pipeline.compare_with_v1(input_path, output_path)
            print(f"Saved comparison: {output_path}")
        elif args.show_components:
            # Save components
            enhanced, components = pipeline.enhance_image(
                input_path,
                return_components=True
            )

            # Save main result
            from nightsight.utils.io import save_image
            save_image((enhanced * 255).astype('uint8'), output_path)
            print(f"Saved enhanced image: {output_path}")

            # Save components
            output_dir = output_path.parent / f"{output_path.stem}_components"
            output_dir.mkdir(exist_ok=True)

            for name, component in components.items():
                if isinstance(component, np.ndarray) and component.ndim in [2, 3]:
                    comp_path = output_dir / f"{name}.png"

                    if component.ndim == 2:
                        # Grayscale
                        comp_vis = (component * 255).astype('uint8')
                    else:
                        # RGB
                        comp_vis = (component * 255).astype('uint8')

                    save_image(comp_vis, comp_path)
                    print(f"  Saved {name}: {comp_path}")

        else:
            # Standard processing
            enhanced = pipeline.enhance_image(input_path, output_path)
            print(f"Saved enhanced image: {output_path}")


if __name__ == '__main__':
    import numpy as np
    main()
