"""
NightSight v2 real-time webcam demo.

Military night vision-style enhancement with live webcam feed.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.v2.pipeline import NightSightV2Pipeline


def main():
    parser = argparse.ArgumentParser(description='NightSight v2 Real-time Demo')

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Optional path to save output video')

    # Module toggles for initial state
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth estimation initially')
    parser.add_argument('--no-zerodce', action='store_true',
                        help='Disable Zero-DCE++ initially')
    parser.add_argument('--no-edges', action='store_true',
                        help='Disable edge outlines initially')
    parser.add_argument('--no-detection', action='store_true',
                        help='Disable object detection initially')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable object tracking initially')
    parser.add_argument('--use-superres', action='store_true',
                        help='Enable super-resolution (much slower)')

    # Model paths
    parser.add_argument('--detector-model', type=str, default=None,
                        help='Path to YOLO weights (default: yolov8n.pt)')
    parser.add_argument('--zerodce-model', type=str, default=None,
                        help='Path to trained Zero-DCE++ model')
    parser.add_argument('--depth-model', type=str, default=None,
                        help='Path to depth estimation model')
    parser.add_argument('--edge-model', type=str, default=None,
                        help='Path to edge detection model')

    args = parser.parse_args()

    print("=" * 60)
    print("NightSight v2 - Real-time Night Vision Demo")
    print("=" * 60)
    print()
    print("Initializing modules...")

    # Create pipeline
    pipeline = NightSightV2Pipeline(
        device=args.device,
        use_all_features=False,
        use_depth=not args.no_depth,
        use_zerodce=not args.no_zerodce,
        use_edges=not args.no_edges,
        zerodce_model_path=args.zerodce_model,
        depth_model_path=args.depth_model,
        edge_model_path=args.edge_model,
        use_detection=not args.no_detection,
        use_tracking=not args.no_tracking,
        use_superres=args.use_superres,
        use_adaptive=True,
        detector_model_path=args.detector_model
    )

    print()
    print("Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  1 - Toggle depth estimation")
    print("  2 - Toggle Zero-DCE++ enhancement")
    print("  3 - Toggle edge outlines")
    print("  4 - Toggle object detection")
    print("  5 - Toggle object tracking")
    print("  6 - Toggle super-resolution")
    print("  7 - Toggle adaptive processing")
    print()
    print("Starting webcam...")
    print()

    # Run webcam processing
    pipeline.process_webcam(
        camera_id=args.camera,
        display_fps=True,
        save_output=args.save_output
    )

    print("\nDemo ended.")


if __name__ == '__main__':
    main()
