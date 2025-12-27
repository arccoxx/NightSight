"""
Create sample images for README from both v1 and v2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2
from nightsight.models.hybrid import NightSightNet
from nightsight.v2 import NightSightV2Pipeline
from nightsight.utils.io import load_image, save_image


def create_samples():
    """Create README samples."""
    print("Creating README samples...")

    # Sample images from LOL dataset
    samples = [
        'data/LOL/eval15/low/146.png',
        'data/LOL/our485/low/171.png',
    ]

    output_dir = Path('outputs/readme_samples')

    # Initialize models
    print("\nInitializing v1 model...")
    v1_model = NightSightNet()
    v1_model.eval()

    print("Initializing v2 pipeline...")
    v2_pipeline = NightSightV2Pipeline(
        device='cpu',
        use_all_features=False,
        use_zerodce=True,
        use_depth=False,
        use_edges=False,
        use_detection=False
    )

    for i, image_path in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Processing {image_path}")

        if not Path(image_path).exists():
            print(f"  Skipping - file not found")
            continue

        # Load image
        image = load_image(image_path, dtype='float32')
        stem = Path(image_path).stem

        # Save original
        original_path = output_dir / 'originals' / f'{stem}_original.png'
        save_image((image * 255).astype(np.uint8), original_path)
        print(f"  Saved original: {original_path.name}")

        # Process with v1
        print("  Processing with v1...")
        with torch.no_grad():
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            v1_enhanced = v1_model(image_tensor).squeeze(0).numpy().transpose(1, 2, 0)

        v1_path = output_dir / 'v1' / f'{stem}_v1.png'
        save_image((np.clip(v1_enhanced, 0, 1) * 255).astype(np.uint8), v1_path)
        print(f"  Saved v1: {v1_path.name}")

        # Process with v2
        print("  Processing with v2...")
        v2_enhanced = v2_pipeline.enhance_image(image)

        v2_path = output_dir / 'v2' / f'{stem}_v2.png'
        save_image((np.clip(v2_enhanced, 0, 1) * 255).astype(np.uint8), v2_path)
        print(f"  Saved v2: {v2_path.name}")

        # Create comparison
        comparison = np.hstack([image, v1_enhanced, v2_enhanced])
        comparison_vis = (np.clip(comparison, 0, 1) * 255).astype(np.uint8)

        # Add labels
        h, w = image.shape[:2]
        cv2.putText(comparison_vis, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_vis, "NightSight v1", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison_vis, "NightSight v2 (Experimental)", (w * 2 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        comparison_path = output_dir / f'{stem}_comparison.png'
        save_image(comparison_vis, comparison_path)
        print(f"  Saved comparison: {comparison_path.name}")

    print("\n" + "=" * 60)
    print("README samples created successfully!")
    print("=" * 60)


if __name__ == '__main__':
    create_samples()
