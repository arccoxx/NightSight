#!/usr/bin/env python3
"""
Visualize validation results from trained model.

Usage:
    python scripts/visualize_results.py --checkpoint outputs/nightsight/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.models.hybrid import NightSightNet
from nightsight.data.datasets import LOLDataset
from nightsight.utils.checkpoint import load_checkpoint


def tensor_to_image(tensor):
    """Convert tensor to numpy image."""
    img = tensor.cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img


def visualize_samples(model, dataset, device, num_samples=5, save_dir=None):
    """Visualize model predictions on validation samples."""
    model.eval()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            # Get sample
            low, high = dataset[i]
            low_input = low.unsqueeze(0).to(device)

            # Enhance
            enhanced = model(low_input)

            # Convert to images
            low_img = tensor_to_image(low)
            enhanced_img = tensor_to_image(enhanced[0])
            high_img = tensor_to_image(high)

            # Plot
            axes[i, 0].imshow(low_img)
            axes[i, 0].set_title(f'Sample {i+1}: Low-light Input', fontsize=12)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(enhanced_img)
            axes[i, 1].set_title(f'Sample {i+1}: Enhanced Output', fontsize=12)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(high_img)
            axes[i, 2].set_title(f'Sample {i+1}: Ground Truth', fontsize=12)
            axes[i, 2].axis('off')

            # Save individual images if requested
            if save_dir:
                Image.fromarray((low_img * 255).astype(np.uint8)).save(
                    save_dir / f"sample_{i+1}_input.png"
                )
                Image.fromarray((enhanced_img * 255).astype(np.uint8)).save(
                    save_dir / f"sample_{i+1}_enhanced.png"
                )
                Image.fromarray((high_img * 255).astype(np.uint8)).save(
                    save_dir / f"sample_{i+1}_ground_truth.png"
                )

    plt.tight_layout()

    if save_dir:
        output_path = save_dir / "comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison saved to: {output_path}")

        # Also save individual images info
        print(f"Individual images saved to: {save_dir}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize validation results")

    parser.add_argument("--checkpoint", type=str,
                        default="outputs/nightsight/checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/LOL",
                        help="Path to LOL dataset")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--save-dir", type=str, default="outputs/nightsight/visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")

    args = parser.parse_args()

    # Get device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    model = NightSightNet()
    model, info = load_checkpoint(args.checkpoint, model, device=str(device))
    model = model.to(device)

    print(f"Loaded model from epoch {info['epoch']}")
    print(f"Metrics: PSNR={info['metrics'].get('psnr', 'N/A'):.2f}, "
          f"SSIM={info['metrics'].get('ssim', 'N/A'):.4f}")

    # Load validation dataset
    print(f"\nLoading validation dataset from: {args.data_dir}")
    dataset = LOLDataset(args.data_dir, split="test", transform=None)
    print(f"Found {len(dataset)} validation samples")

    # Visualize
    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    visualize_samples(model, dataset, device, args.num_samples, args.save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
