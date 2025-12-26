#!/usr/bin/env python3
"""
Populate TensorBoard with Historical Training Data

Reads checkpoint JSON files and populates TensorBoard logs.

Usage:
    python scripts/populate_tensorboard.py --checkpoint-dir outputs/nightsight/checkpoints
"""

import argparse
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def populate_tensorboard(checkpoint_dir: Path, output_dir: Path):
    """Populate TensorBoard with data from checkpoint JSON files."""
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    writer = SummaryWriter(output_dir)

    # Collect all checkpoint JSON files
    json_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.json"))

    if not json_files:
        print("No checkpoint JSON files found.")
        writer.close()
        return

    print(f"Found {len(json_files)} checkpoint files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            epoch = data.get('epoch', -1)
            metrics = data.get('metrics', {})

            if epoch >= 0 and metrics:
                # Log metrics
                if 'loss' in metrics:
                    writer.add_scalar('Loss/train', metrics['loss'], epoch)
                if 'psnr' in metrics:
                    writer.add_scalar('Metrics/PSNR', metrics['psnr'], epoch)
                if 'ssim' in metrics:
                    writer.add_scalar('Metrics/SSIM', metrics['ssim'], epoch)

                print(f"Logged Epoch {epoch}: Loss={metrics.get('loss', 'N/A'):.4f}, "
                      f"PSNR={metrics.get('psnr', 'N/A'):.2f}, "
                      f"SSIM={metrics.get('ssim', 'N/A'):.4f}")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {json_file}: {e}")
            continue

    writer.close()
    print(f"\nTensorBoard logs written to: {output_dir}")
    print(f"Launch with: tensorboard --logdir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Populate TensorBoard with historical training data")

    parser.add_argument("--checkpoint-dir", type=str, default="outputs/nightsight/checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="outputs/nightsight/tensorboard",
                        help="TensorBoard log directory")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    populate_tensorboard(checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()
