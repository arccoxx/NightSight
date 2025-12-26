#!/usr/bin/env python3
"""
Training Monitor Script

Real-time monitoring of NightSightNet training progress.

Usage:
    python scripts/monitor_training.py --log-file path/to/output.log
    python scripts/monitor_training.py --checkpoint-dir outputs/nightsight/checkpoints
"""

import argparse
import json
import time
from pathlib import Path
import re
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_log_file(log_path: Path) -> List[Dict]:
    """Parse training log to extract metrics."""
    metrics = []

    if not log_path.exists():
        return metrics

    with open(log_path, 'r') as f:
        content = f.read()

    # Pattern to match epoch summaries
    pattern = r'Epoch (\d+): Train Loss: ([\d.]+), Val PSNR: ([\d.]+), Val SSIM: ([\d.]+)'

    for match in re.finditer(pattern, content):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_psnr = float(match.group(3))
        val_ssim = float(match.group(4))

        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim
        })

    return metrics


def parse_checkpoint_dir(ckpt_dir: Path) -> List[Dict]:
    """Parse checkpoint JSON files to extract metrics."""
    metrics = []

    if not ckpt_dir.exists():
        return metrics

    for json_file in sorted(ckpt_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            epoch = data.get('epoch', -1)
            metric_data = data.get('metrics', {})

            if epoch >= 0 and metric_data:
                metrics.append({
                    'epoch': epoch,
                    'train_loss': metric_data.get('loss', 0),
                    'val_psnr': metric_data.get('psnr', 0),
                    'val_ssim': metric_data.get('ssim', 0)
                })
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(metrics, key=lambda x: x['epoch'])


def print_summary(metrics: List[Dict]):
    """Print training summary."""
    if not metrics:
        print("No training metrics found yet.")
        return

    latest = metrics[-1]
    best_psnr = max(metrics, key=lambda x: x['val_psnr'])
    best_ssim = max(metrics, key=lambda x: x['val_ssim'])

    print("=" * 60)
    print("NIGHTSIGHTNET TRAINING MONITOR")
    print("=" * 60)
    print(f"\nTotal Epochs Completed: {len(metrics)}")
    print(f"\nLatest Metrics (Epoch {latest['epoch']}):")
    print(f"  Train Loss: {latest['train_loss']:.4f}")
    print(f"  Val PSNR:   {latest['val_psnr']:.2f} dB")
    print(f"  Val SSIM:   {latest['val_ssim']:.4f}")

    print(f"\nBest Metrics:")
    print(f"  Best PSNR: {best_psnr['val_psnr']:.2f} dB (Epoch {best_psnr['epoch']})")
    print(f"  Best SSIM: {best_ssim['val_ssim']:.4f} (Epoch {best_ssim['epoch']})")

    # Training progress
    if len(metrics) >= 2:
        first = metrics[0]
        print(f"\nProgress:")
        print(f"  Loss:  {first['train_loss']:.4f} -> {latest['train_loss']:.4f} "
              f"({((first['train_loss'] - latest['train_loss']) / first['train_loss'] * 100):.1f}% improvement)")
        print(f"  PSNR:  {first['val_psnr']:.2f} -> {latest['val_psnr']:.2f} dB "
              f"(+{latest['val_psnr'] - first['val_psnr']:.2f} dB)")
        print(f"  SSIM:  {first['val_ssim']:.4f} -> {latest['val_ssim']:.4f} "
              f"(+{latest['val_ssim'] - first['val_ssim']:.4f})")

    print("=" * 60)


def plot_metrics(metrics: List[Dict], save_path: Optional[Path] = None):
    """Plot training metrics."""
    if not metrics:
        return

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_psnr = [m['val_psnr'] for m in metrics]
    val_ssim = [m['val_ssim'] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training Loss
    axes[0].plot(epochs, train_loss, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Validation PSNR
    axes[1].plot(epochs, val_psnr, 'g-', linewidth=2)
    axes[1].axhline(y=max(val_psnr), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(val_psnr):.2f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Validation PSNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Validation SSIM
    axes[2].plot(epochs, val_ssim, 'r-', linewidth=2)
    axes[2].axhline(y=max(val_ssim), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(val_ssim):.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Validation SSIM')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def monitor_live(source_path: Path, is_checkpoint_dir: bool = False, interval: int = 10):
    """Live monitoring with auto-refresh."""
    print("Starting live training monitor...")
    print(f"Monitoring: {source_path}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            if is_checkpoint_dir:
                metrics = parse_checkpoint_dir(source_path)
            else:
                metrics = parse_log_file(source_path)

            # Clear screen (works on most terminals)
            print("\033[H\033[J", end="")

            print_summary(metrics)

            print(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Next refresh in {interval}s... (Ctrl+C to stop)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor NightSightNet training progress")

    parser.add_argument("--log-file", type=str,
                        help="Path to training log file")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/nightsight/checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plot")
    parser.add_argument("--plot-output", type=str, default="training_progress.png",
                        help="Output path for plot")
    parser.add_argument("--live", action="store_true",
                        help="Enable live monitoring mode")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds for live mode")

    args = parser.parse_args()

    # Determine source
    if args.log_file:
        source_path = Path(args.log_file)
        is_checkpoint_dir = False
    else:
        source_path = Path(args.checkpoint_dir)
        is_checkpoint_dir = True

    # Live monitoring mode
    if args.live:
        monitor_live(source_path, is_checkpoint_dir, args.interval)
        return

    # One-time check
    if is_checkpoint_dir:
        metrics = parse_checkpoint_dir(source_path)
    else:
        metrics = parse_log_file(source_path)

    print_summary(metrics)

    # Generate plot if requested
    if args.plot and metrics:
        plot_output = Path(args.plot_output)
        plot_metrics(metrics, plot_output)


if __name__ == "__main__":
    main()
