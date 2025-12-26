#!/usr/bin/env python3
"""
NightSight Training Script

Train various models for low-light image enhancement.

Usage:
    python scripts/train.py --config configs/train_nightsight.yaml
    python scripts/train.py --model zerodce --data-dir data/LOL
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nightsight.config import Config, get_default_config
from nightsight.data.datasets import PairedDataset, SyntheticDataset, LOLDataset
from nightsight.data.transforms import get_train_transforms, get_val_transforms
from nightsight.losses.color import CombinedLoss
from nightsight.metrics import MetricTracker
from nightsight.utils.checkpoint import CheckpointManager
from nightsight.core.registry import ModelRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Train NightSight models")

    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, default="nightsight",
                        choices=["zerodce", "retinexformer", "nightsight", "unet", "swinir", "diffusion"],
                        help="Model to train")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(model_name: str, config: Config):
    """Create model based on name."""
    if model_name == "zerodce":
        from nightsight.models.zerodce import ZeroDCE
        return ZeroDCE()
    elif model_name == "retinexformer":
        from nightsight.models.retinexformer import Retinexformer
        return Retinexformer()
    elif model_name == "nightsight":
        from nightsight.models.hybrid import NightSightNet
        return NightSightNet()
    elif model_name == "unet":
        from nightsight.models.unet import UNet
        return UNet()
    elif model_name == "swinir":
        from nightsight.models.swinir import SwinIR
        return SwinIR()
    elif model_name == "diffusion":
        from nightsight.models.diffusion import DDPM
        return DDPM()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_datasets(data_dir: str, config: Config):
    """Create train and validation datasets."""
    data_dir = Path(data_dir)

    train_transform = get_train_transforms(config.data.patch_size)
    val_transform = get_val_transforms(config.data.patch_size)

    # Try LOL dataset first
    lol_path = data_dir / "LOL"
    if lol_path.exists():
        train_dataset = LOLDataset(str(lol_path), split="train", transform=train_transform)
        val_dataset = LOLDataset(str(lol_path), split="test", transform=val_transform)
    else:
        # Fall back to paired dataset
        train_low = data_dir / "train" / "low"
        train_high = data_dir / "train" / "high"
        val_low = data_dir / "val" / "low"
        val_high = data_dir / "val" / "high"

        if train_low.exists() and train_high.exists():
            train_dataset = PairedDataset(str(train_low), str(train_high), transform=train_transform)
            val_dataset = PairedDataset(str(val_low), str(val_high), transform=val_transform)
        else:
            # Fall back to synthetic dataset
            clean_dir = data_dir / "clean"
            if clean_dir.exists():
                train_dataset = SyntheticDataset(str(clean_dir), transform=train_transform)
                val_dataset = SyntheticDataset(str(clean_dir), transform=val_transform)
            else:
                raise ValueError(f"No valid dataset found in {data_dir}")

    return train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            low, high = batch
            low = low.to(device)
            high = high.to(device)
        else:
            low = batch.to(device)
            high = low  # Self-supervised

        optimizer.zero_grad()

        # Forward
        enhanced = model(low)

        # Compute loss
        if isinstance(criterion, CombinedLoss):
            losses = criterion(enhanced, high)
            loss = losses['total']
        else:
            loss = criterion(enhanced, high)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": loss.item()})

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()

    tracker = MetricTracker(['psnr', 'ssim'])

    for batch in tqdm(dataloader, desc="Validating"):
        if isinstance(batch, (list, tuple)):
            low, high = batch
            low = low.to(device)
            high = high.to(device)
        else:
            continue  # Skip unpaired data for validation

        enhanced = model(low)

        # Update metrics
        for i in range(enhanced.shape[0]):
            tracker.update(enhanced[i], high[i])

    return tracker.get_averages()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config(args.model)
        config.training.epochs = args.epochs
        config.training.learning_rate = args.lr
        config.data.batch_size = args.batch_size

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / "config.yaml")

    # Create model
    model = create_model(args.model, config)
    model = model.to(device)
    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(args.data_dir, config)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs
    )

    # Create loss
    criterion = CombinedLoss(
        l1_weight=config.training.l1_weight,
        perceptual_weight=config.training.perceptual_weight,
        ssim_weight=config.training.ssim_weight
    )

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        output_dir / "checkpoints",
        max_to_keep=5,
        best_metric="psnr"
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        model, info = ckpt_manager.load_latest(model, optimizer, scheduler, device)
        start_epoch = info.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_psnr = 0

    for epoch in range(start_epoch, config.training.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
              f"Val PSNR: {val_metrics['psnr']:.2f}, Val SSIM: {val_metrics['ssim']:.4f}")

        # Save checkpoint
        metrics = {**train_metrics, **val_metrics}
        ckpt_manager.save(model, optimizer, epoch, metrics, scheduler)

        # Track best
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            print(f"New best PSNR: {best_psnr:.2f}")

    print(f"Training complete! Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
