"""
Training script for Zero-DCE++ module in NightSight v2.

Trains the low-light enhancement module using zero-reference learning.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from nightsight.v2.modules.zerodce_plus import (
    DCENet,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    ColorConstancyLoss,
    IlluminationSmoothnessLoss
)
from nightsight.utils.io import load_image


class DarkImageDataset(Dataset):
    """Dataset of dark images for Zero-DCE++ training."""

    def __init__(self, image_dir, crop_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size

        # Get all images
        self.image_paths = list(self.image_dir.glob('*.jpg'))
        self.image_paths += list(self.image_dir.glob('*.png'))

        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = load_image(str(self.image_paths[idx]), dtype='float32')

        # Random crop
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if h > crop_h and w > crop_w:
            top = np.random.randint(0, h - crop_h)
            left = np.random.randint(0, w - crop_w)
            img = img[top:top+crop_h, left:left+crop_w]

        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img


def train_epoch(model, dataloader, optimizer, losses, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        images = batch.to(device)

        # Forward
        curves = model(images)
        enhanced = model.apply_curve(images, curves)

        # Compute losses
        loss_spatial = losses['spatial'](enhanced)
        loss_exposure = losses['exposure'](enhanced)
        loss_color = losses['color'](enhanced)
        loss_illum = losses['illumination'](curves)

        # Total loss
        loss = (
            loss_spatial +
            10.0 * loss_exposure +
            5.0 * loss_color +
            200.0 * loss_illum
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train Zero-DCE++ for NightSight v2')

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dark images')
    parser.add_argument('--output-dir', type=str, default='outputs/zerodce_v2',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Dataset
    print("Loading dataset...")
    dataset = DarkImageDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print("Creating model...")
    model = DCENet(num_iterations=8)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Losses
    losses = {
        'spatial': SpatialConsistencyLoss(),
        'exposure': ExposureControlLoss(),
        'color': ColorConstancyLoss(),
        'illumination': IlluminationSmoothnessLoss()
    }

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Training loop
    print("Starting training...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, losses, device, epoch)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                output_dir / 'zerodce_best.pth'
            )
            print(f"Saved best model (loss: {best_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                output_dir / f'zerodce_epoch_{epoch}.pth'
            )

    # Save final model
    torch.save(model.state_dict(), output_dir / 'zerodce_final.pth')
    print("Training complete!")

    writer.close()


if __name__ == '__main__':
    main()
