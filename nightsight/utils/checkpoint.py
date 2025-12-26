"""Checkpoint utilities for saving and loading model states."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime
import numpy as np


def convert_to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(v) for v in obj]
    return obj


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: Union[str, Path],
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    **extra_info
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: Optional optimizer state
        epoch: Current epoch number
        path: Path to save the checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional dictionary of metrics
        config: Optional training configuration
        **extra_info: Additional info to store
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(extra_info)

    torch.save(checkpoint, path)

    # Also save a JSON summary for easy inspection
    summary = {
        "epoch": epoch,
        "timestamp": checkpoint["timestamp"],
        "metrics": convert_to_python_types(metrics) if metrics else {},
    }
    summary.update({k: convert_to_python_types(v) for k, v in extra_info.items() if isinstance(v, (int, float, str, bool, np.integer, np.floating))})

    summary_path = path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = True
) -> Tuple[Optional[torch.nn.Module], Dict[str, Any]]:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint
        model: Optional model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load tensors to
        strict: Whether to strictly enforce matching keys

    Returns:
        Tuple of (model, checkpoint_info)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        # Handle potential key mismatches from DataParallel
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "timestamp": checkpoint.get("timestamp", None),
    }

    return model, info


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "*.pth"
) -> Optional[Path]:
    """
    Get the most recent checkpoint from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]


def get_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric: str = "psnr",
    higher_is_better: bool = True
) -> Optional[Path]:
    """
    Get the best checkpoint based on a metric.

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to use for comparison
        higher_is_better: Whether higher values are better

    Returns:
        Path to the best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    best_path = None
    best_value = float("-inf") if higher_is_better else float("inf")

    for json_path in checkpoint_dir.glob("*.json"):
        try:
            with open(json_path, "r") as f:
                info = json.load(f)

            value = info.get("metrics", {}).get(metric)
            if value is None:
                continue

            is_better = (value > best_value) if higher_is_better else (value < best_value)
            if is_better:
                best_value = value
                best_path = json_path.with_suffix(".pth")

        except (json.JSONDecodeError, KeyError):
            continue

    return best_path if best_path and best_path.exists() else None


class CheckpointManager:
    """
    Manages checkpoints during training.

    Handles saving, loading, and cleanup of checkpoints with
    support for keeping best and last N checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_to_keep: int = 5,
        best_metric: str = "psnr",
        higher_is_better: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            best_metric: Metric to use for determining best checkpoint
            higher_is_better: Whether higher metric values are better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better
        self.best_value = float("-inf") if higher_is_better else float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            scheduler: Optional scheduler
            config: Optional config

        Returns:
            Path to saved checkpoint
        """
        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=path,
            scheduler=scheduler,
            metrics=metrics,
            config=config
        )

        # Check if this is the best
        current_value = metrics.get(self.best_metric)
        if current_value is not None:
            is_better = (
                (current_value > self.best_value) if self.higher_is_better
                else (current_value < self.best_value)
            )
            if is_better:
                self.best_value = current_value
                best_path = self.checkpoint_dir / "best_model.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=best_path,
                    scheduler=scheduler,
                    metrics=metrics,
                    config=config
                )

        # Cleanup old checkpoints
        self._cleanup()

        return path

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for ckpt in checkpoints[self.max_to_keep:]:
            ckpt.unlink(missing_ok=True)
            json_file = ckpt.with_suffix(".json")
            json_file.unlink(missing_ok=True)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load the latest checkpoint."""
        path = get_latest_checkpoint(self.checkpoint_dir)
        if path is None:
            return model, {"epoch": 0, "metrics": {}}
        return load_checkpoint(path, model, optimizer, scheduler, device)

    def load_best(
        self,
        model: torch.nn.Module,
        device: str = "cpu"
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load the best checkpoint."""
        path = self.checkpoint_dir / "best_model.pth"
        if not path.exists():
            path = get_best_checkpoint(
                self.checkpoint_dir,
                self.best_metric,
                self.higher_is_better
            )
        if path is None:
            return model, {"epoch": 0, "metrics": {}}
        return load_checkpoint(path, model, device=device)
