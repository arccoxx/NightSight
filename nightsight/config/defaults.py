"""Default configurations for NightSight models and training."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "retinexformer"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    num_blocks: int = 4
    num_heads: int = 8
    window_size: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pretrained: Optional[str] = None

    # Model-specific configs
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""

    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"

    patch_size: int = 256
    batch_size: int = 8
    num_workers: int = 4

    # Augmentation
    random_crop: bool = True
    random_flip: bool = True
    random_rotation: bool = True

    # For paired datasets
    paired: bool = True
    low_suffix: str = "_low"
    high_suffix: str = "_high"


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, step, plateau

    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.1
    color_weight: float = 0.05

    # Training settings
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class Config:
    """Complete configuration for NightSight."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # General settings
    seed: int = 42
    device: str = "auto"
    debug: bool = False

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "seed": self.seed,
            "device": self.device,
            "debug": self.debug,
        }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "auto"),
            debug=config_dict.get("debug", False),
        )


def get_default_config(model_name: str = "retinexformer") -> Config:
    """
    Get default configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Config instance with model-specific defaults
    """
    config = Config()
    config.model.name = model_name

    if model_name == "zerodce":
        config.model.base_channels = 32
        config.model.num_blocks = 8
        config.training.learning_rate = 1e-4

    elif model_name == "retinexformer":
        config.model.base_channels = 32
        config.model.num_blocks = 4
        config.model.num_heads = 8
        config.training.learning_rate = 2e-4

    elif model_name == "diffusion":
        config.model.base_channels = 64
        config.model.num_blocks = 6
        config.training.learning_rate = 1e-4
        config.training.epochs = 500

    elif model_name == "unet":
        config.model.base_channels = 64
        config.model.num_blocks = 4
        config.training.learning_rate = 1e-4

    elif model_name == "swinir":
        config.model.base_channels = 48
        config.model.num_heads = 6
        config.model.window_size = 8
        config.training.learning_rate = 2e-4

    elif model_name == "hybrid":
        config.model.extra = {
            "use_retinex": True,
            "use_temporal": True,
            "temporal_frames": 5,
        }
        config.training.learning_rate = 1e-4

    return config


# Preset configurations for common use cases
PRESETS = {
    "fast": Config(
        model=ModelConfig(name="zerodce", base_channels=16, num_blocks=4),
        training=TrainingConfig(epochs=100, learning_rate=2e-4),
    ),
    "balanced": Config(
        model=ModelConfig(name="retinexformer", base_channels=32, num_blocks=4),
        training=TrainingConfig(epochs=200, learning_rate=1e-4),
    ),
    "quality": Config(
        model=ModelConfig(name="diffusion", base_channels=64, num_blocks=6),
        training=TrainingConfig(epochs=500, learning_rate=5e-5),
    ),
    "realtime": Config(
        model=ModelConfig(name="zerodce", base_channels=8, num_blocks=3),
        training=TrainingConfig(epochs=50, learning_rate=3e-4),
    ),
}
