"""
Configuration file for inverse behavior model training.

This file contains all hyperparameters for training.
Modify these values to experiment with different settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data generation."""

    # Number of training samples
    num_train_samples: int = 800

    # Number of validation samples
    num_val_samples: int = 200

    # Number of timesteps per motion sequence
    num_steps: int = 100

    # Program generation parameters
    min_units: int = 1  # Min body parts per program
    max_units: int = 3  # Max body parts per program
    min_value: float = 0.1  # Min reward value
    max_value: float = 1.0  # Max reward value

    # Data loading
    batch_size: int = 16
    num_workers: int = 4

    # Optional: path to pre-generated dataset
    dataset_path: Optional[str] = None


@dataclass
class ModelConfig:
    """LLM model configuration."""

    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"  # HuggingFace model name
    motion_dim: int = (
        256  # Dimension of motion tensors (64 poses + 64 actions) * 4 values each
    )
    hidden_dim: int = 1024  # Hidden dimension for motion encoder
    num_layers: int = 6  # Number of transformer layers in motion encoder
    num_prefix_tokens: int = 32  # Number of prefix tokens to add to LLM


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Loss weights
    ce_weight: float = 0.1  # Cross-entropy loss weight
    ot_weight: float = 1.0  # Optimal transport loss weight

    # Training loop
    max_epochs: int = 10
    gradient_clip_val: float = 1.0

    # Hardware
    accelerator: str = "auto"  # "auto", "gpu", "cpu"
    devices: int = 1
    precision: str = "32"  # "32", "16-mixed", "bf16-mixed"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/"
    save_top_k: int = 3  # Save top K checkpoints
    monitor: str = "val/total_loss"  # Metric to monitor

    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Validate every N epochs


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Generation parameters
    num_beams: int = 5  # Beam search width
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-K sampling
    top_p: float = 0.95  # Nucleus sampling

    # Evaluation
    num_eval_samples: int = 10  # Number of samples to evaluate


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment name
    name: str = "inverse_bm_default"

    # Sub-configs
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        assert self.data.num_train_samples > 0
        assert self.data.num_val_samples > 0
        assert self.data.num_steps > 0
        assert self.training.ce_weight >= 0
        assert self.training.ot_weight >= 0
        assert self.training.ce_weight + self.training.ot_weight > 0


# Pre-defined experiment configurations


def get_fast_debug_config() -> ExperimentConfig:
    """Configuration for fast debugging (small model, small dataset)."""
    config = ExperimentConfig(
        name="fast_debug",
        data=DataConfig(
            num_train_samples=50,
            num_val_samples=10,
            num_steps=50,
            batch_size=4,
        ),
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            hidden_dim=256,
            num_layers=2,
        ),
        training=TrainingConfig(
            max_epochs=2,
            log_every_n_steps=5,
        ),
    )
    return config


def get_small_config() -> ExperimentConfig:
    """Configuration for small-scale experiments."""
    config = ExperimentConfig(
        name="small_scale",
        data=DataConfig(
            num_train_samples=200,
            num_val_samples=50,
            num_steps=75,
            batch_size=8,
        ),
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            hidden_dim=384,
            num_layers=3,
        ),
        training=TrainingConfig(
            max_epochs=5,
            learning_rate=5e-5,
        ),
    )
    return config


def get_standard_config() -> ExperimentConfig:
    """Standard configuration for full training."""
    config = ExperimentConfig(
        name="standard",
        data=DataConfig(
            num_train_samples=800,
            num_val_samples=200,
            num_steps=100,
            batch_size=16,
        ),
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            hidden_dim=512,
            num_layers=4,
        ),
        training=TrainingConfig(
            max_epochs=10,
            learning_rate=5e-5,
            ce_weight=0.1,
            ot_weight=1.0,
        ),
    )
    return config


def get_large_config() -> ExperimentConfig:
    """Configuration for large-scale training."""
    config = ExperimentConfig(
        name="large_scale",
        data=DataConfig(
            num_train_samples=2000,
            num_val_samples=500,
            num_steps=150,
            batch_size=32,
        ),
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            hidden_dim=768,
            num_layers=6,
        ),
        training=TrainingConfig(
            max_epochs=20,
            learning_rate=3e-5,
            warmup_steps=2000,
            ce_weight=0.1,
            ot_weight=1.0,
        ),
    )
    return config


def get_syntax_focused_config() -> ExperimentConfig:
    """Configuration emphasizing syntactic correctness."""
    config = ExperimentConfig(
        name="syntax_focused",
        data=DataConfig(
            num_train_samples=800,
            num_val_samples=200,
            num_steps=100,
        ),
        training=TrainingConfig(
            max_epochs=10,
            ce_weight=1.0,  # High CE weight
            ot_weight=0.1,  # Low OT weight
        ),
    )
    return config


def get_semantic_focused_config() -> ExperimentConfig:
    """Configuration emphasizing semantic correctness."""
    config = ExperimentConfig(
        name="semantic_focused",
        data=DataConfig(
            num_train_samples=800,
            num_val_samples=200,
            num_steps=100,
        ),
        training=TrainingConfig(
            max_epochs=10,
            ce_weight=0.05,  # Low CE weight
            ot_weight=2.0,  # High OT weight
        ),
    )
    return config


# Dictionary of available configs
CONFIGS = {
    "debug": get_fast_debug_config,
    "small": get_small_config,
    "standard": get_standard_config,
    "large": get_large_config,
    "syntax": get_syntax_focused_config,
    "semantic": get_semantic_focused_config,
}


def get_config(name: str = "standard") -> ExperimentConfig:
    """Get a pre-defined configuration by name.

    Args:
        name: Configuration name. Options:
            - "debug": Fast debugging
            - "small": Small-scale experiments
            - "standard": Standard training (default)
            - "large": Large-scale training
            - "syntax": Focus on syntactic correctness
            - "semantic": Focus on semantic correctness

    Returns:
        ExperimentConfig instance
    """
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")

    return CONFIGS[name]()


if __name__ == "__main__":
    # Example: Print all configurations
    print("Available Configurations:\n")

    for name, config_fn in CONFIGS.items():
        config = config_fn()
        print(f"=== {name.upper()} ===")
        print(f"Train samples: {config.data.num_train_samples}")
        print(f"Val samples: {config.data.num_val_samples}")
        print(f"Model: {config.model.model_name}")
        print(f"CE weight: {config.training.ce_weight}")
        print(f"OT weight: {config.training.ot_weight}")
        print()
