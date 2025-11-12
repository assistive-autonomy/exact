# Configuration Guide

This directory contains Hydra configuration files for training the inverse behavior model with GRPO.

## Structure

```
configs/
├── config.yaml           # Main configuration file
├── model/               # Model configuration
│   └── qwen-coder.yaml # Qwen2.5-Coder-3B
├── training/           # Training hyperparameters
│   ├── default.yaml    # Standard training config
│   └── fast.yaml       # Fast training for debugging
└── data/               # Data generation configs
    ├── default.yaml    # Standard dataset size
    ├── small.yaml      # Small dataset for debugging
    └── large.yaml      # Large dataset for full training
```

## Usage

### Basic Usage

Run with default configuration:
```bash
python scripts/train_grpo.py
```

### Override Parameters

Override specific parameters from command line:
```bash
python scripts/train_grpo.py training.learning_rate=1e-5 data.num_train_samples=1000
```

### Use Different Config Presets

Use different training or data configurations:
```bash
# Use fast training with small dataset
python scripts/train_grpo.py training=fast data=small

# Use large dataset with default training
python scripts/train_grpo.py data=large
```

### Multi-GPU Training

With Accelerate:
```bash
accelerate launch scripts/train_grpo.py
```

### WandB Configuration

Enable/disable WandB logging:
```bash
# Disable WandB
python scripts/train_grpo.py wandb.mode=disabled

# Set WandB project and entity
python scripts/train_grpo.py wandb.project=my-project wandb.entity=my-team

# Run offline (sync later with `wandb sync`)
python scripts/train_grpo.py wandb.mode=offline
```

### Hydra Sweeps

Run hyperparameter sweeps:
```bash
python scripts/train_grpo.py -m \
    training.learning_rate=1e-6,5e-6,1e-5 \
    training.ot_weight=0.5,1.0,2.0
```

## Configuration Options

### Main Config (`config.yaml`)

- `device`: Device to use (null for auto-detect, "cuda", or "cpu")
- `seed`: Random seed for reproducibility
- `output_dir`: Directory for checkpoints
- `behavior_model`: Settings for the MetaMotivo behavior model
- `wandb`: WandB logging configuration

### Model Config (`model/*.yaml`)

- `name`: HuggingFace model name
- `type`: Model type (causal_lm)
- `motion_encoder`: Configuration for motion encoder
  - `motion_dim`: Input dimension (256)
  - `hidden_dim`: Hidden layer dimension
  - `num_layers`: Number of transformer layers
  - `num_prefix_tokens`: Number of prefix tokens for LLM

### Training Config (`training/*.yaml`)

- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `max_grad_norm`: Gradient clipping threshold
- `warmup_steps`: Number of warmup steps
- `num_generations`: GRPO generations per prompt
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `ot_weight`: Weight for optimal transport reward
- `validity_weight`: Weight for program validity bonus
- `logging_steps`: Log every N steps
- `save_strategy`: When to save checkpoints
- `fp16`: Enable mixed precision training

### Data Config (`data/*.yaml`)

- `num_train_samples`: Number of training samples
- `num_val_samples`: Number of validation samples
- `num_steps`: Timesteps per motion sequence
- `program_generation`: Program generation parameters
  - `min_units`: Minimum program units
  - `max_units`: Maximum program units
  - `min_value`: Minimum parameter value
  - `max_value`: Maximum parameter value

## Examples

### Quick Debug Run
```bash
python scripts/train_grpo.py training=fast data=small wandb.mode=disabled
```

### Full Training Run
```bash
python scripts/train_grpo.py data=large wandb.project=my-project
```

### Hyperparameter Search
```bash
python scripts/train_grpo.py -m \
    training.learning_rate=1e-6,5e-6 \
    training.ot_weight=0.5,1.0,2.0
```

## Output Structure

When you run training, Hydra creates the following structure:
```
outputs/
└── YYYY-MM-DD/          # Date of run
    └── HH-MM-SS/        # Time of run
        ├── .hydra/      # Hydra config files
        ├── checkpoints/ # Model checkpoints
        └── wandb/       # WandB logs (if enabled)
```

For multi-run sweeps:
```
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── 0/           # First run
        ├── 1/           # Second run
        └── ...
```
