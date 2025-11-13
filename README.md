# Executable Activity Models

Learning and reasoning with executable activity models for motion generation and inverse behavior modeling using supervised fine-tuning with PEFT/LoRA.

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Running Experiments

**Basic Training (default config):**
```bash
uv run scripts/train_sft.py
```

**Quick Debug Run (small dataset):**
```bash
uv run scripts/train_sft.py --config-name config_small wandb.mode=disabled
```

**Fast Training (fewer epochs, more frequent evals):**
```bash
uv run scripts/train_sft.py --config-name config_fast wandb.mode=disabled
```

**Override Parameters:**
```bash
uv run scripts/train_sft.py training.learning_rate=1e-5 data.num_train_samples=1000
```

### Configuration

All configuration is now in a single file: `configs/config.yaml`

There are also two variant configs:
- `configs/config_small.yaml` - Small dataset for quick testing (50 train / 10 val samples)
- `configs/config_fast.yaml` - Fast training with frequent evaluation (2 epochs, eval every 10 steps)

### Examples

```bash
# Disable WandB logging
uv run scripts/train_sft.py wandb.mode=disabled

# Set WandB project
uv run scripts/train_sft.py wandb.project=my-project wandb.entity=my-team

# Custom training configuration
uv run scripts/train_sft.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    training.per_device_train_batch_size=8 \
    training.ce_weight=1.0 \
    training.ot_weight=1.5

# Use small dataset variant
uv run scripts/train_sft.py --config-name config_small

# Use fast training variant
uv run scripts/train_sft.py --config-name config_fast

# Combine variant with overrides
uv run scripts/train_sft.py --config-name config_small training.num_train_epochs=5
```





