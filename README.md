# Executable Activity Models

Learning and reasoning with executable activity models for motion generation and inverse behavior modeling using GRPO.

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Running Experiments

**Basic Training:**
```bash
python scripts/train_sft.py
```

**Quick Debug Run:**
```bash
python scripts/train_sft.py training=fast data=small wandb.mode=disabled
```

**Override Parameters:**
```bash
python scripts/train_sft.py training.learning_rate=1e-5 data.num_train_samples=1000
```

**Use Different Config Presets:**
```bash
python scripts/train_sft.py training=fast data=large
```


### Examples

```bash
# Disable WandB logging
python scripts/train_sft.py wandb.mode=disabled

# Set WandB project
python scripts/train_sft.py wandb.project=my-project wandb.entity=my-team

# Custom training configuration
python scripts/train_sft.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    training.per_device_train_batch_size=8 \
    training.ce_weight=1.0 \
    training.ot_weight=1.5
```





