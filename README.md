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

## Quick Start

### 1. Generate Training Data

First, generate training and evaluation datasets:

```bash
# Generate training data
uv run scripts/generate_data.py name=train num_samples=1000

# Generate evaluation data
uv run scripts/generate_data.py name=eval num_samples=200
```

This creates `train.hdf5` and `eval.hdf5` files containing motion-program pairs.

### 2. Train Inverse Behavior Model

Train the model to predict programs from motion sequences:

```bash
# Basic training
uv run scripts/train_ibm.py

# With custom parameters
uv run scripts/train_ibm.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    wandb.mode=disabled
```

## Configuration

The project uses Hydra for configuration management:

- `configs/train.yaml` - Training configuration (model, LoRA, hyperparameters)
- `configs/data.yaml` - Data generation configuration (program parameters)

All parameters can be overridden from the command line using Hydra syntax.

## Usage Examples

```bash
# Generate small dataset for quick testing
uv run scripts/generate_data.py name=train num_samples=50

# Train with disabled WandB logging
uv run scripts/train_ibm.py wandb.mode=disabled

# Train with custom learning rate and batch size
uv run scripts/train_ibm.py \
    training.learning_rate=1e-4 \
    training.batch_size=4

# Train with different loss weights
uv run scripts/train_ibm.py \
    training.ce_weight=1.0 \
    training.ot_weight=0.5
```





