# EXACT: Executable Activity Models

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Ubuntu 22.04](https://img.shields.io/badge/ubuntu-22.04-orange.svg)

Learning inverse motion models that map motion sequences to executable activity programs using supervised fine-tuning with PEFT/LoRA. The system supports both synthetic motion data generation and real-world pose data from the ESK dataset.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Training Data

```bash
# Training data (1000 motion-program pairs)
uv run scripts/generate_data.py name=train num_samples=1000

# Evaluation data (200 pairs)
uv run scripts/generate_data.py name=eval num_samples=200
```

Creates `train.hdf5` and `eval.hdf5` with motion-program pairs from synthetic environment.

### 2. Train Parser

```bash
# Basic training (from synthetic data)
uv run scripts/train_parser.py

# With custom parameters
uv run scripts/train_parser.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    wandb.mode=disabled
```

Trains LoRA-adapted language model to predict activity programs from motion sequences.

### 3. Train on Real ESK Pose Data (Optional)

```bash
# Convert ESK poses to DLC2Action format
uv run exps/esk2dlc.py --annotations-dir data/esk --output-dir data/esk_30

# Run DLC2Action training pipeline
uv run exps/workflow.py
```
