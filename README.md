# EXACT: Executable Activity Models

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Ubuntu 22.04](https://img.shields.io/badge/ubuntu-22.04-orange.svg)

## Installation

```bash
# Using uv (recommended)
uv venv
uv sync
```

## Quick Start

### 1. Generate Synthetic Training Data

```bash
uv run scripts/generate_data.py --name train --num-samples 1000
uv run scripts/generate_data.py --name eval --num-samples 200
```

### 2. Train Parser

```bash
# Train with default config
uv run scripts/train_parser.py

# Override config values (Hydra syntax)
uv run scripts/train_parser.py num_train_epochs=20 learning_rate=1e-5 wandb_mode=online
```

Training uses HuggingFace `Trainer` with standard arguments. All parameters in `configs/train.yaml`.

### 3. Activity Segmentation Baseline (Optional)

```bash
uv run scripts/esk2dlc.py --annotations-dir data/esk --output-dir data/esk_dlc
uv run scripts/exp.py
```
