# Learning and Reasoning with Executable Activity Models

![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)


## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Walktrough

### 1. Generate Training Data

```bash
uv run scripts/generate_data.py name=train num_samples=1000
uv run scripts/generate_data.py name=eval num_samples=200
```

Creates `train.hdf5` and `eval.hdf5` with motion-program pairs.

### 2. Train Parser

```bash
uv run scripts/train_parser.py
```
