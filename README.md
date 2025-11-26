# EXACT: Executable Activity Models

Learning inverse motion models that map motion sequences to reward programs using supervised fine-tuning with PEFT/LoRA.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Generate Training Data

```bash
# Training data
uv run scripts/generate_data.py name=train num_samples=1000

# Evaluation data  
uv run scripts/generate_data.py name=eval num_samples=200
```

Creates `train.hdf5` and `eval.hdf5` with motion-program pairs.

### 2. Train Inverse Motion Model

```bash
# Basic training
uv run scripts/train_ibm.py

# With custom parameters
uv run scripts/train_ibm.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    wandb.mode=disabled
```

## Architecture

- **Motion Model**: Pre-trained MetaMotivo model that generates motion from reward programs
- **Inverse Motion Model**: LLM fine-tuned with LoRA + motion encoder to predict programs from motion
- **Reward Programs**: Time-indexed predicates specifying body part positions over intervals

## Reward Program Grammar

Programs consist of time-indexed motions separated by commas:

```
[start,end]sensor*sensor*...,[start,end]sensor*sensor*...
```

Each sensor specifies a body part, axis, and target value:
```
body_part.axis(value)
```

**Example:**
```
[0,500]head.z(1.4)*lhand.x(0.5),[500,1000]pelvis.y(0.8)
```

This means:
- Timesteps 0-500: head at z=1.4m, left hand at x=0.5m
- Timesteps 500-1000: pelvis at y=0.8m

**Components:**
- **Body parts** (23): `pelvis`, `torso`, `spine`, `chest`, `neck`, `head`, `lhip`, `lknee`, `lankle`, `ltoe`, `rhip`, `rknee`, `rankle`, `rtoe`, `lthorax`, `lshoulder`, `lelbow`, `lwrist`, `lhand`, `rthorax`, `rshoulder`, `relbow`, `rwrist`, `rhand`
- **Axes**: `x`, `y`, `z`
- **Values**: floating point with decimal (e.g., `1.4`, `-0.5`)

### Tolerance Function

Reward uses Gaussian tolerance: $r = \exp\left(-\frac{(h - t)^2}{2\sigma^2}\right)$ where $\sigma = 1.0$

## Configuration

- `configs/train.yaml` - Training config (model, LoRA, motion encoder)
- `configs/data.yaml` - Data generation config (intervals, predicates, body parts)

Override via command line:
```bash
uv run scripts/generate_data.py num_samples=500 num_intervals=3
uv run scripts/train_ibm.py training.learning_rate=1e-4
```
