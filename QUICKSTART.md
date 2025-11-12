# Quick Start Guide

Get up and running with EXACT GRPO training in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd exact

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## 2. Setup WandB (Optional but Recommended)

```bash
# Login to WandB
wandb login

# Or run the setup script
python scripts/setup_wandb.py
```

To skip WandB, add `wandb.mode=disabled` to any training command.

## 3. Quick Test Run

Run a fast training loop with small dataset to verify everything works:

```bash
python scripts/train_grpo.py training=fast data=small
```

This will:
You should see:
- Load Qwen2.5-Coder-3B-Instruct model
- Train motion encoder prefix tokens
- Log rewards, OT distance, and validity
- Save checkpoints
- (Optional) Log to WandB

Expected time: ~5-10 minutes on GPU

## 4. Full Training Run

Once you've verified everything works, run a full training:

```bash
# Default configuration (Qwen Coder, 500 train samples, 10 epochs)
python scripts/train_grpo.py

# Or with larger dataset
python scripts/train_grpo.py data=large
```

## 5. Monitor Training

If WandB is enabled, you can monitor training in real-time:

```bash
# Open the URL printed at the start of training
# Or visit: https://wandb.ai/YOUR_USERNAME/exact-grpo
```

## Common Configurations

### Debug Run (Fastest)
```bash
python scripts/train_grpo.py \
    training=fast \
    data=small \
    wandb.mode=disabled
```

### Development Run
```bash
python scripts/train_grpo.py \
    data=small \
    training.num_train_epochs=5
```

### Production Run
```bash
python scripts/train_grpo.py \
    data=large \
    training.num_train_epochs=20
```

### Multi-GPU Training
```bash
accelerate config  # Run once to configure

accelerate launch scripts/train_grpo.py \
    data=large
```

### Hyperparameter Sweep
```bash
python scripts/train_grpo.py -m \
    training.learning_rate=1e-6,5e-6,1e-5 \
    training.ot_weight=0.5,1.0,2.0
```

## Configuration Overview

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | qwen-coder | Model (Qwen2.5-Coder-3B-Instruct) |
| `training` | default | Training config preset |
| `data` | default | Data generation config |
| `training.learning_rate` | 5e-6 | Learning rate |
| `training.num_train_epochs` | 10 | Number of epochs |
| `training.ot_weight` | 1.0 | Optimal transport weight |
| `data.num_train_samples` | 500 | Training samples |
| `wandb.mode` | online | WandB mode (online/offline/disabled) |

### Override Examples

```bash
# Change learning rate
python scripts/train_grpo.py training.learning_rate=1e-5

# Change dataset size
python scripts/train_grpo.py data.num_train_samples=2000

# Change multiple parameters
python scripts/train_grpo.py \
    training.learning_rate=3e-6 \
    data.num_train_samples=1000 \
    wandb.project=my-project
```

## Output Structure

After training, you'll find:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml          # Full resolved config
        │   └── overrides.yaml       # CLI overrides used
        ├── checkpoints/
        │   └── checkpoint-*/        # Model checkpoints
        └── wandb/                   # WandB logs (if enabled)
```

## Troubleshooting

### "Import error: hydra not found"
```bash
uv sync  # or pip install hydra-core
```

### "Import error: wandb not found"
```bash
uv sync  # or pip install wandb
```

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_grpo.py training.per_device_train_batch_size=1

# Or use gradient accumulation
python scripts/train_grpo.py \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=16
```

### "WandB login failed"
```bash
# Set API key directly
export WANDB_API_KEY=your_key_here

# Or disable WandB
python scripts/train_grpo.py wandb.mode=disabled
```

## Next Steps

1. **Customize Configuration**: Edit `configs/config.yaml` or create your own in `configs/experiment/`
2. **Run Sweeps**: Use `-m` flag for hyperparameter optimization
3. **Scale Up**: Use `accelerate` for multi-GPU training
4. **Monitor**: Check WandB dashboard for training insights
5. **Analyze**: Load checkpoints and evaluate on new data

## Getting Help

- Configuration guide: `configs/README.md`
- Full documentation: `README.md`
- Changelog: `CHANGELOG.md`

## Example Workflow

```bash
# 1. Quick test (5 min)
python scripts/train_grpo.py training=fast data=small wandb.mode=disabled

# 2. Development run (30 min)
python scripts/train_grpo.py data=small

# 3. Full run (1-2 hours)
python scripts/train_grpo.py

# 4. Production run (4-8 hours)
python scripts/train_grpo.py data=large

# 5. Multi-GPU run
accelerate launch scripts/train_grpo.py data=large
```

Happy training! 🚀
