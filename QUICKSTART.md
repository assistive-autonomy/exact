# EXACT - 5-Minute Quickstart

Train an inverse behavior model that generates humanoid motion programs from motion sequences in minutes.

## What It Does

Given a motion sequence (poses + actions), the model learns to generate the program specification that would produce similar motions. Uses supervised fine-tuning (SFT) with a dual loss function:
- **Cross-entropy loss**: Standard token prediction
- **Optimal transport loss**: Motion reconstruction quality
- **Structural loss**: Predicate and argument matching

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
python scripts/train_sft.py training=fast data=small
```

This will:
You should see:
- Load Qwen2.5-Coder-3B-Instruct model
- Train motion encoder prefix tokens
- Log CE loss, OT distance, and structural metrics
- Save checkpoints
- (Optional) Log to WandB

Expected time: ~5-10 minutes on GPU

## 4. Full Training Run

Once you've verified everything works, run a full training:

```bash
# Default configuration (Qwen Coder, 500 train samples, 10 epochs)
python scripts/train_sft.py

# Or with larger dataset
python scripts/train_sft.py data=large
```

## 5. Monitor Training

If WandB is enabled, you can monitor training in real-time:

```bash
# Open the URL printed at the start of training
# Or visit: https://wandb.ai/YOUR_USERNAME/exact-grpo
```

## Common Configurations

## 5. Common Configurations

### Debug Run (Fastest)
```bash
python scripts/train_sft.py \
    training=fast \
    data=small \
    wandb.mode=disabled
```

### Development Run
```bash
python scripts/train_sft.py \
    data=small \
    training.num_train_epochs=5
```

### Production Run
```bash
python scripts/train_sft.py \
    data=large \
    training.num_train_epochs=20
```

### Multi-GPU Training
```bash
accelerate config  # Run once to configure

accelerate launch scripts/train_sft.py \
    data=large
```

### Hyperparameter Sweep
```bash
python scripts/train_sft.py -m \
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
| `training.ce_weight` | 1.0 | Cross-entropy loss weight |
| `training.ot_weight` | 1.0 | Optimal transport weight |
| `training.structural_weight` | 0.5 | Structural loss weight |
| `data.num_train_samples` | 500 | Training samples |
| `wandb.mode` | online | WandB mode (online/offline/disabled) |

### Override Individual Parameters

```bash
# Change learning rate
python scripts/train_sft.py training.learning_rate=1e-5

# Change dataset size
python scripts/train_sft.py data.num_train_samples=2000

# Adjust loss weights
python scripts/train_sft.py \
    training.ce_weight=1.0 \
    training.ot_weight=1.5 \
    training.structural_weight=0.3

# Change multiple parameters
python scripts/train_sft.py \
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
### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_sft.py training.per_device_train_batch_size=1

# Or use gradient accumulation
python scripts/train_sft.py \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=16
```

### "Training too slow"
```bash
# Use FP16 mixed precision
python scripts/train_sft.py training.fp16=true

# Or disable WandB logging
python scripts/train_sft.py wandb.mode=disabled
```
```

### "WandB login failed"
```bash
# Set API key directly
export WANDB_API_KEY=your_key_here

# Or disable WandB
python scripts/train_sft.py wandb.mode=disabled
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
python scripts/train_sft.py training=fast data=small wandb.mode=disabled

# 2. Development run (30 min)
python scripts/train_sft.py data=small

# 3. Full run (1-2 hours)
python scripts/train_sft.py

# 4. Production run (4-8 hours)
python scripts/train_sft.py data=large

# 5. Multi-GPU run
accelerate launch scripts/train_sft.py data=large
```

Happy training! 🚀
