# Executable Activity Models

Learning and reasoning with executable activity models for motion generation and inverse behavior modeling using GRPO.

## Features

- **Behavior Model**: Generate motions from program specifications
- **Program Generator**: Automatically generate diverse program specifications
- **Inverse Behavior Model**: Train LLMs to generate programs from motion tensors using GRPO
- **Optimal Transport Rewards**: Use Wasserstein distance as reward signal for RL training
- **Distributed Training**: Built-in support for Accelerate
- **Configuration Management**: Hydra for flexible configuration
- **Experiment Tracking**: WandB integration for logging and monitoring
- **LLM**: Qwen2.5-Coder-3B-Instruct (optimized for code generation)

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Forward Model (Program → Motion)

Run a random policy:

```bash
uv run scripts/random_policy.py 
```

Run a policy with a custom reward:

```bash
uv run scripts/reward_policy.py --reward move-ego-0-2
```

### Inverse Model (Motion → Program) with GRPO

The inverse model uses Hydra for configuration management and WandB for experiment tracking.

**Basic Training:**
```bash
python scripts/train_grpo.py
```

**Quick Debug Run:**
### Quick Test
```bash
python scripts/train_grpo.py training=fast data=small wandb.mode=disabled
```

**Override Parameters:**
```bash
python scripts/train_grpo.py training.learning_rate=1e-5 data.num_train_samples=1000
```

**Use Different Config Presets:**
```bash
python scripts/train_grpo.py training=fast data=large
```

**Multi-GPU with Accelerate:**
```bash
accelerate launch scripts/train_grpo.py
```

**Hyperparameter Sweeps:**
```bash
python scripts/train_grpo.py -m \
    training.learning_rate=1e-6,5e-6,1e-5 \
    training.ot_weight=0.5,1.0,2.0
```

## Configuration

The project uses Hydra for hierarchical configuration. See `configs/README.md` for detailed documentation.

### Configuration Structure

```
configs/
├── config.yaml           # Main configuration
├── model/               # Model configuration (qwen-coder)
├── training/           # Training hyperparameters (default, fast)
└── data/               # Data generation configs (default, small, large)
```

### Key Configuration Options

- **Model**: Qwen2.5-Coder-3B-Instruct
- **Training**: `training=default` or `training=fast`
- **Data**: `data=default`, `data=small`, or `data=large`
- **WandB**: Configure logging with `wandb.project`, `wandb.entity`, `wandb.mode`
- **Device**: `device=cuda` or `device=cpu` (auto-detect if null)

### Examples

```bash
# Disable WandB logging
python scripts/train_grpo.py wandb.mode=disabled

# Set WandB project
python scripts/train_grpo.py wandb.project=my-project wandb.entity=my-team

# Custom training configuration
python scripts/train_grpo.py \
    training.num_train_epochs=20 \
    training.learning_rate=1e-5 \
    training.per_device_train_batch_size=8
```

## Architecture

### Forward Process
```
Program String → Behavior Model → Motion Tensor [N, 256]
```

### Inverse Process (GRPO Training)
```
Motion Tensor [N, 256] → Motion Encoder → LLM → Generated Program
                                                        ↓
                                                  Reward Function:
                                                  -OT_distance + validity_bonus
                                                        ↓
                                                  GRPO Optimization
```

### Reward Function

The inverse model uses optimal transport distance as the primary reward signal:

```python
reward = -ot_weight * wasserstein_distance(original, reconstructed) + validity_bonus
```

## Programmatic API

### Basic Training

```python
from exact.bm import BehaviourModel
from exact.programs import generate_programs, RewardBuilder
from exact.trainer_grpo import train_motion_to_program_grpo
import torch

# Load behavior model
bm = BehaviourModel(device="cuda")

# Generate training data
programs = generate_programs(num_programs=1000)
motions = []
for prog in programs:
    reward = RewardBuilder.reward_from_name(prog)
    poses, actions = bm.generate(reward, steps=100, render=False)
    motion = torch.cat([poses, actions], dim=-1)
    motions.append(motion)

# Train with GRPO
trainer = train_motion_to_program_grpo(
    train_motions=motions,
    train_programs=programs,
    behaviour_model=bm,
    training_config={
        "num_train_epochs": 10,
        "learning_rate": 5e-6,
        "ot_weight": 1.0,
    },
    wandb_enabled=True,
)
```

### Generate Programs from Motion

```python
from exact.utils import generate_program

# Generate programs from new motions
programs = generate_program(
    model=trainer.model,
    motion_encoder=trainer.motion_encoder,
    tokenizer=trainer.tokenizer,
    motion=motion_tensor,
    num_beams=5,
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_trainer.py -v
```

## Monitoring and Logging

The project integrates with Weights & Biases (WandB) for experiment tracking:

- **Automatic Logging**: Training metrics, rewards, and hyperparameters
- **Visualization**: View training progress in real-time
- **Comparison**: Compare multiple runs and hyperparameters
- **Artifacts**: Save and version model checkpoints

To use WandB:
1. Create a free account at [wandb.ai](https://wandb.ai)
2. Login: `wandb login`
3. Run training with WandB enabled (default)

To disable WandB: `python scripts/train_grpo.py wandb.mode=disabled`

## Output Structure

Training outputs are organized by Hydra:

```
outputs/
└── YYYY-MM-DD/          # Date of run
    └── HH-MM-SS/        # Time of run
        ├── .hydra/      # Hydra config files
        │   └── config.yaml
        ├── checkpoints/ # Model checkpoints
        └── wandb/       # WandB logs (if enabled)
```

## Contributing

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
pytest tests/

# Format code
black exact/ scripts/ tests/
```

## License

See LICENSE file for details.




