# Executable Activity Models

Learning and reasoning with executable activity models for motion generation and inverse behavior modeling using GRPO.

## Features

- **Behavior Model**: Generate motions from program specifications
- **Program Generator**: Automatically generate diverse program specifications
- **Inverse Behavior Model**: Train LLMs to generate programs from motion tensors using SFT
- **Dual Loss Function**: 
  - Cross-entropy loss for program token prediction
  - Optimal transport (Wasserstein distance) for motion reconstruction
  - Structural loss for predicate and argument matching
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

**Multi-GPU with Accelerate:**
```bash
accelerate config  # Run once to configure
accelerate launch scripts/train_sft.py
```

**Hyperparameter Sweeps:**
```bash
python scripts/train_sft.py -m training.learning_rate=1e-6,5e-6,1e-5 training.ot_weight=0.5,1.0,2.0
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
- **Loss Weights**: 
  - `training.ce_weight` - Cross-entropy loss (default: 1.0)
  - `training.ot_weight` - Optimal transport loss (default: 1.0)
  - `training.structural_weight` - Structural similarity loss (default: 0.5)
  - `training.predicate_weight` - Predicate matching weight (default: 1.0)
  - `training.argument_weight` - Argument matching weight (default: 1.0)
- **WandB**: Configure logging with `wandb.project`, `wandb.entity`, `wandb.mode`
- **Device**: `device=cuda` or `device=cpu` (auto-detect if null)

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
    training.ot_weight=1.5 \
    training.structural_weight=0.3
```

## Architecture

### Forward Process
```
Program String → Behavior Model → Motion Tensor [N, 256]
```

### Inverse Process (SFT Training)
```
Motion Tensor [N, 256] → Motion Encoder → LLM → Generated Program
                                                        ↓
                                                  Loss Function:
                                                  1. CE Loss (token prediction)
                                                  2. OT Loss (motion reconstruction)
                                                  3. Structural Loss (predicate + argument)
                                                        ↓
                                                  Gradient Descent
```

### Loss Function Components

The inverse model uses a dual loss function combining three objectives:

```python
# 1. Cross-Entropy Loss: Standard language modeling objective
ce_loss = CrossEntropy(predicted_tokens, target_tokens)

# 2. Optimal Transport Loss: Motion reconstruction quality
generated_motion = BehaviorModel(generated_program)
ot_loss = wasserstein_distance(original_motion, generated_motion)

# 3. Structural Loss: Program structure similarity
predicate_loss = 1 - jaccard_similarity(ref_predicates, gen_predicates)
argument_loss = MSE(ref_arguments[matching], gen_arguments[matching])
structural_loss = predicate_weight * predicate_loss + argument_weight * argument_loss

# Combined Loss
total_loss = ce_weight * ce_loss + ot_weight * ot_loss + structural_weight * structural_loss
```

## Programmatic API

### Basic Training

```python
from exact.bm import BehaviourModel
from exact.programs import generate_programs, RewardBuilder
from exact.trainer_sft import train_motion_to_program_sft
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

# Train with supervised fine-tuning
trainer = train_motion_to_program_sft(
    train_motions=motions,
    train_programs=programs,
    behaviour_model=bm,
    model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    training_config={
        "num_train_epochs": 10,
        "ce_weight": 1.0,
        "ot_weight": 1.0,
        "structural_weight": 0.5,
    },
)

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

To disable WandB: `python scripts/train_sft.py wandb.mode=disabled`

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




