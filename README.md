# EXACT: Executable Activity Models

## Installation

```bash
uv venv && uv sync
```

## ESK Dataset Tasks

### Activity Segmentation

Temporal action segmentation using DLC2Action framework with MS-TCN models.

```bash
# Run segmentation experiment
uv run scripts/segmentation.py

# Override config
uv run scripts/segmentation.py project.annotation_path=esk/D2A_converted_label_activity
```

### Activity Assessment

Pose-based activity quality assessment using STG-NF normalizing flows. Trains on one activity (normal class) and evaluates separation from other activities.

```bash
# List available activities
uv run scripts/assessment.py --list-activities

# Run assessment experiment  
uv run scripts/assessment.py

# Override config
uv run scripts/assessment.py data.target_activity=cleaning training.epochs=50

# Disable wandb
uv run scripts/assessment.py wandb_mode=disabled

# Evaluate from checkpoint
uv run scripts/assessment.py checkpoint.path=results/model.pth checkpoint.eval_only=true
```

## Parser Training

### Generate Training Data

```bash
uv run scripts/generate_data.py --name train --num-samples 1000
uv run scripts/generate_data.py --name eval --num-samples 200
```

### Train Parser

```bash
# Train parser
uv run scripts/parser.py

# Override config
uv run scripts/parser.py num_train_epochs=20 learning_rate=1e-5

# Evaluate from checkpoint
uv run scripts/parser.py --eval-only results/parser/20251222_123456
```

## Configuration

All configs in `configs/`:
- `segmentation.yaml` - Activity segmentation (DLC2Action)
- `assessment.yaml` - Activity assessment (STG-NF)
- `parser.yaml` - Motion-conditioned parser training

Sweep configs for wandb hyperparameter tuning:
- `sweep_segmentation.yaml` - Segmentation sweep
- `sweep_assessment.yaml` - Assessment sweep
- `sweep_parser.yaml` - Parser sweep
