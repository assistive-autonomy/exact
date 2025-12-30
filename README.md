# EXACT: Executable Activity Models

![Python](https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)

## Installation

```bash
uv venv && uv sync
```

## ESK Dataset Tasks

### Activity Segmentation

Temporal action segmentation using DLC2Action framework with multiple models (MS-TCN3, C2F-TCN, ED-TCN, C2F-Transformer).

```bash
# Run segmentation experiment
uv run scripts/segmentation.py

# Override config
uv run scripts/segmentation.py project.annotation_path=esk/D2A_converted_label_activity
```

### Activity Assessment

Pose-based activity quality assessment using STG-NF normalizing flows. Two scripts are provided:

#### Hyperparameter Tuning (`tune_assessment.py`)

Fast script for wandb sweep optimization. Trains N models (one per activity) and evaluates each model only on its target activity test data.

```bash
# Run tuning (trains N models, reports mean_auc for sweep)
uv run scripts/tune_assessment.py

# Run wandb sweep for hyperparameter optimization
wandb sweep configs/sweeps/assessment.yaml
wandb agent <sweep-id>
```

This is optimized for speed:
- Each model evaluated only on its target activity
- Reports `mean_auc` to wandb summary for sweep optimization
- No cross-activity evaluation matrix generation

#### Full Evaluation (`assessment.py`)

Comprehensive evaluation script. Run after hyperparameters are tuned.

```bash
# List available activities for a label type
uv run scripts/assessment.py --list-activities

# Run full assessment (trains N models for N target_activities in config)
uv run scripts/assessment.py

# Override target activities
uv run scripts/assessment.py data.target_activities='[cooking,cleaning]'

# Change label type and activities
uv run scripts/assessment.py data.label_type=verbs data.target_activities='[Cut,Pour,Stir]'

# Disable wandb
uv run scripts/assessment.py wandb_mode=disabled

# Aggregate results from multiple runs
uv run scripts/assessment.py --aggregate results/assessment/run1 results/assessment/run2
```

This performs full cross-activity evaluation:
1. Trains one model per activity in `target_activities`
2. Evaluates each model on **all** target activities
3. Generates separability matrix and summary plots
4. Reports detailed per-activity metrics

Output:
- `separability_matrix.png` - Heatmap showing how each model scores each activity (diagonal should be highest)
- `separation_summary.png` - Bar charts of separation and AUC scores per model
- `aggregated.json` - Raw data for further analysis

#### Recommended Workflow

1. **Tune**: Run sweep with `tune_assessment.py` to find optimal hyperparameters
2. **Evaluate**: Run `assessment.py` with tuned hyperparameters for full evaluation

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

Sweep configs for wandb hyperparameter tuning in `configs/sweeps/`:
- `assessment.yaml` - Assessment hyperparameter tuning sweep
- `parser.yaml` - Parser sweep

## ESK Dataset Labels

The ESK dataset provides three label types that can be configured via `data.label_type` and `project.annotation_path`:

### Activities (`D2A_converted_label_activity`)
High-level activity categories:
- `cleaning` - Kitchen cleanup tasks
- `cooking` - Active cooking on stove
- `experimental procedure` - Study-specific procedures
- `getting ready` - Preparation before cooking
- `other_annot` - Uncategorized actions
- `preparing ingredients` - Ingredient preparation (cutting, peeling, etc.)

### Verbs (`D2A_converted_label_verbs`)
Action primitives (28 classes):
`Add`, `Adjust`, `Carry`, `Clean`, `Close`, `Cut`, `Dry`, `Grab`, `Grate`, `Hold`, `Move`, `Open`, `Peel`, `Pour`, `Press`, `Put`, `Put_on`, `Read`, `Shake`, `Slide`, `Split`, `Stir`, `Switch`, `Take_off`, `Taste`, `Throw`, `Touch`, `Wash`

### Nouns (`D2A_converted_label_nouns`)
Object categories (62 classes):
`Avocado`, `Bottle`, `Bowl`, `Box`, `Broth`, `Brush`, `Butter`, `Button`, `Carrots`, `Cheese`, `Colander`, `Cucumber`, `Cup`, `Cupboard`, `Cutting_board`, `Doser_Glass`, `Drawer`, `Eggplant`, `Fridge`, `Frying_Oil`, `Glove`, `Grater`, `Green_salad`, `Hand`, `Knife`, `Lemon`, `Onions`, `Package`, `Pan`, `Pasta_Spoon`, `Peeler`, `Plate`, `Pot`, `Pot_lid`, `Processed_ingredients`, `Radish`, `Recipe`, `Rice`, `Risotto`, `Salad_bowl`, `Salt`, `Sauce`, `Seasoning`, `Shallots`, `Sink`, `Sink_Sprayer`, `Soap`, `Spatula`, `Sponge`, `Spoon`, `Stock_cube`, `Stoves`, `Surimi`, `Tissue`, `Tomatoes`, `Towel`, `Trash`, `Trivet`, `Water`, `Whip`, `Zucchini`
