# ExAct: Executable Activity Models

![Python](https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)

ExAct learns executable activity models from motion data. The pipeline:
1. **Parser**: Encodes motion sequences into symbolic programs
2. **Executable Models**: Combines programs per activity using disjunctive logic
3. **Data Augmentation**: Generates synthetic motion trajectories from executable models

## Installation

```bash
uv venv && uv sync
```

## Quick Start

```bash
# Train parser on synthetic data
uv run scripts/generate_data.py --name train --num-samples 1000
uv run scripts/parser.py

# Generate augmented data for segmentation
uv run scripts/augment_data.py --train-fraction 0.25 --num-samples 1000 --output-dir data/augmented

# Run segmentation with augmented data
uv run scripts/segmentation.py project.data_path=data/augmented
```

---

## Data Augmentation

Generate synthetic training data using executable activity models. The pipeline:
1. Parses training segments into programs
2. Creates one `ExecutableActivityModel` per activity (disjunctive combination)
3. Generates trajectories using BehaviourModel (MetaMotivo)

```bash
# Generate augmented data (25% of training data → executable models → 1000 synthetic samples)
uv run scripts/augment_data.py \
    --train-fraction 0.25 \
    --num-samples 1000 \
    --output-dir data/augmented \
    --save-models data/augmented/models.json

# Dry run (random poses, no GPU required)
uv run scripts/augment_data.py --train-fraction 0.25 --num-samples 100 --output-dir data/test --dry-run

# Custom label type
uv run scripts/augment_data.py --label-type activity --num-samples 500 --output-dir data/augmented_activity
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-fraction` | 0.25 | Fraction of training videos to parse |
| `--num-samples` | 1000 | Total augmented trajectories to generate |
| `--output-dir` | `data/augmented` | Output directory for ESK-format files |
| `--label-type` | `verbs` | Label type: `verbs`, `nouns`, `activity` |
| `--save-models` | None | Save executable models to JSON |
| `--dry-run` | False | Generate random poses (no BehaviourModel) |

### Output Format

Generates ESK-compatible files:
- `augmented_data_pose3d_smpl.h5` - Pose trajectories (96-dim)
- `augmented_data_labels.pickle` - Activity labels

---

## Activity Segmentation

Temporal action segmentation using DLC2Action framework (MS-TCN3, C2F-TCN, ED-TCN, C2F-Transformer).

```bash
# Run with 100% training data
uv run scripts/segmentation.py

# Run with reduced training data
uv run scripts/segmentation.py train_fraction=0.5
uv run scripts/segmentation.py train_fraction=0.25

# Use augmented data
uv run scripts/segmentation.py project.data_path=data/augmented project.annotation_path=data/augmented
```

### Training Fraction Experiments

Compare baseline (X% real data) vs augmented data:

- **`train_fraction`**: Fraction of training videos (1.0, 0.5, 0.25)
- Creates separate projects: `esk_verbs_100pct`, `esk_verbs_50pct`, `esk_verbs_25pct`
- Each seed uses a different random subset of training videos
- Validation/test sets unchanged

```bash
# Baseline experiments
uv run scripts/segmentation.py train_fraction=0.25   # 25% real data baseline

# Augmented experiments  
uv run scripts/augment_data.py --train-fraction 0.25 --num-samples 1000 --output-dir data/aug25
uv run scripts/segmentation.py project.data_path=data/aug25  # 25% parsed → augmented
```

---

## Activity Assessment

Pose-based activity quality assessment using STG-NF normalizing flows.

### Hyperparameter Tuning

```bash
# Run tuning (trains N models, reports mean_auc for sweep)
uv run scripts/tune_assessment.py

# Run wandb sweep
wandb sweep configs/sweeps/assessment.yaml
wandb agent <sweep-id>
```

### Full Evaluation

```bash
# List available activities
uv run scripts/assessment.py --list-activities

# Run full assessment
uv run scripts/assessment.py

# Custom activities
uv run scripts/assessment.py data.target_activities='[Cut,Pour,Stir]'

# Aggregate results
uv run scripts/assessment.py --aggregate results/assessment/run1 results/assessment/run2
```

Output:
- `separability_matrix.png` - How each model scores each activity
- `separation_summary.png` - Separation and AUC scores per model
- `aggregated.json` - Raw data for analysis

### Program Edit Distance Assessment

Alternative assessment using unordered tree edit distance (UTED) between program structures:

```bash
# Run edit distance assessment
uv run scripts/assessment_edit_dist.py

# With pre-computed models
uv run scripts/assessment_edit_dist.py --load-models data/augmented/models.json

# Save models for later
uv run scripts/assessment_edit_dist.py --save-models results/edit_dist/models.json
```

This method:
1. Parses training data → N programs per activity
2. Parses test data → query programs  
3. Computes M[i,j] = mean(min_edit_dist(test_i, model_j))
4. Lower diagonal = same-activity similarity, higher off-diagonal = separability

Key features:
- **Interval-agnostic**: Ignores temporal intervals, focuses on structure
- **Value tolerance**: Values within 0.3 are considered equal
- Uses `edist` library for constrained UTED

---

## Parser Training

Train a motion-conditioned parser on a single GPU (e.g., H200).

### Prerequisites

1. Generate synthetic data in `../exact_data/`:
```bash
uv run scripts/generate_data.py --name train --num-samples 1000 --output-dir ../exact_data
uv run scripts/generate_data.py --name eval --num-samples 200 --output-dir ../exact_data
```

2. Ensure you have access to the Llama model (via HuggingFace):
```bash
huggingface-cli login
```

### Train Parser

```bash
# Default training (uses configs/parser.yaml)
uv run scripts/parser.py

# Custom hyperparameters
uv run scripts/parser.py num_train_epochs=20 learning_rate=1e-5

# Evaluate checkpoint
uv run scripts/parser.py --eval-only results/parser/20251222_123456
```

### Configuration

Edit `configs/parser.yaml` to customize:
- `train_data` / `eval_data`: Paths to HDF5 data files
- `model_name`: Base LLM (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `per_device_train_batch_size`: Batch size (adjust based on GPU memory)

The model uses **8-bit quantization** (bitsandbytes) by default to reduce memory usage while maintaining quality. LoRA adapters are trained on top of the quantized base model.

---

## Executable Activity Models

Core abstraction for combining multiple programs into a single executable model.

### Architecture

```
Programs p₁, p₂, ..., pₙ  →  ExecutableActivityModel  →  reward r(state)
                              r = 1 - ∏(1 - pᵢ(state))
```

**Disjunctive combination**: Any high-reward program yields high combined reward (soft OR).

### Usage

```python
from exact.models import ExecutableActivityModel, NormalizedProgram
from exact.programs import parse_program

# Create from programs
programs = ["[0,50]torso.x(0.5)*acc;", "[0,100]lhand.y(-0.3)*acc;"]
model = ExecutableActivityModel.from_programs(
    programs=programs,
    activity_name="Grab",
    eval_timesteps=100
)

# Compute reward for a state
reward = model.compute_reward(state, timestep=50)

# Save/load models
from exact.models import ActivityModelCollection
collection = ActivityModelCollection({"Grab": model, "Put": put_model})
collection.save("models.json")
loaded = ActivityModelCollection.load("models.json")
```

### Program Format

Programs follow the grammar in [exact/programs/grammar.lark](exact/programs/grammar.lark):

```
[start,end]joint.axis(value)*sensor;[start,end]motion_type(value);
```

- **Sensor rewards**: `[0,50]lhand.x(0.3)*acc;` - Left hand x-acceleration > 0.3
- **Motion rewards**: `[50,100]walk(0.8);` - Walking motion with magnitude 0.8
- **Temporal intervals**: `[start,end]` normalized to `eval_timesteps`

---

## Configuration

All configs in `configs/`:

| Config | Description |
|--------|-------------|
| `segmentation.yaml` | Activity segmentation (DLC2Action) |
| `assessment.yaml` | Activity assessment (STG-NF) |
| `parser.yaml` | Motion-conditioned parser |

Sweep configs in `configs/sweeps/` for wandb hyperparameter tuning.

---

## ESK Dataset

### Label Types

Configure via `data.label_type` and `project.annotation_path`:

| Type | Path | Classes | Description |
|------|------|---------|-------------|
| `activity` | `D2A_converted_label_activity` | 6 | High-level activities |
| `verbs` | `D2A_converted_label_verbs` | 28 | Action primitives |
| `nouns` | `D2A_converted_label_nouns` | 62 | Object categories |

### Activities
`cleaning`, `cooking`, `experimental procedure`, `getting ready`, `other_annot`, `preparing ingredients`

### Verbs
`Add`, `Adjust`, `Carry`, `Clean`, `Close`, `Cut`, `Dry`, `Grab`, `Grate`, `Hold`, `Move`, `Open`, `Peel`, `Pour`, `Press`, `Put`, `Put_on`, `Read`, `Shake`, `Slide`, `Split`, `Stir`, `Switch`, `Take_off`, `Taste`, `Throw`, `Touch`, `Wash`

### Nouns
`Avocado`, `Bottle`, `Bowl`, `Box`, `Broth`, `Brush`, `Butter`, `Button`, `Carrots`, `Cheese`, `Colander`, `Cucumber`, `Cup`, `Cupboard`, `Cutting_board`, `Doser_Glass`, `Drawer`, `Eggplant`, `Fridge`, `Frying_Oil`, `Glove`, `Grater`, `Green_salad`, `Hand`, `Knife`, `Lemon`, `Onions`, `Package`, `Pan`, `Pasta_Spoon`, `Peeler`, `Plate`, `Pot`, `Pot_lid`, `Processed_ingredients`, `Radish`, `Recipe`, `Rice`, `Risotto`, `Salad_bowl`, `Salt`, `Sauce`, `Seasoning`, `Shallots`, `Sink`, `Sink_Sprayer`, `Soap`, `Spatula`, `Sponge`, `Spoon`, `Stock_cube`, `Stoves`, `Surimi`, `Tissue`, `Tomatoes`, `Towel`, `Trash`, `Trivet`, `Water`, `Whip`, `Zucchini`
