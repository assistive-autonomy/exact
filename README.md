# ExAct: Executable Activity Models

![Python](https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)

ExAct learns executable activity models from motion data. The pipeline:
1. **Parser**: Encodes motion sequences into symbolic programs
2. **Executable Models**: Combines programs per activity using disjunctive logic
3. **Applications**: Data augmentation for segmentation, program-based activity assessment

## Installation

```bash
uv venv && uv sync
```

## Quick Start

```bash
# Train parser on synthetic data
uv run scripts/generate_data.py --name train --num-samples 10000
uv run scripts/generate_data.py --name eval --num-samples 1000
uv run scripts/parser.py

# Once parser is trained, run assessment with program edit distance
uv run scripts/assessment_edit_dist.py --parser-checkpoint results/parser/<checkpoint>

# Generate augmented data for segmentation (with trained parser)
uv run scripts/augment_data.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --train-fraction 0.25 \
    --num-samples 1000

# Run segmentation with augmented data
uv run scripts/segmentation.py project.data_path=data/augmented
```

---

## Data Augmentation

Generate synthetic training data using executable activity models. The pipeline:
1. Parses training segments into programs (using trained parser or mock)
2. Creates one `ExecutableActivityModel` per activity (disjunctive combination)
3. Optionally selects diverse programs using hierarchical clustering
4. Generates trajectories using BehaviourModel (MetaMotivo)

```bash
# Generate augmented data with trained parser
uv run scripts/augment_data.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --train-fraction 0.25 \
    --num-samples 1000 \
    --output-dir data/augmented \
    --save-models data/augmented/models.json

# With program budget (select 100 diverse programs per activity)
uv run scripts/augment_data.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --max-programs-per-activity 500 \
    --program-budget 100 \
    --num-samples 1000

# Dry run with mock parser (for testing pipeline)
uv run scripts/augment_data.py --dry-run --num-samples 100 --output-dir data/test
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-fraction` | 0.25 | Fraction of training videos to parse |
| `--num-samples` | 1000 | Total augmented trajectories to generate |
| `--output-dir` | `data/augmented` | Output directory for ESK-format files |
| `--parser-checkpoint` | None | Path to trained parser (uses mock if not provided) |
| `--program-budget` | None | Select N diverse programs per activity |
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

# Augmented experiments (with trained parser)
uv run scripts/augment_data.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --train-fraction 0.25 \
    --num-samples 1000 \
    --output-dir data/aug25
uv run scripts/segmentation.py project.data_path=data/aug25  # 25% parsed → augmented
```

---

## Activity Assessment

Two approaches for evaluating activity separability:

### 1. Flow-Based Assessment (Baseline)

Pose-based activity quality assessment using STG-NF normalizing flows. Trains density estimators on raw pose data.

```bash
# Run full assessment
uv run scripts/assessment.py

# Custom activities
uv run scripts/assessment.py data.target_activities='[Cut,Pour,Stir]'

# Hyperparameter tuning
wandb sweep configs/sweeps/assessment.yaml
wandb agent <sweep-id>
```

Output:
- `separability_matrix.png` - How each model scores each activity
- `separation_summary.png` - Separation and AUC scores per model

### 2. Program Edit Distance Assessment (ExAct)

Uses unordered tree edit distance (UTED) between program structures. This is the **core ExAct evaluation**.

```bash
# Run with trained parser
uv run scripts/assessment_edit_dist.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --max-train-programs 100 \
    --max-test-programs 50

# With program budget (select diverse subset)
uv run scripts/assessment_edit_dist.py \
    --parser-checkpoint results/parser/<checkpoint> \
    --max-train-programs 500 \
    --program-budget 100

# With pre-computed models (faster iteration)
uv run scripts/assessment_edit_dist.py --load-models data/models.json

# Save models for reuse
uv run scripts/assessment_edit_dist.py --save-models results/models.json
```

**Workflow**:
1. Parse training data → N programs per activity
2. (Optional) Select diverse subset using hierarchical clustering
3. Parse test data → query programs  
4. Compute separability matrix: $M_{i,j}$ = mean min-edit-distance from activity $i$ tests to activity $j$ model
5. Good separability: low diagonal (same-activity), high off-diagonal (cross-activity)

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
from exact.models import ExecutableActivityModel, ActivityModelCollection

# Create from programs
programs = ["[0,50]head.y(1.6)*lhand.x(0.3)", "[0,100]pelvis.y(0.9)"]
model = ExecutableActivityModel.from_programs(
    programs=programs,
    activity_name="Grab",
    eval_timesteps=100
)

# Compute reward for a state (uses MuJoCo)
reward = model.compute_reward(timestep=50, model=mujoco_model, data=mujoco_data)

# Save/load model collections
collection = ActivityModelCollection(eval_timesteps=100)
collection.add_model(model)
collection.save("models.json")
loaded = ActivityModelCollection.load("models.json")
```

### Program Selection

When you have many programs (e.g., 10,000 parsed from a dataset), select a diverse representative subset:

```python
from exact.programs import select_diverse_programs

# Select 100 diverse programs from 10,000 using hierarchical clustering
result = select_diverse_programs(
    programs=all_programs,  # List of 10,000 program strings
    budget=100,
    method="hierarchical",  # or "greedy"
)

print(result.summary())
# Selected 100 programs from 10000 total
# Number of clusters: 100
# Cluster sizes: min=1, max=250, mean=100.0

# Use selected programs for executable model
model = ExecutableActivityModel.from_programs(
    programs=result.selected_programs,
    activity_name="Grab",
)
```

### Program Format

Programs follow the grammar in [exact/programs/grammar.lark](exact/programs/grammar.lark):

```
[start,end]joint.axis(value)*joint.axis(value);[start,end]...
```

Example: `[0,50]head.y(1.65)*rwrist.z(0.45);[50,100]pelvis.y(0.80)`

- **Temporal intervals**: `[start,end]` define when sensor conditions must hold
- **Sensor predicates**: `joint.axis(value)` - e.g., `head.y(1.65)` means head y-position ≈ 1.65
- **Conjunction**: `*` combines multiple sensors (all must be satisfied)
- **Sequence**: `;` chains temporal intervals

---

## Configuration

All configs in `configs/`:

| Config | Description |
|--------|-------------|
| `segmentation.yaml` | Activity segmentation (DLC2Action) |
| `assessment.yaml` | Activity assessment (STG-NF baseline) |
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
