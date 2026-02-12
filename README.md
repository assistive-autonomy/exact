# ExAct: Executable Activity Models

![Python](https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)

ExAct learns executable activity models from motion data. The pipeline:
1. **Parser**: Encodes motion sequences into symbolic programs using a motion-conditioned LLM
2. **Executable Models**: Combines programs per activity using disjunctive logic
3. **Applications**: Data augmentation for segmentation, program-based activity assessment

## Architecture

```
Motion (T, 72) → ST-GCN Encoder → 8 temporal tokens → [prepend to LLM input]
                                   ↓
                            Projection Head → Latent space
                                   ↓                ↓
                              InfoNCE Loss ←  Program embeddings
```

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

# Once parser is trained, run the full ESK pipeline (see below)
```

---

## ESK Dataset Pipeline

After training the parser, use these scripts to process the ESK dataset:

### Full Pipeline (Recommended)

Run all steps with a single command:

```bash
# Full pipeline with trained parser
uv run scripts/run_pipeline.py \
    --parser-checkpoint results/parser/20260122_225017 \
    --esk-path /pvc/esk

# Skip parsing if programs already exist
uv run scripts/run_pipeline.py \
    --skip-parsing \
    --programs /pvc/esk/programs_train.json

# Quick test with fewer programs
uv run scripts/run_pipeline.py \
    --parser-checkpoint results/parser/20260122_225017 \
    --max-programs 20 \
    --max-test-programs 10
```

### Step-by-Step Pipeline

#### Step 1: Parse ESK → Programs

Convert motion segments from the ESK dataset into symbolic programs:

```bash
# Parse training data with trained parser
uv run scripts/parse_esk.py \
    --parser-checkpoint results/parser/20260122_225017 \
    --esk-path /pvc/esk \
    --split train

# Output: /pvc/esk/programs_train.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--parser-checkpoint` | None | Path to trained parser checkpoint |
| `--mock` | False | Use mock parser for testing |
| `--esk-path` | `../esk` | Path to ESK dataset |
| `--split` | `train` | Which split to parse (`train`, `test`, `all`) |
| `--train-fraction` | 1.0 | Fraction of training videos to use |
| `--label-type` | `verbs` | Label type (`verbs`, `nouns`, `activity`) |

Output format:
```json
{
  "metadata": {
    "parser_checkpoint": "results/parser/...",
    "num_videos": 35,
    "num_programs_valid": 37876
  },
  "activity_names": ["Add", "Adjust", "Carry", ...],
  "programs_by_activity": {
    "Cut": [
      {"program": "[0,50]rhand.x(0.3);...", "video": "YH2002...", "start": 100, "end": 250}
    ]
  }
}
```

#### Step 2: Build Executable Models

Compile parsed programs into ActivityModelCollection:

```bash
# Build models from parsed programs
uv run scripts/build_models.py \
    --programs /pvc/esk/programs_train.json

# With program budget (select 50 diverse programs per activity)
uv run scripts/build_models.py \
    --programs /pvc/esk/programs_train.json \
    --program-budget 50 \
    --output /pvc/esk/models.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--programs` | Required | Path to programs JSON (from parse_esk.py) |
| `--output` | Auto | Output path for models JSON |
| `--program-budget` | None | Select N diverse programs per activity |
| `--max-programs` | None | Random sample limit (faster than budget) |

#### Step 3: Run Assessment

Evaluate activity separability using program edit distance:

**Quick Assessment** (uses pre-parsed programs, no LLM inference needed):

```bash
uv run scripts/assessment_quick.py \
    --programs /pvc/esk/programs_train.json \
    --output-dir results/assessment_quick \
    --train-programs 15 \
    --test-programs 10
```

**Full Assessment** (parses test data with LLM):

```bash
uv run scripts/assessment_edit_dist.py \
    --load-models /pvc/esk/models.json \
    --parser-checkpoint results/parser/20260122_225017 \
    --max-train-programs 50 \
    --max-test-programs 50
```

Output:
- `separability_matrix.png` - Distance heatmap
- `results.json` - Full metrics

#### Step 4: Generate Augmented Data (Optional)

```bash
uv run scripts/augment_data.py \
    --load-models /pvc/esk/models.json \
    --num-samples 1000 \
    --output-dir /pvc/esk/augmented
```

#### Step 5: Run Segmentation (Optional)

```bash
# Baseline segmentation
uv run scripts/segmentation.py

# With augmented data
uv run scripts/segmentation.py project.data_path=/pvc/esk/augmented
```

---

## Latest Results

### Program Edit Distance Assessment (ESK Verbs)

**Overall Metrics:**
| Metric | Value |
|--------|-------|
| Same-activity distance (diagonal) | 14.19 |
| Cross-activity distance (off-diagonal) | 14.38 |
| **Separation (higher = better)** | **0.19** |

**Best Separating Activities:**
| Activity | Same | Cross | Separation |
|----------|------|-------|------------|
| Move | 14.1 | 16.1 | +2.01 |
| Press | 16.0 | 17.8 | +1.79 |
| Peel | 11.3 | 12.7 | +1.37 |
| Shake | 8.9 | 9.9 | +1.05 |
| Touch | 11.9 | 12.9 | +1.02 |

---

## Parser Training

Train a motion-conditioned parser on a single GPU (e.g., H200).

### Prerequisites

1. Generate synthetic data:
```bash
uv run scripts/generate_data.py --name train --num-samples 15000 --output-dir ../exact_data
uv run scripts/generate_data.py --name eval --num-samples 1000 --output-dir ../exact_data
```

2. Login to HuggingFace for model access:
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
uv run scripts/parser.py --eval-only results/parser/20260122_225017
```

### Configuration

Edit `configs/parser.yaml` to customize:

**Model:**
- `model_name`: Base LLM (default: `deepseek-ai/deepseek-coder-6.7b-base`)
- `load_in_4bit` / `load_in_8bit`: Quantization options

**Motion Encoder (ST-GCN):**
- `motion_dim`: Input dimensions (default: 72 for 24 SMPL joints × 3)
- `stgcn_hidden_channels`: Hidden dimension (default: 64)
- `stgcn_num_blocks`: Number of ST-GCN blocks (default: 4)
- `graph_strategy`: Adjacency strategy (`uniform`, `distance`, `spatial`)

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

# Save/load model collections
collection = ActivityModelCollection(eval_timesteps=100)
collection.add_model(model)
collection.save("models.json")
loaded = ActivityModelCollection.load("models.json")
```

### Program Selection

Select diverse representative subsets from large program collections:

```python
from exact.programs import select_diverse_programs

# Select 100 diverse programs using hierarchical clustering
result = select_diverse_programs(
    programs=all_programs,  # List of program strings
    budget=100,
    method="hierarchical",
)
print(result.summary())
```

### Program Format

Programs follow the grammar in [exact/programs/grammar.lark](exact/programs/grammar.lark):

```
[start,end]joint.axis(value)*joint.axis(value);[start,end]...
```

Example: `[0,50]head.y(1.65)*rwrist.z(0.45);[50,100]pelvis.y(0.80)`

- **Temporal intervals**: `[start,end]` define when sensor conditions must hold
- **Sensor predicates**: `joint.axis(value)` - e.g., `head.y(1.65)` means head y-position ≈ 1.65
- **Conjunction**: `*` combines multiple sensors (AND)
- **Sequence**: `;` chains temporal intervals

---

## ESK Dataset

### Label Types

| Type | Path | Classes | Description |
|------|------|---------|-------------|
| `activity` | `D2A_converted_label_activity` | 6 | High-level activities |
| `verbs` | `D2A_converted_label_verbs` | 30 | Action primitives |
| `nouns` | `D2A_converted_label_nouns` | 62 | Object categories |

### Verbs (30 classes)
`Add`, `Adjust`, `Carry`, `Clean`, `Close`, `Cut`, `Dry`, `Grab`, `Grate`, `Hold`, `Move`, `Open`, `Peel`, `Pour`, `Press`, `Put`, `Put_on`, `Read`, `Shake`, `Slide`, `Split`, `Stir`, `Switch`, `Take`, `Take_off`, `Tap`, `Taste`, `Throw`, `Touch`, `Wash`

### Data Conversion

To convert raw ESK data to the required format:

```bash
uv run scripts/esk2dlc.py --esk-path /path/to/raw/esk --output-path /pvc/esk
```

---

## Script Reference

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `run_pipeline.py` | Full ExAct pipeline | `--parser-checkpoint`, `--esk-path` |
| `parse_esk.py` | Parse ESK motion → programs | `--parser-checkpoint`, `--label-type`, `--split` |
| `build_models.py` | Build ActivityModelCollection | `--programs`, `--program-budget` |
| `assessment_exec.py` | Executable model assessment (wandb) | `--train-programs`, `--label-type` |
| `assessment_quick.py` | Quick assessment (pre-parsed) | `--programs`, `--train-programs` |
| `assessment_edit_dist.py` | Full assessment (with parsing) | `--load-models`, `--parser-checkpoint` |
| `augment_data.py` | Generate augmented training data | `--load-models`, `--num-samples` |
| `segmentation.py` | Temporal action segmentation | `train_fraction`, `project.data_path` |
| `parser.py` | Train motion-conditioned parser | `--eval-only`, config overrides |
| `generate_data.py` | Generate synthetic training data | `--name`, `--num-samples` |
| `assessment.py` | Flow-based assessment (baseline) | config overrides |

---

## Configuration Files

| Config | Description |
|--------|-------------|
| `configs/parser.yaml` | Motion-conditioned parser training |
| `configs/segmentation.yaml` | Activity segmentation (DLC2Action) |
| `configs/assessment.yaml` | Activity assessment (STG-NF baseline) |
| `configs/sweeps/*.yaml` | Hyperparameter sweep configs for wandb |

---

## Project Structure

```
exact/
├── anomaly/          # STG-NF normalizing flow models (baseline)
├── data/             # Dataset utilities (ESK, DLC format)
├── encoder/          # ST-GCN motion encoder
├── models/           # Executable activity models
├── parser/           # Motion-conditioned LLM parser
└── programs/         # Program grammar, edit distance, selection

scripts/
├── run_pipeline.py       # Full pipeline orchestration
├── parse_esk.py          # ESK → programs
├── build_models.py       # Programs → executable models
├── assessment_quick.py   # Quick assessment (pre-parsed)
├── assessment_edit_dist.py  # Full assessment
├── augment_data.py       # Data augmentation
├── segmentation.py       # Action segmentation
├── parser.py             # Parser training
└── generate_data.py      # Synthetic data generation

configs/
├── parser.yaml           # Parser training config
├── segmentation.yaml     # Segmentation config
├── assessment.yaml       # Assessment config
└── sweeps/               # Hyperparameter sweep configs
```

---

## Typical Workflow

```bash
# 1. Generate synthetic data and train parser
uv run scripts/generate_data.py --name train --num-samples 15000
uv run scripts/generate_data.py --name eval --num-samples 1000
uv run scripts/parser.py

# 2. Parse ESK dataset with trained parser
uv run scripts/parse_esk.py --parser-checkpoint results/parser/<checkpoint>

# 3. Build executable models
uv run scripts/build_models.py --programs /pvc/esk/programs_train.json --program-budget 100

# 4. Run assessment
uv run scripts/assessment_quick.py --programs /pvc/esk/programs_train.json

# 5. (Optional) Generate augmented data and run segmentation
uv run scripts/augment_data.py --load-models /pvc/esk/models.json --num-samples 5000
uv run scripts/segmentation.py project.data_path=/pvc/esk/augmented
```
