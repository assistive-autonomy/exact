# Understanding Human Actions through the Lens of Executable Models

![Python](https://img.shields.io/badge/Python->=3.10-blue?logo=python&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)

This repository contains a staged experimental pipeline for:

1. Generating synthetic program-motion data.
2. Training and running the parser.
3. Building executable models.
4. Running segmentation experiments.
5. Running anomaly detection experiments.

The main workflow is implemented in `scripts/1_generate_data.sh` through `scripts/5_run_anomaly_detection.sh`.

## 1. Setup

Create and sync the environment:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Run all commands from the repository root.

## 2. Data and Paths

The pipeline expects benchmark and generated artifacts under `../exact_data/` (relative to this repository), including:

- `../exact_data/benchmarks/esk`
- `../exact_data/benchmarks/humanact12`
- `../exact_data/programs`
- `../exact_data/models`

Notes:

- `scripts/1_generate_data.sh`, `scripts/2_train_and_parse.sh`, and `scripts/3_generate_augmented.sh` include `cd /pvc/exact` internally.
- If your local repository path is not `/pvc/exact`, update that line in those scripts (or create a matching mount/symlink in your environment).

## 3. End-to-End Pipeline (Scripts 1-5)

### Stage 1: Generate synthetic training data

Script: `scripts/1_generate_data.sh`

Purpose:

- Generates 10 diverse subsets of synthetic motion-program pairs.
- Merges them into one parser training file.

Default run:

```bash
bash scripts/1_generate_data.sh
```

Custom run:

```bash
bash scripts/1_generate_data.sh ../exact_data/programs/synthetic 50000 8
```

Arguments:

- `output_dir` (default: `../exact_data/programs/synthetic`)
- `total_samples` (default: `50000`)
- `num_workers` (default: `8`)

Key output:

- `../exact_data/programs/synthetic/train_diverse.h5`

### Stage 2: Train parser, parse splits, build executable models

Script: `scripts/2_train_and_parse.sh`

Purpose:

- Trains parser (DDP when multiple GPUs are available).
- Parses train/val/test for ESK verbs, ESK activities, HumanAct12.
- Builds executable model collections from train-split programs.

Default run:

```bash
bash scripts/2_train_and_parse.sh
```

Useful variants:

```bash
# Use an existing checkpoint (skip training)
CHECKPOINT=results/parser/<run>/best_generation bash scripts/2_train_and_parse.sh

# Resume interrupted training
RESUME_DIR=results/parser/<run> bash scripts/2_train_and_parse.sh

# Parse only selected splits
SPLITS="val test" bash scripts/2_train_and_parse.sh

# Skip model building
SKIP_BUILD=1 bash scripts/2_train_and_parse.sh
```

Key outputs:

- Parsed programs in `../exact_data/programs/parsed/`
- Executable models in `../exact_data/models/`

### Stage 3: Generate augmented data for segmentation

Script: `scripts/3_generate_augmented.sh`

Purpose:

- Uses executable models from Stage 2 to synthesize augmented benchmark data.
- Produces augmented sets for ESK verbs, ESK activities, and HumanAct12.

Default run:

```bash
bash scripts/3_generate_augmented.sh
```

Useful variants:

```bash
# Global sample count override
NUM_SAMPLES=2000 bash scripts/3_generate_augmented.sh

# Per-dataset sample overrides
NUM_SAMPLES_VERBS=5000 NUM_SAMPLES_ACTIVITY=1500 NUM_SAMPLES_HUMANACT=1000 bash scripts/3_generate_augmented.sh

# Control trajectory lengths
TRAJ_LEN_VERBS=50 TRAJ_LEN_ACTIVITY=350 TRAJ_LEN_HUMANACT=75 bash scripts/3_generate_augmented.sh
```

Key outputs:

- `../exact_data/benchmarks/esk/augmented_verbs/`
- `../exact_data/benchmarks/esk/augmented_activity/`
- `../exact_data/benchmarks/humanact12/augmented/`

### Stage 4: Run segmentation experiments

Script: `scripts/4_run_segmentation.sh`

Purpose:

- Runs segmentation experiments across three conditions:
	- `original` (real data only)
	- `perturbed` (annotation perturbation baseline)
	- `augmented` (real + synthetic data)

Default run:

```bash
bash scripts/4_run_segmentation.sh
```

Useful variants:

```bash
# One condition only
CONDITION=original bash scripts/4_run_segmentation.sh
CONDITION=augmented bash scripts/4_run_segmentation.sh

# One dataset only
DATASET=esk_activities bash scripts/4_run_segmentation.sh
DATASET=esk_verbs bash scripts/4_run_segmentation.sh
DATASET=humanact12 bash scripts/4_run_segmentation.sh
```

This script runs task configs under `configs/segmentation/` via `scripts/tasks/segmentation.py`.

### Stage 5: Run anomaly detection experiments

Script: `scripts/5_run_anomaly_detection.sh`

Purpose:

- Runs distinguishability experiments and logs an NxN AUC matrix (for N classes) to wandb.
- Supports methods:
	- `nf`
	- `mean_sigmoid`
	- `min_sigmoid`

Default run:

```bash
bash scripts/5_run_anomaly_detection.sh
```

Useful variants:

```bash
# One method
METHOD=nf bash scripts/5_run_anomaly_detection.sh
METHOD=mean_sigmoid bash scripts/5_run_anomaly_detection.sh
METHOD=min_sigmoid bash scripts/5_run_anomaly_detection.sh

# One dataset
DATASET=esk_verbs bash scripts/5_run_anomaly_detection.sh
DATASET=esk_activities bash scripts/5_run_anomaly_detection.sh
DATASET=humanact12 bash scripts/5_run_anomaly_detection.sh
```

Results are saved under `results/anomaly_detection/`.

## 4. Recommended Run Order

For a full fresh experimental run:

```bash
bash scripts/1_generate_data.sh
bash scripts/2_train_and_parse.sh
bash scripts/3_generate_augmented.sh
bash scripts/4_run_segmentation.sh
bash scripts/5_run_anomaly_detection.sh
```

## 5. Quick Re-runs

When iterating quickly after you already have parser outputs and models:

```bash
# Segmentation only
bash scripts/4_run_segmentation.sh

# Anomaly detection only
bash scripts/5_run_anomaly_detection.sh
```

