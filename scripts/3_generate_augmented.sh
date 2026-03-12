#!/usr/bin/env bash
# =============================================================================
# Stage 3: Generate Augmented Motion Data for Segmentation
#
# Takes the executable activity models built in Stage 2 and generates
# synthetic ESK-format motion sequences.  Used as data augmentation for the
# segmentation experiments (Stage 4).  CPU-only — no GPU required.
#
# Usage:
#   bash scripts/3_generate_augmented.sh
#
# Environment variables:
#   NUM_SAMPLES=1000     Total synthetic samples per model file  (default: 1000)
#   NUM_SAMPLES_VERBS=1000   Override sample count for ESK verbs  (default: NUM_SAMPLES)
#   NUM_SAMPLES_ACTIVITY=1000  Override for ESK activities        (default: NUM_SAMPLES)
#   NUM_SAMPLES_HUMANACT=1000  Override for HumanAct12            (default: NUM_SAMPLES)
#   TRAJ_LEN_VERBS=50    Steps per trajectory for ESK verbs      (default: 50)
#   TRAJ_LEN_ACTIVITY=350 Steps per trajectory for ESK activities(default: 350)
#   TRAJ_LEN_HUMANACT=75 Steps per trajectory for HumanAct12     (default: 75)
#   RELABEL_WORKERS=4    MuJoCo relabel threads per worker       (default: 4)
#   Z_VARIANTS=4         Diversity: z-vector variants per sample (default: 4)
#   BATCH_SIZE=256       Buffer batch size for z computation     (default: 256)
#   SEED=42              Random seed                             (default: 42)
#   NUM_WORKERS=<N>      Override CPU worker count               (default: auto)
#
# Outputs:
#   ../exact_data/benchmarks/esk/augmented_verbs/
#   ../exact_data/benchmarks/esk/augmented_activity/
#   ../exact_data/benchmarks/humanact12/augmented/
# =============================================================================
set -euo pipefail

cd /pvc/exact

NUM_CPUS=$(nproc)
echo "Detected ${NUM_CPUS} CPU cores (no GPU required)"

# ─── Configuration ──────────────────────────────────────────────────────────
ESK_PATH="${ESK_PATH:-../exact_data/benchmarks/esk}"
HUMANACT12_PATH="${HUMANACT12_PATH:-../exact_data/benchmarks/humanact12}"
MODELS_DIR="${MODELS_DIR:-../exact_data/models}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
# Per-dataset sample counts — set independently to control aug/real ratio.
# ESK has ~2M real train frames → need more samples to reach meaningful ratio.
# HumanAct12 has ~56K real frames → fewer samples suffice.
NUM_SAMPLES_VERBS="${NUM_SAMPLES_VERBS:-$NUM_SAMPLES}"
NUM_SAMPLES_ACTIVITY="${NUM_SAMPLES_ACTIVITY:-$NUM_SAMPLES}"
NUM_SAMPLES_HUMANACT="${NUM_SAMPLES_HUMANACT:-$NUM_SAMPLES}"
# Per-dataset trajectory lengths based on median segment durations in training data:
#   ESK verbs:    median=31, p75~50  → 50 steps
#   ESK activity: median=333         → 350 steps
#   HumanAct12:   median=60          → 75 steps
TRAJ_LEN_VERBS="${TRAJ_LEN_VERBS:-50}"
TRAJ_LEN_ACTIVITY="${TRAJ_LEN_ACTIVITY:-350}"
TRAJ_LEN_HUMANACT="${TRAJ_LEN_HUMANACT:-75}"
RELABEL_WORKERS="${RELABEL_WORKERS:-4}"
Z_VARIANTS="${Z_VARIANTS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""   # Force CPU-only

WORKER_ARGS=""
if [[ -n "${NUM_WORKERS:-}" ]]; then
    WORKER_ARGS="--num-workers $NUM_WORKERS"
fi

COMMON_ARGS=(
    --relabel-workers "$RELABEL_WORKERS"
    --z-variants "$Z_VARIANTS"
    --buffer-batch-size "$BATCH_SIZE"
    --seed "$SEED"
    $WORKER_ARGS
)

# Check at least one model file exists
FOUND=0
for f in "$MODELS_DIR/models_verbs.json" \
         "$MODELS_DIR/models_activity.json" \
         "$MODELS_DIR/models_humanact12.json"; do
    [[ -f "$f" ]] && FOUND=1 && break
done
if [[ "$FOUND" -eq 0 ]]; then
    echo "ERROR: No model JSON files found in $MODELS_DIR"
    echo "  Run scripts/2_train_and_parse.sh first."
    exit 1
fi

echo ""
echo "============================================"
echo "Stage 3: Generating Augmented Data (CPU-only, ${NUM_CPUS} cores)"
echo "============================================"
echo "  Samples (verbs/activity/humanact): $NUM_SAMPLES_VERBS / $NUM_SAMPLES_ACTIVITY / $NUM_SAMPLES_HUMANACT"
echo "  Trajectory length (verbs/activity/humanact): $TRAJ_LEN_VERBS / $TRAJ_LEN_ACTIVITY / $TRAJ_LEN_HUMANACT"
echo ""

if [[ -f "$MODELS_DIR/models_verbs.json" ]]; then
    echo "[1/3] ESK verbs  →  $ESK_PATH/augmented_verbs/  (samples=$NUM_SAMPLES_VERBS, traj_len=$TRAJ_LEN_VERBS)"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_verbs.json" \
        --output-dir "$ESK_PATH/augmented_verbs" \
        --video-name augmented_verbs \
        --trajectory-length "$TRAJ_LEN_VERBS" \
        --num-samples "$NUM_SAMPLES_VERBS" \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: models_verbs.json not found"
fi

if [[ -f "$MODELS_DIR/models_activity.json" ]]; then
    echo "[2/3] ESK activities  →  $ESK_PATH/augmented_activity/  (samples=$NUM_SAMPLES_ACTIVITY, traj_len=$TRAJ_LEN_ACTIVITY)"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_activity.json" \
        --output-dir "$ESK_PATH/augmented_activity" \
        --video-name augmented_activity \
        --trajectory-length "$TRAJ_LEN_ACTIVITY" \
        --num-samples "$NUM_SAMPLES_ACTIVITY" \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: models_activity.json not found"
fi

if [[ -f "$MODELS_DIR/models_humanact12.json" ]]; then
    echo "[3/3] HumanAct12  →  $HUMANACT12_PATH/augmented/  (samples=$NUM_SAMPLES_HUMANACT, traj_len=$TRAJ_LEN_HUMANACT)"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_humanact12.json" \
        --output-dir "$HUMANACT12_PATH/augmented" \
        --video-name augmented_humanact12 \
        --trajectory-length "$TRAJ_LEN_HUMANACT" \
        --num-samples "$NUM_SAMPLES_HUMANACT" \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: models_humanact12.json not found"
fi

echo "============================================"
echo "Stage 3 complete"
echo "============================================"
[[ -f "$MODELS_DIR/models_verbs.json" ]] && \
    echo "  ESK verbs:      $ESK_PATH/augmented_verbs/"
[[ -f "$MODELS_DIR/models_activity.json" ]] && \
    echo "  ESK activities:  $ESK_PATH/augmented_activity/"
[[ -f "$MODELS_DIR/models_humanact12.json" ]] && \
    echo "  HumanAct12:      $HUMANACT12_PATH/augmented/"
echo ""
echo "Next: bash scripts/4_run_segmentation.sh"
