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
#   TRAJECTORY_LEN=100   Steps per trajectory                    (default: 100)
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
TRAJECTORY_LEN="${TRAJECTORY_LEN:-100}"
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
    --num-samples "$NUM_SAMPLES"
    --trajectory-length "$TRAJECTORY_LEN"
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
echo "  Samples per model: $NUM_SAMPLES"
echo "  Trajectory length: $TRAJECTORY_LEN"
echo ""

if [[ -f "$MODELS_DIR/models_verbs.json" ]]; then
    echo "[1/3] ESK verbs  →  $ESK_PATH/augmented_verbs/"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_verbs.json" \
        --output-dir "$ESK_PATH/augmented_verbs" \
        --video-name augmented_verbs \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: models_verbs.json not found"
fi

if [[ -f "$MODELS_DIR/models_activity.json" ]]; then
    echo "[2/3] ESK activities  →  $ESK_PATH/augmented_activity/"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_activity.json" \
        --output-dir "$ESK_PATH/augmented_activity" \
        --video-name augmented_activity \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: models_activity.json not found"
fi

if [[ -f "$MODELS_DIR/models_humanact12.json" ]]; then
    echo "[3/3] HumanAct12  →  $HUMANACT12_PATH/augmented/"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_humanact12.json" \
        --output-dir "$HUMANACT12_PATH/augmented" \
        --video-name augmented_humanact12 \
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
