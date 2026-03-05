#!/usr/bin/env bash
# ============================================================================
# Generate augmented data from Executable Activity Models (CPU-only)
#
# This script runs on a CPU-only machine with many cores.  It takes the
# executable model JSON files produced by train_and_parse.sh and generates
# augmented ESK-format data using multiprocessing (no GPU required).
#
# The heavy lifting is done by generate_augmented_data.py, which spawns one
# worker per activity class.  Each worker loads its own BehaviourModel on
# CPU, pre-computes z vectors, and generates trajectories.
#
# Usage:
#   bash scripts/generate_augmented.sh
#
# Customisation (via environment variables):
#   NUM_SAMPLES=5000     Total samples per model file (default: 1000)
#   RELABEL_WORKERS=8    MuJoCo relabel threads per worker (default: 4)
#   NUM_WORKERS=32       Override auto-detected worker count
#   TRAJECTORY_LEN=100   Steps per trajectory (default: 100)
#   Z_VARIANTS=4         z-vector diversity variants (default: 4)
#   BATCH_SIZE=256       Buffer sample size for z computation (default: 256)
#   SEED=42              Random seed (default: 42)
#
# Example (high-throughput, 188-core machine):
#   NUM_SAMPLES=10000 RELABEL_WORKERS=4 NUM_WORKERS=40 \
#       bash scripts/generate_augmented.sh
# ============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
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

# Limit per-process threading to keep cores free for multiprocessing workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Force CPU-only even if CUDA libs are present
export CUDA_VISIBLE_DEVICES=""

# If user didn't set NUM_WORKERS, let the Python script auto-detect
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

# ─── Discover model files ──────────────────────────────────────────────────
# Verify at least one model file exists
FOUND=0
for f in "$MODELS_DIR/models_verbs.json" \
         "$MODELS_DIR/models_activity.json" \
         "$MODELS_DIR/models_humanact12.json"; do
    if [[ -f "$f" ]]; then
        FOUND=1
        break
    fi
done
if [[ "$FOUND" -eq 0 ]]; then
    echo "ERROR: No model JSON files found. Expected at least one of:"
    echo "  $MODELS_DIR/models_verbs.json"
    echo "  $MODELS_DIR/models_activity.json"
    echo "  $MODELS_DIR/models_humanact12.json"
    echo ""
    echo "Run scripts/train_and_parse.sh first to build models."
    exit 1
fi

echo ""
echo "============================================"
echo "Generating augmented data (CPU-only, ${NUM_CPUS} cores)"
echo "============================================"
echo "  Samples per model: $NUM_SAMPLES"
echo "  Trajectory length: $TRAJECTORY_LEN"
echo "  Relabel workers/process: $RELABEL_WORKERS"
echo "  z variants: $Z_VARIANTS"
echo ""

# ─── Step 1: ESK verbs ─────────────────────────────────────────────────────
if [[ -f "$MODELS_DIR/models_verbs.json" ]]; then
    echo "──────────────────────────────────────────"
    echo "[1] ESK verbs → $ESK_PATH/augmented_verbs/"
    echo "──────────────────────────────────────────"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_verbs.json" \
        --output-dir "$ESK_PATH/augmented_verbs" \
        --video-name augmented_verbs \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: $MODELS_DIR/models_verbs.json not found"
fi

# ─── Step 2: ESK activities ────────────────────────────────────────────────
if [[ -f "$MODELS_DIR/models_activity.json" ]]; then
    echo "──────────────────────────────────────────"
    echo "[2] ESK activities → $ESK_PATH/augmented_activity/"
    echo "──────────────────────────────────────────"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_activity.json" \
        --output-dir "$ESK_PATH/augmented_activity" \
        --video-name augmented_activity \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: $MODELS_DIR/models_activity.json not found"
fi

# ─── Step 3: HumanAct12 ───────────────────────────────────────────────────
if [[ -f "$MODELS_DIR/models_humanact12.json" ]]; then
    echo "──────────────────────────────────────────"
    echo "[3] HumanAct12 → $HUMANACT12_PATH/augmented/"
    echo "──────────────────────────────────────────"
    uv run scripts/data/generate_augmented_data.py \
        --models "$MODELS_DIR/models_humanact12.json" \
        --output-dir "$HUMANACT12_PATH/augmented" \
        --video-name augmented_humanact12 \
        "${COMMON_ARGS[@]}"
    echo ""
else
    echo "SKIP: $MODELS_DIR/models_humanact12.json not found"
fi

echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Outputs:"
[[ -f "$MODELS_DIR/models_verbs.json" ]] && \
    echo "  ESK verbs:      $ESK_PATH/augmented_verbs/"
[[ -f "$MODELS_DIR/models_activity.json" ]] && \
    echo "  ESK activities:  $ESK_PATH/augmented_activity/"
[[ -f "$MODELS_DIR/models_humanact12.json" ]] && \
    echo "  HumanAct12:      $HUMANACT12_PATH/augmented/"
echo ""
echo "Next steps:"
echo "  - Action segmentation:  uv run scripts/tasks/segmentation.py --config-name <config>"
echo "  - Anomaly detection:    uv run scripts/tasks/anomaly_detection_exec.py --config <config>"
