#!/usr/bin/env bash
# ============================================================================
# Train parser on train_diverse.h5, then parse ESK + HumanAct12 datasets,
# then build executable activity models for both.
#
# Optimized for multi-GPU (2× H200) and high CPU count (188 vCPUs):
#   - Training uses DDP via torchrun across all GPUs
#   - Parsing steps run in parallel on separate GPUs
#   - Build-models steps run in parallel (CPU-only)
#
# Usage:
#   bash scripts/train_and_parse.sh
#
# To resume from a previous run (e.g. after a crash):
#   RESUME_DIR=results/parser/20260222_230000 bash scripts/train_and_parse.sh
#
# To skip training and only parse (provide existing checkpoint):
#   CHECKPOINT=results/parser/20260222_230000 bash scripts/train_and_parse.sh
# ============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc)
echo "Detected ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/parser/parser_v4.yaml}"
ESK_PATH="../exact_data/benchmarks/esk"
HUMANACT12_PATH="../exact_data/benchmarks/humanact12"
PROGRAMS_DIR="../exact_data/programs/parsed"
MODELS_DIR="../exact_data/models"
mkdir -p "$PROGRAMS_DIR" "$MODELS_DIR"

# Performance: set CPU threading for data loading & NCCL tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_P2P_LEVEL=NVL         # Use NVLink for inter-GPU P2P if available
export NCCL_ASYNC_ERROR_HANDLING=1 # Graceful NCCL error reporting
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce CUDA memory fragmentation

# Parsing batch size — H200 144GB can handle large batches for inference
BATCH_SIZE="${BATCH_SIZE:-256}"

# ─── Step 1: Train the parser (multi-GPU DDP) ──────────────────────────────
if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "============================================"
    echo "SKIPPING TRAINING — using existing checkpoint"
    echo "  Checkpoint: $CHECKPOINT"
    echo "============================================"
    BEST_CHECKPOINT="$CHECKPOINT"
else
    echo "============================================"
    echo "Step 1/3: Training parser on train_diverse.h5 (${NUM_GPUS} GPUs, DDP)"
    echo "============================================"

    TRAIN_ARGS="--config $CONFIG"

    if [[ -n "${RESUME_DIR:-}" ]]; then
        echo "  Resuming from: $RESUME_DIR"
        TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME_DIR"
    fi

    # Use torchrun for multi-GPU DDP training when >1 GPU available
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        uv run torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port="${MASTER_PORT:-29500}" \
            scripts/parsing/train_parser.py $TRAIN_ARGS
    else
        uv run scripts/parsing/train_parser.py $TRAIN_ARGS
    fi

    # Find the output directory (most recent in results/parser/)
    OUTPUT_DIR=$(ls -dt results/parser/2* 2>/dev/null | head -1)
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: Could not find training output directory"
        exit 1
    fi
    echo "  Training output: $OUTPUT_DIR"

    BEST_CHECKPOINT="$OUTPUT_DIR"

    # Prefer best-by-generation checkpoint if available (tracks edit distance,
    # which is a better proxy for downstream quality than teacher-forced eval_loss)
    if [[ -f "$OUTPUT_DIR/best_generation/prefix_parser.pt" ]]; then
        BEST_CHECKPOINT="$OUTPUT_DIR/best_generation"
        echo "  Using best-generation checkpoint (by edit distance)"
    fi

    # Verify checkpoint files exist
    if [[ ! -f "$BEST_CHECKPOINT/prefix_parser.pt" ]]; then
        echo "ERROR: prefix_parser.pt not found in $BEST_CHECKPOINT"
        echo "  Available files:"
        ls -la "$BEST_CHECKPOINT/"
        exit 1
    fi

    echo "  Best checkpoint: $BEST_CHECKPOINT"
fi

echo ""
echo "============================================"
echo "Step 2/3: Parsing all datasets in parallel (${NUM_GPUS} GPUs)"
echo "============================================"

# Run parsing jobs in parallel across GPUs.
# Each job gets its own GPU via CUDA_VISIBLE_DEVICES.
PARSE_PIDS=()

echo "  [GPU 0] Parsing ESK verbs..."
CUDA_VISIBLE_DEVICES=0 uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$ESK_PATH" \
    --split train \
    --label-type verbs \
    --batch-size "$BATCH_SIZE" \
    --output "$PROGRAMS_DIR/programs_verbs_train.json" &
PARSE_PIDS+=($!)

echo "  [GPU $((NUM_GPUS > 1 ? 1 : 0))] Parsing ESK activities..."
CUDA_VISIBLE_DEVICES=$((NUM_GPUS > 1 ? 1 : 0)) uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$ESK_PATH" \
    --split train \
    --label-type activity \
    --batch-size "$BATCH_SIZE" \
    --output "$PROGRAMS_DIR/programs_activity_train.json" &
PARSE_PIDS+=($!)

# Wait for one GPU to free up, then launch HumanAct12 parsing
wait "${PARSE_PIDS[0]}" || { echo "ERROR: ESK verbs parsing failed"; exit 1; }
echo "  ESK verbs done → [GPU 0] Parsing HumanAct12..."

CUDA_VISIBLE_DEVICES=0 uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$HUMANACT12_PATH" \
    --split train \
    --label-type actions \
    --batch-size "$BATCH_SIZE" \
    --output "$PROGRAMS_DIR/programs_humanact12_train.json" &
PARSE_PIDS+=($!)

# Wait for remaining parse jobs
wait "${PARSE_PIDS[1]}" || { echo "ERROR: ESK activities parsing failed"; exit 1; }
echo "  ESK activities done."
wait "${PARSE_PIDS[2]}" || { echo "ERROR: HumanAct12 parsing failed"; exit 1; }
echo "  HumanAct12 done."

echo "  All parsing complete."
echo "    ESK verbs:      $PROGRAMS_DIR/programs_verbs_train.json"
echo "    ESK activities:  $PROGRAMS_DIR/programs_activity_train.json"
echo "    HumanAct12:      $PROGRAMS_DIR/programs_humanact12_train.json"

echo ""
echo "============================================"
echo "Step 3/3: Building executable models in parallel (CPU, ${NUM_CPUS} cores)"
echo "============================================"

# Build-models is CPU-only — run all 3 concurrently
# Use --program-budget with TF-IDF diversity selection (not random sampling)
BUILD_PIDS=()

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_verbs_train.json" \
    --output "$MODELS_DIR/models_verbs.json" \
    --program-budget 110 \
    --selection-method tfidf \
    --validate &
BUILD_PIDS+=($!)

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_activity_train.json" \
    --output "$MODELS_DIR/models_activity.json" \
    --program-budget 110 \
    --selection-method tfidf \
    --validate &
BUILD_PIDS+=($!)

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_humanact12_train.json" \
    --output "$MODELS_DIR/models_humanact12.json" \
    --program-budget 110 \
    --selection-method tfidf \
    --validate &
BUILD_PIDS+=($!)

# Wait for all build jobs
FAIL=0
for i in "${!BUILD_PIDS[@]}"; do
    wait "${BUILD_PIDS[$i]}" || { echo "ERROR: build_models job $i failed"; FAIL=1; }
done
[[ "$FAIL" -eq 0 ]] || exit 1

echo "  All models built."

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Parser checkpoint:     $BEST_CHECKPOINT"
echo "  ESK verbs programs:    $PROGRAMS_DIR/programs_verbs_train.json"
echo "  ESK verbs models:      $MODELS_DIR/models_verbs.json"
echo "  ESK activity programs: $PROGRAMS_DIR/programs_activity_train.json"
echo "  ESK activity models:   $MODELS_DIR/models_activity.json"
echo "  HA12 programs:         $PROGRAMS_DIR/programs_humanact12_train.json"
echo "  HA12 models:           $MODELS_DIR/models_humanact12.json"
echo ""
echo "Next steps:"
echo "  - Data augmentation:   uv run scripts/parsing/augment_data.py --models <models.json>"
echo "  - Action segmentation: uv run scripts/tasks/segmentation.py --config-name segmentation/humanact12_100pt.yaml"
echo "  - Anomaly detection (edit distance): uv run scripts/tasks/anomaly_detection.py --config configs/anomaly_detection.yaml"
