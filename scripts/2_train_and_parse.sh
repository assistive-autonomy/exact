#!/usr/bin/env bash
# =============================================================================
# Stage 2: Train Parser, Parse All Splits, Build Executable Models
#
# Optimised for multi-GPU (tested on 2× H200): training uses DDP, parsing
# jobs are distributed across available GPUs, model building is CPU-parallel.
#
# Steps:
#   1. Train motion-prefix parser on train_diverse.h5 (DDP, all GPUs)
#   2. Parse train / val / test splits for ESK verbs, ESK activities, HumanAct12
#   3. Build executable ActivityModelCollection from the train-split programs
#
# Usage:
#   # Full run (train + parse + build)
#   bash scripts/2_train_and_parse.sh
#
#   # Skip training, use existing checkpoint
#   CHECKPOINT=results/parser/20260304_090844/best_generation \
#       bash scripts/2_train_and_parse.sh
#
#   # Resume interrupted training
#   RESUME_DIR=results/parser/20260304_090844 \
#       bash scripts/2_train_and_parse.sh
#
#   # Parse only specific splits (space-separated)
#   SPLITS="val test" bash scripts/2_train_and_parse.sh
#
#   # Skip building executable models
#   SKIP_BUILD=1 bash scripts/2_train_and_parse.sh
# =============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc)
echo "Detected ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/parser/parser.yaml}"
ESK_PATH="../exact_data/benchmarks/esk"
HUMANACT12_PATH="../exact_data/benchmarks/humanact12"
PROGRAMS_DIR="../exact_data/programs/parsed"
MODELS_DIR="../exact_data/models"
SPLITS="${SPLITS:-train val test}"   # Which splits to parse (space-separated)
SKIP_BUILD="${SKIP_BUILD:-0}"

mkdir -p "$PROGRAMS_DIR" "$MODELS_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE="${BATCH_SIZE:-256}"

# ─── Step 1: Train parser ────────────────────────────────────────────────────
if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "============================================"
    echo "Step 1/3: SKIPPING training — using checkpoint"
    echo "  Checkpoint: $CHECKPOINT"
    echo "============================================"
    BEST_CHECKPOINT="$CHECKPOINT"
else
    echo "============================================"
    echo "Step 1/3: Training parser (${NUM_GPUS} GPUs, DDP)"
    echo "============================================"

    TRAIN_ARGS="--config $CONFIG"
    if [[ -n "${RESUME_DIR:-}" ]]; then
        echo "  Resuming from: $RESUME_DIR"
        TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME_DIR"
    fi

    if [[ "$NUM_GPUS" -gt 1 ]]; then
        uv run torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port="${MASTER_PORT:-29500}" \
            scripts/parsing/train_parser.py $TRAIN_ARGS
    else
        uv run scripts/parsing/train_parser.py $TRAIN_ARGS
    fi

    OUTPUT_DIR=$(ls -dt results/parser/2* 2>/dev/null | head -1)
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: Could not find training output directory"
        exit 1
    fi
    echo "  Training output: $OUTPUT_DIR"

    BEST_CHECKPOINT="$OUTPUT_DIR"
    # Prefer best-generation checkpoint (tracks edit distance, the actual target metric)
    if [[ -f "$OUTPUT_DIR/best_generation/prefix_parser.pt" ]]; then
        BEST_CHECKPOINT="$OUTPUT_DIR/best_generation"
        echo "  Using best-generation checkpoint (by edit distance)"
    fi

    if [[ ! -f "$BEST_CHECKPOINT/prefix_parser.pt" ]]; then
        echo "ERROR: prefix_parser.pt not found in $BEST_CHECKPOINT"
        ls -la "$BEST_CHECKPOINT/"
        exit 1
    fi
    echo "  Best checkpoint: $BEST_CHECKPOINT"
fi

# ─── Step 2: Parse all splits in parallel ───────────────────────────────────
echo ""
echo "============================================"
echo "Step 2/3: Parsing all splits in parallel (${NUM_GPUS} GPUs)"
echo "  Splits: $SPLITS"
echo "============================================"

PARSE_PIDS=()
GPU_IDX=0

for SPLIT in $SPLITS; do
    echo "  [GPU $((GPU_IDX % NUM_GPUS))] ESK verbs ($SPLIT)..."
    CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$ESK_PATH" \
        --split "$SPLIT" \
        --label-type verbs \
        --batch-size "$BATCH_SIZE" \
        --output "$PROGRAMS_DIR/programs_verbs_${SPLIT}.json" &
    PARSE_PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    echo "  [GPU $((GPU_IDX % NUM_GPUS))] ESK activities ($SPLIT)..."
    CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$ESK_PATH" \
        --split "$SPLIT" \
        --label-type activity \
        --batch-size "$BATCH_SIZE" \
        --output "$PROGRAMS_DIR/programs_activity_${SPLIT}.json" &
    PARSE_PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    echo "  [GPU $((GPU_IDX % NUM_GPUS))] HumanAct12 ($SPLIT)..."
    CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$HUMANACT12_PATH" \
        --split "$SPLIT" \
        --label-type actions \
        --batch-size "$BATCH_SIZE" \
        --output "$PROGRAMS_DIR/programs_humanact12_${SPLIT}.json" &
    PARSE_PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))
done

FAIL=0
for i in "${!PARSE_PIDS[@]}"; do
    wait "${PARSE_PIDS[$i]}" || { echo "ERROR: parse job $i failed"; FAIL=1; }
done
[[ "$FAIL" -eq 0 ]] || exit 1

echo "  All parsing complete."
for SPLIT in $SPLITS; do
    echo "    programs_verbs_${SPLIT}.json"
    echo "    programs_activity_${SPLIT}.json"
    echo "    programs_humanact12_${SPLIT}.json"
done

# ─── Step 3: Build executable models (train split only) ──────────────────────
if [[ "$SKIP_BUILD" -eq 1 ]]; then
    echo ""
    echo "Skipping model building (SKIP_BUILD=1)"
else
    echo ""
    echo "============================================"
    echo "Step 3/3: Building executable models from train split (CPU, ${NUM_CPUS} cores)"
    echo "============================================"

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

    FAIL=0
    for i in "${!BUILD_PIDS[@]}"; do
        wait "${BUILD_PIDS[$i]}" || { echo "ERROR: build_models job $i failed"; FAIL=1; }
    done
    [[ "$FAIL" -eq 0 ]] || exit 1

    echo "  All models built."
fi

echo ""
echo "============================================"
echo "Stage 2 complete"
echo "============================================"
echo "  Parser checkpoint:    $BEST_CHECKPOINT"
echo "  Parsed programs:      $PROGRAMS_DIR/"
echo "  Executable models:    $MODELS_DIR/"
echo ""
echo "Next steps:"
echo "  Segmentation augmentation:  bash scripts/3_generate_augmented.sh"
echo "  Segmentation experiments:   bash scripts/4_run_segmentation.sh"
echo "  Anomaly detection:          bash scripts/5_run_anomaly_detection.sh"
