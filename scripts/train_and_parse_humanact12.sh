#!/usr/bin/env bash
# ============================================================================
# Train parser then parse HumanAct12 only (faster development iteration).
#
# This is a lightweight version of train_and_parse.sh that skips ESK parsing
# to enable faster experimentation on a smaller dataset.
#
# Usage:
#   bash scripts/train_and_parse_humanact12.sh
#
# With config override:
#   CONFIG=configs/parser/parser_v2.yaml bash scripts/train_and_parse_humanact12.sh
#
# To resume from a previous run:
#   RESUME_DIR=results/parser/20260222_230000 bash scripts/train_and_parse_humanact12.sh
#
# To skip training and only parse (provide existing checkpoint):
#   CHECKPOINT=results/parser/20260222_230000 bash scripts/train_and_parse_humanact12.sh
# ============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc)
echo "Detected ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/parser/parser_v2.yaml}"
HUMANACT12_PATH="../humanact12"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    echo "Step 1/3: Training parser (${NUM_GPUS} GPUs, DDP)"
    echo "  Config: $CONFIG"
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

    # Find the output directory (most recent in results/parser/)
    OUTPUT_DIR=$(ls -dt results/parser/2* 2>/dev/null | head -1)
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: Could not find training output directory"
        exit 1
    fi
    echo "  Training output: $OUTPUT_DIR"

    BEST_CHECKPOINT="$OUTPUT_DIR"

    if [[ ! -f "$BEST_CHECKPOINT/prefix_parser.pt" ]]; then
        echo "ERROR: prefix_parser.pt not found in $BEST_CHECKPOINT"
        ls -la "$BEST_CHECKPOINT/"
        exit 1
    fi

    echo "  Best checkpoint: $BEST_CHECKPOINT"
fi

echo ""
echo "============================================"
echo "Step 2/3: Parsing HumanAct12 only (${NUM_GPUS} GPUs)"
echo "============================================"

# Parse with first available GPU
echo "  [GPU 0] Parsing HumanAct12..."
CUDA_VISIBLE_DEVICES=0 uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$HUMANACT12_PATH" \
    --split train \
    --label-type actions \
    --batch-size "$BATCH_SIZE" \
    --output "$HUMANACT12_PATH/programs_train.json"

echo "  HumanAct12 parsing complete."
echo "    Output: $HUMANACT12_PATH/programs_train.json"

echo ""
echo "============================================"
echo "Step 3/3: Building executable models (CPU)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$HUMANACT12_PATH/programs_train.json" \
    --output "$HUMANACT12_PATH/models.json" \
    --validate

echo "  Models built."

echo ""
echo "============================================"
echo "ALL DONE (HumanAct12 only)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Parser checkpoint:  $BEST_CHECKPOINT"
echo "  HA12 programs:      $HUMANACT12_PATH/programs_train.json"
echo "  HA12 models:        $HUMANACT12_PATH/models.json"
echo ""
echo "Next steps:"
echo "  - Evaluate segmentation: uv run scripts/tasks/segmentation.py --config-name segmentation/humanact12_100pt.yaml"
echo "  - Or run full pipeline:  bash scripts/train_and_parse.sh"
