#!/usr/bin/env bash
# ============================================================================
# Train parser on train_diverse.h5, then parse ESK + HumanAct12 datasets,
# then build executable activity models for both.
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

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG="configs/parser/parser.yaml"
ESK_PATH="../exact_data/benchmarks/esk"
HUMANACT12_PATH="../exact_data/benchmarks/humanact12"
PROGRAMS_DIR="../exact_data/programs/parsed"
MODELS_DIR="../exact_data/models"
mkdir -p "$PROGRAMS_DIR" "$MODELS_DIR"

# ─── Multi-GPU / performance environment ────────────────────────────────────
NUM_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "Detected $NUM_GPUS GPU(s)"

# NCCL optimizations for multi-GPU
export NCCL_P2P_LEVEL=NVL              # Use NVLink for peer-to-peer if available
export NCCL_IB_DISABLE=0               # Enable InfiniBand (if present)
export CUDA_DEVICE_MAX_CONNECTIONS=1    # Overlap compute and communication

# Prevent CPU thread oversubscription: each DataLoader worker + CUDA runtime
# spawns threads; cap OMP to avoid 188 vCPUs × N_workers contention.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Reduce CUDA memory fragmentation (recommended by PyTorch for large models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Step 1: Train the parser ──────────────────────────────────────────────
if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "============================================"
    echo "SKIPPING TRAINING — using existing checkpoint"
    echo "  Checkpoint: $CHECKPOINT"
    echo "============================================"
    BEST_CHECKPOINT="$CHECKPOINT"
else
    echo "============================================"
    echo "Step 1/7: Training parser on train_diverse.h5"
    echo "  GPUs: $NUM_GPUS"
    echo "============================================"

    if [[ "$NUM_GPUS" -gt 1 ]]; then
        TRAIN_CMD="uv run torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/parsing/train_parser.py --config $CONFIG"
    else
        TRAIN_CMD="uv run scripts/parsing/train_parser.py --config $CONFIG"
    fi

    if [[ -n "${RESUME_DIR:-}" ]]; then
        echo "  Resuming from: $RESUME_DIR"
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_DIR"
    fi

    $TRAIN_CMD

    # Find the output directory (most recent in results/parser/)
    OUTPUT_DIR=$(ls -dt results/parser/2* 2>/dev/null | head -1)
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: Could not find training output directory"
        exit 1
    fi
    echo "  Training output: $OUTPUT_DIR"

    # Find the best checkpoint
    # The trainer with load_best_model_at_end saves the best model directly
    # in the output directory, so we use that.
    BEST_CHECKPOINT="$OUTPUT_DIR"

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
echo "Step 2/7: Parsing ESK dataset (verbs)"
echo "============================================"

uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$ESK_PATH" \
    --split train \
    --label-type verbs \
    --batch-size "${BATCH_SIZE:-8}" \
    --output "$PROGRAMS_DIR/programs_verbs_train.json"

echo "  ESK verbs programs saved to: $PROGRAMS_DIR/programs_verbs_train.json"

echo ""
echo "============================================"
echo "Step 3/7: Parsing ESK dataset (activities)"
echo "============================================"

uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$ESK_PATH" \
    --split train \
    --label-type activity \
    --batch-size "${BATCH_SIZE:-8}" \
    --output "$PROGRAMS_DIR/programs_activity_train.json"

echo "  ESK activity programs saved to: $PROGRAMS_DIR/programs_activity_train.json"

echo ""
echo "============================================"
echo "Step 4/7: Parsing HumanAct12 dataset"
echo "============================================"

uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$HUMANACT12_PATH" \
    --split train \
    --label-type actions \
    --batch-size "${BATCH_SIZE:-8}" \
    --output "$PROGRAMS_DIR/programs_humanact12_train.json"

echo "  HumanAct12 programs saved to: $PROGRAMS_DIR/programs_humanact12_train.json"

echo ""
echo "============================================"
echo "Step 5/7: Building executable models (ESK verbs)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_verbs_train.json" \
    --output "$MODELS_DIR/models_verbs.json" \
    --validate

echo "  ESK verbs models saved to: $MODELS_DIR/models_verbs.json"

echo ""
echo "============================================"
echo "Step 6/7: Building executable models (ESK activities)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_activity_train.json" \
    --output "$MODELS_DIR/models_activity.json" \
    --validate

echo "  ESK activity models saved to: $MODELS_DIR/models_activity.json"

echo ""
echo "============================================"
echo "Step 7/7: Building executable models (HumanAct12)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$PROGRAMS_DIR/programs_humanact12_train.json" \
    --output "$MODELS_DIR/models_humanact12.json" \
    --validate

echo "  HumanAct12 models saved to: $MODELS_DIR/models_humanact12.json"

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
echo "  - Assessment (edit distance): uv run scripts/tasks/assessment.py --config configs/assessment.yaml"
