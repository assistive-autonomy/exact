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
CONFIG="configs/parser.yaml"
ESK_PATH="../esk"
HUMANACT12_PATH="../humanact12"

# ─── Step 1: Train the parser ──────────────────────────────────────────────
if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "============================================"
    echo "SKIPPING TRAINING — using existing checkpoint"
    echo "  Checkpoint: $CHECKPOINT"
    echo "============================================"
    BEST_CHECKPOINT="$CHECKPOINT"
else
    echo "============================================"
    echo "Step 1/5: Training parser on train_diverse.h5"
    echo "============================================"

    TRAIN_CMD="uv run scripts/parsing/train_parser.py --config $CONFIG"

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
echo "Step 2/5: Parsing ESK dataset"
echo "============================================"

uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$ESK_PATH" \
    --split train \
    --label-type verbs \
    --output "$ESK_PATH/programs_train.json"

echo "  ESK programs saved to: $ESK_PATH/programs_train.json"

echo ""
echo "============================================"
echo "Step 3/5: Parsing HumanAct12 dataset"
echo "============================================"

uv run scripts/parsing/parse_esk.py \
    --parser-checkpoint "$BEST_CHECKPOINT" \
    --esk-path "$HUMANACT12_PATH" \
    --split train \
    --label-type actions \
    --output "$HUMANACT12_PATH/programs_train.json"

echo "  HumanAct12 programs saved to: $HUMANACT12_PATH/programs_train.json"

echo ""
echo "============================================"
echo "Step 4/5: Building executable models (ESK)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$ESK_PATH/programs_train.json" \
    --output "$ESK_PATH/models.json" \
    --validate

echo "  ESK models saved to: $ESK_PATH/models.json"

echo ""
echo "============================================"
echo "Step 5/5: Building executable models (HumanAct12)"
echo "============================================"

uv run scripts/parsing/build_models.py \
    --programs "$HUMANACT12_PATH/programs_train.json" \
    --output "$HUMANACT12_PATH/models.json" \
    --validate

echo "  HumanAct12 models saved to: $HUMANACT12_PATH/models.json"

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Parser checkpoint: $BEST_CHECKPOINT"
echo "  ESK programs:      $ESK_PATH/programs_train.json"
echo "  ESK models:        $ESK_PATH/models.json"
echo "  HA12 programs:     $HUMANACT12_PATH/programs_train.json"
echo "  HA12 models:       $HUMANACT12_PATH/models.json"
echo ""
echo "Next steps:"
echo "  - Data augmentation:   uv run scripts/parsing/augment_data.py --models <models.json>"
echo "  - Action segmentation: uv run scripts/tasks/segmentation.py --config-name segmentation/humanact12_100pt.yaml"
echo "  - Assessment (edit distance): uv run scripts/tasks/assessment.py --config configs/assessment.yaml"
