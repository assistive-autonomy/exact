#!/usr/bin/env bash
# ============================================================================
# Parse validation and test splits of ESK + HumanAct12 datasets using an
# already-trained parser checkpoint, then build executable activity models.
#
# This complements train_and_parse.sh which only parses the train split.
#
# Optimized for multi-GPU: parsing jobs run in parallel across GPUs.
# Build-models steps run in parallel (CPU-only).
#
# Usage:
#   # Using the most recent checkpoint (auto-detected)
#   bash scripts/parse_val_test.sh
#
#   # Using a specific checkpoint
#   CHECKPOINT=results/parser/20260304_090844/best_generation bash scripts/parse_val_test.sh
#
#   # Parse only val or only test
#   SPLITS=val bash scripts/parse_val_test.sh
#   SPLITS=test bash scripts/parse_val_test.sh
#
#   # Skip building executable models (just parse)
#   SKIP_BUILD=1 bash scripts/parse_val_test.sh
# ============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_CPUS=$(nproc)
echo "Detected ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"

# ─── Configuration ──────────────────────────────────────────────────────────
ESK_PATH="../exact_data/benchmarks/esk"
HUMANACT12_PATH="../exact_data/benchmarks/humanact12"
SPLITS="${SPLITS:-val test}"    # Space-separated list of splits to parse
SKIP_BUILD="${SKIP_BUILD:-0}"
PROGRAMS_DIR="../exact_data/programs/parsed"
MODELS_DIR="../exact_data/models"
mkdir -p "$PROGRAMS_DIR" "$MODELS_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Parsing batch size — H200 144GB can handle large batches for inference
BATCH_SIZE="${BATCH_SIZE:-256}"

# ─── Resolve checkpoint ────────────────────────────────────────────────────
if [[ -n "${CHECKPOINT:-}" ]]; then
    BEST_CHECKPOINT="$CHECKPOINT"
else
    # Auto-detect most recent training output
    OUTPUT_DIR=$(ls -dt results/parser/2* 2>/dev/null | head -1)
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: No training output found in results/parser/"
        echo "  Provide a checkpoint via: CHECKPOINT=<path> bash scripts/parse_val_test.sh"
        exit 1
    fi
    # Prefer best-generation checkpoint if available
    if [[ -f "$OUTPUT_DIR/best_generation/prefix_parser.pt" ]]; then
        BEST_CHECKPOINT="$OUTPUT_DIR/best_generation"
    else
        BEST_CHECKPOINT="$OUTPUT_DIR"
    fi
fi

# Verify checkpoint
if [[ ! -f "$BEST_CHECKPOINT/prefix_parser.pt" ]]; then
    echo "ERROR: prefix_parser.pt not found in $BEST_CHECKPOINT"
    ls -la "$BEST_CHECKPOINT/" 2>/dev/null || true
    exit 1
fi
echo "Using checkpoint: $BEST_CHECKPOINT"
echo "Splits to parse:  $SPLITS"
echo ""

# ─── Step 1: Parse val/test splits in parallel ─────────────────────────────
echo "============================================"
echo "Step 1/2: Parsing val/test splits (${NUM_GPUS} GPUs)"
echo "============================================"

PARSE_PIDS=()
GPU_IDX=0

for SPLIT in $SPLITS; do
    echo "  Parsing ESK verbs ($SPLIT)..."
    CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$ESK_PATH" \
        --split "$SPLIT" \
        --label-type verbs \
        --batch-size "$BATCH_SIZE" \
        --output "$PROGRAMS_DIR/programs_verbs_${SPLIT}.json" &
    PARSE_PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    echo "  Parsing ESK activities ($SPLIT)..."
    CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$ESK_PATH" \
        --split "$SPLIT" \
        --label-type activity \
        --batch-size "$BATCH_SIZE" \
        --output "$PROGRAMS_DIR/programs_activity_${SPLIT}.json" &
    PARSE_PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    echo "  Parsing HumanAct12 ($SPLIT)..."
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

# Wait for all parse jobs
FAIL=0
for i in "${!PARSE_PIDS[@]}"; do
    wait "${PARSE_PIDS[$i]}" || { echo "ERROR: parse job $i (PID ${PARSE_PIDS[$i]}) failed"; FAIL=1; }
done
[[ "$FAIL" -eq 0 ]] || exit 1

echo ""
echo "  All parsing complete."
for SPLIT in $SPLITS; do
    echo "    ESK verbs ($SPLIT):      $PROGRAMS_DIR/programs_verbs_${SPLIT}.json"
    echo "    ESK activities ($SPLIT):  $PROGRAMS_DIR/programs_activity_${SPLIT}.json"
    echo "    HumanAct12 ($SPLIT):      $PROGRAMS_DIR/programs_humanact12_${SPLIT}.json"
done

# ─── Step 2: Build executable models ───────────────────────────────────────
if [[ "$SKIP_BUILD" -eq 1 ]]; then
    echo ""
    echo "Skipping model building (SKIP_BUILD=1)"
else
    echo ""
    echo "============================================"
    echo "Step 2/2: Building executable models (CPU, ${NUM_CPUS} cores)"
    echo "============================================"

    BUILD_PIDS=()

    for SPLIT in $SPLITS; do
        uv run scripts/parsing/build_models.py \
            --programs "$PROGRAMS_DIR/programs_verbs_${SPLIT}.json" \
            --output "$MODELS_DIR/models_verbs_${SPLIT}.json" \
            --program-budget 110 \
            --selection-method tfidf \
            --validate &
        BUILD_PIDS+=($!)

        uv run scripts/parsing/build_models.py \
            --programs "$PROGRAMS_DIR/programs_activity_${SPLIT}.json" \
            --output "$MODELS_DIR/models_activity_${SPLIT}.json" \
            --program-budget 110 \
            --selection-method tfidf \
            --validate &
        BUILD_PIDS+=($!)

        uv run scripts/parsing/build_models.py \
            --programs "$PROGRAMS_DIR/programs_humanact12_${SPLIT}.json" \
            --output "$MODELS_DIR/models_humanact12_${SPLIT}.json" \
            --program-budget 110 \
            --selection-method tfidf \
            --validate &
        BUILD_PIDS+=($!)
    done

    FAIL=0
    for i in "${!BUILD_PIDS[@]}"; do
        wait "${BUILD_PIDS[$i]}" || { echo "ERROR: build_models job $i failed"; FAIL=1; }
    done
    [[ "$FAIL" -eq 0 ]] || exit 1

    echo "  All models built."
fi

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Parser checkpoint: $BEST_CHECKPOINT"
for SPLIT in $SPLITS; do
    echo "  --- $SPLIT ---"
    echo "    ESK verbs programs:    $PROGRAMS_DIR/programs_verbs_${SPLIT}.json"
    echo "    ESK activity programs: $PROGRAMS_DIR/programs_activity_${SPLIT}.json"
    echo "    HumanAct12 programs:   $PROGRAMS_DIR/programs_humanact12_${SPLIT}.json"
    if [[ "$SKIP_BUILD" -ne 1 ]]; then
        echo "    ESK verbs models:      $MODELS_DIR/models_verbs_${SPLIT}.json"
        echo "    ESK activity models:   $MODELS_DIR/models_activity_${SPLIT}.json"
        echo "    HumanAct12 models:     $MODELS_DIR/models_humanact12_${SPLIT}.json"
    fi
done
