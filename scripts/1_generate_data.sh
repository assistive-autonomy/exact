#!/usr/bin/env bash
# =============================================================================
# Stage 1: Generate Synthetic Training Data
#
# Generates 10 diverse subsets of synthetic motion-program pairs and merges
# them into a single train_diverse.h5 for parser training.
#
# Usage:
#   bash scripts/1_generate_data.sh [output_dir] [total_samples] [num_workers]
#
# Outputs:
#   ../exact_data/programs/synthetic/train_diverse.h5   (merged training file)
#   ../exact_data/programs/synthetic/diverse_subsets/   (individual HDF5 subsets)
#
# GPU not required; this is a CPU-only step.
# =============================================================================
set -euo pipefail

cd /pvc/exact

OUTPUT_DIR="${1:-../exact_data/programs/synthetic}"
TOTAL_SAMPLES="${2:-50000}"
NUM_WORKERS="${3:-8}"

mkdir -p "$OUTPUT_DIR/diverse_subsets"

echo "=============================================="
echo "Stage 1: Generating Synthetic Training Data"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Total samples:    $TOTAL_SAMPLES"
echo "Workers per job:  $NUM_WORKERS"
echo ""

SAMPLES_PER_SUBSET=$((TOTAL_SAMPLES / 10))
SUBSET_DIR="$OUTPUT_DIR/diverse_subsets"

echo "Generating $SAMPLES_PER_SUBSET samples per subset (10 subsets)"
echo ""

# Subset 1: Simple programs (1-2 segments, 1-2 predicates)
echo "[1/10] Simple programs (1-2 segments, 1-2 predicates)..."
uv run scripts/data/generate_data.py \
    --name subset_01_simple \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 1001 \
    --num-workers $NUM_WORKERS \
    --min-preds 1 \
    --max-preds 2 \
    --num-intervals 2 \
    --min-interval-time 200

# Subset 2: Complex programs (8-10 segments, 4-5 predicates)
echo "[2/10] Complex programs (8-10 segments, 4-5 predicates)..."
uv run scripts/data/generate_data.py \
    --name subset_02_complex \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 2002 \
    --num-workers $NUM_WORKERS \
    --min-preds 4 \
    --max-preds 5 \
    --num-intervals 10 \
    --min-interval-time 50

# Subset 3: Upper body focus
echo "[3/10] Upper body focus..."
uv run scripts/data/generate_data.py \
    --name subset_03_upper_body \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 3003 \
    --num-workers $NUM_WORKERS \
    --allowed-parts "chest,neck,head,lthorax,lshoulder,lelbow,lwrist,lhand,rthorax,rshoulder,relbow,rwrist,rhand"

# Subset 4: Lower body focus
echo "[4/10] Lower body focus..."
uv run scripts/data/generate_data.py \
    --name subset_04_lower_body \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 4004 \
    --num-workers $NUM_WORKERS \
    --allowed-parts "pelvis,lhip,lknee,lankle,ltoe,rhip,rknee,rankle,rtoe"

# Subset 5: Subtle movements (low multipliers 0.0-1.0)
echo "[5/10] Subtle movements (low multipliers)..."
uv run scripts/data/generate_data.py \
    --name subset_05_subtle \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 5005 \
    --num-workers $NUM_WORKERS \
    --min-value 0.0 \
    --max-value 1.0 \
    --value-step 0.1

# Subset 6: Extreme movements (high multipliers 1.0-2.0)
echo "[6/10] Extreme movements (high multipliers)..."
uv run scripts/data/generate_data.py \
    --name subset_06_extreme \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 6006 \
    --num-workers $NUM_WORKERS \
    --min-value 1.0 \
    --max-value 2.0 \
    --value-step 0.1

# Subset 7: Short precise segments (many short intervals)
echo "[7/10] Short precise segments..."
uv run scripts/data/generate_data.py \
    --name subset_07_short_segments \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 7007 \
    --num-workers $NUM_WORKERS \
    --num-intervals 15 \
    --min-interval-time 30 \
    --max-preds 3

# Subset 8: Long broad segments (few long intervals)
echo "[8/10] Long broad segments..."
uv run scripts/data/generate_data.py \
    --name subset_08_long_segments \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 8008 \
    --num-workers $NUM_WORKERS \
    --num-intervals 3 \
    --min-interval-time 300

# Subset 9: Single axis focus (x / y / z)
echo "[9/10] Single axis focus (x, y, z)..."
AXIS_SAMPLES=$((SAMPLES_PER_SUBSET / 3))

uv run scripts/data/generate_data.py \
    --name subset_09a_x_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9009 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "x"

uv run scripts/data/generate_data.py \
    --name subset_09b_y_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9010 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "y"

uv run scripts/data/generate_data.py \
    --name subset_09c_z_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9011 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "z"

# Subset 10: Mixed diversity
echo "[10/10] Mixed diversity (balanced)..."
uv run scripts/data/generate_data.py \
    --name subset_10_mixed \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 10010 \
    --num-workers $NUM_WORKERS \
    --min-preds 1 \
    --max-preds 5 \
    --num-intervals 10 \
    --min-interval-time 50

echo ""
echo "=============================================="
echo "Merging all subsets into single training file"
echo "=============================================="

uv run scripts/data/merge_datasets.py \
    --input-dir "$SUBSET_DIR" \
    --output "$OUTPUT_DIR/train_diverse.h5" \
    --shuffle

echo ""
echo "=============================================="
echo "Stage 1 complete"
echo "=============================================="
echo "  Merged training data: $OUTPUT_DIR/train_diverse.h5"
echo ""
echo "Next: bash scripts/2_train_and_parse.sh"
