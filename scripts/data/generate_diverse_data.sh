#!/bin/bash
#
# Generate Diverse Training Data
#
# This script generates multiple datasets with different parameter configurations
# to create a more diverse training set for motion-to-program generation.
#
# Diversity strategies:
#   1. Varying number of segments (simple to complex programs)
#   2. Varying number of predicates per segment
#   3. Different body part subsets (upper body, lower body, full body)
#   4. Different value ranges (subtle to extreme movements)
#   5. Different interval lengths (short/precise to long/broad segments)
#
# Usage: ./scripts/generate_diverse_data.sh [output_dir] [total_samples]
#

set -e  # Exit on error

OUTPUT_DIR="${1:-../exact_data/programs/synthetic}"
TOTAL_SAMPLES="${2:-50000}"
NUM_WORKERS="${3:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR/diverse_subsets"

echo "=============================================="
echo "Generating Diverse Training Data"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Total samples: $TOTAL_SAMPLES"
echo "Workers per job: $NUM_WORKERS"
echo ""

# Calculate samples per subset (we have 10 diverse subsets)
SAMPLES_PER_SUBSET=$((TOTAL_SAMPLES / 10))

# Temporary directory for subset files
SUBSET_DIR="$OUTPUT_DIR/diverse_subsets"

echo "Generating $SAMPLES_PER_SUBSET samples per subset (10 subsets)"
echo ""

# ----------------------------------------------------------
# Subset 1: Simple programs (1-2 segments, 1-2 predicates)
# ----------------------------------------------------------
echo "[1/10] Simple programs (1-2 segments, 1-2 predicates)..."
uv run python scripts/data/generate_data.py \
    --name subset_01_simple \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 1001 \
    --num-workers $NUM_WORKERS \
    --min-preds 1 \
    --max-preds 2 \
    --num-intervals 2 \
    --min-interval-time 200

# ----------------------------------------------------------
# Subset 2: Complex programs (8-10 segments, 4-5 predicates)
# ----------------------------------------------------------
echo "[2/10] Complex programs (8-10 segments, 4-5 predicates)..."
uv run python scripts/data/generate_data.py \
    --name subset_02_complex \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 2002 \
    --num-workers $NUM_WORKERS \
    --min-preds 4 \
    --max-preds 5 \
    --num-intervals 10 \
    --min-interval-time 50

# ----------------------------------------------------------
# Subset 3: Upper body focus
# ----------------------------------------------------------
echo "[3/10] Upper body focus..."
uv run python scripts/data/generate_data.py \
    --name subset_03_upper_body \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 3003 \
    --num-workers $NUM_WORKERS \
    --allowed-parts "chest,neck,head,lthorax,lshoulder,lelbow,lwrist,lhand,rthorax,rshoulder,relbow,rwrist,rhand"

# ----------------------------------------------------------
# Subset 4: Lower body focus
# ----------------------------------------------------------
echo "[4/10] Lower body focus..."
uv run python scripts/data/generate_data.py \
    --name subset_04_lower_body \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 4004 \
    --num-workers $NUM_WORKERS \
    --allowed-parts "pelvis,lhip,lknee,lankle,ltoe,rhip,rknee,rankle,rtoe"

# ----------------------------------------------------------
# Subset 5: Subtle movements (low multipliers 0.0-1.0)
# ----------------------------------------------------------
echo "[5/10] Subtle movements (low multipliers)..."
uv run python scripts/data/generate_data.py \
    --name subset_05_subtle \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 5005 \
    --num-workers $NUM_WORKERS \
    --min-value 0.0 \
    --max-value 1.0 \
    --value-step 0.1

# ----------------------------------------------------------
# Subset 6: Extreme movements (high multipliers 1.0-2.0)
# ----------------------------------------------------------
echo "[6/10] Extreme movements (high multipliers)..."
uv run python scripts/data/generate_data.py \
    --name subset_06_extreme \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 6006 \
    --num-workers $NUM_WORKERS \
    --min-value 1.0 \
    --max-value 2.0 \
    --value-step 0.1

# ----------------------------------------------------------
# Subset 7: Short precise segments (many short intervals)
# ----------------------------------------------------------
echo "[7/10] Short precise segments..."
uv run python scripts/data/generate_data.py \
    --name subset_07_short_segments \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 7007 \
    --num-workers $NUM_WORKERS \
    --num-intervals 15 \
    --min-interval-time 30 \
    --max-preds 3

# ----------------------------------------------------------
# Subset 8: Long broad segments (few long intervals)
# ----------------------------------------------------------
echo "[8/10] Long broad segments..."
uv run python scripts/data/generate_data.py \
    --name subset_08_long_segments \
    --num-samples $SAMPLES_PER_SUBSET \
    --output-dir "$SUBSET_DIR" \
    --seed 8008 \
    --num-workers $NUM_WORKERS \
    --num-intervals 3 \
    --min-interval-time 300

# ----------------------------------------------------------
# Subset 9: Single axis focus (x-axis only, y-axis only, z-axis only mixed)
# ----------------------------------------------------------
echo "[9/10] Single axis focus (mixed)..."
# Split into 3 sub-batches for each axis
AXIS_SAMPLES=$((SAMPLES_PER_SUBSET / 3))

uv run python scripts/data/generate_data.py \
    --name subset_09a_x_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9009 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "x"

uv run python scripts/data/generate_data.py \
    --name subset_09b_y_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9010 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "y"

uv run python scripts/data/generate_data.py \
    --name subset_09c_z_axis \
    --num-samples $AXIS_SAMPLES \
    --output-dir "$SUBSET_DIR" \
    --seed 9011 \
    --num-workers $NUM_WORKERS \
    --allowed-axes "z"

# ----------------------------------------------------------
# Subset 10: Mixed diversity (balanced parameters)
# ----------------------------------------------------------
echo "[10/10] Mixed diversity (balanced)..."
uv run python scripts/data/generate_data.py \
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

# Merge all HDF5 files
uv run python scripts/data/merge_datasets.py \
    --input-dir "$SUBSET_DIR" \
    --output "$OUTPUT_DIR/train_diverse.h5" \
    --shuffle

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Generated diverse training data:"
echo "  - Subsets: $SUBSET_DIR/"
echo "  - Merged:  $OUTPUT_DIR/train_diverse.h5"
echo ""
echo "To use for training, update your config.yaml:"
echo "  train_data: $OUTPUT_DIR/train_diverse.h5"
echo ""
