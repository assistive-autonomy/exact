#!/usr/bin/env bash
# =============================================================================
# Stage 2: Train Parser, Parse All Splits, Build Executable Models
#
# Optimised for multi-GPU (tested on 4× H200): training uses DDP, parsing
# jobs are distributed across available GPUs, model building is CPU-parallel.
#
# Steps:
#   1. Train motion-prefix parser on train_diverse.h5 (DDP, all GPUs)
#   2. Parse train / val / test splits for ESK verbs, ESK activities, HumanAct12
#      (uses enhanced single-sample parsing for improved yield)
#   3. Build executable ActivityModelCollection from the train-split programs
#   4. Upload parsed programs, models and parser checkpoint to HF Hub via git
#
# Usage:
#   # Full run (train + parse + build)
#   bash scripts/2_train_and_parse.sh
#
#   # Skip training, use existing checkpoint
#   CHECKPOINT=results/parser/20260309_101643/best_generation \
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
#
#   # Skip uploading to HF Hub
#   SKIP_UPLOAD=1 bash scripts/2_train_and_parse.sh
#
# Standalone parsing (use existing checkpoint without training):
#   # Parse verbs train split
#   CUDA_VISIBLE_DEVICES=0 uv run scripts/parsing/parse_esk.py \
#       --parser-checkpoint results/parser/20260309_101643/best_generation \
#       --esk-path ../exact_data/benchmarks/esk \
#       --split train --label-type verbs \
#       --output ../exact_data/programs/parsed/programs_verbs_train.json \
#       --num-passes 5 --num-retries 15
#
#   # Parse activity train split
#   CUDA_VISIBLE_DEVICES=1 uv run scripts/parsing/parse_esk.py \
#       --parser-checkpoint results/parser/20260309_101643/best_generation \
#       --esk-path ../exact_data/benchmarks/esk \
#       --split train --label-type activity \
#       --output ../exact_data/programs/parsed/programs_activity_train.json \
#       --num-passes 5 --num-retries 15
#
#   # Parse humanact12 train split
#   CUDA_VISIBLE_DEVICES=2 uv run scripts/parsing/parse_esk.py \
#       --parser-checkpoint results/parser/20260309_101643/best_generation \
#       --esk-path ../exact_data/benchmarks/humanact12 \
#       --split train --label-type actions \
#       --output ../exact_data/programs/parsed/programs_humanact12_train.json \
#       --num-passes 5 --num-retries 15
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
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
HF_DATA_REPO="../exact_data"          # HuggingFace Hub git clone of the data repo

# Enhanced parsing parameters
NUM_PASSES="${NUM_PASSES:-5}"
NUM_RETRIES="${NUM_RETRIES:-15}"

mkdir -p "$PROGRAMS_DIR" "$MODELS_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# ─── Step 2: Parse all splits ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "Step 2/3: Parsing all splits (${NUM_GPUS} GPUs)"
echo "  Splits: $SPLITS"
echo "============================================"

# Collect parse tasks: (GPU_ID, label_type, data_path, split)
PARSE_TASKS=()
GPU_IDX=0

for SPLIT in $SPLITS; do
    PARSE_TASKS+=("$((GPU_IDX % NUM_GPUS)) verbs     $ESK_PATH        $SPLIT")
    GPU_IDX=$((GPU_IDX + 1))
    PARSE_TASKS+=("$((GPU_IDX % NUM_GPUS)) activity  $ESK_PATH        $SPLIT")
    GPU_IDX=$((GPU_IDX + 1))
    PARSE_TASKS+=("$((GPU_IDX % NUM_GPUS)) actions   $HUMANACT12_PATH $SPLIT")
    GPU_IDX=$((GPU_IDX + 1))
done

launch_parse_job() {
    local gpu_id="$1" label_type="$2" data_path="$3" split="$4"
    local out_label="$label_type"
    [[ "$label_type" == "actions" ]] && out_label="humanact12"
    
    local output_file="$PROGRAMS_DIR/programs_${out_label}_${split}.json"
    echo "  [GPU ${gpu_id}] ${out_label} (${split})..."
    
    CUDA_VISIBLE_DEVICES="$gpu_id" uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint "$BEST_CHECKPOINT" \
        --esk-path "$data_path" \
        --split "$split" \
        --label-type "$label_type" \
        --output "$output_file" \
        --num-passes "$NUM_PASSES" \
        --num-retries "$NUM_RETRIES" \
        --greedy-fallback &
}

# Run parse tasks - one per GPU at a time (memory intensive)
PARSE_PIDS=()
for task in "${PARSE_TASKS[@]}"; do
    # shellcheck disable=SC2086
    launch_parse_job $task
    PARSE_PIDS+=($!)
    
    # Run only NUM_GPUS jobs concurrently
    if (( ${#PARSE_PIDS[@]} >= NUM_GPUS )); then
        wait "${PARSE_PIDS[0]}" || true
        PARSE_PIDS=("${PARSE_PIDS[@]:1}")
    fi
done

FAIL=0
for pid in "${PARSE_PIDS[@]}"; do
    wait "$pid" || { echo "ERROR: a parse job failed (pid=$pid)"; FAIL=1; }
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
        --program-budget 10 \
        --selection-method tfidf \
        --validate &
    BUILD_PIDS+=($!)

    uv run scripts/parsing/build_models.py \
        --programs "$PROGRAMS_DIR/programs_activity_train.json" \
        --output "$MODELS_DIR/models_activity.json" \
        --program-budget 10 \
        --selection-method tfidf \
        --validate &
    BUILD_PIDS+=($!)

    uv run scripts/parsing/build_models.py \
        --programs "$PROGRAMS_DIR/programs_humanact12_train.json" \
        --output "$MODELS_DIR/models_humanact12.json" \
        --program-budget 10 \
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

# ─── Step 4: Upload to Hugging Face Hub ──────────────────────────────────────
if [[ "$SKIP_UPLOAD" -eq 1 ]]; then
    echo ""
    echo "Skipping HF Hub upload (SKIP_UPLOAD=1)"
else
    echo ""
    echo "============================================"
    echo "Step 4/4: Uploading to Hugging Face Hub"
    echo "  Repo: $HF_DATA_REPO"
    echo "============================================"

    # Copy the parser checkpoint into the data repo so it is versioned
    # alongside the programs and models it produced.
    HF_CHECKPOINT_DIR="$HF_DATA_REPO/checkpoints/parser"
    mkdir -p "$HF_CHECKPOINT_DIR"
    echo "  Copying parser checkpoint → $HF_CHECKPOINT_DIR/"
    cp -r "$BEST_CHECKPOINT/." "$HF_CHECKPOINT_DIR/"

    pushd "$HF_DATA_REPO" > /dev/null

    # Ensure git-lfs tracks the binary / potentially-large file types before
    # staging.  This is a no-op when the patterns are already in .gitattributes.
    git lfs track "*.pt" "*.safetensors" "*.h5" "*.hdf5" "*.bin" 2>/dev/null || true

    # Stage changed artefacts (programs, models, checkpoint, updated .gitattributes)
    git add programs/ models/ checkpoints/ .gitattributes 2>/dev/null || \
        git add programs/ models/ checkpoints/

    if git diff --cached --quiet; then
        echo "  Nothing new to commit — HF data repo already up to date."
    else
        COMMIT_TS=$(date +'%Y-%m-%d %H:%M:%S')
        COMMIT_MSG="Add parsed programs, models and parser checkpoint (${COMMIT_TS})"
        git commit -m "$COMMIT_MSG"
       echo "  Pushing to HF Hub…"
        git push
        echo "  Upload complete."
    fi

    popd > /dev/null
fi

echo ""
echo "============================================"
echo "Stage 2 complete"
echo "============================================"
echo "  Parser checkpoint:    $BEST_CHECKPOINT"
echo "  Parsed programs:      $PROGRAMS_DIR/"
echo "  Executable models:    $MODELS_DIR/"
echo "  HF Hub repo:          $HF_DATA_REPO"
echo ""
echo "Next steps:"
echo "  Segmentation augmentation:  bash scripts/3_generate_augmented.sh"
echo "  Segmentation experiments:   bash scripts/4_run_segmentation.sh"
echo "  Anomaly detection:          bash scripts/5_run_anomaly_detection.sh"
