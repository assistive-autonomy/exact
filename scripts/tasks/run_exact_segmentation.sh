#!/usr/bin/env bash
# ============================================================================
# run_exact_segmentation.sh
#
# Generate synthetic training data from EXACT executable activity models
# using augment_data.py + BehaviourModel/MuJoCo rollouts.
#
# Optimized for multi-GPU + high CPU count:
#   - Both synth fractions per dataset run in parallel on separate GPU halves
#   - Each half uses batched MuJoCo envs + relabel workers scaled to CPU count
#   - z-vector pre-computation with configurable variants
#
# Synthetic data is written to self-contained output directories:
#   ../esk_synthetic/        (ESK activities + verbs)
#   ../humanact12_synthetic/ (HumanAct12)
#
# These directories mirror the original data layout so you can copy/symlink
# the synthetic files into the original data dirs on another machine and
# run segmentation directly:
#
#   # On the segmentation machine:
#   cp ../esk_synthetic/D2A_converted_pose_smpl/*       ../esk/D2A_converted_pose_smpl/
#   cp ../esk_synthetic/D2A_converted_label_activity/*   ../esk/D2A_converted_label_activity/
#   cp ../esk_synthetic/D2A_converted_label_verbs/*      ../esk/D2A_converted_label_verbs/
#   cp ../humanact12_synthetic/D2A_converted_pose_smpl/* ../humanact12/D2A_converted_pose_smpl/
#   cp ../humanact12_synthetic/D2A_converted_label_actions/* ../humanact12/D2A_converted_label_actions/
#
#   uv run scripts/tasks/segmentation.py --config-name segmentation/augmented/<config>
#
# Variations per dataset (3 datasets × 2 synth fractions = 6 outputs):
#   - 10% synthetic (relative to training set size)
#   - 20% synthetic
#
# Datasets:
#   - ESK activities  (6 classes)
#   - ESK verbs       (30 classes)
#   - HumanAct12      (12 classes)
#
# Prerequisites:
#   - Executable models already built (from scripts/train_and_parse.sh):
#       ../esk/models_activity.json
#       ../esk/models_verbs.json
#       ../humanact12/models.json
#
# Usage:
#   bash scripts/tasks/run_exact_segmentation.sh
#
# To use dry-run mode (random placeholder data, no MuJoCo needed):
#   DRY_RUN=1 bash scripts/tasks/run_exact_segmentation.sh
#
# To run a subset of datasets:
#   DATASETS="esk_activities humanact12" bash scripts/tasks/run_exact_segmentation.sh
# ============================================================================
set -euo pipefail

cd /pvc/exact

# ─── Hardware detection ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NUM_CPUS=$(nproc)
echo "Detected ${NUM_GPUS} GPUs, ${NUM_CPUS} CPUs"

# ─── Configuration ──────────────────────────────────────────────────────────
ESK_PATH="../esk"
HA12_PATH="../humanact12"
ESK_SYNTH_PATH="../esk_synthetic"
HA12_SYNTH_PATH="../humanact12_synthetic"

TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-100}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-esk_activities esk_verbs humanact12}"

SYNTH_FRACTIONS=(10 20)   # Percentage of training set to synthesize

# Performance tuning
BATCH_ENVS="${BATCH_ENVS:-32}"           # Parallel MuJoCo envs per GPU
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1024}"  # z-computation buffer size
Z_VARIANTS="${Z_VARIANTS:-4}"            # z-vector variants per timestep

# Scale relabel workers across all GPU workers: total CPUs / num_gpus
RELABEL_WORKERS=$(( NUM_CPUS / NUM_GPUS ))
if [[ "$RELABEL_WORKERS" -lt 4 ]]; then RELABEL_WORKERS=4; fi

echo "Using all $NUM_GPUS GPUs per generation job (sequential)"
echo "Relabel workers per GPU: $RELABEL_WORKERS"
echo "Batch envs per GPU: $BATCH_ENVS"
echo "Buffer batch size: $BUFFER_BATCH_SIZE"
echo "z-variants: $Z_VARIANTS"
echo ""

# Threading / NCCL tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Helper functions ───────────────────────────────────────────────────────

count_training_segments() {
    local programs_json="$1"
    python3 -c "
import json
d = json.load(open('$programs_json'))
total = sum(len(p) for p in d['programs_by_activity'].values())
print(total)
"
}

generate_synthetic_data() {
    # Generate synthetic data from an executable activity model collection.
    # Uses all available GPUs for maximum throughput.
    #
    # Args:
    #   $1 - models JSON path
    #   $2 - programs JSON path (to count training segments)
    #   $3 - synth percentage (e.g., 10 or 20)
    #   $4 - data directory (where pose .h5 will be placed)
    #   $5 - label directory (where label .pickle will be placed)
    #   $6 - video name for the synthetic data
    local models_json="$1"
    local programs_json="$2"
    local synth_pct="$3"
    local data_dir="$4"
    local label_dir="$5"
    local video_name="$6"

    # Create output directories
    mkdir -p "$data_dir" "$label_dir"

    # Check if already generated
    if [[ -f "$data_dir/${video_name}_pose3d_smpl.h5" ]] && \
       [[ -f "$label_dir/${video_name}_labels.pickle" ]]; then
        echo "    ✓ Already exists: $video_name (skipping)"
        return 0
    fi

    # Compute number of synthetic samples
    local total_segments
    total_segments=$(count_training_segments "$programs_json")
    local num_samples=$(( total_segments * synth_pct / 100 ))
    if [[ "$num_samples" -lt 1 ]]; then
        num_samples=1
    fi

    echo "    Generating $num_samples synthetic samples (${synth_pct}% of $total_segments segments) on $NUM_GPUS GPUs"

    local tmpdir
    tmpdir=$(mktemp -d /tmp/synth_XXXXXX)

    local dry_run_flag=""
    if [[ "$DRY_RUN" == "1" ]]; then
        dry_run_flag="--dry-run"
    fi

    uv run scripts/parsing/augment_data.py \
        --load-models "$models_json" \
        --num-samples "$num_samples" \
        --trajectory-length "$TRAJECTORY_LENGTH" \
        --output-dir "$tmpdir" \
        --video-name "$video_name" \
        --seed "$SEED" \
        --num-gpus "$NUM_GPUS" \
        --batch-envs "$BATCH_ENVS" \
        --buffer-batch-size "$BUFFER_BATCH_SIZE" \
        --relabel-workers "$RELABEL_WORKERS" \
        --z-variants "$Z_VARIANTS" \
        $dry_run_flag

    # Move generated files to output directories
    mv "$tmpdir/${video_name}_pose3d_smpl.h5" "$data_dir/"
    mv "$tmpdir/${video_name}_labels.pickle" "$label_dir/"
    rm -rf "$tmpdir"

    echo "    ✓ Placed: $data_dir/${video_name}_pose3d_smpl.h5"
    echo "    ✓ Placed: $label_dir/${video_name}_labels.pickle"
}

# ─── Generate synthetic training data ───────────────────────────────────────

echo "============================================"
echo "Generating synthetic training data"
echo "============================================"
echo ""

for dataset in $DATASETS; do
    case "$dataset" in
        esk_activities)
            models_json="$ESK_PATH/models_activity.json"
            programs_json="$ESK_PATH/programs_activity_train.json"
            data_dir="$ESK_SYNTH_PATH/D2A_converted_pose_smpl"
            label_dir="$ESK_SYNTH_PATH/D2A_converted_label_activity"
            video_prefix="synth_act"
            ;;
        esk_verbs)
            models_json="$ESK_PATH/models_verbs.json"
            programs_json="$ESK_PATH/programs_verbs_train.json"
            data_dir="$ESK_SYNTH_PATH/D2A_converted_pose_smpl"
            label_dir="$ESK_SYNTH_PATH/D2A_converted_label_verbs"
            video_prefix="synth_verb"
            ;;
        humanact12)
            models_json="$HA12_PATH/models.json"
            programs_json="$HA12_PATH/programs_train.json"
            data_dir="$HA12_SYNTH_PATH/D2A_converted_pose_smpl"
            label_dir="$HA12_SYNTH_PATH/D2A_converted_label_actions"
            video_prefix="synth_ha12"
            ;;
        *)
            echo "ERROR: Unknown dataset '$dataset'"
            exit 1
            ;;
    esac

    echo "  Dataset: $dataset"

    for synth_pct in "${SYNTH_FRACTIONS[@]}"; do
        video_name="${video_prefix}_${synth_pct}pct"
        generate_synthetic_data \
            "$models_json" \
            "$programs_json" \
            "$synth_pct" \
            "$data_dir" \
            "$label_dir" \
            "$video_name"
    done

    echo "  ✓ $dataset: all synth fractions done"
    echo ""
done

echo "  All synthetic data generated."
echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Synthetic data output:"
echo "  $ESK_SYNTH_PATH/"
find "$ESK_SYNTH_PATH" -type f 2>/dev/null | sort | sed 's/^/    /'
echo "  $HA12_SYNTH_PATH/"
find "$HA12_SYNTH_PATH" -type f 2>/dev/null | sort | sed 's/^/    /'
echo ""
echo "To run segmentation on another machine, copy synthetic files"
echo "into the original data directories:"
echo ""
echo "  cp $ESK_SYNTH_PATH/D2A_converted_pose_smpl/*         $ESK_PATH/D2A_converted_pose_smpl/"
echo "  cp $ESK_SYNTH_PATH/D2A_converted_label_activity/*    $ESK_PATH/D2A_converted_label_activity/"
echo "  cp $ESK_SYNTH_PATH/D2A_converted_label_verbs/*       $ESK_PATH/D2A_converted_label_verbs/"
echo "  cp $HA12_SYNTH_PATH/D2A_converted_pose_smpl/*        $HA12_PATH/D2A_converted_pose_smpl/"
echo "  cp $HA12_SYNTH_PATH/D2A_converted_label_actions/*    $HA12_PATH/D2A_converted_label_actions/"
echo ""
echo "Then run segmentation:"
echo "  uv run scripts/tasks/segmentation.py --config-name segmentation/augmented/<config>"
