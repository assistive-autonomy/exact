#!/usr/bin/env bash
# =============================================================================
# Stage 4: Run Segmentation Experiments
#
# Evaluates the value of executable-model data augmentation for activity
# segmentation using the DLC2Action framework.
#
# Three experimental conditions per dataset:
#   original   — baseline: train on real data only
#   perturbed  — robustness: train on temporally perturbed annotations
#   augmented  — proposed:  train on original + all synthetic data
#
# Datasets: ESK verbs, ESK activities, HumanAct12
#
# Prerequisites:
#   - Stage 3 complete (augmented data generated into benchmarks dirs)
#
# Usage:
#   # Run all conditions
#   bash scripts/4_run_segmentation.sh
#
#   # Run only one condition
#   CONDITION=original bash scripts/4_run_segmentation.sh
#   CONDITION=augmented bash scripts/4_run_segmentation.sh
#
#   # Run one dataset
#   DATASET=esk_activities bash scripts/4_run_segmentation.sh
# =============================================================================
set -euo pipefail

#cd /pvc/exact

SCRIPT="scripts/tasks/segmentation.py"

# Filter: set to a specific condition or "all"
CONDITION="${CONDITION:-all}"
# Filter: set to a specific dataset or "all"
DATASET="${DATASET:-all}"

run_config() {
    local config_name="$1"
    local label="$2"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Running: $label"
    echo "  config: configs/${config_name}.yaml"
    echo "──────────────────────────────────────────────"
    uv run "$SCRIPT" --config-name "$config_name"
}

should_run() {
    local cond="$1"
    local dataset="$2"
    { [[ "$CONDITION" == "all" ]] || [[ "$CONDITION" == "$cond" ]]; } && \
    { [[ "$DATASET"    == "all" ]] || [[ "$DATASET"    == "$dataset" ]]; }
}

echo "============================================"
echo "Stage 4: Segmentation Experiments"
echo "  Condition filter: $CONDITION"
echo "  Dataset filter:   $DATASET"
echo "============================================"

# ─── Condition 1: Original (baseline) ───────────────────────────────────────
if should_run original esk_activities; then
    run_config "segmentation/original/esk_activities" "ESK Activities — original"
fi
if should_run original esk_verbs; then
    run_config "segmentation/original/esk_verbs" "ESK Verbs — original"
fi
if should_run original humanact12; then
    run_config "segmentation/original/humanact12" "HumanAct12 — original"
fi

# ─── Condition 2: Perturbed (robustness baseline) ────────────────────────────
if should_run perturbed esk_activities; then
    run_config "segmentation/perturbed/esk_activities" "ESK Activities — perturbed"
fi
if should_run perturbed esk_verbs; then
    run_config "segmentation/perturbed/esk_verbs" "ESK Verbs — perturbed"
fi
if should_run perturbed humanact12; then
    run_config "segmentation/perturbed/humanact12" "HumanAct12 — perturbed"
fi

# ─── Condition 3: Augmented (original + all synthetic data) ─────────────────
if should_run augmented esk_activities; then
    run_config "segmentation/augmented/esk_activities" "ESK Activities — augmented"
fi
if should_run augmented esk_verbs; then
    run_config "segmentation/augmented/esk_verbs" "ESK Verbs — augmented"
fi
if should_run augmented humanact12; then
    run_config "segmentation/augmented/humanact12" "HumanAct12 — augmented"
fi

echo ""
echo "============================================"
echo "Stage 4 complete"
echo "============================================"
echo "Results stored in the DLC2Action project directories."
echo ""
echo "Next: bash scripts/5_run_anomaly_detection.sh"
