#!/usr/bin/env bash
# =============================================================================
# Stage 5: Run Anomaly Detection Experiments
#
# Evaluates three methods for activity distinguishability, all producing an
# N × N AUC matrix logged to wandb:
#
#   nf           — Normalising Flow (STG-NF, SOTA density estimator).
#                  Trains one model per target activity.
#
#   mean_sigmoid — Executable models + program edit distance, mean aggregation.
#
#   min_sigmoid  — Executable models + program edit distance, min aggregation.
#
# All three methods share a single script; the method is specified inside
# the config file (or overridden on the command line).
#
# Datasets: ESK verbs, ESK activities, HumanAct12
#
# Prerequisites:
#   Stage 2 complete — parsed programs and executable models available.
#
# Usage:
#   bash scripts/5_run_anomaly_detection.sh                   # all methods + datasets
#   METHOD=nf          bash scripts/5_run_anomaly_detection.sh
#   METHOD=mean_sigmoid bash scripts/5_run_anomaly_detection.sh
#   METHOD=min_sigmoid  bash scripts/5_run_anomaly_detection.sh
#   DATASET=esk_verbs   bash scripts/5_run_anomaly_detection.sh
#   DATASET=esk_activities bash scripts/5_run_anomaly_detection.sh
#   DATASET=humanact12  bash scripts/5_run_anomaly_detection.sh
# =============================================================================
set -euo pipefail

#cd /pvc/exact

SCRIPT="scripts/tasks/anomaly_detection.py"

METHOD="${METHOD:-all}"    # nf | mean_sigmoid | min_sigmoid | all
DATASET="${DATASET:-all}"  # esk_verbs | esk_activities | humanact12 | all

run() {
    local config="$1"
    local method_override="${2:-}"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Running: $config"
    [[ -n "$method_override" ]] && echo "  method override: $method_override"
    echo "──────────────────────────────────────────────"
    if [[ -n "$method_override" ]]; then
        uv run "$SCRIPT" --config "$config" "method=$method_override"
    else
        uv run "$SCRIPT" --config "$config"
    fi
}

want_method()  { [[ "$METHOD"  == "all" ]] || [[ "$METHOD"  == "$1" ]]; }
want_dataset() { [[ "$DATASET" == "all" ]] || [[ "$DATASET" == "$1" ]]; }

echo "============================================"
echo "Stage 5: Anomaly Detection"
echo "  Method filter:  $METHOD"
echo "  Dataset filter: $DATASET"
echo "============================================"

# ─── ESK Verbs ──────────────────────────────────────────────────────────────
if want_dataset esk_verbs; then
    want_method nf           && run configs/anomaly_detection/nf_esk_verbs.yaml
    want_method mean_sigmoid && run configs/anomaly_detection/exec_esk_verbs.yaml mean_sigmoid
    want_method min_sigmoid  && run configs/anomaly_detection/exec_esk_verbs.yaml min_sigmoid
fi

# ─── ESK Activities ──────────────────────────────────────────────────────────
if want_dataset esk_activities; then
    want_method nf           && run configs/anomaly_detection/nf_esk_activities.yaml
    want_method mean_sigmoid && run configs/anomaly_detection/exec_esk_activities.yaml mean_sigmoid
    want_method min_sigmoid  && run configs/anomaly_detection/exec_esk_activities.yaml min_sigmoid
fi

# ─── HumanAct12 ─────────────────────────────────────────────────────────────
if want_dataset humanact12; then
    want_method nf           && run configs/anomaly_detection/nf_humanact12.yaml
    want_method mean_sigmoid && run configs/anomaly_detection/exec_humanact12.yaml mean_sigmoid
    want_method min_sigmoid  && run configs/anomaly_detection/exec_humanact12.yaml min_sigmoid
fi

echo ""
echo "============================================"
echo "Stage 5 complete"
echo "============================================"
echo "Results: results/anomaly_detection/"
