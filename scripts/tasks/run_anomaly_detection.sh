#!/usr/bin/env bash
# Anomaly detection evaluation with normalising flows (STG-NF).
# Trains one model per target activity and generates separability matrices.
#
# Usage:
#   bash scripts/tasks/run_anomaly_detection.sh          # run all three
#   bash scripts/tasks/run_anomaly_detection.sh verbs     # run one dataset
#
# Uncomment / comment lines as needed before launching.
set -euo pipefail

SCRIPT="scripts/tasks/anomaly_detection.py"

run_config() {
    local config="$1"
    echo "============================================"
    echo "Running: ${config}"
    echo "============================================"
    uv run "${SCRIPT}" --config "${config}"
}

# --- ESK Verbs (10 targets) ---
run_esk_verbs() {
    run_config configs/anomaly_detection/nf_esk_verbs.yaml
}

# --- ESK Activities (6 targets) ---
run_esk_activities() {
    run_config configs/anomaly_detection/nf_esk_activities.yaml
}

# --- HumanAct12 Actions (select 6 of 12 before running) ---
run_humanact12() {
    run_config configs/anomaly_detection/nf_humanact12.yaml
}

# ---- Main dispatcher ----
if [[ $# -gt 0 ]]; then
    case "$1" in
        verbs)        run_esk_verbs ;;
        activities)   run_esk_activities ;;
        humanact12)   run_humanact12 ;;
        all)          run_esk_verbs; run_esk_activities; run_humanact12 ;;
        *)            echo "Unknown dataset: $1. Choose from: verbs, activities, humanact12, all"; exit 1 ;;
    esac
else
    # Default: run all
    # run_esk_verbs
    run_esk_activities
    run_humanact12
fi
