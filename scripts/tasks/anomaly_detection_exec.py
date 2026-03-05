#!/usr/bin/env python
"""Anomaly Detection using Executable Activity Models with Program Edit Distance.

This script evaluates activity separability using edit distance between
programs parsed from motion data. It supports both fine-grained (verbs)
and coarse-grained (activity) labels.

Key features:
- Logs all metrics and matrices to Weights & Biases
- Computes separability matrix using minimum edit distance
- Supports pre-parsed programs (fast) or real-time parsing (slow)

Workflow:
1. Load/parse programs for training set (N programs per activity)
2. Load/parse programs for test set (all available)  
3. For each test program, compute min edit distance to each activity's model
4. Build separability matrix: M[i,j] = mean min-dist from activity i tests to activity j model
5. Log results to wandb

Usage:
    # Using pre-parsed programs (fast)
    uv run scripts/tasks/anomaly_detection_exec.py \
        --train-programs /pvc/esk/programs_train.json \
        --label-type verbs

    # Parse on the fly (slower, uses LLM)
    uv run scripts/tasks/anomaly_detection_exec.py \
        --parser-checkpoint results/parser/20260122_225017 \
        --label-type activity
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.programs import (
    ProgramDistanceMatrix,
    parse_to_tree,
    VALUE_TOLERANCE,
)


# =============================================================================
# Wandb Configuration
# =============================================================================
WANDB_PROJECT = "exact"
WANDB_ENTITY = "assistive-autonomy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection using Executable Activity Models"
    )
    
    # Data sources
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Path to saved ActivityModelCollection JSON (e.g., models_activity.json). "
             "Uses these as the exact model programs. Auto-detected from --esk-path if not set.",
    )
    parser.add_argument(
        "--train-programs",
        type=str,
        default=None,
        help="Path to pre-parsed programs JSON for training (ignored if --models is set)",
    )
    parser.add_argument(
        "--test-programs",
        type=str,
        default=None,
        help="Path to pre-parsed programs JSON for testing (default: auto-detected)",
    )
    parser.add_argument(
        "--parser-checkpoint",
        type=str,
        default=None,
        help="Path to parser checkpoint (for parsing on the fly)",
    )
    parser.add_argument(
        "--esk-path",
        type=str,
        default="../exact_data/benchmarks/esk",
        help="Path to dataset directory",
    )
    
    # Label configuration
    parser.add_argument(
        "--label-type",
        type=str,
        default="verbs",
        choices=["verbs", "activity", "actions"],
        help="Label type: verbs (fine-grained), activity (coarse-grained), or actions (humanact12)",
    )
    parser.add_argument(
        "--activities",
        type=str,
        nargs="+",
        default=None,
        help="Specific activities to assess (default: all). E.g., --activities Grab Carry Put",
    )
    
    # Budget settings
    parser.add_argument(
        "--train-budget",
        type=int,
        default=100,
        help="Number of programs per activity for model (training)",
    )
    parser.add_argument(
        "--max-test-per-activity",
        type=int,
        default=None,
        help="Maximum test programs per activity (default: all)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/anomaly_detection_exec/<label_type>_<timestamp>)",
    )
    
    # Wandb
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom wandb run name",
    )
    
    # Performance
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for distance computation",
    )
    parser.add_argument(
        "--sample-test",
        type=int,
        default=None,
        help="Sample this many test programs per activity (for faster testing)",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_programs_from_json(path: str) -> dict:
    """Load programs from JSON file.
    
    Returns:
        Dict with 'activity_names', 'programs_by_activity', 'metadata'
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    return {
        "activity_names": data.get("activity_names", []),
        "programs_by_activity": data.get("programs_by_activity", {}),
        "metadata": data.get("metadata", {}),
    }


def filter_valid_programs(programs: list[dict], max_count: int = None) -> list[str]:
    """Filter programs, keeping only syntactically valid ones.
    
    Args:
        programs: List of program dicts with 'program' key
        max_count: Maximum number to return (None = all)
        
    Returns:
        List of valid program strings
    """
    valid = []
    for p in programs:
        prog_str = p.get("program", "") if isinstance(p, dict) else str(p)
        try:
            tree = parse_to_tree(prog_str)
            if tree is not None:
                valid.append(prog_str)
        except Exception:
            continue
        
        if max_count and len(valid) >= max_count:
            break
    
    return valid


def plot_separability_matrix(
    matrix: np.ndarray,
    activity_names: list[str],
    output_path: str,
    title: str = "Program Edit Distance Separability Matrix",
    cmap: str = "coolwarm",
):
    """Plot separability matrix as heatmap (matching notebook style)."""
    n = len(activity_names)
    
    # Wrap activity labels
    wrapped_names = [name.replace("_", "\n").replace(" ", "\n") for name in activity_names]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # For edit distance: lower = more similar, so use reversed colormap
    # But coolwarm works well as-is (blue=low, red=high)
    im = sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Min Edit Distance (lower = more similar)", "shrink": 0.8},
        annot_kws={"size": 10},
        ax=ax,
        xticklabels=wrapped_names,
        yticklabels=wrapped_names,
    )
    
    ax.set_xlabel("Model Activity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Query Activity", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved matrix plot to {output_path}")
    
    return fig


def plot_separation_summary(
    activity_names: list[str],
    metrics: dict,
    output_path: str,
):
    """Plot per-activity separation scores."""
    per_activity = metrics.get("per_activity", {})
    
    separations = []
    for act in activity_names:
        act_metrics = per_activity.get(act, {})
        separations.append(act_metrics.get("separation", 0))
    
    separations = np.array(separations)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by positive/negative separation
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in separations]
    
    bars = ax.bar(
        np.arange(len(activity_names)),
        separations,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    
    ax.set_xticks(np.arange(len(activity_names)))
    ax.set_xticklabels(activity_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Separation (cross - same distance)", fontsize=12)
    ax.set_title("Per-Activity Separation (higher = better)", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, separations)):
        offset = 0.05 if val > 0 else -0.05
        ax.annotate(
            f"{val:.2f}",
            xy=(i, val + offset),
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=8,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved separation summary to {output_path}")
    
    return fig


def plot_auc_matrix(
    auc_matrix: np.ndarray,
    activity_names: list[str],
    output_path: str,
    title: str = "AUC Matrix",
    cmap: str = "RdYlGn",
):
    """Plot AUC matrix as heatmap.
    
    AUC[i,j] = how well model i separates its own activity i (positive) from activity j (negative).
    Diagonal = one-vs-rest AUROC for model i.
    Same convention as the NF anomaly detection matrix.
    """
    n = len(activity_names)
    wrapped_names = [name.replace("_", "\n").replace(" ", "\n") for name in activity_names]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask NaN for display
    mask = np.isnan(auc_matrix)
    
    im = sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        mask=mask,
        cbar_kws={"label": "AUC (higher = better separation)", "shrink": 0.8},
        annot_kws={"size": 9},
        ax=ax,
        xticklabels=wrapped_names,
        yticklabels=wrapped_names,
    )
    
    ax.set_xlabel("Query Activity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Target Activity (model)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved AUC matrix plot to {output_path}")
    
    return fig


def plot_auc_comparison(
    activity_names: list[str],
    mean_sigmoid_aucs: dict[str, float],
    min_sigmoid_aucs: dict[str, float],
    output_path: str,
):
    """Plot per-model AUC comparison between mean-sigmoid and min-sigmoid."""
    n = len(activity_names)
    x = np.arange(n)
    width = 0.35
    
    mean_vals = [mean_sigmoid_aucs.get(a, 0) for a in activity_names]
    min_vals = [min_sigmoid_aucs.get(a, 0) for a in activity_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width / 2, mean_vals, width, label="mean-sigmoid", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, min_vals, width, label="min-sigmoid", color="#e67e22", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(activity_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("AUC (one-vs-rest)", fontsize=12)
    ax.set_title("Per-Activity AUC: mean-sigmoid vs min-sigmoid", fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.7, label="Random")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                           ha="center", va="bottom", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved AUC comparison plot to {output_path}")
    
    return fig


def main():
    args = parse_args()
    
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_wandb = args.wandb and not args.no_wandb
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/anomaly_detection_exec/{args.label_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Executable Activity Model Anomaly Detection")
    logger.info("=" * 60)
    logger.info(f"Label type: {args.label_type}")
    logger.info(f"Train budget: {args.train_budget} programs/activity")
    logger.info(f"Output: {output_dir}")
    
    # Initialize wandb
    if use_wandb:
        import wandb
        
        run_name = args.wandb_run_name or f"exec_{args.label_type}_{timestamp}"
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "label_type": args.label_type,
                "model_source": "saved_models" if (args.models or not args.train_programs) else "train_programs",
                "train_budget": args.train_budget,
                "max_test_per_activity": args.max_test_per_activity,
                "value_tolerance": VALUE_TOLERANCE,
                "seed": args.seed,
                "method": "executable_activity_models",
                "metric": "program_edit_distance",
            },
        )
        logger.info(f"Wandb run: {run_name}")
    
    # =========================================================================
    # Resolve paths
    # =========================================================================
    esk_path = Path(args.esk_path)
    
    # --- Model programs (the "exact models") ---
    models_path = None
    if args.models:
        models_path = Path(args.models)
    else:
        # Auto-detect model file from esk_path + label_type
        if args.label_type == "activity":
            candidate = esk_path / "models_activity.json"
        elif args.label_type == "verbs":
            candidate = esk_path / "models_verbs.json"
        else:
            candidate = esk_path / "models.json"
        if candidate.exists():
            models_path = candidate
    
    use_saved_models = models_path is not None and models_path.exists()
    
    # --- Test programs file ---
    if args.test_programs:
        test_path = Path(args.test_programs)
    else:
        # Auto-detect test programs file
        if args.label_type == "activity":
            test_path = esk_path / "programs_activity_test.json"
        elif args.label_type == "verbs":
            test_path = esk_path / "programs_verbs_test.json"
            if not test_path.exists():
                test_path = esk_path / "programs_test.json"
        else:
            test_path = esk_path / "programs_test.json"
    
    if not test_path.exists():
        logger.error(f"Test programs file not found: {test_path}")
        return 1
    
    # --- Train programs file (fallback if no saved models) ---
    train_path = None
    if not use_saved_models:
        if args.train_programs:
            train_path = Path(args.train_programs)
        else:
            if args.label_type == "activity":
                train_path = esk_path / "programs_activity_train.json"
            elif args.label_type == "verbs":
                train_path = esk_path / "programs_verbs_train.json"
                if not train_path.exists():
                    train_path = esk_path / "programs_train.json"
            else:
                train_path = esk_path / "programs_train.json"
        
        if not train_path.exists():
            logger.error(f"No model file or train programs found. Tried: {models_path}, {train_path}")
            return 1
    
    # =========================================================================
    # Load model programs
    # =========================================================================
    if use_saved_models:
        logger.info(f"Loading saved ExAct models from: {models_path}")
        with open(models_path, "r") as f:
            models_data = json.load(f)
        
        all_activity_names = list(models_data["models"].keys())
        model_programs_by_activity = {}
        for act_name, model_info in models_data["models"].items():
            model_programs_by_activity[act_name] = model_info["original_programs"]
        
        logger.info(f"Loaded models for {len(all_activity_names)} activities")
        for act, progs in model_programs_by_activity.items():
            logger.info(f"  {act}: {len(progs)} model programs")
    else:
        logger.info(f"Loading train programs from: {train_path}")
        train_data = load_programs_from_json(train_path)
        all_activity_names = train_data["activity_names"]
        model_programs_by_activity = None  # will sample below
    
    # Load test programs
    logger.info(f"Loading test programs from: {test_path}")
    test_data = load_programs_from_json(test_path)
    
    # =========================================================================
    # Filter activities
    # =========================================================================
    if args.activities:
        invalid = [a for a in args.activities if a not in all_activity_names]
        if invalid:
            logger.error(f"Unknown activities: {invalid}")
            logger.error(f"Available: {all_activity_names}")
            return 1
        activity_names = args.activities
        logger.info(f"Filtering to {len(activity_names)} activities: {activity_names}")
    else:
        activity_names = all_activity_names
        logger.info(f"Using all {len(activity_names)} activities: {activity_names}")
    
    # =========================================================================
    # Build distance matrix: model + test programs
    # =========================================================================
    dist_matrix = ProgramDistanceMatrix(activity_names)
    
    train_counts = {}
    test_counts = {}
    
    for activity in activity_names:
        # --- Model programs ---
        if use_saved_models:
            # Use exact saved model programs (no budget, no sampling)
            raw_progs = model_programs_by_activity.get(activity, [])
            model_progs = filter_valid_programs(
                [{"program": p} if isinstance(p, str) else p for p in raw_progs]
            )
        else:
            # Sample from train programs up to budget
            train_progs_raw = train_data["programs_by_activity"].get(activity, [])
            if not train_progs_raw:
                logger.warning(f"No training programs for {activity}")
                continue
            random.shuffle(train_progs_raw)
            model_progs = filter_valid_programs(train_progs_raw, max_count=args.train_budget)
        
        if model_progs:
            dist_matrix.set_model_programs(activity, model_progs)
            train_counts[activity] = len(model_progs)
        else:
            logger.warning(f"No valid model programs for {activity}")
            continue
        
        # --- Test programs ---
        test_progs_raw = test_data["programs_by_activity"].get(activity, [])
        test_limit = args.sample_test if args.sample_test else args.max_test_per_activity
        test_progs = filter_valid_programs(test_progs_raw, max_count=test_limit)
        
        for prog in test_progs:
            dist_matrix.add_test_program(activity, prog)
        test_counts[activity] = len(test_progs)
        
        logger.info(f"  {activity}: {train_counts.get(activity, 0)} model, {len(test_progs)} test")
    
    # Compute raw distance matrix (existing behavior)
    logger.info(f"Computing raw distance matrix with {args.num_workers} worker(s)...")
    matrix = dist_matrix.compute_matrix(verbose=True, num_workers=args.num_workers)
    
    # Get raw distance metrics
    metrics = dist_matrix.get_separability_metrics()
    
    logger.info("=" * 60)
    logger.info("Raw Distance Results")
    logger.info("=" * 60)
    logger.info(f"Diagonal mean (same-activity): {metrics['diagonal_mean']:.2f}")
    logger.info(f"Off-diagonal mean (cross-activity): {metrics['off_diagonal_mean']:.2f}")
    logger.info(f"Separation (higher = better): {metrics['separation']:.2f}")
    
    # =========================================================================
    # Sigmoid-based scoring and AUC computation
    # =========================================================================
    scoring_methods = ["mean-sigmoid", "min-sigmoid"]
    auc_results = {}
    
    for method in scoring_methods:
        logger.info(f"\nComputing {method} scores and AUC...")
        auc_matrix, score_matrix, auc_metrics = dist_matrix.compute_auc_matrix(
            method=method, verbose=True
        )
        
        auc_results[method] = {
            "auc_matrix": auc_matrix,
            "score_matrix": score_matrix,
            "metrics": auc_metrics,
        }
        
        logger.info(f"  {method} mean AUC (one-vs-rest): {auc_metrics['mean_auc']:.4f}")
        logger.info(f"  {method} mean pairwise AUC: {auc_metrics['mean_pairwise_auc']:.4f}")
        
        # Per-model AUC
        for act, auc_val in auc_metrics["per_model_auc"].items():
            logger.info(f"    {act}: AUC = {auc_val:.4f}")
    
    # =========================================================================
    # Build results dict
    # =========================================================================
    results = {
        "activity_names": activity_names,
        "distance_matrix": matrix.tolist(),
        "distance_metrics": metrics,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "config": {
            "label_type": args.label_type,
            "model_source": str(models_path) if use_saved_models else "train_programs",
            "train_programs": str(train_path) if train_path else None,
            "test_programs": str(test_path),
            "train_budget": args.train_budget if not use_saved_models else None,
            "max_test_per_activity": args.max_test_per_activity,
            "value_tolerance": VALUE_TOLERANCE,
            "seed": args.seed,
            "timestamp": timestamp,
        },
    }
    
    # Add sigmoid/AUC results
    for method in scoring_methods:
        method_key = method.replace("-", "_")
        r = auc_results[method]
        results[f"{method_key}_score_matrix"] = r["score_matrix"].tolist()
        results[f"{method_key}_auc_matrix"] = np.where(
            np.isnan(r["auc_matrix"]), None, r["auc_matrix"]
        ).tolist()
        results[f"{method_key}_metrics"] = r["metrics"]
    
    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")
    
    # =========================================================================
    # Plots
    # =========================================================================
    
    # 1. Raw distance separability matrix
    matrix_plot_path = output_dir / "separability_matrix.png"
    plot_separability_matrix(
        matrix,
        activity_names,
        str(matrix_plot_path),
        title=f"Program Edit Distance Separability ({args.label_type})",
    )
    
    # 2. Raw distance separation summary
    summary_plot_path = output_dir / "separation_summary.png"
    plot_separation_summary(activity_names, metrics, str(summary_plot_path))
    
    # 3. AUC matrices for each method
    auc_plot_paths = {}
    score_plot_paths = {}
    for method in scoring_methods:
        method_key = method.replace("-", "_")
        r = auc_results[method]
        
        # AUC matrix heatmap
        auc_path = output_dir / f"auc_matrix_{method_key}.png"
        plot_auc_matrix(
            r["auc_matrix"],
            activity_names,
            str(auc_path),
            title=f"AUC Matrix ({method}, {args.label_type})",
        )
        auc_plot_paths[method] = auc_path
        
        # Score matrix heatmap
        score_path = output_dir / f"score_matrix_{method_key}.png"
        plot_separability_matrix(
            r["score_matrix"],
            activity_names,
            str(score_path),
            title=f"Sigmoid Score Matrix ({method}, {args.label_type})",
            cmap="RdYlGn",
        )
        score_plot_paths[method] = score_path
    
    # 4. AUC comparison (mean-sigmoid vs min-sigmoid)
    comparison_path = output_dir / "auc_comparison.png"
    plot_auc_comparison(
        activity_names,
        auc_results["mean-sigmoid"]["metrics"]["per_model_auc"],
        auc_results["min-sigmoid"]["metrics"]["per_model_auc"],
        str(comparison_path),
    )
    
    # =========================================================================
    # Log to wandb
    # =========================================================================
    if use_wandb:
        import wandb
        
        # Raw distance metrics
        wandb.summary["distance/diagonal_mean"] = metrics["diagonal_mean"]
        wandb.summary["distance/off_diagonal_mean"] = metrics["off_diagonal_mean"]
        wandb.summary["distance/separation"] = metrics["separation"]
        wandb.summary["n_activities"] = len(activity_names)
        wandb.summary["total_train_programs"] = sum(train_counts.values())
        wandb.summary["total_test_programs"] = sum(test_counts.values())
        
        # Per-activity raw metrics
        per_activity = metrics.get("per_activity", {})
        for act, act_metrics in per_activity.items():
            wandb.summary[f"{act}/separation"] = act_metrics.get("separation", 0)
            wandb.summary[f"{act}/same_dist"] = act_metrics.get("same_activity_dist", 0)
            wandb.summary[f"{act}/cross_dist"] = act_metrics.get("cross_activity_dist", 0)
        
        # Sigmoid/AUC metrics for each method
        for method in scoring_methods:
            method_key = method.replace("-", "_")
            r = auc_results[method]
            m = r["metrics"]
            
            wandb.summary[f"{method_key}/mean_auc"] = m["mean_auc"]
            wandb.summary[f"{method_key}/mean_pairwise_auc"] = m["mean_pairwise_auc"]
            
            for act, auc_val in m["per_model_auc"].items():
                wandb.summary[f"{method_key}/{act}/auc"] = auc_val
            
            # Log AUC matrix as table
            auc_mat = r["auc_matrix"]
            auc_table_data = []
            for i, row_act in enumerate(activity_names):
                row_data = [row_act]
                for j in range(len(activity_names)):
                    val = auc_mat[i, j]
                    row_data.append(float(val) if not np.isnan(val) else None)
                auc_table_data.append(row_data)
            
            auc_table = wandb.Table(
                columns=["target_activity"] + activity_names,
                data=auc_table_data,
            )
            wandb.log({f"{method_key}_auc_matrix": auc_table})
        
        # Log raw distance matrix as table
        matrix_table_data = []
        for i, row_act in enumerate(activity_names):
            row = {"model_trained_on": row_act}
            for j, col_act in enumerate(activity_names):
                row[col_act] = float(matrix[i, j])
            matrix_table_data.append(row)
        
        matrix_table = wandb.Table(
            columns=["model_trained_on"] + activity_names,
            data=[[row["model_trained_on"]] + [row[a] for a in activity_names] 
                  for row in matrix_table_data]
        )
        wandb.log({"distance_matrix": matrix_table})
        
        # Log results summary table
        results_table_data = []
        for act in activity_names:
            act_metrics = per_activity.get(act, {})
            mean_sig_auc = auc_results["mean-sigmoid"]["metrics"]["per_model_auc"].get(act, float("nan"))
            min_sig_auc = auc_results["min-sigmoid"]["metrics"]["per_model_auc"].get(act, float("nan"))
            results_table_data.append([
                act,
                act_metrics.get("separation", 0),
                act_metrics.get("same_activity_dist", 0),
                act_metrics.get("cross_activity_dist", 0),
                mean_sig_auc,
                min_sig_auc,
                train_counts.get(act, 0),
                test_counts.get(act, 0),
            ])
        
        results_table = wandb.Table(
            columns=["activity", "separation", "same_dist", "cross_dist",
                     "mean_sigmoid_auc", "min_sigmoid_auc", "n_train", "n_test"],
            data=results_table_data,
        )
        wandb.log({"results_table": results_table})
        
        # Log all plots
        plot_log = {
            "separability_matrix_plot": wandb.Image(str(matrix_plot_path)),
            "separation_summary_plot": wandb.Image(str(summary_plot_path)),
            "auc_comparison_plot": wandb.Image(str(comparison_path)),
        }
        for method in scoring_methods:
            method_key = method.replace("-", "_")
            plot_log[f"{method_key}_auc_matrix_plot"] = wandb.Image(str(auc_plot_paths[method]))
            plot_log[f"{method_key}_score_matrix_plot"] = wandb.Image(str(score_plot_paths[method]))
        wandb.log(plot_log)
        
        wandb.finish()
        logger.info("Logged results to wandb")
    
    logger.success(f"Anomaly detection complete! Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
