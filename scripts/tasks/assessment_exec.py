#!/usr/bin/env python
"""Assessment using Executable Activity Models with Program Edit Distance.

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
    uv run scripts/assessment_exec.py \
        --train-programs /pvc/esk/programs_train.json \
        --label-type verbs

    # Parse on the fly (slower, uses LLM)
    uv run scripts/assessment_exec.py \
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
        description="Assessment using Executable Activity Models"
    )
    
    # Data sources
    parser.add_argument(
        "--train-programs",
        type=str,
        default=None,
        help="Path to pre-parsed programs JSON for training",
    )
    parser.add_argument(
        "--test-programs",
        type=str,
        default=None,
        help="Path to pre-parsed programs JSON for testing (default: same as train)",
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
        default="/pvc/esk",
        help="Path to ESK dataset",
    )
    
    # Label configuration
    parser.add_argument(
        "--label-type",
        type=str,
        default="verbs",
        choices=["verbs", "activity"],
        help="Label type: verbs (fine-grained) or activity (coarse-grained)",
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
        help="Output directory (default: results/assessment_exec/<label_type>_<timestamp>)",
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
        output_dir = Path(f"results/assessment_exec/{args.label_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Executable Activity Model Assessment")
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
                "train_budget": args.train_budget,
                "max_test_per_activity": args.max_test_per_activity,
                "value_tolerance": VALUE_TOLERANCE,
                "seed": args.seed,
                "method": "executable_activity_models",
                "metric": "program_edit_distance",
            },
        )
        logger.info(f"Wandb run: {run_name}")
    
    # Load programs
    # Determine correct programs file based on label type
    if args.train_programs:
        train_path = args.train_programs
    else:
        # Auto-detect based on label type
        esk_path = Path(args.esk_path)
        if args.label_type == "activity":
            train_path = esk_path / "programs_activity_train.json"
        else:
            train_path = esk_path / "programs_train.json"
    
    if not Path(train_path).exists():
        logger.error(f"Programs file not found: {train_path}")
        logger.error("Run parse_esk.py first to generate programs")
        return 1
    
    logger.info(f"Loading programs from: {train_path}")
    train_data = load_programs_from_json(train_path)
    
    # Use same file for test if not specified
    if args.test_programs:
        test_data = load_programs_from_json(args.test_programs)
    else:
        test_data = train_data
    
    # Filter activities if specified
    all_activity_names = train_data["activity_names"]
    if args.activities:
        # Validate specified activities exist
        invalid = [a for a in args.activities if a not in all_activity_names]
        if invalid:
            logger.error(f"Unknown activities: {invalid}")
            logger.error(f"Available: {all_activity_names}")
            return 1
        activity_names = args.activities
        logger.info(f"Filtering to {len(activity_names)} activities: {activity_names}")
    else:
        activity_names = all_activity_names
        logger.info(f"Found {len(activity_names)} activities: {activity_names}")
    
    # Create distance matrix calculator
    dist_matrix = ProgramDistanceMatrix(activity_names)
    
    # Build train/test sets
    logger.info("Building train/test program sets...")
    
    train_counts = {}
    test_counts = {}
    
    for activity in activity_names:
        # Get all programs for this activity
        train_progs_raw = train_data["programs_by_activity"].get(activity, [])
        test_progs_raw = test_data["programs_by_activity"].get(activity, [])
        
        if not train_progs_raw:
            logger.warning(f"No training programs for {activity}")
            continue
        
        # Shuffle for random selection
        random.shuffle(train_progs_raw)
        
        # Filter to valid programs and limit to budget
        train_progs = filter_valid_programs(train_progs_raw, max_count=args.train_budget)
        
        # For test: use all remaining programs (or separate test set)
        # Apply sample_test limit if specified
        test_limit = args.sample_test if args.sample_test else args.max_test_per_activity
        if args.test_programs:
            # Separate test set - use all valid programs
            test_progs = filter_valid_programs(test_progs_raw, max_count=test_limit)
        else:
            # Same file - use programs not in training set
            remaining = train_progs_raw[args.train_budget:]
            test_progs = filter_valid_programs(remaining, max_count=test_limit)
        
        # Set model programs
        if train_progs:
            dist_matrix.set_model_programs(activity, train_progs)
            train_counts[activity] = len(train_progs)
        
        # Add test programs
        for prog in test_progs:
            dist_matrix.add_test_program(activity, prog)
        test_counts[activity] = len(test_progs)
        
        logger.info(f"  {activity}: {len(train_progs)} train, {len(test_progs)} test")
    
    # Compute distance matrix
    logger.info(f"Computing separability matrix with {args.num_workers} worker(s)...")
    matrix = dist_matrix.compute_matrix(verbose=True, num_workers=args.num_workers)
    
    # Get metrics
    metrics = dist_matrix.get_separability_metrics()
    
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Diagonal mean (same-activity): {metrics['diagonal_mean']:.2f}")
    logger.info(f"Off-diagonal mean (cross-activity): {metrics['off_diagonal_mean']:.2f}")
    logger.info(f"Separation (higher = better): {metrics['separation']:.2f}")
    
    # Build results dict
    results = {
        "activity_names": activity_names,
        "matrix": matrix.tolist(),
        "metrics": metrics,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "config": {
            "label_type": args.label_type,
            "train_programs": str(train_path),
            "train_budget": args.train_budget,
            "max_test_per_activity": args.max_test_per_activity,
            "value_tolerance": VALUE_TOLERANCE,
            "seed": args.seed,
            "timestamp": timestamp,
        },
    }
    
    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Plot and save visualizations
    matrix_plot_path = output_dir / "separability_matrix.png"
    plot_separability_matrix(
        matrix,
        activity_names,
        str(matrix_plot_path),
        title=f"Program Edit Distance Separability ({args.label_type})",
    )
    
    summary_plot_path = output_dir / "separation_summary.png"
    plot_separation_summary(activity_names, metrics, str(summary_plot_path))
    
    # Log to wandb
    if use_wandb:
        import wandb
        
        # Log metrics
        wandb.summary["diagonal_mean"] = metrics["diagonal_mean"]
        wandb.summary["off_diagonal_mean"] = metrics["off_diagonal_mean"]
        wandb.summary["separation"] = metrics["separation"]
        wandb.summary["n_activities"] = len(activity_names)
        wandb.summary["total_train_programs"] = sum(train_counts.values())
        wandb.summary["total_test_programs"] = sum(test_counts.values())
        
        # Log per-activity metrics
        per_activity = metrics.get("per_activity", {})
        for act, act_metrics in per_activity.items():
            wandb.summary[f"{act}/separation"] = act_metrics.get("separation", 0)
            wandb.summary[f"{act}/same_dist"] = act_metrics.get("same_activity_dist", 0)
            wandb.summary[f"{act}/cross_dist"] = act_metrics.get("cross_activity_dist", 0)
        
        # Log separability matrix as table
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
        wandb.log({"separability_matrix": matrix_table})
        
        # Log results table
        results_table_data = []
        for act in activity_names:
            act_metrics = per_activity.get(act, {})
            results_table_data.append({
                "activity": act,
                "separation": act_metrics.get("separation", 0),
                "same_activity_dist": act_metrics.get("same_activity_dist", 0),
                "cross_activity_dist": act_metrics.get("cross_activity_dist", 0),
                "n_train": train_counts.get(act, 0),
                "n_test": test_counts.get(act, 0),
            })
        
        results_table = wandb.Table(
            columns=["activity", "separation", "same_activity_dist", "cross_activity_dist", "n_train", "n_test"],
            data=[[r["activity"], r["separation"], r["same_activity_dist"], 
                   r["cross_activity_dist"], r["n_train"], r["n_test"]] 
                  for r in results_table_data]
        )
        wandb.log({"results_table": results_table})
        
        # Log plots
        wandb.log({
            "separability_matrix_plot": wandb.Image(str(matrix_plot_path)),
            "separation_summary_plot": wandb.Image(str(summary_plot_path)),
        })
        
        wandb.finish()
        logger.info("Logged results to wandb")
    
    logger.success(f"Assessment complete! Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
