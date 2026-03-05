#!/usr/bin/env python
"""Quick Anomaly Detection using pre-parsed programs.

This script runs anomaly detection using already-parsed programs from programs_train.json,
avoiding the slow LLM inference step. It splits the programs into train/test 
and computes the separability matrix.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.programs import (
    ProgramDistanceMatrix,
    parse_to_tree,
    VALUE_TOLERANCE,
)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_separability_matrix(
    matrix: np.ndarray,
    activity_names: list[str],
    output_path: str,
    title: str = "Program Edit Distance Separability Matrix",
):
    """Plot separability matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(matrix, cmap="RdYlGn_r")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Min Edit Distance (lower = more similar)")
    
    ax.set_xticks(np.arange(len(activity_names)))
    ax.set_yticks(np.arange(len(activity_names)))
    ax.set_xticklabels(activity_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(activity_names, fontsize=8)
    
    for i in range(len(activity_names)):
        for j in range(len(activity_names)):
            text = ax.text(j, i, f"{matrix[i, j]:.1f}",
                          ha="center", va="center", 
                          color="white" if matrix[i, j] > matrix.mean() else "black",
                          fontsize=6)
    
    ax.set_xlabel("Model Activity")
    ax.set_ylabel("Query Activity")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved matrix plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick Anomaly Detection using pre-parsed programs")
    parser.add_argument("--programs", type=str, default="../exact_data/programs/parsed/programs_verbs_train.json",
                       help="Path to programs JSON")
    parser.add_argument("--output-dir", type=str, default="results/anomaly_detection_quick",
                       help="Output directory")
    parser.add_argument("--train-programs", type=int, default=20,
                       help="Number of programs per activity for model")
    parser.add_argument("--test-programs", type=int, default=10,
                       help="Number of programs per activity for testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Quick Anomaly Detection using Pre-Parsed Programs")
    logger.info("=" * 60)
    
    # Load programs
    logger.info(f"Loading programs from {args.programs}")
    with open(args.programs, "r") as f:
        data = json.load(f)
    
    activity_names = data.get("activity_names", [])
    programs_by_activity = data.get("programs_by_activity", {})
    
    logger.info(f"Found {len(activity_names)} activities")
    
    # Create distance matrix calculator
    dist_matrix = ProgramDistanceMatrix(activity_names)
    
    # Split programs into train/test for each activity
    logger.info("Splitting programs into train/test...")
    for activity in activity_names:
        activity_programs = programs_by_activity.get(activity, [])
        if not activity_programs:
            logger.warning(f"No programs for {activity}")
            continue
        
        # Extract program strings
        prog_strings = [p["program"] for p in activity_programs]
        
        # Shuffle and split
        random.shuffle(prog_strings)
        
        n_train = min(args.train_programs, len(prog_strings) // 2)
        n_test = min(args.test_programs, len(prog_strings) - n_train)
        
        train_progs = prog_strings[:n_train]
        test_progs = prog_strings[n_train:n_train + n_test]
        
        # Set model programs
        dist_matrix.set_model_programs(activity, train_progs)
        
        # Add test programs
        for prog in test_progs:
            dist_matrix.add_test_program(activity, prog)
        
        logger.info(f"  {activity}: {len(train_progs)} train, {len(test_progs)} test")
    
    # Compute matrix
    logger.info("Computing distance matrix...")
    matrix = dist_matrix.compute_matrix(verbose=True)
    
    # Get metrics
    metrics = dist_matrix.get_separability_metrics()
    
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Diagonal mean (same-activity): {metrics['diagonal_mean']:.2f}")
    logger.info(f"Off-diagonal mean (cross-activity): {metrics['off_diagonal_mean']:.2f}")
    logger.info(f"Separation (higher = better): {metrics['separation']:.2f}")
    
    # Save results
    results = dist_matrix.to_dict()
    results["config"] = {
        "programs_file": str(args.programs),
        "train_programs": args.train_programs,
        "test_programs": args.test_programs,
        "value_tolerance": VALUE_TOLERANCE,
        "seed": args.seed,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_dir / 'results.json'}")
    
    # Plot matrix
    plot_separability_matrix(
        matrix, activity_names,
        str(output_dir / "separability_matrix.png"),
        title="Program Edit Distance Separability (Pre-parsed)"
    )
    
    logger.success(f"Anomaly detection complete! Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
