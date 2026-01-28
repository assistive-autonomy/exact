#!/usr/bin/env python
"""Activity Assessment using Program Edit Distance.

This script evaluates activity separability using unordered tree edit distance (UTED)
between activity motion programs. Unlike the flow-based approach that operates on 
raw pose data, this method compares the symbolic structure of parsed programs.

Workflow:
1. Parse training data → create executable models (N programs per activity)
2. Parse test data → query programs
3. For each test program, compute min edit distance to each activity's model
4. Build separability matrix: M[i,j] = mean min-dist from activity i tests to activity j model

A good separability shows:
- Low diagonal (same-activity programs are structurally similar)
- High off-diagonal (cross-activity programs are structurally different)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.programs import (
    ProgramDistanceMatrix,
    parse_to_tree,
    program_edit_distance,
    VALUE_TOLERANCE,
)
from exact.models import ActivityModelCollection

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_config(config_path: str, overrides: list[str] = None) -> DictConfig:
    """Load config file and apply CLI overrides."""
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg
    return cfg


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def load_split_file(split_path: str) -> tuple[list[str], list[str], list[str]]:
    """Load train/val/test split from file.
    
    Supports two formats:
    1. Header-based: "Train videos:", "Val videos:", "Test videos:" followed by video names
    2. Tab-separated: "video_name\ttrain/val/test"
    """
    train_videos, val_videos, test_videos = [], [], []
    
    with open(split_path, "r") as f:
        content = f.read()
    
    # Check format
    if "Train videos:" in content or "Val videos:" in content or "Test videos:" in content:
        # Header-based format
        current_split = None
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train"):
                current_split = "train"
            elif line.startswith("Val"):
                current_split = "val"
            elif line.startswith("Test"):
                current_split = "test"
            elif current_split:
                if current_split == "train":
                    train_videos.append(line)
                elif current_split == "val":
                    val_videos.append(line)
                elif current_split == "test":
                    test_videos.append(line)
    else:
        # Tab-separated format
        for line in content.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                video_name, split = parts[0], parts[1]
                if split == "train":
                    train_videos.append(video_name)
                elif split == "val":
                    val_videos.append(video_name)
                elif split == "test":
                    test_videos.append(video_name)
    
    return train_videos, val_videos, test_videos


def load_training_segments(
    train_videos: list[str],
    label_dir: str,
    pose_dir: str,
) -> dict[str, list[tuple[np.ndarray, int, int]]]:
    """Load training segments per activity.
    
    Returns:
        Dict mapping activity name -> list of (poses, start, end) tuples
    """
    import pickle
    import h5py
    
    label_path = Path(label_dir)
    pose_path = Path(pose_dir)
    
    segments_by_activity: dict[str, list] = {}
    
    for video_name in tqdm(train_videos, desc="Loading training data"):
        # Load labels
        label_file = label_path / f"{video_name}_labels.pickle"
        if not label_file.exists():
            continue
            
        with open(label_file, "rb") as f:
            labels_data = pickle.load(f)
        
        # labels_data format: (meta, activity_names, video_names, segments)
        if isinstance(labels_data, tuple) and len(labels_data) >= 4:
            _, activity_names, _, segments = labels_data
        else:
            continue
        
        # Load poses - use SMPL format
        pose_file = None
        for suffix in ["_pose3d_smpl.h5", "_body_hand_eye_pose_norm.h5", "_pose_norm.h5"]:
            candidate = pose_path / f"{video_name}{suffix}"
            if candidate.exists():
                pose_file = candidate
                break
        
        if pose_file is None:
            logger.debug(f"No pose file found for {video_name}")
            continue
        
        with h5py.File(pose_file, "r") as f:
            if "tracks" in f and "table" in f["tracks"]:
                table = f["tracks"]["table"]
                # Handle structured array format (SMPL files)
                if table.dtype.names and "values_block_0" in table.dtype.names:
                    poses = table["values_block_0"][:]
                else:
                    poses = table[:]
            else:
                # Fallback: try direct array
                keys = list(f.keys())
                if keys:
                    poses = f[keys[0]][:]
                else:
                    logger.debug(f"Cannot read poses from {pose_file}")
                    continue
        
        # Convert SMPL format (96 features: 24 joints * 4 [x,y,z,likelihood]) 
        # to parser format (72 features: 24 joints * 3 [x,y,z])
        if poses.shape[1] == 96:
            # Remove likelihood column every 4th feature
            n_joints = 24
            poses_xyz = np.zeros((poses.shape[0], n_joints * 3), dtype=np.float32)
            for j in range(n_joints):
                poses_xyz[:, j*3:j*3+3] = poses[:, j*4:j*4+3]
            poses = poses_xyz
        
        # Extract segments
        # segments[0] is for this video, segments[0][activity_idx] is list of (start, end)
        video_segments = segments[0] if segments else []
        
        for activity_idx, activity_name in enumerate(activity_names):
            if activity_idx >= len(video_segments):
                continue
                
            activity_name = str(activity_name)
            if activity_name not in segments_by_activity:
                segments_by_activity[activity_name] = []
            
            for segment in video_segments[activity_idx]:
                if len(segment) >= 2:
                    start, end = int(segment[0]), int(segment[1])
                    if start < end <= len(poses):
                        segment_poses = poses[start:end]
                        segments_by_activity[activity_name].append((segment_poses, start, end))
    
    return segments_by_activity


def create_parser(checkpoint_path: str | None = None):
    """Create parser instance (mock or trained).
    
    Args:
        checkpoint_path: Path to trained parser checkpoint.
                        If None, uses mock parser for testing.
    
    Returns:
        Parser instance with parse() method
    """
    from exact.parser import load_parser
    return load_parser(checkpoint_path=checkpoint_path)


def parse_segments_to_programs(
    segments: list[tuple[np.ndarray, int, int]],
    parser,
    max_programs: int | None = None,
) -> list[str]:
    """Parse pose segments into programs using the parser.
    
    Args:
        segments: List of (poses, start, end) tuples
        parser: Parser instance with parse() method
        max_programs: Maximum number of programs to return
        
    Returns:
        List of program strings
    """
    programs = []
    
    for poses, start, end in segments:
        if max_programs and len(programs) >= max_programs:
            break
        
        try:
            program = parser.parse(poses)
            if program:
                programs.append(program)
        except Exception as e:
            logger.debug(f"Failed to parse segment: {e}")
            continue
    
    return programs


def plot_separability_matrix(
    matrix: np.ndarray,
    activity_names: list[str],
    output_path: str,
    title: str = "Program Edit Distance Separability Matrix",
):
    """Plot separability matrix as heatmap.
    
    Note: Lower values on diagonal = better (same activity is more similar)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix, cmap="RdYlGn_r")  # Reversed: green=low (good), red=high
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Min Edit Distance (lower = more similar)")
    
    # Set ticks
    ax.set_xticks(np.arange(len(activity_names)))
    ax.set_yticks(np.arange(len(activity_names)))
    ax.set_xticklabels(activity_names, rotation=45, ha="right")
    ax.set_yticklabels(activity_names)
    
    # Add text annotations
    for i in range(len(activity_names)):
        for j in range(len(activity_names)):
            text = ax.text(j, i, f"{matrix[i, j]:.1f}",
                          ha="center", va="center", 
                          color="white" if matrix[i, j] > matrix.mean() else "black",
                          fontsize=8)
    
    ax.set_xlabel("Model Activity")
    ax.set_ylabel("Query Activity")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved matrix plot to {output_path}")


def plot_separation_summary(
    metrics: dict,
    output_path: str,
):
    """Plot per-activity separation metrics."""
    per_activity = metrics["per_activity"]
    activities = list(per_activity.keys())
    
    separations = [per_activity[a]["separation"] for a in activities]
    same_dist = [per_activity[a]["same_activity_dist"] for a in activities]
    cross_dist = [per_activity[a]["cross_activity_dist"] for a in activities]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Separation scores
    ax = axes[0]
    colors = ["green" if s > 0 else "red" for s in separations]
    ax.barh(activities, separations, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Separation (cross - same)")
    ax.set_title("Separation Score (higher = better)")
    
    # Plot 2: Same vs Cross distances
    ax = axes[1]
    x = np.arange(len(activities))
    width = 0.35
    ax.bar(x - width/2, same_dist, width, label="Same Activity", color="blue", alpha=0.7)
    ax.bar(x + width/2, cross_dist, width, label="Cross Activity", color="orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(activities, rotation=45, ha="right")
    ax.set_ylabel("Edit Distance")
    ax.set_title("Same vs Cross Activity Distance")
    ax.legend()
    
    # Plot 3: Summary stats
    ax = axes[2]
    summary = {
        "Diagonal\n(same-act)": metrics["diagonal_mean"],
        "Off-diag\n(cross-act)": metrics["off_diagonal_mean"],
        "Separation": metrics["separation"],
    }
    colors = ["blue", "orange", "green" if metrics["separation"] > 0 else "red"]
    ax.bar(summary.keys(), summary.values(), color=colors, alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title("Overall Metrics")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved summary plot to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Activity Assessment using Program Edit Distance")
    parser.add_argument("--config", type=str, default="configs/assessment.yaml",
                       help="Path to config file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: results/assessment_edit_dist/<timestamp>)")
    parser.add_argument("--max-train-programs", type=int, default=50,
                       help="Maximum programs per activity for training model")
    parser.add_argument("--max-test-programs", type=int, default=50,
                       help="Maximum test programs per activity")
    parser.add_argument("--load-models", type=str, default=None,
                       help="Load pre-computed models from JSON instead of parsing")
    parser.add_argument("--save-models", type=str, default=None,
                       help="Save computed models to JSON")
    parser.add_argument("--parser-checkpoint", type=str, default=None,
                       help="Path to trained parser checkpoint (uses mock if not provided)")
    parser.add_argument("--program-budget", type=int, default=None,
                       help="Select diverse subset of programs per activity (None = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/assessment_edit_dist/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Activity Assessment using Program Edit Distance")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Value tolerance: {VALUE_TOLERANCE}")
    
    # Load config
    cfg = load_config(args.config, args.overrides)
    
    # Paths - use absolute paths or paths relative to esk_dir from config
    esk_dir = Path(cfg.data.get("esk_dir", "/pvc/esk"))
    label_type = cfg.data.get("label_type", "verbs")
    label_dir = str(esk_dir / f"D2A_converted_label_{label_type}")
    pose_dir = str(esk_dir / "D2A_converted_pose_smpl")
    split_path = str(esk_dir / "traintest_split.txt")
    
    logger.info(f"Label type: {label_type}")
    logger.info(f"Label dir: {label_dir}")
    
    # Load splits
    train_videos, val_videos, test_videos = load_split_file(split_path)
    logger.info(f"Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")
    
    # Get activity names
    import pickle
    sample_label = Path(label_dir) / f"{train_videos[0]}_labels.pickle"
    with open(sample_label, "rb") as f:
        sample_data = pickle.load(f)
    activity_names = [str(a) for a in sample_data[1]]
    logger.info(f"Activities: {activity_names}")
    
    # Create distance matrix calculator
    dist_matrix = ProgramDistanceMatrix(activity_names)
    
    # Load or create model programs
    if args.load_models:
        logger.info(f"Loading models from {args.load_models}")
        # Load programs directly from JSON without parsing into rewards (much faster)
        with open(args.load_models, "r") as f:
            models_data = json.load(f)
        
        for activity in activity_names:
            if activity in models_data.get("models", {}):
                model_data = models_data["models"][activity]
                programs = model_data.get("original_programs", [])
                # Limit programs if requested
                if args.max_train_programs and len(programs) > args.max_train_programs:
                    programs = programs[:args.max_train_programs]
                dist_matrix.set_model_programs(activity, programs)
                logger.info(f"  {activity}: {len(programs)} programs")
    else:
        logger.info("Creating model programs from training data...")
        
        # Load training segments
        train_segments = load_training_segments(train_videos, label_dir, pose_dir)
        
        # Create parser (trained or mock)
        parser_model = create_parser(checkpoint_path=args.parser_checkpoint)
        if args.parser_checkpoint:
            logger.info(f"Using trained parser from {args.parser_checkpoint}")
        else:
            logger.info("Using mock parser (no checkpoint provided)")
        
        # Parse training segments
        for activity in activity_names:
            if activity not in train_segments:
                logger.warning(f"No training segments for {activity}")
                continue
            
            segments = train_segments[activity]
            programs = parse_segments_to_programs(
                segments, parser_model, args.max_train_programs
            )
            
            # Apply program budget selection if specified
            if programs and args.program_budget and len(programs) > args.program_budget:
                from exact.programs import select_diverse_programs
                logger.info(f"  Selecting {args.program_budget} diverse programs from {len(programs)}...")
                result = select_diverse_programs(
                    programs,
                    budget=args.program_budget,
                    method="hierarchical",
                    show_progress=False,
                )
                programs = result.selected_programs
                logger.info(f"  {activity}: {len(programs)} programs (selected from {len(result.cluster_labels)})")
            
            if programs:
                dist_matrix.set_model_programs(activity, programs)
                if not args.program_budget:
                    logger.info(f"  {activity}: {len(programs)} programs")
    
    # Save models if requested
    if args.save_models:
        logger.info(f"Saving models to {args.save_models}")
        # Convert to ActivityModelCollection format
        from exact.models import ExecutableActivityModel
        save_collection = ActivityModelCollection(eval_timesteps=100)
        for activity, programs in dist_matrix.model_programs.items():
            # Extract original program strings from ProgramTree objects
            program_strings = [p.program for p in programs]
            model = ExecutableActivityModel.from_programs(
                programs=program_strings,
                activity_name=activity,
                eval_timesteps=100,
            )
            save_collection.add_model(model)
        save_collection.save(args.save_models)
    
    # Parse test programs
    logger.info("Creating test programs from test data...")
    test_segments = load_training_segments(test_videos, label_dir, pose_dir)
    parser_model = create_parser(checkpoint_path=args.parser_checkpoint)
    
    for activity in activity_names:
        if activity not in test_segments:
            logger.warning(f"No test segments for {activity}")
            continue
        
        segments = test_segments[activity]
        programs = parse_segments_to_programs(
            segments, parser_model, args.max_test_programs
        )
        
        for prog in programs:
            dist_matrix.add_test_program(activity, prog)
        
        logger.info(f"  {activity}: {len(programs)} test programs")
    
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
        "label_type": label_type,
        "max_train_programs": args.max_train_programs,
        "max_test_programs": args.max_test_programs,
        "value_tolerance": VALUE_TOLERANCE,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_dir / 'results.json'}")
    
    # Plot matrix
    plot_separability_matrix(
        matrix, activity_names,
        str(output_dir / "separability_matrix.png"),
        title=f"Program Edit Distance Separability ({label_type})"
    )
    
    # Plot summary
    plot_separation_summary(metrics, str(output_dir / "separation_summary.png"))
    
    logger.success(f"Assessment complete! Results saved to {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
