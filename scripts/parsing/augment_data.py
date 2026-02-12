#!/usr/bin/env python
"""Data augmentation using Executable Activity Models.

This script generates augmented training data for activity segmentation
by using trained parsers to create executable activity models, then
generating new motion trajectories guided by these models.

Workflow:
1. Load training subset (e.g., 25% of data)
2. Parse activity segments into programs (mock for now, real parser later)
3. Aggregate programs by activity class into ExecutableActivityModels
4. Generate augmented samples using BehaviourModel with random init poses
5. Save augmented data in ESK format for DLC2Action pipeline

Usage:
    python scripts/parsing/augment_data.py \
        --load-models /pvc/esk/models.json \
        --num-samples 1000 \
        --output-dir /pvc/esk/augmented
"""

import argparse
import random
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

# Conditional imports for heavy dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_split_file(split_path: str) -> dict:
    """Parse a train/val/test split file.
    
    Returns:
        dict with keys 'train', 'validation', 'test' containing lists of video names
    """
    splits = {"train": [], "validation": [], "test": []}
    current_section = None
    
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train videos:"):
                current_section = "train"
            elif line.startswith("Validation videos:"):
                current_section = "validation"
            elif line.startswith("Test videos:"):
                current_section = "test"
            elif current_section:
                splits[current_section].append(line)
    
    return splits


def subsample_videos(videos: list[str], fraction: float, seed: int) -> list[str]:
    """Subsample videos to a given fraction."""
    if fraction >= 1.0:
        return videos
    
    n_keep = max(1, int(len(videos) * fraction))
    rng = random.Random(seed)
    return rng.sample(videos, n_keep)


def load_training_segments(
    data_path: str,
    label_path: str,
    video_names: list[str],
    data_suffix: str = "_pose3d_smpl.h5",
    label_suffix: str = "_labels.pickle",
) -> dict:
    """Load all activity segments from training videos.
    
    Args:
        data_path: Path to pose data directory
        label_path: Path to label directory
        video_names: List of video names to load
        data_suffix: Suffix for pose files
        label_suffix: Suffix for label files
        
    Returns:
        Dictionary mapping activity_name -> list of segment info dicts
    """
    from exact.data.utils import load_labels, extract_segment_poses
    from exact.data.esk import load_pose_file
    from exact.data.env import smpl_rotations_to_positions
    
    all_segments = {}
    activity_names = None
    
    for video_name in tqdm(video_names, desc="Loading training data"):
        pose_file = Path(data_path) / f"{video_name}{data_suffix}"
        label_file = Path(label_path) / f"{video_name}{label_suffix}"
        
        if not pose_file.exists() or not label_file.exists():
            logger.warning(f"Missing files for {video_name}, skipping")
            continue
        
        # Load SMPL axis-angle rotations (T, 24, 3) and convert to positions (T, 72)
        pose_data, _ = load_pose_file(pose_file)
        poses = smpl_rotations_to_positions(pose_data)  # (T, 72)
        labels = load_labels(str(label_file))
        
        if activity_names is None:
            activity_names = labels["activity_names"]
            for name in activity_names:
                all_segments[name] = []
        
        # Extract segments for each activity
        video_segments = labels["segments"][0]  # First (only) video
        for activity_idx, activity_name in enumerate(activity_names):
            segs = video_segments[activity_idx]
            for seg in segs:
                start, end, flag = seg
                if end > start:  # Valid segment
                    segment_poses = extract_segment_poses(poses, start, end)
                    all_segments[activity_name].append({
                        "video": video_name,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "poses": segment_poses,
                    })
    
    return all_segments


def create_executable_models(
    segments_by_activity: dict,
    parser,
    eval_timesteps: int = 100,
    max_programs_per_activity: int = 50,
    program_budget: int | None = None,
    seed: int = 42,
) -> dict:
    """Create ExecutableActivityModels from parsed segments.
    
    Args:
        segments_by_activity: Dict mapping activity -> list of segment info
        parser: Parser to convert segments to programs (trained or mock)
        eval_timesteps: Common evaluation timesteps for all models
        max_programs_per_activity: Maximum programs to parse per activity
        program_budget: If set, select diverse subset using hierarchical clustering
        seed: Random seed for sampling
        
    Returns:
        Dictionary mapping activity_name -> ExecutableActivityModel
    """
    from exact.models import create_executable_model
    
    rng = random.Random(seed)
    models = {}
    
    for activity_name, segments in segments_by_activity.items():
        if not segments:
            logger.info(f"No segments for {activity_name}, skipping")
            continue
        
        # Sample segments if too many
        if len(segments) > max_programs_per_activity:
            segments = rng.sample(segments, max_programs_per_activity)
        
        # Parse segments into programs
        programs = []
        for seg in segments:
            poses = seg.get("poses")
            duration = seg["duration"]
            
            # Use parser (trained or mock)
            if poses is not None and hasattr(parser, 'parse') and not hasattr(parser, 'duration'):
                # Trained parser uses motion data
                program = parser.parse(poses)
            else:
                # Mock parser uses duration
                program = parser.parse(duration=duration)
            programs.append(program)
        
        # Apply program budget selection if specified
        if program_budget and len(programs) > program_budget:
            from exact.programs import select_diverse_programs
            logger.info(f"  Selecting {program_budget} diverse programs from {len(programs)} for {activity_name}...")
            result = select_diverse_programs(
                programs,
                budget=program_budget,
                method="hierarchical",
                show_progress=False,
            )
            programs = result.selected_programs
        
        # Create executable model
        model = create_executable_model(
            activity_name=activity_name,
            programs=programs,
            eval_timesteps=eval_timesteps,
            metadata={
                "num_source_segments": len(segments),
                "source_videos": list(set(s["video"] for s in segments)),
            },
        )
        models[activity_name] = model
        
        logger.info(f"Created model for {activity_name}: {len(programs)} programs")
    
    return models


def observation_to_pose96(obs: np.ndarray) -> np.ndarray:
    """Convert HumEnv observation (358-dim) to ESK-compatible pose format (96-dim).
    
    Extracts root-relative 3D joint positions from the HumEnv observation
    (24 joints × 3 xyz = 72 dims), then appends a likelihood column per
    joint to produce the 96-dim format expected by DLC2Action:
    24 joints × 4 (x, y, z, likelihood).
    
    Args:
        obs: HumEnv observation [358] or [N, 358]
        
    Returns:
        ESK-compatible pose [96] or [N, 96]
    """
    from exact.data.env import extract_joint_positions
    
    is_batched = obs.ndim > 1
    if not is_batched:
        obs = obs[np.newaxis, :]
    
    # Extract root-relative joint positions: (N, 72) = 24 joints × 3 xyz
    joint_positions = extract_joint_positions(obs)  # (N, 72)
    
    # Reshape to (N, 24, 3) for per-joint processing
    positions_3d = joint_positions.reshape(-1, 24, 3)
    
    # Add likelihood column (1.0 = confident) per joint
    likelihood = np.ones((*positions_3d.shape[:-1], 1), dtype=np.float32)
    
    # Concatenate: (x, y, z, likelihood) per joint → (N, 24, 4) → flatten to (N, 96)
    pose_with_lik = np.concatenate([positions_3d, likelihood], axis=-1)
    pose96 = pose_with_lik.reshape(-1, 96)
    
    if not is_batched:
        pose96 = pose96[0]
    
    return pose96.astype(np.float32)


def generate_augmented_data(
    executable_models: dict,
    num_samples: int,
    output_dir: str,
    behaviour_model: "BehaviourModel" = None,
    env: "HumEnv" = None,
    trajectory_length: int = 100,
    seed: int = 42,
    dry_run: bool = False,
) -> str:
    """Generate augmented data using executable activity models.
    
    Args:
        executable_models: Dict mapping activity -> ExecutableActivityModel
        num_samples: Total number of samples to generate
        output_dir: Output directory for augmented data
        behaviour_model: BehaviourModel for trajectory generation (None for dry run)
        env: HumEnv environment (None for dry run)
        trajectory_length: Length of each generated trajectory
        seed: Random seed
        dry_run: If True, generate random placeholder data instead of real trajectories
        
    Returns:
        Path to output directory with generated files
    """
    from exact.data import ESKDatasetWriter
    
    rng = random.Random(seed)
    
    # Get list of activities with models
    activities = list(executable_models.keys())
    num_activities = len(activities)
    
    if num_activities == 0:
        raise ValueError("No executable models provided")
    
    # Distribute samples equally across activities
    samples_per_activity = num_samples // num_activities
    remainder = num_samples % num_activities
    
    logger.info(f"Generating {num_samples} samples across {num_activities} activities")
    logger.info(f"  {samples_per_activity} per activity (+{remainder} extra)")
    
    # Initialize dataset writer
    writer = ESKDatasetWriter(
        activity_names=activities,
        output_dir=output_dir,
        video_name="augmented_data",
    )
    
    # Generate trajectories for each activity
    for activity_idx, activity_name in enumerate(activities):
        model = executable_models[activity_name]
        
        # Add one extra sample to first 'remainder' activities
        n_samples = samples_per_activity + (1 if activity_idx < remainder else 0)
        
        logger.info(f"Generating {n_samples} samples for {activity_name}")
        
        for _ in tqdm(range(n_samples), desc=activity_name):
            if dry_run or behaviour_model is None or env is None:
                # Generate random placeholder trajectory
                poses = np.random.randn(trajectory_length, 96).astype(np.float32) * 0.1
            else:
                # Generate real trajectory using behaviour model
                result = behaviour_model.generate_trajectory(
                    env=env,
                    executable_model=model,
                    num_steps=trajectory_length,
                )
                # Convert observations to ESK pose format
                poses = observation_to_pose96(result["observations"])
            
            writer.add_trajectory(poses, activity_name)
    
    # Save to files
    pose_path, label_path = writer.save()
    
    logger.info(f"Saved augmented data:")
    logger.info(f"  Poses: {pose_path}")
    logger.info(f"  Labels: {label_path}")
    logger.info(f"  Summary: {writer.summary()}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented data using Executable Activity Models"
    )
    
    # Model loading options (alternative to parsing from scratch)
    parser.add_argument(
        "--load-models",
        type=str,
        default=None,
        help="Load pre-built models from JSON (from build_models.py). "
             "If provided, skips parsing and uses these models directly.",
    )
    
    # Data paths (only needed if not using --load-models)
    parser.add_argument(
        "--data-path",
        type=str,
        default="esk/D2A_converted_pose_smpl",
        help="Path to pose data directory",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="esk/D2A_converted_label_verbs",
        help="Path to label directory",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="esk/trainvaltest_split.txt",
        help="Path to train/val/test split file",
    )
    
    # Augmentation parameters
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.25,
        help="Fraction of training data to use for parsing (default: 0.25)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total number of augmented samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=100,
        help="Length of each generated trajectory (default: 100)",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=100,
        help="Evaluation timesteps for executable models (default: 100)",
    )
    parser.add_argument(
        "--max-programs-per-activity",
        type=int,
        default=50,
        help="Maximum programs per activity model (default: 50)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Output directory for augmented data (default: data/augmented)",
    )
    
    # Model options
    parser.add_argument(
        "--behaviour-model",
        type=str,
        default="facebook/metamotivo-M-1",
        help="HuggingFace model name for BehaviourModel",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model inference (auto, cpu, cuda)",
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate random data instead of real trajectories (for testing)",
    )
    parser.add_argument(
        "--save-models",
        type=str,
        default=None,
        help="Path to save executable models JSON (optional)",
    )
    parser.add_argument(
        "--parser-checkpoint",
        type=str,
        default=None,
        help="Path to trained parser checkpoint (uses mock if not provided)",
    )
    parser.add_argument(
        "--program-budget",
        type=int,
        default=None,
        help="Select diverse subset of programs per activity using hierarchical clustering",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger.info("=== Data Augmentation with Executable Activity Models ===")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
    
    # Option 1: Load pre-built models from JSON
    if args.load_models:
        logger.info(f"Loading pre-built models from: {args.load_models}")
        from exact.models import ActivityModelCollection
        import json
        
        with open(args.load_models, "r") as f:
            models_data = json.load(f)
        
        collection = ActivityModelCollection.from_dict(models_data)
        executable_models = collection.models
        logger.info(f"Loaded {len(executable_models)} activity models")
        for name, model in executable_models.items():
            logger.info(f"  {name}: {model.num_programs} programs")
    
    # Option 2: Parse from ESK data
    else:
        logger.info(f"Parsing from ESK data (train_fraction={args.train_fraction})")
        
        # Load split file and get training videos
        logger.info("Loading split file...")
        splits = parse_split_file(args.split_path)
        train_videos = splits["train"]
        logger.info(f"Found {len(train_videos)} training videos")
    
    # Subsample training videos
        # Subsample training videos
        train_subset = subsample_videos(train_videos, args.train_fraction, args.seed)
        logger.info(f"Using {len(train_subset)} videos ({args.train_fraction*100:.0f}%)")
        
        # Load training segments
        logger.info("Loading training segments...")
        segments = load_training_segments(
            data_path=args.data_path,
            label_path=args.label_path,
            video_names=train_subset,
        )
        
        # Log segment statistics
        total_segments = sum(len(segs) for segs in segments.values())
        logger.info(f"Loaded {total_segments} segments across {len(segments)} activities")
        for name, segs in sorted(segments.items(), key=lambda x: -len(x[1])):
            if segs:
                logger.info(f"  {name}: {len(segs)} segments")
        
        # Create parser (trained or mock)
        logger.info("Creating parser...")
        from exact.parser import load_parser
        parser_model = load_parser(checkpoint_path=args.parser_checkpoint)
        if args.parser_checkpoint:
            logger.info(f"Using trained parser from {args.parser_checkpoint}")
        else:
            logger.info("Using mock parser (no checkpoint provided)")
        
        logger.info("Creating executable activity models...")
        
        # Create executable models
        executable_models = create_executable_models(
            segments_by_activity=segments,
            parser=parser_model,
            eval_timesteps=args.eval_timesteps,
            max_programs_per_activity=args.max_programs_per_activity,
            program_budget=args.program_budget,
            seed=args.seed,
        )
        
        # Optionally save models
        if args.save_models:
            from exact.models import ActivityModelCollection
            collection = ActivityModelCollection(eval_timesteps=args.eval_timesteps)
            for model in executable_models.values():
                collection.add_model(model)
            collection.save(args.save_models)
            logger.info(f"Saved models to {args.save_models}")
    
    # Set up behaviour model and environment (unless dry run)
    behaviour_model = None
    env = None
    
    if not args.dry_run:
        logger.info("Loading behaviour model and environment...")
        
        if args.device == "auto":
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        from exact import BehaviourModel, HumEnv
        
        try:
            behaviour_model = BehaviourModel(
                model_name=args.behaviour_model,
                device=device,
            )
            env = HumEnv()
            logger.info(f"Loaded model on {device}")
        except Exception as e:
            logger.warning(f"Failed to load behaviour model: {e}")
            logger.warning("Falling back to dry run mode")
            args.dry_run = True
    
    # Generate augmented data
    logger.info("Generating augmented data...")
    output_dir = generate_augmented_data(
        executable_models=executable_models,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        behaviour_model=behaviour_model,
        env=env,
        trajectory_length=args.trajectory_length,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    
    logger.success(f"Augmentation complete! Data saved to {output_dir}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("To use augmented data with segmentation:")
    print(f"  python scripts/tasks/segmentation.py \\")
    print(f"    project.data_path={output_dir} \\")
    print(f"    project.annotation_path={output_dir} \\")
    print(f"    training.split_path=<new_split_file>")
    print("="*60)


if __name__ == "__main__":
    main()
