#!/usr/bin/env python
"""Parse ESK dataset into motion programs.

This script loads motion segments from the ESK dataset, parses them using
the trained motion-conditioned parser, and saves the resulting programs
as a JSON file for downstream use (executable models, augmentation, etc.).

Output format:
    {
        "metadata": {
            "parser_checkpoint": "results/parser/...",
            "esk_data_path": "../esk",
            "split": "train",
            "num_videos": 35,
            "num_segments": 1234,
            "timestamp": "2026-01-22T12:00:00"
        },
        "activity_names": ["Cut", "Pour", "Stir", ...],
        "programs_by_activity": {
            "Cut": [
                {"program": "[0,50]rhand.x(0.3);...", "video": "YH2002...", "start": 100, "end": 250},
                ...
            ],
            ...
        }
    }

Usage:
    # Parse training data with trained parser
    uv run scripts/parsing/parse_esk.py --parser-checkpoint results/parser/<checkpoint>
    
    # Parse 50% of training data (for reduced experiments)
    uv run scripts/parsing/parse_esk.py --parser-checkpoint results/parser/<checkpoint> --train-fraction 0.5
    
    # Use mock parser for testing pipeline
    uv run scripts/parsing/parse_esk.py --mock
"""

import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data.esk import load_pose_file
from exact.data.env import smpl_rotations_to_positions


def parse_args():
    parser = argparse.ArgumentParser(description="Parse ESK dataset into programs")
    parser.add_argument(
        "--parser-checkpoint",
        type=str,
        default=None,
        help="Path to trained parser checkpoint directory",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock parser (for testing pipeline without trained model)",
    )
    parser.add_argument(
        "--esk-path",
        type=str,
        default="../esk",
        help="Path to ESK dataset (relative to this script or absolute)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Which split to parse",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training videos to use (for reduced experiments)",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="verbs",
        choices=["verbs", "nouns", "activity"],
        help="Type of labels to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <esk_path>/programs_<split>.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--max-segments-per-activity",
        type=int,
        default=None,
        help="Maximum segments to parse per activity (for faster testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parsing (if parser supports batching)",
    )
    return parser.parse_args()


def load_split_file(split_path: str) -> dict[str, list[str]]:
    """Load train/test split from file.
    
    Returns:
        Dict with 'train' and 'test' keys containing video names
    """
    splits = {"train": [], "test": []}
    current_section = None
    
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train"):
                current_section = "train"
            elif line.startswith("Test"):
                current_section = "test"
            elif line.startswith("Val"):
                current_section = "val"
            elif current_section and current_section in splits:
                splits[current_section].append(line)
    
    return splits


def subsample_videos(videos: list[str], fraction: float, seed: int) -> list[str]:
    """Subsample videos to a given fraction."""
    if fraction >= 1.0:
        return videos
    
    n_keep = max(1, int(len(videos) * fraction))
    rng = random.Random(seed)
    return rng.sample(videos, n_keep)


def load_segments_from_video(
    video_name: str,
    pose_dir: Path,
    label_dir: Path,
    pose_suffix: str = "_pose3d_smpl.h5",
    label_suffix: str = "_labels.pickle",
) -> dict[str, list[dict]]:
    """Load all segments from a single video.
    
    Loads SMPL axis-angle rotations from ESK HDF5, converts them to
    root-relative 3D joint positions via MuJoCo forward kinematics,
    and returns segments as (T, 72) position arrays matching the format
    used for parser training.
    
    Returns:
        Dict mapping activity_name -> list of segment dicts with 'poses', 'start', 'end'
    """
    pose_file = pose_dir / f"{video_name}{pose_suffix}"
    label_file = label_dir / f"{video_name}{label_suffix}"
    
    if not pose_file.exists() or not label_file.exists():
        logger.warning(f"Missing files for {video_name}")
        return {}
    
    # Load SMPL axis-angle rotations (T, 24, 3) via pandas reader
    try:
        pose_data, _ = load_pose_file(pose_file)
    except Exception as e:
        logger.warning(f"Cannot read poses from {pose_file}: {e}")
        return {}
    
    # Convert axis-angle rotations → root-relative 3D positions (T, 72)
    # This matches the format produced by generate_data.py / extract_joint_positions()
    logger.debug(f"Converting {video_name}: {pose_data.shape[0]} frames, rotations → positions")
    poses = smpl_rotations_to_positions(pose_data)  # (T, 72)
    
    # Load labels
    with open(label_file, "rb") as f:
        labels_data = pickle.load(f)
    
    # Parse labels format: (meta, activity_names, video_names, segments)
    if isinstance(labels_data, tuple) and len(labels_data) >= 4:
        _, activity_names, _, segments = labels_data
    else:
        logger.warning(f"Unexpected label format in {label_file}")
        return {}
    
    # Extract segments per activity
    segments_by_activity = {}
    video_segments = segments[0] if segments else []  # First (only) video
    
    for activity_idx, activity_name in enumerate(activity_names):
        if activity_idx >= len(video_segments):
            continue
        
        activity_name = str(activity_name)
        if activity_name not in segments_by_activity:
            segments_by_activity[activity_name] = []
        
        for seg in video_segments[activity_idx]:
            if len(seg) >= 2:
                start, end = int(seg[0]), int(seg[1])
                if start < end <= len(poses):
                    segment_poses = poses[start:end].copy()
                    segments_by_activity[activity_name].append({
                        "poses": segment_poses,
                        "start": start,
                        "end": end,
                        "video": video_name,
                    })
    
    return segments_by_activity


def create_parser(checkpoint_path: str | None = None, use_mock: bool = False):
    """Create parser instance (trained or mock).
    
    Args:
        checkpoint_path: Path to trained parser checkpoint
        use_mock: If True, use mock parser instead
        
    Returns:
        Parser with parse() method
    """
    if use_mock or checkpoint_path is None:
        from exact.models import MockParser
        logger.info("Using mock parser")
        return MockParser()
    
    from exact.parser import load_parser
    logger.info(f"Loading trained parser from: {checkpoint_path}")
    return load_parser(checkpoint_path=checkpoint_path)


def validate_program(program: str) -> bool:
    """Check if a program is syntactically valid.
    
    Args:
        program: Program string to validate
        
    Returns:
        True if program is valid, False otherwise
    """
    from exact.parser.utils import validate_program as _validate
    return _validate(program)


def main():
    args = parse_args()
    
    # Resolve paths - relative paths are relative to workspace root (parent of scripts/)
    workspace_root = Path(__file__).parent.parent
    if args.esk_path.startswith("..") or args.esk_path.startswith("."):
        esk_path = (workspace_root / args.esk_path).resolve()
    else:
        esk_path = Path(args.esk_path).resolve()
    
    logger.info(f"ESK data path: {esk_path}")
    
    # Setup directories based on label type
    label_type_map = {
        "verbs": "D2A_converted_label_verbs",
        "nouns": "D2A_converted_label_nouns",
        "activity": "D2A_converted_label_activity",
    }
    pose_dir = esk_path / "D2A_converted_pose_smpl"
    label_dir = esk_path / label_type_map[args.label_type]
    split_file = esk_path / "traintest_split.txt"
    
    # Check paths exist
    if not esk_path.exists():
        logger.error(f"ESK path does not exist: {esk_path}")
        return 1
    if not pose_dir.exists():
        logger.error(f"Pose directory does not exist: {pose_dir}")
        return 1
    if not label_dir.exists():
        logger.error(f"Label directory does not exist: {label_dir}")
        return 1
    
    # Load split
    splits = load_split_file(str(split_file))
    logger.info(f"Loaded split: {len(splits['train'])} train, {len(splits['test'])} test videos")
    
    # Select videos based on split argument
    if args.split == "train":
        videos = splits["train"]
    elif args.split == "test":
        videos = splits["test"]
    else:
        videos = splits["train"] + splits["test"]
    
    # Subsample if requested
    if args.train_fraction < 1.0 and args.split in ["train", "all"]:
        original_count = len(videos)
        videos = subsample_videos(videos, args.train_fraction, args.seed)
        logger.info(f"Subsampled to {len(videos)}/{original_count} videos (fraction={args.train_fraction})")
    
    # Create parser
    parser = create_parser(
        checkpoint_path=args.parser_checkpoint,
        use_mock=args.mock,
    )
    
    # Load all segments
    logger.info(f"Loading segments from {len(videos)} videos...")
    all_segments_by_activity: dict[str, list[dict]] = {}
    activity_names_set = set()
    
    for video_name in tqdm(videos, desc="Loading videos"):
        video_segments = load_segments_from_video(
            video_name,
            pose_dir,
            label_dir,
        )
        for activity_name, segments in video_segments.items():
            activity_names_set.add(activity_name)
            if activity_name not in all_segments_by_activity:
                all_segments_by_activity[activity_name] = []
            all_segments_by_activity[activity_name].extend(segments)
    
    activity_names = sorted(activity_names_set)
    total_segments = sum(len(segs) for segs in all_segments_by_activity.values())
    logger.info(f"Loaded {total_segments} segments across {len(activity_names)} activities")
    
    # Parse segments into programs
    logger.info("Parsing segments into programs...")
    programs_by_activity: dict[str, list[dict]] = {}
    parse_stats = {"total": 0, "valid": 0, "invalid": 0}
    
    for activity_name in tqdm(activity_names, desc="Activities"):
        segments = all_segments_by_activity.get(activity_name, [])
        
        # Limit segments if requested
        if args.max_segments_per_activity and len(segments) > args.max_segments_per_activity:
            rng = random.Random(args.seed)
            segments = rng.sample(segments, args.max_segments_per_activity)
        
        programs_by_activity[activity_name] = []
        
        for seg in tqdm(segments, desc=f"  {activity_name}", leave=False):
            poses = seg["poses"]
            
            # Parse motion to program
            try:
                program = parser.parse(poses)
            except Exception as e:
                logger.warning(f"Parse error for {seg['video']}[{seg['start']}:{seg['end']}]: {e}")
                parse_stats["invalid"] += 1
                continue
            
            parse_stats["total"] += 1
            
            # Validate program
            if validate_program(program):
                parse_stats["valid"] += 1
                programs_by_activity[activity_name].append({
                    "program": program,
                    "video": seg["video"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["end"] - seg["start"],
                })
            else:
                parse_stats["invalid"] += 1
                logger.debug(f"Invalid program: {program[:50]}...")
    
    # Log stats
    logger.info(f"Parsing complete:")
    logger.info(f"  Total: {parse_stats['total']}")
    logger.info(f"  Valid: {parse_stats['valid']} ({100*parse_stats['valid']/max(1,parse_stats['total']):.1f}%)")
    logger.info(f"  Invalid: {parse_stats['invalid']}")
    
    for activity_name in activity_names:
        n_programs = len(programs_by_activity.get(activity_name, []))
        logger.info(f"  {activity_name}: {n_programs} programs")
    
    # Build output
    output_data = {
        "metadata": {
            "parser_checkpoint": args.parser_checkpoint,
            "esk_data_path": str(esk_path),
            "split": args.split,
            "train_fraction": args.train_fraction,
            "label_type": args.label_type,
            "num_videos": len(videos),
            "num_segments_total": total_segments,
            "num_programs_valid": parse_stats["valid"],
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
        },
        "activity_names": activity_names,
        "programs_by_activity": programs_by_activity,
    }
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_{args.split}"
        if args.train_fraction < 1.0:
            suffix += f"_{int(args.train_fraction * 100)}pct"
        output_path = esk_path / f"programs{suffix}.json"
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved programs to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
