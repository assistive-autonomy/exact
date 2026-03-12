#!/usr/bin/env python
"""Parse ESK/HumanAct12 motion segments into symbolic programs.

This script parses motion segments using a trained motion-prefix parser with:
1. Single-sample processing for reliability
2. Multiple passes and retries with temperature scaling
3. Greedy decoding fallback after sampling fails
4. Prefix extraction from malformed programs

Usage:
    # Parse ESK activity train split
    uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint results/parser/20260309_101643/best_generation \
        --esk-path ../exact_data/benchmarks/esk \
        --split train \
        --label-type activity \
        --output ../exact_data/programs/parsed/programs_activity_train.json
    
    # Parse HumanAct12
    uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint results/parser/20260309_101643/best_generation \
        --esk-path ../exact_data/benchmarks/humanact12 \
        --split train \
        --label-type actions \
        --output ../exact_data/programs/parsed/programs_humanact12_train.json
    
    # Fast mode (uses batching, less reliable but faster)
    uv run scripts/parsing/parse_esk.py \
        --parser-checkpoint results/parser/20260309_101643/best_generation \
        --split train \
        --fast
"""

import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data.esk import load_pose_file
from exact.data.env import smpl_rotations_to_positions
from exact.parser.utils import validate_program, post_process_program


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced ESK parsing with improved yield")
    parser.add_argument("--parser-checkpoint", type=str, required=True)
    parser.add_argument("--esk-path", type=str, default="../exact_data/benchmarks/esk")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--label-type", type=str, default="activity", 
                        choices=["verbs", "nouns", "activity", "actions"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Use batched mode (faster but lower yield)")
    parser.add_argument("--max-segments-per-activity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Enhanced parsing parameters
    parser.add_argument("--num-passes", type=int, default=5, help="Number of independent parsing attempts")
    parser.add_argument("--num-retries", type=int, default=15, help="Retries per pass with increasing temp")
    parser.add_argument("--greedy-fallback", action="store_true", default=True,
                        help="Use greedy decoding as last resort")
    return parser.parse_args()


def load_split_file(split_path: str) -> dict[str, list[str]]:
    """Load train/val/test split from file."""
    splits = {"train": [], "test": [], "val": []}
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


def load_segments_from_video(
    video_name: str,
    pose_dir: Path,
    label_dir: Path,
) -> dict[str, list[dict]]:
    """Load all segments from a single video."""
    pose_file = pose_dir / f"{video_name}_pose3d_smpl.h5"
    label_file = label_dir / f"{video_name}_labels.pickle"
    
    if not pose_file.exists() or not label_file.exists():
        logger.warning(f"Missing files for {video_name}")
        return {}
    
    try:
        pose_data, _ = load_pose_file(pose_file)
    except Exception as e:
        logger.warning(f"Cannot read poses from {pose_file}: {e}")
        return {}
    
    poses = smpl_rotations_to_positions(pose_data)
    
    with open(label_file, "rb") as f:
        labels_data = pickle.load(f)
    
    if isinstance(labels_data, tuple) and len(labels_data) >= 4:
        _, activity_names, _, segments = labels_data
    else:
        logger.warning(f"Unexpected label format in {label_file}")
        return {}
    
    segments_by_activity = {}
    video_segments = segments[0] if segments else []
    
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


class EnhancedParser:
    """Parser wrapper with improved retry and fallback strategies."""
    
    def __init__(
        self,
        checkpoint_path: str,
        num_passes: int = 5,
        num_retries: int = 15,
        greedy_fallback: bool = True,
        device: str = "auto",
    ):
        from exact.parser import load_parser
        
        self.num_passes = num_passes
        self.num_retries = num_retries
        self.greedy_fallback = greedy_fallback
        
        # Load parser with enhanced settings
        self.parser = load_parser(
            checkpoint_path=checkpoint_path,
            device=device,
            use_grammar_constraint=True,
            temperature=0.3,  # Lower base temperature for more focused sampling
            top_p=0.95,       # Slightly wider nucleus
            num_retries=num_retries,
            retry_temperature=0.4,  # Start retries lower
            num_passes=num_passes,
        )
        
        # Store settings for greedy fallback (reuse same parser to save memory)
        self.greedy_fallback = greedy_fallback
        self._original_temperature = 0.3
        self._original_top_p = 0.95
        self._original_num_retries = num_retries
        self._original_num_passes = num_passes
    
    def _set_greedy_mode(self, enable: bool):
        """Switch parser between sampling and greedy mode."""
        if enable:
            self.parser.temperature = 0.0
            self.parser.top_p = 1.0
            self.parser.num_retries = 0
            self.parser.num_passes = 1
        else:
            self.parser.temperature = self._original_temperature
            self.parser.top_p = self._original_top_p
            self.parser.num_retries = self._original_num_retries
            self.parser.num_passes = self._original_num_passes
    
    def parse_single(self, motion: np.ndarray) -> tuple[str, bool]:
        """Parse a single motion with all fallback strategies.
        
        Returns:
            Tuple of (program_string, is_valid)
        """
        program = ""
        
        # Try sampling-based parsing
        try:
            program = self.parser.parse(motion)
            if validate_program(program):
                return program, True
        except Exception as e:
            logger.debug(f"Sampling parse failed: {e}")
        
        # Fallback to greedy decoding (reuse same parser)
        if self.greedy_fallback:
            try:
                self._set_greedy_mode(True)
                program = self.parser.parse(motion)
                self._set_greedy_mode(False)
                if validate_program(program):
                    logger.debug("Greedy fallback succeeded")
                    return program, True
            except Exception as e:
                self._set_greedy_mode(False)
                logger.debug(f"Greedy parse failed: {e}")
        
        # Return last attempt even if invalid
        return program, False
    
    def parse_batch_safe(self, motions: list[np.ndarray]) -> list[tuple[str, bool]]:
        """Parse a batch with fallback to single parsing for failures.
        
        For maximum reliability, this does single parsing for all segments.
        This avoids the issue where a single exception can invalidate an
        entire batch in the original parser.
        """
        results = []
        for i, motion in enumerate(tqdm(motions, desc="Parsing", leave=False)):
            try:
                program, is_valid = self.parse_single(motion)
                results.append((program, is_valid))
            except Exception as e:
                logger.warning(f"Parse failed for segment {i}: {e}")
                results.append(("", False))
        return results
    
    def parse_batch_hybrid(self, motions: list[np.ndarray], batch_size: int = 8) -> list[tuple[str, bool]]:
        """Try batch parsing first, fall back to single parsing for failures.
        
        This is faster than parse_batch_safe but still catches individual failures.
        """
        results: list[tuple[str, bool]] = [("", False)] * len(motions)
        
        # Try batch parsing first
        try:
            batch_programs = self.parser.parse_batch(motions)
            for i, prog in enumerate(batch_programs):
                is_valid = validate_program(prog)
                results[i] = (prog, is_valid)
        except Exception as e:
            logger.warning(f"Batch parse failed: {e}, falling back to single parsing")
            # Fall back to single parsing for ALL
            return self.parse_batch_safe(motions)
        
        # Re-try failures individually with more attempts
        failed_indices = [i for i, (_, valid) in enumerate(results) if not valid]
        if failed_indices:
            logger.info(f"Re-trying {len(failed_indices)} failed segments individually...")
            for i in tqdm(failed_indices, desc="Re-parsing failures", leave=False):
                try:
                    program, is_valid = self.parse_single(motions[i])
                    if is_valid:
                        results[i] = (program, is_valid)
                except Exception as e:
                    logger.debug(f"Retry failed for segment {i}: {e}")
        
        return results


def main():
    args = parse_args()
    
    logger.info(f"Enhanced parsing with:")
    logger.info(f"  num_passes={args.num_passes}")
    logger.info(f"  num_retries={args.num_retries}")
    logger.info(f"  greedy_fallback={args.greedy_fallback}")
    logger.info(f"  fast={args.fast}")
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if args.esk_path.startswith("..") or args.esk_path.startswith("."):
        esk_path = (workspace_root / args.esk_path).resolve()
    else:
        esk_path = Path(args.esk_path).resolve()
    
    # Setup directories
    label_type_map = {
        "verbs": "D2A_converted_label_verbs",
        "nouns": "D2A_converted_label_nouns",
        "activity": "D2A_converted_label_activity",
        "actions": "D2A_converted_label_actions",
    }
    pose_dir = esk_path / "D2A_converted_pose_smpl"
    label_dir = esk_path / label_type_map[args.label_type]
    
    split_file = esk_path / "trainvaltest_split.txt"
    if not split_file.exists():
        split_file = esk_path / "traintest_split.txt"
    
    # Load split
    splits = load_split_file(str(split_file))
    
    if args.split == "train":
        videos = splits["train"]
    elif args.split == "val":
        videos = splits.get("val", [])
    else:
        videos = splits["test"]
    
    logger.info(f"Processing {len(videos)} videos from {args.split} split")
    
    # Create enhanced parser
    parser = EnhancedParser(
        checkpoint_path=args.parser_checkpoint,
        num_passes=args.num_passes,
        num_retries=args.num_retries,
        greedy_fallback=args.greedy_fallback,
    )
    
    # Load all segments
    logger.info("Loading segments...")
    all_segments_by_activity: dict[str, list[dict]] = {}
    
    for video_name in tqdm(videos, desc="Loading videos"):
        video_segments = load_segments_from_video(video_name, pose_dir, label_dir)
        for activity_name, segments in video_segments.items():
            if activity_name not in all_segments_by_activity:
                all_segments_by_activity[activity_name] = []
            all_segments_by_activity[activity_name].extend(segments)
    
    total_segments = sum(len(segs) for segs in all_segments_by_activity.values())
    logger.info(f"Loaded {total_segments} segments across {len(all_segments_by_activity)} activities")
    
    # Parse segments
    programs_by_activity: dict[str, list[dict]] = {}
    stats = {"total": 0, "valid": 0, "invalid": 0}
    
    for activity_name in tqdm(all_segments_by_activity.keys(), desc="Activities"):
        segments = all_segments_by_activity[activity_name]
        
        # Limit segments if requested
        if args.max_segments_per_activity and len(segments) > args.max_segments_per_activity:
            rng = random.Random(args.seed)
            segments = rng.sample(segments, args.max_segments_per_activity)
        
        programs_by_activity[activity_name] = []
        
        # Parse segments
        motions = [seg["poses"] for seg in segments]
        
        if args.fast:
            # Use batch parsing (faster but potentially lower yield)
            results = [(parser.parser.parse(m), True) for m in motions]
            results = [(p, validate_program(p)) for p, _ in results]
        else:
            # Use safe single-sample parsing
            results = parser.parse_batch_safe(motions)
        
        for seg, (program, is_valid) in zip(segments, results):
            stats["total"] += 1
            
            if is_valid:
                stats["valid"] += 1
                programs_by_activity[activity_name].append({
                    "program": program,
                    "video": seg["video"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["end"] - seg["start"],
                })
            else:
                stats["invalid"] += 1
                logger.debug(f"Invalid: {activity_name} [{seg['start']}-{seg['end']}]: {program[:50]}...")
    
    # Log results
    success_rate = stats["valid"] / max(1, stats["total"]) * 100
    logger.info(f"Parsing complete: {stats['valid']}/{stats['total']} valid ({success_rate:.1f}%)")
    
    for activity_name in all_segments_by_activity.keys():
        n_programs = len(programs_by_activity.get(activity_name, []))
        n_segments = len(all_segments_by_activity.get(activity_name, []))
        logger.info(f"  {activity_name}: {n_programs}/{n_segments} programs")
    
    # Build output
    output_data = {
        "metadata": {
            "parser_checkpoint": args.parser_checkpoint,
            "esk_data_path": str(esk_path),
            "split": args.split,
            "label_type": args.label_type,
            "num_videos": len(videos),
            "num_segments_total": total_segments,
            "num_programs_valid": stats["valid"],
            "success_rate": success_rate,
            "timestamp": datetime.now().isoformat(),
            "parsing_params": {
                "num_passes": args.num_passes,
                "num_retries": args.num_retries,
                "greedy_fallback": args.greedy_fallback,
                "fast_mode": args.fast,
            },
        },
        "activity_names": sorted(all_segments_by_activity.keys()),
        "programs_by_activity": programs_by_activity,
    }
    
    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = esk_path / f"programs_{args.label_type}_{args.split}_enhanced.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
