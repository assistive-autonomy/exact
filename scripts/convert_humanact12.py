"""Convert HumanAct12 dataset to DLC2Action format for action segmentation.

HumanAct12 contains 1191 short pre-segmented clips (shape: T, 24, 3) with 24 SMPL
joints in raw SMPL parameter ordering, across 12 subjects and 12 main action classes
(34 sub-classes). This script converts it into the DLC2Action-compatible format
used by the ESK pipeline:

  - Pose files: HDF5 with multi-level columns (scorer, individuals, bodyparts, coords)
  - Label files: Pickle tuples (metadata, label_names, individuals, annotations)
  - Split file:  Train/Validation/Test video lists

Clips are concatenated per-subject into long pseudo-videos suitable for temporal
action segmentation. Joint ordering is remapped from HumanAct12's raw SMPL parameter
order to the project's SMPL_KEYPOINTS layout (see exact.encoder.utils.SMPL_KEYPOINTS).

Usage:
    uv run python scripts/convert_humanact12.py [--src ../HumanAct12] [--dst ../humanact12_d2a]
"""

from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# SMPL_KEYPOINTS — canonical joint ordering (from exact/encoder/utils.py)
# ---------------------------------------------------------------------------
SMPL_KEYPOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# ---------------------------------------------------------------------------
# HumanAct12 raw order → SMPL_KEYPOINTS permutation
# ---------------------------------------------------------------------------
# HumanAct12 .npy files use the canonical SMPL parameter ordering:
#   0:Pelvis 1:L_Hip 2:R_Hip 3:Spine1 4:L_Knee 5:R_Knee 6:Spine2
#   7:L_Ankle 8:R_Ankle 9:Spine3 10:L_Foot 11:R_Foot 12:Neck
#   13:L_Collar 14:R_Collar 15:Head 16:L_Shoulder 17:R_Shoulder
#   18:L_Elbow 19:R_Elbow 20:L_Wrist 21:R_Wrist 22:L_Hand 23:R_Hand
#
# _HA12_TO_SMPL[raw_idx] = SMPL_KEYPOINTS index
_HA12_TO_SMPL: dict[int, int] = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3, 8: 7,
    9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19, 15: 13,
    16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22, 22: 18, 23: 23,
}

# Pre-compute permutation array for vectorised reordering
_PERM = np.empty(24, dtype=int)
for _raw, _out in _HA12_TO_SMPL.items():
    _PERM[_out] = _raw  # reordered[:, out, :] = original[:, raw, :]

# ---------------------------------------------------------------------------
# HumanAct12 action label mapping (action code first 2 digits -> class name)
# ---------------------------------------------------------------------------
ACTION_CLASSES: dict[str, str] = {
    "01": "warm_up",
    "02": "walk",
    "03": "run",
    "04": "jump",
    "05": "drink",
    "06": "lift_dumbbell",
    "07": "sit",
    "08": "eat",
    "09": "steer",
    "10": "call",
    "11": "box",
    "12": "throw",
}

LABEL_NAMES = sorted(set(ACTION_CLASSES.values()))  # alphabetical, 12 labels

# ---------------------------------------------------------------------------
# Train / Validation / Test split (by subject)
# ---------------------------------------------------------------------------
TRAIN_SUBJECTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]
VAL_SUBJECTS = ["P09"]
TEST_SUBJECTS = ["P10", "P11", "P12"]

SCORER = "HumanAct12"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_filename(fname: str) -> dict:
    """Parse HumanAct12 filename into components.

    Example: P10G01R02F0345T0464A0201.npy
    """
    name = Path(fname).stem
    parts: dict[str, str] = {}
    parts["subject"] = name[: name.index("G")]       # P10
    rest = name[name.index("G"):]
    parts["group"] = rest[: rest.index("R")]          # G01
    rest = rest[rest.index("R"):]
    parts["repetition"] = rest[: rest.index("F")]     # R02
    rest = rest[rest.index("F"):]
    parts["start_frame"] = rest[: rest.index("T")]    # F0345
    rest = rest[rest.index("T"):]
    parts["end_frame"] = rest[: rest.index("A")]      # T0464
    parts["action_code"] = rest[rest.index("A") + 1:] # 0201
    parts["main_action"] = parts["action_code"][:2]   # 02
    parts["filename"] = fname
    return parts


def reorder_joints(pose: np.ndarray) -> np.ndarray:
    """Reorder from HumanAct12 raw order to SMPL_KEYPOINTS order.

    Args:
        pose: (T, 24, 3) in HumanAct12 raw SMPL parameter ordering.

    Returns:
        (T, 24, 3) in SMPL_KEYPOINTS ordering.
    """
    return pose[:, _PERM, :]


def build_pose_dataframe(
    pose: np.ndarray,
    video_id: str,
) -> pd.DataFrame:
    """Build a DLC2Action-compatible multi-level-column DataFrame.

    Args:
        pose: (T, 24, 3) in SMPL_KEYPOINTS ordering.
        video_id: Identifier used as the individual name.

    Returns:
        DataFrame with shape (T, 96) and 4-level MultiIndex columns:
        (scorer, individuals, bodyparts, coords) where coords = [x, y, z, likelihood].
    """
    n_frames = pose.shape[0]
    n_joints = 24
    coords = ["x", "y", "z", "likelihood"]

    # Build flat data array: (T, 24*4)
    data = np.zeros((n_frames, n_joints * 4), dtype=np.float32)
    for j in range(n_joints):
        data[:, j * 4] = pose[:, j, 0]      # x
        data[:, j * 4 + 1] = pose[:, j, 1]  # y
        data[:, j * 4 + 2] = pose[:, j, 2]  # z
        data[:, j * 4 + 3] = 1.0            # likelihood (always confident)

    # Build multi-level column index
    tuples = []
    for bp in SMPL_KEYPOINTS:
        for c in coords:
            tuples.append((SCORER, video_id, bp, c))

    columns = pd.MultiIndex.from_tuples(
        tuples, names=["scorer", "individuals", "bodyparts", "coords"]
    )

    return pd.DataFrame(data, columns=columns)


def build_label_pickle(
    segments: list[tuple[int, int, str]],
    n_frames: int,
    video_id: str,
) -> tuple:
    """Build a DLC2Action-compatible label pickle tuple.

    Args:
        segments: List of (start_frame, end_frame, action_class_name).
        n_frames: Total number of frames in the video.
        video_id: Video identifier.

    Returns:
        Tuple (metadata, label_names_array, individuals, annotations)
        matching the ESK label format.
    """
    metadata = {
        "datetime": "2026-02-08",
        "annotator": "HumanAct12",
        "video_file": video_id,
    }

    label_names = np.array(LABEL_NAMES)
    individuals = [video_id]

    # Build per-label segment arrays: annotations[individual][label] -> (N, 3)
    label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    per_label: list[list[list[int]]] = [[] for _ in LABEL_NAMES]

    for start, end, action_name in segments:
        idx = label_to_idx[action_name]
        per_label[idx].append([start, end, 0])  # 0 = no confusion

    # Convert to numpy arrays
    label_arrays = []
    for segs in per_label:
        if segs:
            label_arrays.append(np.array(segs, dtype=np.int32))
        else:
            label_arrays.append(np.empty((0, 3), dtype=np.int32))

    annotations = [label_arrays]  # single individual per video

    return (metadata, label_names, individuals, annotations)


def convert_subject(
    subject: str,
    clip_infos: list[dict],
    data_dir: Path,
    gap_frames: int = 10,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Concatenate all clips for one subject into a pseudo-video.

    Clips are sorted by (group, repetition, start_frame) for a reproducible
    canonical ordering.  A small gap of ``gap_frames`` zero-velocity frames
    (holding the last pose of the previous clip) is inserted between consecutive
    clips to avoid artificial discontinuities at action boundaries.

    Returns:
        pose: (T_total, 24, 3) in SMPL_KEYPOINTS ordering.
        segments: list of (start, end, action_class_name) with frame indices
                  into the concatenated sequence (end is exclusive).
    """
    # Sort clips by group, repetition, then original start frame
    clip_infos = sorted(
        clip_infos,
        key=lambda c: (c["group"], c["repetition"], int(c["start_frame"][1:])),
    )

    all_poses: list[np.ndarray] = []
    segments: list[tuple[int, int, str]] = []
    cursor = 0

    for i, info in enumerate(clip_infos):
        raw = np.load(str(data_dir / info["filename"]))  # (T, 24, 3)
        reordered = reorder_joints(raw.astype(np.float32))
        T = reordered.shape[0]

        # Insert gap between clips (hold last pose of previous clip)
        if i > 0 and gap_frames > 0:
            last_pose = all_poses[-1][-1:]  # (1, 24, 3)
            gap = np.repeat(last_pose, gap_frames, axis=0)
            all_poses.append(gap)
            cursor += gap_frames

        action_name = ACTION_CLASSES[info["main_action"]]
        segments.append((cursor, cursor + T, action_name))
        all_poses.append(reordered)
        cursor += T

    pose = np.concatenate(all_poses, axis=0)
    assert pose.shape[0] == cursor
    return pose, segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert HumanAct12 → DLC2Action format")
    parser.add_argument("--src", type=str, default="../HumanAct12",
                        help="Path to HumanAct12 root (contains data/HumanAct12/)")
    parser.add_argument("--dst", type=str, default="../humanact12_d2a",
                        help="Output directory for converted dataset")
    parser.add_argument("--gap-frames", type=int, default=10,
                        help="Transition frames inserted between clips (hold last pose)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for any future stochastic steps)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    data_dir = src / "data" / "HumanAct12"

    if not data_dir.exists():
        raise FileNotFoundError(f"HumanAct12 data dir not found: {data_dir}")

    logger.info(f"Source: {src}")
    logger.info(f"Destination: {dst}")

    # Create output directories
    pose_dir = dst / "D2A_converted_pose_smpl"
    label_dir = dst / "D2A_converted_label_actions"
    pose_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Parse all filenames
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".npy")
    clip_infos = [parse_filename(f.name) for f in files]
    logger.info(f"Found {len(clip_infos)} clips across "
                f"{len({c['subject'] for c in clip_infos})} subjects")

    # Group by subject
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for info in clip_infos:
        by_subject[info["subject"]].append(info)

    # Convert each subject into a pseudo-video
    video_ids: list[str] = []
    for subject in sorted(by_subject.keys()):
        video_id = f"HA12_{subject}"
        video_ids.append(video_id)

        clips = by_subject[subject]
        logger.info(f"  {video_id}: {len(clips)} clips ...")

        pose, segments = convert_subject(subject, clips, data_dir, args.gap_frames)
        logger.info(f"    -> {pose.shape[0]} frames, {len(segments)} segments")

        # Save pose HDF5
        df = build_pose_dataframe(pose, video_id)
        pose_path = pose_dir / f"{video_id}_pose3d_smpl.h5"
        df.to_hdf(str(pose_path), key="df_with_missing", mode="w")
        logger.info(f"    Saved pose: {pose_path.name}")

        # Save label pickle
        label_tuple = build_label_pickle(segments, pose.shape[0], video_id)
        label_path = label_dir / f"{video_id}_labels.pickle"
        with open(label_path, "wb") as f:
            pickle.dump(label_tuple, f)
        logger.info(f"    Saved labels: {label_path.name}")

    # Write train/val/test split file
    splits = {"train": [], "validation": [], "test": []}
    for vid in video_ids:
        subject = vid.split("_")[1]  # HA12_P01 -> P01
        if subject in TRAIN_SUBJECTS:
            splits["train"].append(vid)
        elif subject in VAL_SUBJECTS:
            splits["validation"].append(vid)
        elif subject in TEST_SUBJECTS:
            splits["test"].append(vid)

    split_path = dst / "trainvaltest_split.txt"
    with open(split_path, "w") as f:
        f.write("Train videos:\n")
        for v in splits["train"]:
            f.write(f"{v}\n")
        f.write("Validation videos:\n")
        for v in splits["validation"]:
            f.write(f"{v}\n")
        f.write("Test videos:\n")
        for v in splits["test"]:
            f.write(f"{v}\n")
    logger.info(f"Split file: {split_path}")
    logger.info(f"  Train: {len(splits['train'])} videos ({TRAIN_SUBJECTS})")
    logger.info(f"  Val:   {len(splits['validation'])} videos ({VAL_SUBJECTS})")
    logger.info(f"  Test:  {len(splits['test'])} videos ({TEST_SUBJECTS})")

    # Print label statistics
    logger.info(f"\nAction classes ({len(LABEL_NAMES)}):")
    for name in LABEL_NAMES:
        count = sum(
            1 for info in clip_infos
            if ACTION_CLASSES[info["main_action"]] == name
        )
        logger.info(f"  {name}: {count} clips")

    logger.success(f"Conversion complete! Output at {dst}")
    logger.info(f"\nTo use with segmentation.py, set in your config:")
    logger.info(f"  data_path: {pose_dir}")
    logger.info(f"  annotation_path: {label_dir}")
    logger.info(f"  split_path: {split_path}")


if __name__ == "__main__":
    main()
