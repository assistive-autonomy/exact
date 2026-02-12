#!/usr/bin/env python
"""Preprocess ESK data: convert SMPL axis-angle rotations to 3D joint positions.

ESK HDF5 files store SMPL axis-angle rotation parameters (24 joints × 3) in
the x/y/z coordinate columns.  This script runs MuJoCo forward kinematics on
the HumEnv SMPL humanoid to produce root-relative 3D joint positions and
rewrites the files in the same DLC2Action-compatible format.

The converted files are **fully compatible** with:
  - The parser pipeline (parse_esk.py), which expects (T, 72) positions.
  - The segmentation pipeline (DLC2Action), which reads the 96-dim HDF5.
  - Augmented data from augment_data.py (also position-based).

Usage:
    # Convert all ESK pose files in-place (backs up originals)
    uv run python scripts/preprocess_esk.py --esk-path ../esk

    # Convert to a new output directory (non-destructive)
    uv run python scripts/preprocess_esk.py --esk-path ../esk --output-dir ../esk_positions

    # Dry run — report what would be converted
    uv run python scripts/preprocess_esk.py --esk-path ../esk --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data.esk import load_pose_file
from exact.data.env import smpl_rotations_to_positions
from exact.encoder import SMPL_KEYPOINTS


def convert_pose_file(
    input_path: Path,
    output_path: Path,
    standing_height: float = 0.94,
) -> bool:
    """Convert a single ESK pose file from rotations to positions.

    Args:
        input_path: Path to input HDF5 file (axis-angle rotations).
        output_path: Path for output HDF5 file (3D positions).
        standing_height: Root z-position for FK.

    Returns:
        True if conversion succeeded.
    """
    try:
        # Load axis-angle rotations (T, 24, 3)
        pose_data, individual_id = load_pose_file(input_path)
        n_frames = pose_data.shape[0]

        # Convert to root-relative 3D positions (T, 72)
        positions = smpl_rotations_to_positions(pose_data, standing_height)
        positions_3d = positions.reshape(n_frames, 24, 3)

        # Add likelihood column (all 1.0)
        likelihood = np.ones((n_frames, 24, 1), dtype=np.float32)
        pose_with_likelihood = np.concatenate([positions_3d, likelihood], axis=-1)
        pose_reshaped = pose_with_likelihood.reshape(n_frames, -1)  # (T, 96)

        # Rebuild DLC2Action-compatible DataFrame
        scorer = "ESK"
        columnindex = pd.MultiIndex.from_product(
            [[scorer], [individual_id], SMPL_KEYPOINTS, ["x", "y", "z", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df_pose = pd.DataFrame(pose_reshaped, columns=columnindex)

        # Write HDF5
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pose.to_hdf(str(output_path), key="tracks", format="table", mode="w")
        return True

    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ESK axis-angle rotations to 3D joint positions"
    )
    parser.add_argument(
        "--esk-path",
        type=str,
        required=True,
        help="Path to ESK dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite in-place with backup)",
    )
    parser.add_argument(
        "--pose-dir-name",
        type=str,
        default="D2A_converted_pose_smpl",
        help="Name of the pose subdirectory",
    )
    parser.add_argument(
        "--standing-height",
        type=float,
        default=0.94,
        help="Standing height for FK (default: 0.94m)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of original files when overwriting (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Skip backup when overwriting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without writing",
    )
    args = parser.parse_args()

    esk_path = Path(args.esk_path).resolve()
    pose_dir = esk_path / args.pose_dir_name

    if not pose_dir.exists():
        logger.error(f"Pose directory not found: {pose_dir}")
        return 1

    pose_files = sorted(pose_dir.glob("*.h5"))
    logger.info(f"Found {len(pose_files)} pose files in {pose_dir}")

    if args.dry_run:
        for f in pose_files:
            logger.info(f"  Would convert: {f.name}")
        logger.info(f"Dry run complete. {len(pose_files)} files would be converted.")
        return 0

    # Determine output directory
    if args.output_dir:
        out_pose_dir = Path(args.output_dir).resolve() / args.pose_dir_name
    else:
        out_pose_dir = pose_dir  # in-place

    # Backup originals if overwriting
    if out_pose_dir == pose_dir and args.backup:
        backup_dir = esk_path / f"{args.pose_dir_name}_rotations_backup"
        if not backup_dir.exists():
            logger.info(f"Backing up originals to {backup_dir}")
            shutil.copytree(pose_dir, backup_dir)
        else:
            logger.info(f"Backup already exists at {backup_dir}, skipping")

    converted = 0
    failed = 0

    for pose_file in tqdm(pose_files, desc="Converting"):
        output_file = out_pose_dir / pose_file.name
        if convert_pose_file(pose_file, output_file, args.standing_height):
            converted += 1
        else:
            failed += 1

    logger.info(f"Conversion complete: {converted} succeeded, {failed} failed")
    if args.output_dir:
        logger.info(f"Output: {out_pose_dir}")

        # Copy label files and split file to output directory if needed
        for subdir in esk_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("D2A_converted_label"):
                dest = Path(args.output_dir).resolve() / subdir.name
                if not dest.exists():
                    shutil.copytree(subdir, dest)
                    logger.info(f"Copied {subdir.name} to output directory")

        split_file = esk_path / "traintest_split.txt"
        if split_file.exists():
            dest = Path(args.output_dir).resolve() / split_file.name
            if not dest.exists():
                shutil.copy2(split_file, dest)
                logger.info("Copied traintest_split.txt to output directory")

    return 0


if __name__ == "__main__":
    sys.exit(main())
