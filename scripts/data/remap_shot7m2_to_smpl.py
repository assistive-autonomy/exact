"""
Remap shot7m2 3D poses (26 joints) to SMPL skeleton format (24 joints).

shot7m2 skeleton (26 joints):
  center, l_hip, l_knee, l_ankle, l_foot, l_toes, r_hip, r_knee, r_ankle,
  r_foot, r_toes, lumbars, low_thorax, high_thorax, cervicals,
  l_shoulder_blade, l_shoulder, l_elbow, l_wrist, neck, head, head_top,
  r_shoulder_blade, r_shoulder, r_elbow, r_wrist

SMPL skeleton (24 joints, from exact/encoder/utils.py):
  Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe,
  Torso, Spine, Chest, Neck, Head, L_Thorax, L_Shoulder, L_Elbow, L_Wrist,
  L_Hand, R_Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand

Mapping strategy:
  - 22 joints have direct or near-direct correspondences
  - L_Hand / R_Hand are missing in shot7m2 and are extrapolated from
    the elbow→wrist vector direction
  - Dropped shot7m2 joints: l_foot, r_foot, cervicals, head_top
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Joint mapping: SMPL joint name → shot7m2 source joint name ──────────────
# For joints that exist in shot7m2 under a different name.
# L_Hand and R_Hand are handled separately (extrapolated).
SMPL_TO_SHOT7M2 = {
    "Pelvis": "center",
    "L_Hip": "l_hip",
    "L_Knee": "l_knee",
    "L_Ankle": "l_ankle",
    "L_Toe": "l_toes",
    "R_Hip": "r_hip",
    "R_Knee": "r_knee",
    "R_Ankle": "r_ankle",
    "R_Toe": "r_toes",
    "Torso": "lumbars",
    "Spine": "low_thorax",
    "Chest": "high_thorax",
    "Neck": "neck",
    "Head": "head",
    "L_Thorax": "l_shoulder_blade",
    "L_Shoulder": "l_shoulder",
    "L_Elbow": "l_elbow",
    "L_Wrist": "l_wrist",
    # L_Hand → extrapolated
    "R_Thorax": "r_shoulder_blade",
    "R_Shoulder": "r_shoulder",
    "R_Elbow": "r_elbow",
    "R_Wrist": "r_wrist",
    # R_Hand → extrapolated
}

# Canonical SMPL joint order (must match exact/encoder/utils.py)
SMPL_JOINT_ORDER = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

COORDS = ["x", "y", "z", "likelihood"]

# Fraction of elbow→wrist length used to extrapolate hand position
HAND_EXTRAPOLATION_FACTOR = 0.25


def extrapolate_hand(
    elbow: np.ndarray, wrist: np.ndarray, factor: float = HAND_EXTRAPOLATION_FACTOR
) -> np.ndarray:
    """
    Extrapolate hand position beyond the wrist along the elbow→wrist direction.

    Parameters
    ----------
    elbow : array of shape (N, 3)
    wrist : array of shape (N, 3)
    factor : how far beyond the wrist to place the hand, as a fraction
             of the elbow→wrist segment length.

    Returns
    -------
    hand : array of shape (N, 3)
    """
    direction = wrist - elbow
    return wrist + factor * direction


def remap_file(src_path: str, dst_path: str, scorer: str = "SHOT7M2_SMPL") -> None:
    """Convert a single shot7m2 H5 pose file to SMPL format."""
    df = pd.read_hdf(src_path)

    # Extract the individual name from the source file
    individual = df.columns.get_level_values("individuals").unique()[0]
    src_bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()

    # Build xyz arrays for convenience
    def get_xyz(bodypart: str) -> np.ndarray:
        cols = [(df.columns.get_level_values("bodyparts") == bodypart) &
                (df.columns.get_level_values("coords").isin(["x", "y", "z"]))]
        return df.loc[:, cols[0]].values  # shape (T, 3)

    def get_likelihood(bodypart: str) -> np.ndarray:
        cols = [(df.columns.get_level_values("bodyparts") == bodypart) &
                (df.columns.get_level_values("coords") == "likelihood")]
        return df.loc[:, cols[0]].values.ravel()  # shape (T,)

    T = len(df)
    result = np.zeros((T, 24 * 4), dtype=np.float64)

    for joint_idx, smpl_joint in enumerate(SMPL_JOINT_ORDER):
        col_start = joint_idx * 4

        if smpl_joint in SMPL_TO_SHOT7M2:
            src_joint = SMPL_TO_SHOT7M2[smpl_joint]
            xyz = get_xyz(src_joint)
            lh = get_likelihood(src_joint)
            result[:, col_start:col_start + 3] = xyz
            result[:, col_start + 3] = lh

        elif smpl_joint == "L_Hand":
            elbow_xyz = get_xyz("l_elbow")
            wrist_xyz = get_xyz("l_wrist")
            hand_xyz = extrapolate_hand(elbow_xyz, wrist_xyz)
            result[:, col_start:col_start + 3] = hand_xyz
            result[:, col_start + 3] = get_likelihood("l_wrist")

        elif smpl_joint == "R_Hand":
            elbow_xyz = get_xyz("r_elbow")
            wrist_xyz = get_xyz("r_wrist")
            hand_xyz = extrapolate_hand(elbow_xyz, wrist_xyz)
            result[:, col_start:col_start + 3] = hand_xyz
            result[:, col_start + 3] = get_likelihood("r_wrist")

    # Build multi-index columns matching the ESK/SMPL convention
    tuples = []
    for joint in SMPL_JOINT_ORDER:
        for coord in COORDS:
            tuples.append((scorer, individual, joint, coord))

    columns = pd.MultiIndex.from_tuples(
        tuples, names=["scorer", "individuals", "bodyparts", "coords"]
    )

    out_df = pd.DataFrame(result, index=df.index, columns=columns)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    out_df.to_hdf(dst_path, key="tracks", mode="w")


def main():
    parser = argparse.ArgumentParser(
        description="Remap shot7m2 3D poses to SMPL 24-joint skeleton format."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="../shot7m2/poses",
        help="Source directory with shot7m2 *_3D_poses.h5 files.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="../shot7m2/D2A_converted_pose_smpl",
        help="Destination directory for SMPL-format H5 files.",
    )
    parser.add_argument(
        "--suffix_in",
        type=str,
        default="_3D_poses.h5",
        help="Input file suffix to match.",
    )
    parser.add_argument(
        "--suffix_out",
        type=str,
        default="_pose3d_smpl.h5",
        help="Output file suffix.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="SHOT7M2_SMPL",
        help="Scorer name to use in the output multi-index.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    src_files = sorted(src_dir.glob(f"*{args.suffix_in}"))

    if not src_files:
        print(f"No files matching *{args.suffix_in} found in {src_dir}")
        sys.exit(1)

    print(f"Remapping {len(src_files)} files from {src_dir} → {dst_dir}")
    print(f"  Source skeleton:  26 joints (shot7m2)")
    print(f"  Target skeleton:  24 joints (SMPL)")
    print(f"  Input suffix:     {args.suffix_in}")
    print(f"  Output suffix:    {args.suffix_out}")
    print()

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_file in tqdm(src_files, desc="Remapping"):
        stem = src_file.name.replace(args.suffix_in, "")
        dst_file = dst_dir / f"{stem}{args.suffix_out}"
        remap_file(str(src_file), str(dst_file), scorer=args.scorer)

    # ── Validation ──────────────────────────────────────────────────────────
    print("\n✓ Remapping complete. Running validation on first output file...")
    first_out = sorted(dst_dir.glob(f"*{args.suffix_out}"))[0]
    df = pd.read_hdf(str(first_out))

    out_bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    assert len(out_bodyparts) == 24, f"Expected 24 joints, got {len(out_bodyparts)}"
    assert out_bodyparts == SMPL_JOINT_ORDER, (
        f"Joint order mismatch!\n  Expected: {SMPL_JOINT_ORDER}\n  Got: {out_bodyparts}"
    )
    assert df.shape[1] == 96, f"Expected 96 columns (24×4), got {df.shape[1]}"

    print(f"  Shape:      {df.shape}")
    print(f"  Joints:     {len(out_bodyparts)} (SMPL ✓)")
    print(f"  Joint order matches exact/encoder/utils.py ✓")
    print(f"  Columns:    {df.shape[1]} (24 joints × 4 coords ✓)")


if __name__ == "__main__":
    main()
