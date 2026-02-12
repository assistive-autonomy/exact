import ast
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# SMPL keypoint names (24 keypoints) — canonical ordering used everywhere
SMPL_KEYPOINTS = [
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

# ---------------------------------------------------------------------------
# Raw SMPL fitting order → SMPL_KEYPOINTS permutation
#
# The raw CSV from SMPL body fitting uses the canonical SMPL parameter order:
#   0:Pelvis 1:L_Hip 2:R_Hip 3:Spine1 4:L_Knee 5:R_Knee 6:Spine2
#   7:L_Ankle 8:R_Ankle 9:Spine3 10:L_Foot 11:R_Foot 12:Neck
#   13:L_Collar 14:R_Collar 15:Head 16:L_Shoulder 17:R_Shoulder
#   18:L_Elbow 19:R_Elbow 20:L_Wrist 21:R_Wrist 22:L_Hand 23:R_Hand
#
# _RAW_TO_SMPL_PERM[j] gives the raw index whose data belongs at
# SMPL_KEYPOINTS[j].
# ---------------------------------------------------------------------------
_RAW_TO_SMPL = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3, 8: 7,
    9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19, 15: 13,
    16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22, 22: 18, 23: 23,
}
_RAW_TO_SMPL_PERM = np.empty(24, dtype=int)
for _raw, _out in _RAW_TO_SMPL.items():
    _RAW_TO_SMPL_PERM[_out] = _raw


def extract_episode_info(file_path: Path) -> tuple[str, str, str] | None:
    """
    Extract split, subject, and datetime from file path.

    Path format: annotations_dir/split/subject/datetime/...

    Returns:
        Tuple of (split, subject, datetime) or None if extraction fails
    """
    parts = file_path.parts

    # Find split folder (train or test)
    split_name = None
    split_idx = None
    for i, part in enumerate(parts):
        if part in ["train", "test"]:
            split_name = part
            split_idx = i
            break

    if split_idx is None or split_idx + 2 >= len(parts):
        return None

    subject = parts[split_idx + 1]
    datetime = parts[split_idx + 2]

    return split_name, subject, datetime


def convert_pose_file(pose_file: Path, output_path: Path) -> bool:
    """
    Convert a single pose3d_smpl.csv file to HDF5 format.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract episode info from path
        episode_info = extract_episode_info(pose_file)
        if episode_info is None:
            logger.warning(f"Could not determine subject/datetime for: {pose_file}")
            return False

        split_name, subject, datetime_formatted = episode_info
        # Individual ID should be the full video ID for proper annotation matching
        individual_id = f"{subject}_{datetime_formatted}"

        # Read pose data from CSV
        pose_df = pd.read_csv(pose_file)
        poses_list = pose_df["poses"].apply(ast.literal_eval).tolist()
        pose_array = np.array(poses_list, dtype=np.float32)

        n_frames, n_params = pose_array.shape
        n_keypoints = len(SMPL_KEYPOINTS)
        expected_size = n_keypoints * 3  # 24 * 3 = 72

        # Pad or trim to expected size
        if n_params < expected_size:
            pose_padded = np.pad(
                pose_array, ((0, 0), (0, expected_size - n_params)), mode="constant"
            )
        else:
            pose_padded = pose_array[:, :expected_size]

        # Reshape to (frames, keypoints, 3) — still in raw SMPL fitting order
        pose_array = pose_padded.reshape(n_frames, n_keypoints, 3)

        # Reorder from raw SMPL fitting order to SMPL_KEYPOINTS order.
        # Raw CSV:        Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, ...
        # SMPL_KEYPOINTS: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, ...
        pose_array = pose_array[:, _RAW_TO_SMPL_PERM, :]

        # Create likelihood column based on coordinate variance
        # Padded coordinates (0, 0, 0) get low likelihood, real coordinates get high likelihood
        # This adds variance needed for DLC2Action's feature importance calculations
        likelihood = np.ones((n_frames, n_keypoints, 1), dtype=np.float32)
        for kp in range(n_keypoints):
            # If all coords are 0 (padded), likelihood is low; otherwise high
            is_zero = (
                (pose_array[:, kp, 0] == 0)
                & (pose_array[:, kp, 1] == 0)
                & (pose_array[:, kp, 2] == 0)
            )
            likelihood[is_zero, kp, 0] = 0.1

        pose_with_likelihood = np.concatenate([pose_array, likelihood], axis=-1)

        # Reshape to (frames, keypoints * 4) for DataFrame
        pose_reshaped = pose_with_likelihood.reshape(n_frames, -1)

        # Create MultiIndex columns for DLC2Action format
        individuals = [individual_id]
        columnindex = pd.MultiIndex.from_product(
            [["ESK"], individuals, SMPL_KEYPOINTS, ["x", "y", "z", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

        df_pose = pd.DataFrame(pose_reshaped, columns=columnindex)

        # Save as HDF5
        pose_suffix = "_pose3d_smpl.h5"
        pose_filename = f"{individual_id}{pose_suffix}"
        pose_h5_file = output_path / "D2A_converted_pose_smpl" / pose_filename
        df_pose.to_hdf(pose_h5_file, key="tracks", format="table", mode="w")

        logger.info(f"✓ Pose converted: {subject}/{datetime_formatted}")
        logger.info(f"  → Saved: D2A_converted_pose_smpl/{pose_filename}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to convert pose: {pose_file}")
        logger.error(f"  Error: {str(e)}")
        return False


def main(
    annotations_dir: str = "/pvc/esk_annotations", output_dir: str = "/pvc/esk"
) -> None:
    """
    Convert ESK pose data for DLC2Action.

    Creates esk_30/D2A_converted_pose_smpl/ folder with HDF5 pose files.

    Args:
        annotations_dir: Path to the annotations directory
        output_dir: Path to the output directory
    """
    annotations_path = Path(annotations_dir)
    output_path = Path(output_dir)

    if not annotations_path.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return

    # Create output directory structure
    (output_path / "D2A_converted_pose_smpl").mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Created DLC poses directory at {output_path}/D2A_converted_pose_smpl\n"
    )

    # Process pose files
    pose_files = sorted(list(annotations_path.rglob("pose3d_smpl.csv")))
    logger.info(f"Found {len(pose_files)} pose files to convert")
    poses_converted = sum(convert_pose_file(f, output_path) for f in pose_files)
    poses_failed = len(pose_files) - poses_converted

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Conversion complete!")
    logger.info(f"  Pose files: {poses_converted} converted, {poses_failed} failed")
    logger.info(f"  Output directory: {output_path}/D2A_converted_pose_smpl")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
