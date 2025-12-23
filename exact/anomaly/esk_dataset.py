"""
ESK Dataset loader for anomaly detection.

Supports:
- SMPL pose data (24 keypoints)
- Activity, verb, and noun labels
- Activity-level anomaly detection: train on one activity, evaluate on all
"""

import pickle
from pathlib import Path
from typing import Optional, Literal, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


LabelType = Literal["activity", "verbs", "nouns"]


# SMPL keypoints for reference
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


def load_train_test_split(split_file: Path) -> tuple[list[str], list[str]]:
    """Load train/test video IDs from split file."""
    train_videos = []
    test_videos = []
    current_section = None

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Train"):
                current_section = "train"
            elif line.startswith("Test"):
                current_section = "test"
            elif line and current_section:
                if current_section == "train":
                    train_videos.append(line)
                else:
                    test_videos.append(line)

    return train_videos, test_videos


def load_pose_file(pose_path: Path) -> tuple[np.ndarray, str]:
    """
    Load SMPL pose data from HDF5 file.

    Returns:
        pose_data: Array of shape (T, V, C) where V=24 keypoints, C=3 (x,y,z)
        individual_id: Video/person identifier
    """
    df = pd.read_hdf(pose_path)

    # Get individual ID from columns
    individual_id = df.columns.get_level_values("individuals").unique()[0]

    # Reshape: (T, V*4) -> (T, V, 4) -> (T, V, 3) (drop likelihood)
    n_frames = len(df)
    n_keypoints = len(SMPL_KEYPOINTS)

    # Extract x, y, z coordinates (skip likelihood)
    pose_data = np.zeros((n_frames, n_keypoints, 3), dtype=np.float32)
    for i, kp in enumerate(SMPL_KEYPOINTS):
        pose_data[:, i, 0] = (
            df.iloc[:, df.columns.get_level_values("bodyparts") == kp].iloc[:, 0].values
        )  # x
        pose_data[:, i, 1] = (
            df.iloc[:, df.columns.get_level_values("bodyparts") == kp].iloc[:, 1].values
        )  # y
        pose_data[:, i, 2] = (
            df.iloc[:, df.columns.get_level_values("bodyparts") == kp].iloc[:, 2].values
        )  # z

    return pose_data, individual_id


def load_labels(label_path: Path) -> tuple[list[str], list[str], list]:
    """
    Load activity labels from pickle file.

    Returns:
        label_names: List of label class names
        individuals: List of individual IDs
        annotations: List of time segments per individual per label
    """
    with open(label_path, "rb") as f:
        data = pickle.load(f)

    # (info_dict, label_names, individuals, annotations)
    label_names = list(data[1])
    individuals = list(data[2])
    annotations = data[3]  # [individuals][labels] -> list of [start, end, confusion]

    return label_names, individuals, annotations


def annotations_to_frame_labels(
    annotations: list,
    n_frames: int,
    label_names: list[str],
) -> np.ndarray:
    """
    Convert segment annotations to per-frame labels.

    Returns:
        frame_labels: Array of shape (n_frames, n_labels) with binary labels
    """
    n_labels = len(label_names)
    frame_labels = np.zeros((n_frames, n_labels), dtype=np.float32)

    # annotations[individual_idx][label_idx] = list of [start, end, confusion]
    if len(annotations) > 0:
        individual_annot = annotations[0]  # Single individual per file
        for label_idx, segments in enumerate(individual_annot):
            if isinstance(segments, np.ndarray) and len(segments) > 0:
                for segment in segments:
                    start, end = int(segment[0]), int(segment[1])
                    if start < n_frames and end <= n_frames:
                        frame_labels[start:end, label_idx] = 1.0

    return frame_labels


def get_dominant_activity(segment_labels: np.ndarray, label_names: list[str]) -> str:
    """
    Get the dominant activity for a segment.

    Args:
        segment_labels: Array of shape (n_labels,) with mean activity presence
        label_names: List of activity names

    Returns:
        Name of the dominant activity, or "none" if no activity
    """
    if segment_labels.max() < 0.5:
        return "none"
    return label_names[segment_labels.argmax()]


class ESKPoseDataset(Dataset):
    """
    ESK Pose Dataset for activity-based anomaly detection.

    For training: loads only segments where the target activity is present (>50% of frames)
    For testing: loads ALL segments and tracks which activity each belongs to

    Args:
        esk_dir: Path to ESK dataset directory
        split: "train" or "test"
        label_type: Type of labels ("activity", "verbs", "nouns")
        seg_len: Length of pose segments (temporal window)
        seg_stride: Stride between segments
        normalize: Whether to normalize pose coordinates
        target_activity: Activity to train on (required for training)
        test_activities: List of activities to include in test (None = all)
    """

    def __init__(
        self,
        esk_dir: Union[str, Path],
        split: Literal["train", "test"] = "train",
        label_type: LabelType = "activity",
        seg_len: int = 24,
        seg_stride: int = 6,
        normalize: bool = True,
        target_activity: Optional[str] = None,
        test_activities: Optional[list[str]] = None,
    ):
        self.esk_dir = Path(esk_dir)
        self.split = split
        self.label_type = label_type
        self.seg_len = seg_len
        self.seg_stride = seg_stride if split == "train" else 1
        self.normalize = normalize
        self.target_activity = target_activity
        self.test_activities = test_activities

        # Paths
        self.pose_dir = self.esk_dir / "D2A_converted_pose_smpl"
        self.label_dir = self.esk_dir / f"D2A_converted_label_{label_type}"
        self.split_file = self.esk_dir / "traintest_split.txt"

        # Load train/test split
        train_videos, test_videos = load_train_test_split(self.split_file)
        self.video_ids = train_videos if split == "train" else test_videos

        # Get label names from first available file
        self.label_names = self._get_label_names()

        # Load data
        self.segments, self.labels, self.activity_names, self.metadata = (
            self._load_data()
        )

    def _get_label_names(self) -> list[str]:
        """Get consistent label names from first available label file."""
        for video_id in self.video_ids:
            label_path = self.label_dir / f"{video_id}_labels.pickle"
            if label_path.exists():
                label_names, _, _ = load_labels(label_path)
                return label_names
        return []

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
        """
        Load and segment all pose data.

        Returns:
            segments: Pose segments (N, T, V, C)
            labels: Activity label vectors (N, n_labels)
            activity_names: Dominant activity name for each segment
            metadata: List of segment metadata dicts
        """
        all_segments = []
        all_labels = []
        all_activity_names = []
        all_metadata = []

        n_labels = len(self.label_names) if self.label_names else 1

        for video_id in self.video_ids:
            # Find pose file
            pose_files = list(self.pose_dir.glob(f"{video_id}*.h5"))
            if not pose_files:
                continue
            pose_path = pose_files[0]

            # Load pose
            try:
                pose_data, individual_id = load_pose_file(pose_path)
            except Exception as e:
                print(f"Error loading {pose_path}: {e}")
                continue

            # Load labels
            label_path = self.label_dir / f"{video_id}_labels.pickle"
            if label_path.exists():
                label_names, _, annotations = load_labels(label_path)
                frame_labels = annotations_to_frame_labels(
                    annotations, len(pose_data), label_names
                )
                # Ensure consistent dimensions
                if frame_labels.shape[1] != n_labels:
                    new_labels = np.zeros((len(pose_data), n_labels), dtype=np.float32)
                    min_labels = min(frame_labels.shape[1], n_labels)
                    new_labels[:, :min_labels] = frame_labels[:, :min_labels]
                    frame_labels = new_labels
            else:
                frame_labels = np.zeros((len(pose_data), n_labels), dtype=np.float32)
                label_names = self.label_names if self.label_names else ["unknown"]

            # Normalize if requested
            if self.normalize:
                pose_data = self._normalize_pose(pose_data)

            # Segment the data
            n_frames = len(pose_data)
            for start in range(0, n_frames - self.seg_len + 1, self.seg_stride):
                end = start + self.seg_len

                segment = pose_data[start:end]
                segment_labels = frame_labels[start:end].mean(
                    axis=0
                )  # Average over time

                # Get dominant activity for this segment
                dominant_activity = get_dominant_activity(
                    segment_labels, self.label_names
                )

                # Training: only keep segments from target activity
                if self.split == "train":
                    if self.target_activity is None:
                        raise ValueError(
                            "target_activity must be specified for training"
                        )
                    if dominant_activity != self.target_activity:
                        continue

                # Testing: optionally filter by test_activities
                if self.split == "test" and self.test_activities is not None:
                    if dominant_activity not in self.test_activities:
                        continue

                all_segments.append(segment)
                all_labels.append(segment_labels)
                all_activity_names.append(dominant_activity)
                all_metadata.append(
                    {
                        "video_id": video_id,
                        "individual_id": individual_id,
                        "start_frame": start,
                        "end_frame": end,
                        "dominant_activity": dominant_activity,
                    }
                )

        if len(all_segments) == 0:
            return np.zeros((0, self.seg_len, 24, 3)), np.zeros((0, n_labels)), [], []

        return (
            np.stack(all_segments, axis=0),
            np.stack(all_labels, axis=0),
            all_activity_names,
            all_metadata,
        )

    def _normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        """Normalize pose: center on pelvis and scale to unit cube."""
        pelvis = pose[:, 0:1, :]
        pose_centered = pose - pelvis
        max_val = np.abs(pose_centered).max() + 1e-6
        return (pose_centered / max_val).astype(np.float32)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            pose: Tensor of shape (C, T, V) for STG-NF input
            label: Tensor of shape (n_labels,) with activity presence
            activity: Dominant activity name for this segment
        """
        segment = self.segments[idx]  # (T, V, C)
        label = self.labels[idx]
        activity = self.activity_names[idx]

        # Transpose to (C, T, V) for model input
        pose_tensor = torch.from_numpy(segment).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label).float()

        return pose_tensor, label_tensor, activity


def collate_fn(batch):
    """Custom collate function that handles activity names."""
    poses = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    activities = [item[2] for item in batch]
    return poses, labels, activities


def get_esk_dataloaders(
    esk_dir: Union[str, Path],
    label_type: LabelType = "activity",
    target_activity: str = "cooking",
    test_activities: Optional[list[str]] = None,
    seg_len: int = 24,
    seg_stride: int = 6,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train and test data loaders for ESK activity-based anomaly detection.

    Training data contains ONLY segments from the target activity.
    Test data contains segments from all activities (or specified test_activities).

    Args:
        esk_dir: Path to ESK dataset
        label_type: Type of labels (activity, verbs, nouns)
        target_activity: Activity to use for training (normal class)
        test_activities: Activities to include in test (None = all)
        seg_len: Segment length
        seg_stride: Segment stride for training
        batch_size: Batch size
        num_workers: Data loader workers

    Returns:
        train_loader: Training data loader (target activity only)
        test_loader: Test data loader (all or specified activities)
        info: Dataset information dict
    """
    train_dataset = ESKPoseDataset(
        esk_dir=esk_dir,
        split="train",
        label_type=label_type,
        seg_len=seg_len,
        seg_stride=seg_stride,
        target_activity=target_activity,
    )

    test_dataset = ESKPoseDataset(
        esk_dir=esk_dir,
        split="test",
        label_type=label_type,
        seg_len=seg_len,
        seg_stride=1,  # Dense sampling for test
        target_activity=target_activity,
        test_activities=test_activities,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Count samples per activity in test set
    activity_counts = {}
    for act in test_dataset.activity_names:
        activity_counts[act] = activity_counts.get(act, 0) + 1

    info = {
        "label_names": train_dataset.label_names,
        "target_activity": target_activity,
        "n_train_samples": len(train_dataset),
        "n_test_samples": len(test_dataset),
        "test_activity_counts": activity_counts,
        "n_keypoints": 24,
        "seg_len": seg_len,
    }

    return train_loader, test_loader, info


def get_unique_activities(
    esk_dir: Union[str, Path], label_type: LabelType = "activity"
) -> list[str]:
    """Get list of unique activity labels."""
    label_dir = Path(esk_dir) / f"D2A_converted_label_{label_type}"
    label_files = list(label_dir.glob("*_labels.pickle"))

    if label_files:
        label_names, _, _ = load_labels(label_files[0])
        return label_names
    return []
