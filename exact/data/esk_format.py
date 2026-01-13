"""ESK dataset format utilities.

This module provides utilities for reading and writing data in the ESK dataset
format used by DLC2Action for activity segmentation.

Data Format:
- Pose files: HDF5 with 'tracks/table' containing (index, values_block_0[96])
- Label files: Pickle tuple of (metadata_dict, activity_names, video_names, segments)
  where segments[video_idx][activity_idx] is array of [start, end, flag]
"""

import pickle
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def load_pose_data(pose_path: str) -> np.ndarray:
    """Load pose data from ESK HDF5 file.
    
    Args:
        pose_path: Path to pose HDF5 file
        
    Returns:
        Pose array of shape [num_frames, 96]
    """
    with h5py.File(pose_path, "r") as f:
        table = f["tracks/table"][:]
        poses = table["values_block_0"]
    return poses


def load_labels(label_path: str) -> dict:
    """Load labels from ESK pickle file.
    
    Args:
        label_path: Path to label pickle file
        
    Returns:
        Dictionary with:
            - metadata: dict
            - activity_names: list of activity names
            - video_names: list of video names
            - segments: list[list[np.ndarray]] - segments[video][activity]
    """
    with open(label_path, "rb") as f:
        data = pickle.load(f)
    
    metadata, activity_names, video_names, segments = data
    
    return {
        "metadata": metadata,
        "activity_names": list(activity_names),
        "video_names": video_names,
        "segments": segments,
    }


def save_pose_data(
    pose_path: str,
    poses: np.ndarray,
) -> None:
    """Save pose data to ESK HDF5 format.
    
    Args:
        pose_path: Output path for pose HDF5 file
        poses: Pose array of shape [num_frames, 96]
    """
    num_frames = len(poses)
    
    # Create structured array matching ESK format
    dtype = np.dtype([
        ("index", "<i8"),
        ("values_block_0", "<f4", (96,)),
    ])
    
    table = np.zeros(num_frames, dtype=dtype)
    table["index"] = np.arange(num_frames)
    table["values_block_0"] = poses.astype(np.float32)
    
    # Create HDF5 file with proper structure
    Path(pose_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(pose_path, "w") as f:
        # Create tracks group
        tracks = f.create_group("tracks")
        
        # Create table dataset
        tracks.create_dataset("table", data=table)
        
        # Create minimal index structure (required by DLC2Action)
        _i_table = tracks.create_group("_i_table")
        index = _i_table.create_group("index")
        
        # Empty index arrays (not used but required for format compatibility)
        index.create_dataset("abounds", data=np.array([], dtype=np.int64))
        index.create_dataset("bounds", data=np.zeros((0, 127), dtype=np.int64))
        index.create_dataset("indices", data=np.zeros((0, 131072), dtype=np.uint32))
        index.create_dataset("indicesLR", data=np.zeros(131072, dtype=np.uint32))
        index.create_dataset("mbounds", data=np.array([], dtype=np.int64))
        index.create_dataset("mranges", data=np.array([], dtype=np.int64))
        index.create_dataset("ranges", data=np.zeros((0, 2), dtype=np.int64))
        index.create_dataset("sorted", data=np.zeros((0, 131072), dtype=np.int64))
        index.create_dataset("sortedLR", data=np.zeros(131201, dtype=np.int64))
        index.create_dataset("zbounds", data=np.array([], dtype=np.int64))


def save_labels(
    label_path: str,
    activity_names: list[str],
    video_name: str,
    segments: list[np.ndarray],
    metadata: Optional[dict] = None,
) -> None:
    """Save labels to ESK pickle format.
    
    Args:
        label_path: Output path for label pickle file
        activity_names: List of activity names
        video_name: Name of the video
        segments: List of segment arrays, one per activity
                  Each array has shape [N, 3] with [start, end, flag]
        metadata: Optional metadata dictionary
    """
    if metadata is None:
        metadata = {}
    
    # Convert activity names to numpy array of strings
    activity_names_arr = np.array(activity_names, dtype=str)
    
    # Ensure segments are proper numpy arrays
    segments_processed = []
    for seg in segments:
        if len(seg) == 0:
            segments_processed.append(np.array([], dtype=np.int32).reshape(0, 3))
        else:
            segments_processed.append(np.array(seg, dtype=np.int32))
    
    # Create tuple structure: (metadata, names, video_names, [[segments]])
    data = (
        metadata,
        activity_names_arr,
        [video_name],
        [segments_processed],
    )
    
    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, "wb") as f:
        pickle.dump(data, f)


def get_activity_segments(
    labels: dict,
    activity_name: str,
    video_idx: int = 0,
) -> np.ndarray:
    """Get segments for a specific activity.
    
    Args:
        labels: Labels dictionary from load_labels()
        activity_name: Name of the activity
        video_idx: Index of the video (default 0)
        
    Returns:
        Array of segments with shape [N, 3] containing [start, end, flag]
    """
    activity_idx = labels["activity_names"].index(activity_name)
    return labels["segments"][video_idx][activity_idx]


def extract_segment_poses(
    poses: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Extract poses for a specific segment.
    
    Args:
        poses: Full pose array [num_frames, 96]
        start: Start frame index
        end: End frame index
        
    Returns:
        Segment poses [end - start, 96]
    """
    return poses[start:end].copy()


class ESKDatasetWriter:
    """Helper class for creating augmented ESK datasets.
    
    This class accumulates generated trajectories and labels, then
    writes them out in ESK format for use with DLC2Action.
    """
    
    def __init__(
        self,
        activity_names: list[str],
        output_dir: str,
        video_name: str = "augmented",
    ):
        """Initialize dataset writer.
        
        Args:
            activity_names: List of all activity names
            output_dir: Output directory for pose and label files
            video_name: Name for the augmented video
        """
        self.activity_names = activity_names
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        
        # Accumulate poses and segments
        self.poses: list[np.ndarray] = []
        self.segments: dict[str, list[tuple[int, int]]] = {
            name: [] for name in activity_names
        }
        self.current_frame = 0
    
    def add_trajectory(
        self,
        poses: np.ndarray,
        activity_name: str,
    ) -> None:
        """Add a generated trajectory with its activity label.
        
        Args:
            poses: Pose trajectory [num_frames, 96]
            activity_name: Activity label for this trajectory
        """
        num_frames = len(poses)
        start = self.current_frame
        end = self.current_frame + num_frames
        
        self.poses.append(poses)
        self.segments[activity_name].append((start, end))
        self.current_frame = end
    
    def save(
        self,
        pose_suffix: str = "_pose3d_smpl.h5",
        label_suffix: str = "_labels.pickle",
    ) -> tuple[str, str]:
        """Save accumulated data to ESK format files.
        
        Args:
            pose_suffix: Suffix for pose file
            label_suffix: Suffix for label file
            
        Returns:
            Tuple of (pose_path, label_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all poses
        all_poses = np.concatenate(self.poses, axis=0)
        
        # Convert segments to ESK format
        segments_arrays = []
        for activity_name in self.activity_names:
            segs = self.segments[activity_name]
            if segs:
                arr = np.array([[s, e, 0] for s, e in segs], dtype=np.int32)
            else:
                arr = np.array([], dtype=np.int32).reshape(0, 3)
            segments_arrays.append(arr)
        
        # Save files
        pose_path = self.output_dir / f"{self.video_name}{pose_suffix}"
        label_path = self.output_dir / f"{self.video_name}{label_suffix}"
        
        save_pose_data(str(pose_path), all_poses)
        save_labels(
            str(label_path),
            self.activity_names,
            self.video_name,
            segments_arrays,
        )
        
        return str(pose_path), str(label_path)
    
    @property
    def num_trajectories(self) -> int:
        """Total number of trajectories added."""
        return sum(len(segs) for segs in self.segments.values())
    
    @property
    def total_frames(self) -> int:
        """Total number of frames."""
        return self.current_frame
    
    def summary(self) -> dict:
        """Get summary of accumulated data."""
        return {
            "num_trajectories": self.num_trajectories,
            "total_frames": self.total_frames,
            "segments_per_activity": {
                name: len(segs) for name, segs in self.segments.items()
            },
        }
