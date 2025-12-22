"""
Anomaly Detection: Spatio-Temporal Graph Normalizing Flows for Pose-Based Anomaly Detection.

A lightweight implementation based on the paper:
"STG-NF: Normalizing Flows for Video Anomaly Detection" (CVPR 2023)

This package provides:
- Graph-based normalizing flow models for skeleton pose data
- Support for various skeleton layouts (body, hands, etc.)
- Training and evaluation utilities for anomaly detection
- ESK dataset loader with activity-based partitioning
"""

from exact.anomaly.models import STG_NF, FlowNet
from exact.anomaly.graph import Graph, get_skeleton_layout, SMPL_KEYPOINTS
from exact.anomaly.trainer import Trainer
from exact.anomaly.esk_dataset import (
    ESKPoseDataset,
    get_esk_dataloaders,
    get_unique_activities,
)

__version__ = "0.1.0"
__all__ = [
    "STG_NF",
    "FlowNet",
    "Graph",
    "get_skeleton_layout",
    "SMPL_KEYPOINTS",
    "Trainer",
    "ESKPoseDataset",
    "get_esk_dataloaders",
    "get_unique_activities",
]
