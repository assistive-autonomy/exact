"""
Graph definitions for skeleton-based pose data.

Supports various skeleton layouts including SMPL body model (24 joints).
"""

import numpy as np
from typing import Optional, Literal


# SMPL skeleton layout (24 joints)
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

# SMPL skeleton edges (parent-child relationships)
SMPL_EDGES = [
    # Spine chain
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),  # Pelvis -> Torso -> Spine -> Chest -> Neck -> Head
    # Left arm
    (11, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),  # Chest -> L_Thorax -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand
    # Right arm
    (11, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),  # Chest -> R_Thorax -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand
    # Left leg
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Toe
    # Right leg
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Toe
]

# OpenPose 18-joint skeleton
OPENPOSE_KEYPOINTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

OPENPOSE_EDGES = [
    (4, 3),
    (3, 2),
    (7, 6),
    (6, 5),
    (13, 12),
    (12, 11),
    (10, 9),
    (9, 8),
    (11, 5),
    (8, 2),
    (5, 1),
    (2, 1),
    (0, 1),
    (15, 0),
    (14, 0),
    (17, 15),
    (16, 14),
]


SkeletonLayout = Literal["smpl", "openpose", "custom"]
AdjacencyStrategy = Literal["uniform", "distance", "spatial"]


def get_skeleton_layout(
    layout: SkeletonLayout = "smpl",
    custom_edges: Optional[list] = None,
    custom_num_nodes: Optional[int] = None,
) -> tuple[int, list, int]:
    """
    Get skeleton layout configuration.

    Args:
        layout: Skeleton type ("smpl", "openpose", or "custom")
        custom_edges: Custom edge list for "custom" layout
        custom_num_nodes: Number of nodes for "custom" layout

    Returns:
        Tuple of (num_nodes, edge_list, center_node)
    """
    if layout == "smpl":
        return len(SMPL_KEYPOINTS), SMPL_EDGES, 0  # Pelvis as center
    elif layout == "openpose":
        return len(OPENPOSE_KEYPOINTS), OPENPOSE_EDGES, 1  # Neck as center
    elif layout == "custom":
        if custom_edges is None or custom_num_nodes is None:
            raise ValueError(
                "custom_edges and custom_num_nodes required for custom layout"
            )
        return custom_num_nodes, custom_edges, 0
    else:
        raise ValueError(f"Unknown layout: {layout}")


class Graph:
    """
    Graph representation of skeleton structure for ST-GCN operations.

    Creates adjacency matrices with various partitioning strategies.

    Args:
        layout: Skeleton layout type
        strategy: Adjacency partitioning strategy
        max_hop: Maximum hop distance for neighbors
        custom_edges: Custom edges for "custom" layout
        custom_num_nodes: Number of nodes for "custom" layout
    """

    def __init__(
        self,
        layout: SkeletonLayout = "smpl",
        strategy: AdjacencyStrategy = "uniform",
        max_hop: int = 1,
        custom_edges: Optional[list] = None,
        custom_num_nodes: Optional[int] = None,
    ):
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop

        # Get skeleton configuration
        self.num_node, edges, self.center = get_skeleton_layout(
            layout, custom_edges, custom_num_nodes
        )

        # Build edge list with self-loops
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.edge = self.self_link + list(edges)

        # Compute hop distances
        self.hop_dis = self._get_hop_distance()

        # Build adjacency matrix
        self.A = self._get_adjacency(strategy)

    def _get_hop_distance(self) -> np.ndarray:
        """Compute shortest path distances between all node pairs."""
        # Build adjacency for distance computation
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[i, j] = 1
            A[j, i] = 1

        # Compute hop distances using matrix powers
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = np.stack(transfer_mat) > 0

        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d

        return hop_dis

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix using degree normalization."""
        Dl = np.sum(A, axis=0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return np.dot(A, Dn)

    def _get_adjacency(self, strategy: AdjacencyStrategy) -> np.ndarray:
        """Build adjacency matrix with specified partitioning strategy."""
        valid_hop = range(0, self.max_hop + 1)

        # Base adjacency from valid hops
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self._normalize_adjacency(adjacency)

        if strategy == "uniform":
            # Single adjacency matrix
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency

        elif strategy == "distance":
            # Separate matrix for each hop distance
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]

        elif strategy == "spatial":
            # Spatial partitioning (root, close, further from center)
            A_list = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A_list.append(a_root)
                else:
                    A_list.append(a_root + a_close)
                    A_list.append(a_further)

            A = np.stack(A_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return A
