import numpy as np
from typing import Literal

import torch
import torch.nn as nn

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
    # Spine chain:  Pelvis -> Torso -> Spine -> Chest -> Neck -> Head
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),  
    # Left arm: Chest -> L_Thorax -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # Right arm: Chest -> R_Thorax -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23), 
    # Left leg: Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Toe
    (0, 1), (1, 2), (2, 3), (3, 4), 
    # Right leg: Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Toe
    (0, 5), (5, 6), (6, 7), (7, 8),
]

# Adjacency partitioning strategies
AdjacencyStrategy = Literal["uniform", "distance", "spatial"]

# SMPL skeleton constants
NUM_JOINTS = len(SMPL_KEYPOINTS)
CENTER_JOINT = 0  # Pelvis


class Graph:
    """
    Graph representation of SMPL skeleton structure for ST-GCN operations.

    Creates adjacency matrices with various partitioning strategies for the
    24-joint SMPL body model.

    Args:
        strategy: Adjacency partitioning strategy ('uniform', 'distance', or 'spatial')
        max_hop: Maximum hop distance for neighbors
    """

    def __init__(
        self,
        strategy: AdjacencyStrategy = "uniform",
        max_hop: int = 1,
    ):
        self.strategy = strategy
        self.max_hop = max_hop

        # SMPL skeleton configuration
        self.num_node = NUM_JOINTS
        self.center = CENTER_JOINT
        edges = SMPL_EDGES

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


class GraphConv(nn.Module):
    """
    Graph convolution layer for skeleton data.

    Applies spatial graph convolution followed by temporal convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,  # spatial kernel size (number of adjacency matrices)
        t_kernel_size: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_kernel_size // 2, 0),
            stride=(stride, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, V)
            A: Adjacency matrix (K, V, V)
        """
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, A)

        return x.contiguous()



class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolution block.

    Graph convolution followed by temporal convolution with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],  # (temporal, spatial)
        stride: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        t_kernel, s_kernel = kernel_size

        self.gcn = GraphConv(in_channels, out_channels, s_kernel, t_kernel_size=1)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel, 1),
                (stride, 1),
                (t_kernel // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)