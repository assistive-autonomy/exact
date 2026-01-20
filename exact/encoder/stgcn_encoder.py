"""
Spatial-Temporal Graph Convolutional Network encoder for motion encoding.

Leverages the ST-GCN architecture to encode skeletal motion sequences into embeddings
suitable for downstream tasks (anomaly detection with normalizing flows, or LLM conditioning).
"""

import torch
import torch.nn as nn

from exact.encoder.utils import Graph, STGCNBlock

class STGCNEncoder(nn.Module):
    """
    ST-GCN based trajectory encoder for SMPL skeletal motion encoding.

    Processes 24-joint SMPL skeletal motion using spatial-temporal graph convolutions
    followed by temporal pooling and projection to output embedding space.

    Architecture:
        1. Input reshaping: [B, T, 72] -> [B, 3, T, 24]
        2. ST-GCN blocks: Learn spatio-temporal motion patterns on SMPL skeleton
        3. Temporal pooling: Aggregate into num_temporal_tokens time windows
        4. Output projection: Map each window to target output dimension

    Args:
        num_nodes: Number of skeleton joints (24 for SMPL)
        input_channels: Input feature dimension per joint (3 for x,y,z)
        hidden_channels: Hidden dimension for ST-GCN blocks
        output_dim: Output dimension (must match downstream task requirement)
        num_blocks: Number of ST-GCN blocks
        num_temporal_tokens: Number of temporal tokens to output (preserves time structure)
        temporal_kernel_size: Temporal convolution kernel size
        spatial_kernel_size: Number of adjacency matrix partitions
        dropout: Dropout probability
        graph_strategy: Adjacency partitioning strategy ('uniform', 'distance', 'spatial')
    """

    def __init__(
        self,
        num_nodes: int = 24,
        input_channels: int = 3,
        hidden_channels: int = 64,
        output_dim: int = 2048,
        num_blocks: int = 4,
        num_temporal_tokens: int = 8,
        temporal_kernel_size: int = 9,
        spatial_kernel_size: int = 3,
        dropout: float = 0.1,
        graph_strategy: str = "spatial",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_temporal_tokens = num_temporal_tokens

        # Build SMPL skeleton graph
        graph = Graph(strategy=graph_strategy, max_hop=1)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # Input projection
        self.input_bn = nn.BatchNorm2d(input_channels)

        # ST-GCN blocks
        channels = [input_channels] + [hidden_channels] * num_blocks
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(
                STGCNBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=(temporal_kernel_size, spatial_kernel_size),
                    stride=1,
                    residual=i > 0,  # No residual for first block
                )
            )

        # Temporal pooling to num_temporal_tokens windows (preserves temporal structure)
        self.temporal_pool = nn.AdaptiveAvgPool2d((num_temporal_tokens, num_nodes))

        # Output projection to target embedding space (per temporal token)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels * num_nodes, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Encode motion sequence into temporal embeddings.

        Args:
            motion: [batch_size, seq_len, num_nodes * input_channels]
                   Motion sequence with flattened joint features

        Returns:
            embeddings: [batch_size, num_temporal_tokens, output_dim]
                       Multiple embeddings preserving temporal structure
        """
        # Compute in float32 for BatchNorm numerical stability
        motion = motion.float()
        
        batch_size, seq_len, features = motion.shape
        assert features == self.num_nodes * self.input_channels, (
            f"Expected {self.num_nodes * self.input_channels} features, "
            f"got {features}"
        )

        # Reshape to ST-GCN format: [B, C, T, V]
        x = motion.view(batch_size, seq_len, self.num_nodes, self.input_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]

        # Input normalization
        x = self.input_bn(x)

        # ST-GCN processing
        for block in self.blocks:
            x = block(x, self.A)

        # Temporal pooling: [B, C, T, V] -> [B, C, num_temporal_tokens, V]
        x = self.temporal_pool(x)

        # Reshape for projection: [B, C, T', V] -> [B, T', C*V]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T', C, V]
        x = x.view(batch_size, self.num_temporal_tokens, -1)  # [B, T', C*V]

        # Project each temporal token: [B, T', C*V] -> [B, T', output_dim]
        embeddings = self.output_projection(x)

        return embeddings
