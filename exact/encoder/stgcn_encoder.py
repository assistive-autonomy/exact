"""
Spatial-Temporal Graph Convolutional Network encoder for motion encoding.

Leverages the ST-GCN architecture to encode skeletal motion sequences into embeddings
suitable for downstream tasks (anomaly detection with normalizing flows, or LLM conditioning).
"""

import torch
import torch.nn as nn

from exact.encoder.utils import Graph, STGCNBlock


class MotionNormalizer(nn.Module):
    """
    Per-feature running normalization for raw motion input.

    Unlike BatchNorm, this module ALWAYS normalizes using running statistics
    (both in train and eval mode), eliminating the distribution mismatch that
    occurs when BatchNorm switches from batch stats (train) to running stats
    (eval) on out-of-distribution data.

    During training, running mean/var are updated via exponential moving average.
    During eval, the stored statistics are used without updates.

    Args:
        num_features: Number of input features (e.g., 72 for 24 joints × 3 coords)
        momentum: EMA momentum for running stats updates (default: 0.1)
        eps: Small constant for numerical stability
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using running statistics.

        Args:
            x: Input tensor of shape [..., num_features] (e.g. [B, T, 72])
        Returns:
            Normalized tensor of same shape
        """
        if self.training:
            with torch.no_grad():
                flat = x.reshape(-1, x.shape[-1]).float()
                batch_mean = flat.mean(dim=0)
                batch_var = flat.var(dim=0, unbiased=False)

                self.num_batches_tracked += 1
                if self.num_batches_tracked == 1:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    self.running_mean.lerp_(batch_mean, self.momentum)
                    self.running_var.lerp_(batch_var, self.momentum)

        # Always normalize with running stats (identical behavior train & eval)
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)

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
        joint_embedding: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_temporal_tokens = num_temporal_tokens
        self.joint_embedding = joint_embedding

        # Build SMPL skeleton graph
        graph = Graph(strategy=graph_strategy, max_hop=1)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # Per-feature normalization (always uses running stats — no train/eval gap)
        self.motion_normalizer = MotionNormalizer(num_nodes * input_channels)

        # Input projection (InstanceNorm: no running stats, immune to train/eval mismatch)
        self.input_bn = nn.InstanceNorm2d(input_channels, affine=True)

        # ── Per-joint positional encoding ─────────────────────────────────
        # After spatial graph convolutions, joint identity gets blurred.
        # A learnable per-joint embedding is added after the first block so
        # that the downstream blocks can maintain joint-specific features.
        if joint_embedding:
            self.joint_embed = nn.Parameter(
                torch.randn(1, hidden_channels, 1, num_nodes) * 0.02
            )
        else:
            self.joint_embed = None

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
        # Direct projection avoids bottleneck: hidden_channels*num_nodes → output_dim
        # (Previous hidden_channels*4=256 intermediate caused 6:1 compression of joint info)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels * num_nodes, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
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
        # Compute in float32 for numerical stability
        motion = motion.float()
        
        batch_size, seq_len, features = motion.shape
        assert features == self.num_nodes * self.input_channels, (
            f"Expected {self.num_nodes * self.input_channels} features, "
            f"got {features}"
        )

        # Per-feature normalization (running mean/std, identical train & eval)
        motion = self.motion_normalizer(motion)

        # Reshape to ST-GCN format: [B, C, T, V]
        x = motion.view(batch_size, seq_len, self.num_nodes, self.input_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]

        # Input normalization
        x = self.input_bn(x)

        # ST-GCN processing
        for i, block in enumerate(self.blocks):
            x = block(x, self.A)
            # Inject per-joint positional encoding after the first block.
            # At this point x is [B, hidden_channels, T, V] — adding the
            # joint embedding injects a unique per-joint signal that survives
            # the subsequent spatial mixing layers.
            if i == 0 and self.joint_embed is not None:
                x = x + self.joint_embed

        # Temporal pooling: [B, C, T, V] -> [B, C, num_temporal_tokens, V]
        x = self.temporal_pool(x)

        # Reshape for projection: [B, C, T', V] -> [B, T', C*V]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T', C, V]
        x = x.view(batch_size, self.num_temporal_tokens, -1)  # [B, T', C*V]

        # Project each temporal token: [B, T', C*V] -> [B, T', output_dim]
        embeddings = self.output_projection(x)

        return embeddings
