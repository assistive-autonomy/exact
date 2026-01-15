import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTrajectoryEncoder(nn.Module):
    """Temporal-aware trajectory encoder using learned queries and cross-attention.

    Instead of global average pooling (which destroys temporal structure),
    this encoder uses learned query tokens that attend to the motion sequence,
    preserving temporal information crucial for segment-based program generation.

    Architecture:
        1. Input projection + positional encoding
        2. Transformer encoder for motion self-attention
        3. Learned queries cross-attend to encoded motion
        4. Output projection to LLM embedding space

    This design allows the model to:
        - Preserve temporal structure through positional encodings
        - Learn segment-relevant features via cross-attention queries
        - Generate variable-length prefix embeddings aligned with motion segments
    """

    def __init__(
        self,
        trajectory_dim: int = 72,
        hidden_dim: int = 256,
        output_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        num_queries: int = 32,
        nhead: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        """Initialize temporal trajectory encoder.

        Args:
            trajectory_dim: Input motion feature dimension (24 joints * 3 axes = 72)
            hidden_dim: Hidden dimension for transformer layers
            output_dim: Output dimension (must match LLM hidden size)
            num_encoder_layers: Number of self-attention layers for motion encoding
            num_decoder_layers: Number of cross-attention layers for query decoding
            num_queries: Number of learned query tokens (prefix length for LLM)
            nhead: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum motion sequence length
        """
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection with layer norm for stable training
        self.input_projection = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Motion encoder (self-attention over motion sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better gradient flow
        )
        self.motion_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Learned query tokens that will cross-attend to motion
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Query positional encoding (learned, for query ordering)
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Cross-attention decoder (queries attend to motion)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection to LLM embedding space
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # For auxiliary reconstruction loss
        self.reconstruction_head = None

    def enable_reconstruction(self, trajectory_dim: int):
        """Enable motion reconstruction head for auxiliary loss."""
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, trajectory_dim),
        )

    def forward(
        self, motion: torch.Tensor, return_encoded_motion: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode motion sequence into LLM-compatible prefix embeddings.

        Args:
            motion: [batch_size, seq_len, trajectory_dim] motion sequence
            return_encoded_motion: If True, also return encoded motion for reconstruction

        Returns:
            prefix_embeddings: [batch_size, num_queries, output_dim]
            encoded_motion (optional): [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = motion.shape

        # Project input and add positional encoding
        x = self.input_projection(motion)  # [B, T, hidden_dim]
        x = self.pos_encoding(x)

        # Encode motion with self-attention
        encoded_motion = self.motion_encoder(x)  # [B, T, hidden_dim]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, num_queries, H]
        query_pos = self.query_pos.expand(batch_size, -1, -1)

        # Cross-attend queries to encoded motion
        # Queries attend to temporal motion features, learning to extract
        # segment-relevant information
        decoded_queries = self.query_decoder(
            tgt=queries + query_pos,
            memory=encoded_motion,
        )  # [B, num_queries, hidden_dim]

        # Project to LLM embedding space
        prefix_embeddings = self.output_projection(decoded_queries)

        if return_encoded_motion:
            return prefix_embeddings, encoded_motion

        return prefix_embeddings

    def reconstruct_motion(self, encoded_motion: torch.Tensor) -> torch.Tensor:
        """Reconstruct motion from encoded representation (for auxiliary loss).

        Args:
            encoded_motion: [batch_size, seq_len, hidden_dim]

        Returns:
            reconstructed: [batch_size, seq_len, trajectory_dim]
        """
        if self.reconstruction_head is None:
            raise RuntimeError("Reconstruction head not enabled. Call enable_reconstruction() first.")
        return self.reconstruction_head(encoded_motion)


# Keep the old encoder for backward compatibility
class TrajectoryEncoder(nn.Module):
    """Legacy encoder - use TemporalTrajectoryEncoder for new training."""

    def __init__(
        self,
        trajectory_dim: int = 214,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens
        self.output_dim = output_dim

        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, output_dim * num_prefix_tokens)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        batch_size = motion.shape[0]

        x = self.input_projection(motion)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        x = self.output_projection(x)
        embeddings = x.view(batch_size, self.num_prefix_tokens, self.output_dim)

        return embeddings
