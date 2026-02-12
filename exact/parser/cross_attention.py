"""Gated cross-attention module for injecting motion conditioning into LLM layers.

Provides a gated cross-attention block that can be injected after selected LLM
decoder layers. Uses a learnable gate initialized to zero so the model starts as
a pure LLM and gradually learns to attend to motion features during training.
"""

import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention layer for motion conditioning.

    Inserted after selected LLM decoder layers to inject motion information.
    Uses a learnable gate initialized to zero for training stability — the
    model starts as a pure LLM and gradually learns to use motion features.

    Architecture:
        h' = h + tanh(gate) * CrossAttn(LayerNorm(h), motion_features)

    Args:
        hidden_dim: LLM hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability in cross-attention
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Initialize gate to zero — model starts as pure LLM
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention.

        Args:
            hidden_states: [B, T_text, D] text hidden states (query)
            encoder_hidden_states: [B, T_motion, D] motion features (key/value)

        Returns:
            [B, T_text, D] updated hidden states with cross-attended motion info
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        cross_out, _ = self.cross_attn(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
        )
        return residual + torch.tanh(self.gate) * cross_out
