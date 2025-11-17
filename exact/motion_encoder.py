import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
        
    def __init__(
        self,
        motion_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        
        self.num_prefix_tokens = num_prefix_tokens
        self.output_dim = output_dim
        
        self.input_projection = nn.Linear(motion_dim, hidden_dim)
        
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
    