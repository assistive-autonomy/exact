import ot
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any

from transformers import PreTrainedTokenizer, PreTrainedModel

def wasserstein_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """wasserstein_distance computes the 2-Wasserstein distance between two point clouds"""
    X_norm = X.pow(2).sum(-1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(-1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    # clamp is needed to avoid negative values due to numerical errors
    M = torch.sqrt(torch.clamp(val, min=0))

    fst_pot = torch.ones(X.shape[0]) / X.shape[0]
    snd_pot = torch.ones(Y.shape[0]) / Y.shape[0]
    return ot.emd2(fst_pot, snd_pot, M)


class MotionProgramDataset(Dataset):
    """Dataset of motion sequences and their corresponding programs for SFT training."""
    
    def __init__(
        self,
        motions: List[torch.Tensor],
        programs: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.motions = motions
        self.programs = programs if programs is not None else [None] * len(motions)
        self.tokenizer = tokenizer
        
        assert len(self.motions) == len(self.programs)
    
    def __len__(self) -> int:
        return len(self.motions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "motion": self.motions[idx],
            "reference_program": self.programs[idx],
        }


class MotionEncoder(nn.Module):
    """Encodes motion sequences into embeddings compatible with LLM."""
    
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
    

def generate_program(
    model: PreTrainedModel,
    motion_encoder: MotionEncoder,
    tokenizer: PreTrainedTokenizer,
    motion: torch.Tensor,
    num_beams: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> List[str]:
    model.eval()
    motion_encoder.eval()
    
    with torch.no_grad():
        motion_embeddings = motion_encoder(motion)
        
        outputs = model.generate(
            inputs_embeds=motion_embeddings,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        programs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return programs
