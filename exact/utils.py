from typing import List, Optional, Dict, Any

import ot
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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


def generate_program(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    motion: torch.Tensor,
    num_beams: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> List[str]:
    """Generate programs from motion sequences using the inverse behavior model.

    Args:
        model: The trained inverse behavior model
        tokenizer: Tokenizer for the model
        motion: Input motion sequence tensor of shape [batch_size, seq_len, motion_dim]
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling

    Returns:
        List of generated program strings
    """
    model.eval()

    with torch.no_grad():
        # Average pool the motion sequence to get a fixed-size representation
        motion_embeddings = motion.mean(dim=1)  # [batch_size, motion_dim]

        # Project motion embeddings to match the model's hidden size
        motion_projection = nn.Linear(
            motion_embeddings.size(-1),
            model.config.hidden_size,
            device=motion_embeddings.device,
        )
        hidden_states = motion_projection(motion_embeddings).unsqueeze(
            1
        )  # [batch_size, 1, hidden_size]

        # Generate programs
        outputs = model.generate(
            inputs_embeds=hidden_states,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        programs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return programs
