from typing import List, Optional, Dict, Any

import ot
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


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
        item = {
            "motion": self.motions[idx].cpu() if self.motions[idx].is_cuda else self.motions[idx],
        }
        
        # Tokenize the program if tokenizer is provided
        if self.tokenizer is not None and self.programs[idx] is not None:
            tokenized = self.tokenizer(
                self.programs[idx],
                truncation=True,
                padding=False,  # Padding handled by data collator
                return_tensors=None,  # Return lists, not tensors
            )
            item["input_ids"] = tokenized["input_ids"]
            item["attention_mask"] = tokenized["attention_mask"]
            # Labels are the same as input_ids for causal language modeling
            item["labels"] = tokenized["input_ids"]
        
        return item




