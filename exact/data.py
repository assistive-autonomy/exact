"""Data module for loading motion-program pairs."""
from typing import List

import h5py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from exact.env import OBS_SLICES


class TrajectoryGenerationDataset(Dataset):
    """Dataset for program-to-pose pairs from HDF5 files.

    HDF5 format:
        motion_{i}/motion: numpy array of shape (T, 358 + 69)
            - First 358 dims: observation (use local_body_pos slice 1:70)
            - Last 69 dims: action
        motion_{i}.attrs['program']: program string
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        """Initialize dataset from HDF5 file.

        Args:
            path: Path to HDF5 file
            tokenizer: Tokenizer for encoding programs
            max_seq_length: Maximum sequence length for tokenization
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Get local_body_pos slice indices
        self.pos_start, self.pos_end = OBS_SLICES["local_body_pos"]

        # Load the HDF5 file and extract data
        self.programs: List[str] = []
        self.local_body_pos: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []

        with h5py.File(path, "r") as f:
            for key in f.keys():
                motion = f[key]["motion"][()]
                program = f[key].attrs["program"]

                # Extract local_body_pos from obs (dims 1:70 of first 358)
                obs = motion[..., :358]
                local_pos = obs[..., self.pos_start : self.pos_end]

                # Extract actions (last 69 dims)
                action = motion[..., 358:]

                self.programs.append(program)
                self.local_body_pos.append(torch.tensor(local_pos, dtype=torch.float32))
                self.actions.append(torch.tensor(action, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.programs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - input_ids: tokenized program (max_seq_length,)
                - attention_mask: attention mask (max_seq_length,)
                - local_body_pos: body positions (T, 69)
                - actions: actions (T, 69)
        """
        program = self.programs[idx]
        local_pos = self.local_body_pos[idx]
        action = self.actions[idx]

        # Tokenize the program
        encoded = self.tokenizer(
            program,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "local_body_pos": local_pos,
            "actions": action,
        }
