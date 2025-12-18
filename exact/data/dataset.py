from typing import List

import h5py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TrajectoryGenerationDataset(Dataset):
    """Dataset for program-to-pose pairs from HDF5 files.

    HDF5 format:
        motion_{i}/motion: numpy array of shape (T, 72)
            - 24 SMPL joints * 3 (x, y, z) world positions
            - Root joint at (0, 0, 0), other joints relative to world
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length

        self.programs: List[str] = []
        self.obs: List[torch.Tensor] = []

        with h5py.File(path, "r") as f:
            for key in f.keys():
                motion = f[key]["motion"][()]
                program = f[key].attrs["program"]

                self.programs.append(program)
                self.obs.append(torch.tensor(motion, dtype=torch.float32))

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
                - obs: padded observations (T, 361)
        """
        program = self.programs[idx]
        obs = self.obs[idx]

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
            "obs": obs,
        }
