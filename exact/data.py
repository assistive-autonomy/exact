"""Data module for loading motion-program pairs."""
from typing import List

import h5py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class Program2PoseDataset(Dataset):
    """Dataset for program-to-pose pairs from HDF5 files."""
    
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
        
        # Load the HDF5 file and extract data
        self.programs: List[str] = []
        self.poses: List[torch.Tensor] = []
        
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                pose = f[key]['motion'][()][..., :214]
                program = f[key].attrs['program']
                self.programs.append(program)
                self.poses.append(torch.tensor(pose, dtype=torch.float32))
        
    def __len__(self) -> int:
        return len(self.programs)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and poses
        """
        program = self.programs[idx]
        pose = self.poses[idx]
        
        # Tokenize the program
        encoded = self.tokenizer(
            program,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'poses': pose,
        }