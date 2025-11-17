import h5py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class Program2PoseDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load the HDF5 file and extract data
        self.programs = []
        self.poses = []
        
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                pose =  f[key]['motion'][()][..., :214]
                program = f[key].attrs['program']
                self.programs.append(program)
                self.poses.append(pose)
        
    def __len__(self):
        return len(self.programs)
    
    def __getitem__(self, idx):
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
            'poses': torch.tensor(pose, dtype=torch.float32)
        }