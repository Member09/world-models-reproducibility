import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BouncingBallDataset(Dataset):
    """
    A unified dataset for Bouncing Ball physics.
    Supports 'frame' mode for VAE and 'sequence' mode for RNN.
    """
    def __init__(self, data_dir, mode='frame', seq_len=30):
        self.data_dir = data_dir
        self.mode = mode
        self.seq_len = seq_len
        
        # List all .npy files (assuming each file is one episode/rollout)
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load a single rollout: Shape [Total_Frames, Channels, H, W]
        # Example: [100, 1, 32, 32]
        data = np.load(os.path.join(self.data_dir, self.filenames[idx]))
        data = torch.from_numpy(data).float() / 255.0  # Normalize to [0, 1]

        if self.mode == 'frame':
            # For VAE: Pick a random frame from this rollout
            random_idx = np.random.randint(0, data.shape[0])
            return data[random_idx]

        elif self.mode == 'sequence':
            # For RNN: Return a chunk of frames of length 'seq_len'
            # We need (seq_len + 1) to have a target for prediction
            start_idx = np.random.randint(0, data.shape[0] - self.seq_len)
            sequence = data[start_idx : start_idx + self.seq_len + 1]
            
            # Input sequence (0 to T-1) and Target sequence (1 to T)
            x = sequence[:-1]
            y = sequence[1:]
            return x, y

        return data