"""
Convert data compatible to classifier.
"""

import numpy as np
from torch.utils.data import Dataset
import torch


class VectorizedDataset(Dataset):
    def __init__(self, X_data: np.array, y_data: np.array):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

        self.len = len(X_data)

    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]