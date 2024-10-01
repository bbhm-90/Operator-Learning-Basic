import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import Dataset

class DeepONetDatasetV0(Dataset):
    def __init__(self,
        u:ndarray,
        y:ndarray,
        s:ndarray):
        """
            u: (N, Du)
            y: (My, Dy)
            s: (N, My, Ds)
        """
        assert  s.ndim == 3, s.shape
        assert u.ndim == y.ndim == 2
        N = u.shape[0]
        assert s.shape[1] == y.shape[0]
        assert s.shape[0] == N
        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()
        self.s = torch.from_numpy(s).float()
        self.N = N
        self.indx = torch.arange(self.N).long()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.u[idx, ...],\
               self.y,\
               self.s[idx, ...],\
               self.indx[idx, ...]