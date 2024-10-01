import numpy as np
from torch.utils.data import DataLoader

def test_0():
    from src.data_processing.dataset import DeepONetDatasetV0
    N, Du = 10, 17
    My, Dy, Ds = 13, 2, 1
    batch_size = 6
    u = np.random.rand(N, Du)
    y = np.random.rand(My, Dy)
    s = np.random.rand(N, My, Ds)
    dataset = DeepONetDatasetV0(u = u,y = y, s = s)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    u_, y_, s_, idx_ = next(iter(dataloader))
    assert u_.shape == (batch_size, Du), u_.shape
    assert y_.shape == (batch_size, My, Dy), y_.shape
    assert s_.shape == (batch_size, My, Ds), s_.shape
    assert idx_.shape == (batch_size, ), idx_.shape

if __name__ == "__main__":
    test_0()