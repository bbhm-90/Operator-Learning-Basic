import torch
from src.models.base import MLPBasic

def test_0():
    x = torch.rand(10, 3)
    model = MLPBasic(
        layers=[3, 10, 10, 2],
        acts=['relu', 'relu', 'iden'])
    y = model(x)
    assert y.shape == (10, 2)

if __name__ == "__main__":
    test_0()