import torch
import numpy as np

def test_0():
    from src.utils.loss_function import getLossMSE
    x = torch.rand(10, 7)
    x_ = torch.rand(10, 7)
    loss = getLossMSE(x, x_).item()
    loss_ = torch.nn.MSELoss()(x, x_).item()
    np.allclose(loss, loss_)

def test_1():
    from src.utils.loss_function import getLossMSE
    x = torch.rand(10, 7, 4)
    x_ = torch.rand(10, 7, 4)
    loss = getLossMSE(x, x_).item()
    loss_ = torch.nn.MSELoss()(x, x_).item()
    np.allclose(loss, loss_)

def test_2():
    from src.utils.loss_function import getLossMSERel
    x = torch.rand(10, 7, 4)
    x_ = torch.rand(10, 7, 4)
    loss = getLossMSERel(x, x_, agg=False)
    assert loss.shape == (10, )


if __name__ == "__main__":
    test_0()
    test_1()
    test_2()