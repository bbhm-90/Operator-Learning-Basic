from torch import Tensor

def getLossMSE(
        x:Tensor, 
        x_:Tensor,
        agg=True
        )->Tensor:
    """
        x: (B,...)
        x_: (B,...)
        -----
        this is equivalent to 
            torch.nn.MSELoss()(x, x_)
    """
    assert x.shape == x_.shape, x.shape
    B = x.shape[0]
    x = x.view(B, -1)
    x_ = x_.view(B, -1)
    loss = (x - x_).pow(2).mean(1)
    if agg:
        return loss.mean()
    else:
        return loss

def getLossMSERel(
        x:Tensor, 
        x_:Tensor, 
        agg=True
        )->Tensor:
    """
        x: (B,...)
        x_: (B,...)
    """
    assert x.shape == x_.shape
    B = x.shape[0]
    x = x.view(B, -1)
    x_ = x_.view(B, -1)
    scale = x.pow(2).max(1)[0].detach()
    loss = (x - x_).pow(2).mean(1) / scale
    if agg:
        return loss.mean()
    else:
        return loss