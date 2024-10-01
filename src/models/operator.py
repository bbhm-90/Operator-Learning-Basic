import torch
from torch import nn
from torch import Tensor

class DeepONet(nn.Module):
    def __init__(self,
            branch_net:nn.Module,
            trunk_net:nn.Module,
            bias=True,
            ) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        try:
            next(trunk_net.parameters())
            device = next(trunk_net.parameters()).device
        except (StopIteration):
            try:
                next(branch_net.parameters())
                device = next(branch_net.parameters()).device
            except (StopIteration):
                device = 'cpu'
        if bias:
            self.bias = torch.nn.Parameter(
                torch.tensor([0.]).float().to(device)
                ).requires_grad_(True)
        else:
            self.bias = torch.tensor([0.]).float().to(device)
    
    def forward(self, u:Tensor, y_shared:Tensor, y_indx:Tensor=None):
        N, My, _ = y_shared.shape
        assert u.ndim == 2
        assert y_shared.ndim == 3
        # NOTE (assumption): num of y_shared points are the same for all trajs
        coeffs = self.branch_net(u)
        coeffs = coeffs.unsqueeze(1).expand(N, My, coeffs.shape[1])
        # NOTE (assumption): y is the same for all trajectories
        bases = self.trunk_net(y_shared[0, ...])
        if y_indx is not None and len(y_indx) < bases.shape[0]:
            bases = bases[y_indx, ...]
        bases = bases.tile([N, 1, 1])
        assert bases.shape == coeffs.shape, (bases.shape == coeffs.shape)
        s_ = (bases * coeffs).sum(dim=2) + self.bias
        assert s_.shape[:2] == (N, My)
        return s_