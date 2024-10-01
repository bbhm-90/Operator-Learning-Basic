import torch
from torch import nn

def get_act_func(act: str) -> callable:
    if act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'sin':
        return torch.sin
    elif act == 'silu':
        return nn.SiLU()
    elif act == 'selu':
        return nn.SELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'iden':
        return lambda x: x
    else:
        raise NotImplementedError(act)

class MLPBasic(nn.Module):
    def __init__(
        self,
        layers=[1, 50, 50, 1],
        acts=['relu', 'relu', 'iden']
        ) -> None:
        assert len(layers) == len(acts) + 1
        super(MLPBasic, self).__init__()
        act_funcs = []
        for act in acts:
            act_funcs.append(get_act_func(act))
        self.act_funcs = act_funcs

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, inputs):
        for layer, act in zip(self.layers, self.act_funcs):
            inputs = act(layer(inputs))
        return inputs