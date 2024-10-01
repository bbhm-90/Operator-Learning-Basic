import torch
from src.models.operator import DeepONet

def test_1():
    from src.models.base import MLPBasic
    bsz = 10
    num_input_sensors = 15
    num_output_sensors = 25
    y_dim = 1
    num_trunk_basis = 7
    
    u = torch.rand(bsz, num_input_sensors)
    y = torch.rand(bsz, num_output_sensors, y_dim)
    
    branch_net = MLPBasic(
        layers=[num_input_sensors, 10, 10, num_trunk_basis],
        acts=['relu', 'relu', 'iden'])
    trunk_net = MLPBasic(
        layers=[y_dim, 10, 10, num_trunk_basis],
        acts=['relu', 'relu', 'iden'])
    op_model = DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net)
    s_ = op_model.forward(
        u=u, y_shared=y)
    assert s_.shape == (bsz, num_output_sensors)

if __name__ == "__main__":
    test_1()