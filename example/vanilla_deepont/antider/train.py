import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.antider.reader import read_data
from src.data_processing.dataset import DeepONetDatasetV0
from src.models.operator import DeepONet
from src.utils.loss_function import getLossMSERel
pjoin = os.path.join

def get_batch_loss(u, y_share, s):
        u = u.float()
        s = s.float().squeeze()
        y_share = y_share.float()
        s_ = model.forward(u=u, y_shared=y_share, y_indx=None)
        assert s_.shape == s.shape, (s_.shape, s.shape)
        return getLossMSERel(x=s, x_=s_)

branch_net = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
)
trunk_net = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
)

if __name__ == "__main__":
    batch_size = int(10000)
    lr = 1e-4
    num_epoch = int(10**4)
    #######
    #######
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    out_dir = pjoin(script_dir, 'results')
    if not os.path.exists(out_dir):
         os.makedirs(out_dir)
    #######
    #######
    np.random.seed(0)
    torch.manual_seed(0)

    data = read_data()
    dataloader = {}
    for tag in ['trn', 'tst']:
        dataset = DeepONetDatasetV0(
            u=data['trn']['u'],
            y=data['trn']['x'].reshape(-1, 1),
            s=data['trn']['s'][:, :, np.newaxis]
        )
        dataloader[tag] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        bias=True
    )
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    for epoch in range(num_epoch):
        loss_ep = 0.
        ndata = 0
        for u, y_share, s, _ in dataloader['trn']:
            loss = get_batch_loss(u, y_share, s)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_ep += loss.item() * u.shape[0]
            ndata += u.shape[0]
        loss_ep /= ndata
        with torch.no_grad():
            loss_ep_tst, ndata = 0., 0
            for u, y_share, s, _ in dataloader['tst']:
                loss_tst = get_batch_loss(u, y_share, s).item()
                loss_ep_tst += loss_tst * u.shape[0]
                ndata += u.shape[0]
            loss_ep_tst /= ndata
        print(f"{epoch}, {loss_ep:.3e}, {loss_ep_tst:.3e}")
    torch.save(model.state_dict(), pjoin(out_dir, "operator.pth"))