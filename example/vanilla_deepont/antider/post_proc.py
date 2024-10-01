import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import (
    read_data,
    DeepONetDatasetV0,
    DeepONet,
    branch_net,
    trunk_net
)
import matplotlib.pyplot as plt
pjoin = os.path.join


if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    out_dir = pjoin(script_dir, 'results')
    if not os.path.exists(out_dir):
         os.makedirs(out_dir)
    #######
    #######

    data = read_data()
    dataloader = {}
    for tag in ['trn', 'tst']:
        dataset = DeepONetDatasetV0(
            u=data['trn']['u'],
            y=data['trn']['x'].reshape(-1, 1),
            s=data['trn']['s'][:, :, np.newaxis]
        )
        dataloader[tag] = DataLoader(dataset, batch_size=int(10**5), shuffle=False)

    model = DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        bias=True
    )
    model.load_state_dict(torch.load(pjoin(out_dir, 'operator.pth')))
    with torch.no_grad():
        for u, y_share, s, _ in dataloader['tst']:
            u = u.float()
            s = s.float().squeeze()
            y_share = y_share.float()
            s_ = model.forward(u=u, y_shared=y_share, y_indx=None)
    for i in range(s.shape[0]):
        plt.plot(data['trn']['x'], s[i, :], label='ground truth')
        plt.plot(data['trn']['x'], s_[i, :], label='prediction')
        plt.xlabel('y')
        plt.ylabel('s(y)')
        plt.legend()
        plt.show()