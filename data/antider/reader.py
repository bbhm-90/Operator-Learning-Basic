"""
    Data should be downloaded firts from here:
    https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
"""
import os
from typing import Dict
import numpy as np
from numpy import ndarray

def read_data()->Dict:
    """
        output:
        x: (Mx, )
        u: (N, Mx)
        s: (N, My)
    """
    url = 'https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html'
    dir_pth = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for tag in ['trn', 'tst']:
        data_path = dir_pth + f'/{tag}.npz'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"\n The file must be downloaded from \n \t {url} \n and located in \n \t {dir_pth}")
        d = np.load(data_path, allow_pickle=True)
        X = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
        Nsamples = X[0].shape[0]
        Nsensors = X[0].shape[1]
        s = d["y"].astype(np.float32).reshape(Nsamples, Nsensors)
        u = X[0].reshape(Nsamples, Nsensors)
        x = X[1].reshape(Nsensors)
        data[tag] = {'x':x, 'u':u, 's':s}
        print(f"num of {tag} realizations: {Nsamples}")
        print(f"num of {tag} sensors: {Nsensors}")
        print("-"*10)
    return data