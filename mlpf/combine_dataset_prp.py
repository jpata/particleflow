import h5py
import pandas
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data, Batch
from glob import glob

import pickle
import multiprocessing

from pyg import make_file_loaders
from pyg.dataset import PFGraphDataset, one_hot_embedding


path_from = '/particleflowvol/data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/'
path_to = '/particleflowvol/all/'

os.makedirs(path_to, exist_ok=True)

dataset = PFGraphDataset(path_from, 'cms')

dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=500))
loader = make_file_loaders(dataset, 1, num_workers=10, prefetch_factor=10)

all = []
c = 0
j = 0

for file in loader:
    print(f'file # {i}')
    all = all + file
    c = c + 1

    if c == 10:
        print(f'SAVING FILE # {j}')
        torch.save(all, path_to + f'data_{j}.pt')
        c = 0
        all = []
        j = j + 1
