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


path_from = '../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed/'
path_to = '../../data/10/processed/'

all = []
c = 0
j = 0
for i in range(500):
    print(f'file # {i}')
    a = torch.load(path_from + f'data_{i}.pt')
    all = all + a
    c = c + 1

    if c == 10:
        print(f'SAVING FILE # {j}')
        torch.save(all, path_to + f'data_{j}.pt')
        c = 0
        all = []
        j = j + 1
