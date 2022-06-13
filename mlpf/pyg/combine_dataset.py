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


path = '../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed/'

all = []
for i in range(500):
    print(f'file # {i}')
    a = torch.load(path + f'data_{i}.pt')
    all = all + a

torch.save(all, 'all_ttbar.pt')
