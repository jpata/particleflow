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


path = '../../data/cms_ttbar_raw/processed/'

all = []
for i in range(50):
    print(f'file # {i}')
    a = torch.load(path + f'data_{450+i}.pt')
    all = all + a

torch.save(all, 'ttbar_valid.pt')
