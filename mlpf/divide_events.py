import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import matplotlib.pyplot as plt
import os
import pickle as pkl
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use("Agg")

# path = '../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed/'
path = '/particleflowvol/particleflow/data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed/'


num_files = 499

c = 1
for j in range(num_files):
    data = list(torch.load(path + f'data_{j}.pt'))
    # divide the file into 10
    for i in range(10):
        print(f'{i * 10}:{(i + 1) * 10}')

        if i == 0:
            torch.save(data[(i * 10):((i + 1) * 10)], path + f'processed2/data_{j}.pt')
        else:
            torch.save(data[(i * 10):((i + 1) * 10)], path + f'processed2/data_{num_files+c}.pt')
            c = c + 1
