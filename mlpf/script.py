from torch_geometric.data import Data, Batch
from pyg import parse_args
from pyg import MLPF, training_loop, make_predictions, make_plots
from pyg import get_model_fname, save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg.dataset import PFGraphDataset

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import matplotlib.pyplot as plt
from glob import glob
import sys
import os
import os.path as osp
import shutil
import pickle as pkl
import json
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

# Check if the GPU configuration has been provided
use_gpu = torch.cuda.device_count() > 0
multi_gpu = torch.cuda.device_count() > 1

if multi_gpu or use_gpu:
    print(f'Will use {torch.cuda.device_count()} gpu(s)')
else:
    print('Will use cpu')

# define the global base device
if use_gpu:
    device = torch.device('cuda:0')
    print("GPU model:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

t1 = time.time()
dataset = PFGraphDataset(device, '/particleflowvol/particleflow/data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/', 'cms')
t2 = time.time()
print('data', t2 - t1, 's')


t1 = time.time()
loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=2)
t2 = time.time()
print('loader', t2 - t1, 's')
