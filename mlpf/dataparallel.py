from torch_geometric.data import Data, Batch
from pyg import parse_args
from pyg import PFGraphDataset, one_hot_embedding
from pyg import MLPF, training_loop, make_predictions, make_plots
from pyg import save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg import make_file_loaders

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import time
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


import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import time
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

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def training_loop():
    device = torch.device('cuda:0')

    dataset = PFGraphDataset('/particleflowvol/particleflow/data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/', 'cms')
    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=2))
    # construct file loaders
    file_loader = make_file_loaders(train_dataset)

    model = MLPF(input_dim=len(features_cms), num_classes=9)
    model.to(device)
    model = torch_geometric.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    t0, tt0 = time.time(), time.time()
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} is {round(time.time() - t0, 3)}s')

        file = [x for t in file for x in t]     # unpack the list of tuples to a list
        loader = DataListLoader(file, batch_size=4)

        for i, batch in enumerate(loader):
            print(f'batch # {i+1}/{len(loader)}')
            pred, target = model(batch)

            loss_clf = torch.nn.functional.cross_entropy(pred[:, :9], target['ygen_id'])  # for classifying PID

            optimizer.zero_grad()
            loss_clf.backward()
            optimizer.step()

        t0 = time.time()

    print(f'total time is {round(time.time()-tt0,3)}s')


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    training_loop()
