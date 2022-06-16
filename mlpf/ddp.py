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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def training_loop(rank, world_size):
    print(f"Running training_loop DDP example on rank {rank}.")
    setup(rank, world_size)

    dataset = PFGraphDataset('/particleflowvol/particleflow/data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/', 'cms')
    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=rank * 5, stop=rank * 5 + 5))
    # construct file loaders
    loader = make_file_loaders(train_dataset, num_workers=2, prefetch_factor=10)

    # create model and move it to GPU with id rank
    model = MLPF(input_dim=len(features_cms), num_classes=9).to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    t0, tt0 = time.time(), time.time()
    for num, file in enumerate(loader):
        print(f'Time to load file {num+1}/{len(loader)} is {round(time.time() - t0, 3)}s')

        file = [x for t in file for x in t]     # unpack the list of tuples to a list
        loader = DataLoader(file, batch_size=50)

        t = 0
        for i, batch in enumerate(loader):
            tb = time.time()
            pred, target = ddp_model(batch.to(rank))
            print(f'batch {i}/{len(loader)}, forward pass = {round(time.time() - tb, 3)}s')
            t = t + (time.time() - tb)

            loss_clf = torch.nn.functional.cross_entropy(pred[:, :9], target['ygen_id'])  # for classifying PID

            optimizer.zero_grad()
            loss_clf.backward()
            optimizer.step()

        print(f'Average inference time per batch is {round((t / len(loader)), 3)}s')

        t0 = time.time()

    print(f'total time is {round(time.time()-tt0,3)}s')

    cleanup()


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    run_demo(training_loop, world_size)
    # run_demo(demo_checkpoint, world_size)
