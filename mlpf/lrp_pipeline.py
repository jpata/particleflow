from pytorch_delphes import PFGraphDataset, dataloader_qcd, load_model
from lrp import MLPF, LRP_MLPF, make_Rmaps
import argparse
import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import numpy as np
import mplhep as hep
import pandas as pd

import torch
import torch_geometric
from torch_geometric.nn import GravNetConv

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch


# this script runs lrp on a trained MLPF model

parser = argparse.ArgumentParser()

# for saving the model
parser.add_argument("--dataset",        type=str,           default='../data/test_tmp_delphes/data/pythia8_qcd',   help="testing dataset path")
parser.add_argument("--outpath",        type=str,           default='../data/test_tmp_delphes/experiments/',       help="path to the trained model directory")
parser.add_argument("--load_model",     type=str,           default="",     help="Which model to load")
parser.add_argument("--load_epoch",     type=int,           default=0,      help="Which epoch of the model to load")
parser.add_argument("--out_neuron",     type=int,           default=0,      help="the output neuron you wish to explain")
parser.add_argument("--n_test",         type=int,           default=50,      help="number of data files to use for testing.. each file contains 100 events")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run lrp and make Rmaps
    python -u lrp_pipeline.py --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --out_neuron=0 --n_test=1
    """

    # Check if the GPU configuration and define the global base device
    if torch.cuda.device_count() > 0:
        print(f'Will use {torch.cuda.device_count()} gpu(s)')
        print("GPU model:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print('Will use cpu')
        device = torch.device('cpu')

    # get sample dataset
    print('Fetching the data..')
    full_dataset_qcd = PFGraphDataset(args.dataset)
    loader = dataloader_qcd(full_dataset_qcd, multi_gpu=False, n_test=args.n_test, batch_size=1)

    # load a pretrained model
    state_dict, model_kwargs, outpath = load_model(device, args.outpath, args.load_model, args.load_epoch)
    model = MLPF(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)

    # run lrp
    Rtensors_list, preds_list, inputs_list = [], [], []

    for i, event in enumerate(loader):
        print(f'Explaining event # {i}')

        # run lrp on sample model
        model.eval()
        lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
        Rtensor, pred, input = lrp_instance.explain(event, neuron_to_explain=args.out_neuron)

        Rtensors_list.append(Rtensor.detach().to('cpu'))
        preds_list.append(pred.detach().to('cpu'))
        inputs_list.append(input.detach().to('cpu').to_dict())

    with open(f'{args.outpath}/Rtensors_list.pkl', 'wb') as f:
        pkl.dump(Rtensors_list, f)
    with open(f'{args.outpath}/preds_list.pkl', 'wb') as f:
        pkl.dump(preds_list, f)
    with open(f'{args.outpath}/inputs_list.pkl', 'wb') as f:
        pkl.dump(inputs_list, f)
        break
    make_Rmaps(Rtensors_list, pid='chhadron', neighbors=3)
