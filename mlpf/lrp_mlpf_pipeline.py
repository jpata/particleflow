import argparse
import os
import os.path as osp
import pickle as pkl
import sys

import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch_geometric
from lrp import LRP_MLPF, MLPF, make_Rmaps
from pyg import PFGraphDataset, dataloader_qcd, load_model
from torch_geometric.data import Batch, Data, DataListLoader, DataLoader

# this script runs lrp on a trained MLPF model

parser = argparse.ArgumentParser()

# for saving the model
parser.add_argument("--dataset_qcd", type=str, default="../data/delphes/pythia8_qcd", help="testing dataset path")
parser.add_argument("--outpath", type=str, default="../experiments/", help="path to the trained model directory")
parser.add_argument("--load_model", type=str, default="", help="Which model to load")
parser.add_argument("--load_epoch", type=int, default=0, help="Which epoch of the model to load")
parser.add_argument("--out_neuron", type=int, default=0, help="the output neuron you wish to explain")
parser.add_argument("--pid", type=str, default="chhadron", help="Which model to load")
parser.add_argument(
    "--n_test", type=int, default=50, help="number of data files to use for testing.. each file contains 100 events"
)
parser.add_argument("--run_lrp", dest="run_lrp", action="store_true", help="runs lrp")
parser.add_argument("--make_rmaps", dest="make_rmaps", action="store_true", help="makes rmaps")

args = parser.parse_args()


if __name__ == "__main__":

    if args.run_lrp:
        # Check if the GPU configuration and define the global base device
        if torch.cuda.device_count() > 0:
            print(f"Will use {torch.cuda.device_count()} gpu(s)")
            print("GPU model:", torch.cuda.get_device_name(0))
            device = torch.device("cuda:0")
        else:
            print("Will use cpu")
            device = torch.device("cpu")

        # get sample dataset
        print("Fetching the data..")
        full_dataset_qcd = PFGraphDataset(args.dataset_qcd)
        loader = dataloader_qcd(full_dataset_qcd, multi_gpu=False, n_test=args.n_test, batch_size=1)

        # load a pretrained model and update the outpath
        outpath = args.outpath + args.load_model
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.load_model, args.load_epoch)
        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # initialize placeholders for Rscores, the event inputs, and the event predictions
        Rtensors_list, preds_list, inputs_list = [], [], []

        # define the lrp instance
        lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)

        # loop over events to explain them
        for i, event in enumerate(loader):
            print(f"Explaining event # {i}")

            # run lrp on the event
            Rtensor, pred, input = lrp_instance.explain(event, neuron_to_explain=args.out_neuron)

            # store the Rscores, the event inputs, and the event predictions
            Rtensors_list.append(Rtensor.detach().to("cpu"))
            preds_list.append(pred.detach().to("cpu"))
            inputs_list.append(input.detach().to("cpu").to_dict())

        with open(f"{outpath}/Rtensors_list.pkl", "wb") as f:
            pkl.dump(Rtensors_list, f)
        with open(f"{outpath}/inputs_list.pkl", "wb") as f:
            pkl.dump(inputs_list, f)
        with open(f"{outpath}/preds_list.pkl", "wb") as f:
            pkl.dump(preds_list, f)

    if args.make_rmaps:
        outpath = args.outpath + args.load_model
        with open(f"{outpath}/Rtensors_list.pkl", "rb") as f:
            Rtensors_list = pkl.load(f)
        with open(f"{outpath}/inputs_list.pkl", "rb") as f:
            inputs_list = pkl.load(f)
        with open(f"{outpath}/preds_list.pkl", "rb") as f:
            preds_list = pkl.load(f)

        print("Making Rmaps..")
        make_Rmaps(
            args.outpath, Rtensors_list, inputs_list, preds_list, pid=args.pid, neighbors=3, out_neuron=args.out_neuron
        )
