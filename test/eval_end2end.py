#import setGPU
import torch
import torch_geometric
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
import pandas
import mplhep
import pickle

import graph_data
import train_end2end
import time

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=sorted(train_end2end.model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--path", type=str, help="path to model", default="data/PFNet7_TTbar_14TeV_TuneCUETP8M1_cfi_gen__npar_221073__cfg_ee19d91068__user_jovyan__ntrain_400__lr_0.0001__1588215695")
    parser.add_argument("--epoch", type=str, default="best", help="Epoch to use; could be 'last' or 'best'")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--start", type=int, default=3800, help="first file index to evaluate")
    parser.add_argument("--stop", type=int, default=4000, help="last file index to evaluate")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
   
    epoch = args.epoch
    model = args.model
    path = args.path
    weights = torch.load("{}/epoch_{}/weights.pth".format(path, epoch), map_location=device)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}

    with open('{}/model_kwargs.pkl'.format(path),'rb') as f:
        model_kwargs = pickle.load(f)
        
    model_class = train_end2end.model_classes[args.model]
    model = model_class(**model_kwargs)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    
    print(args.dataset)    
    full_dataset = graph_data.PFGraphDataset(root=args.dataset)
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.start, stop=args.stop))
    
    loader = DataListLoader(test_dataset, batch_size=1, pin_memory=False, shuffle=False)
    loader.collate_fn = collate
    
    big_df, edges_df = train_end2end.prepare_dataframe(model, loader, False, device)
    
    big_df.to_csv("{}/test.csv".format(path))
    edges_df.to_csv("{}/edges.csv".format(path))
    print(big_df)
    print(edges_df)
