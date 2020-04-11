import setGPU
import torch
import torch_geometric
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import pandas
import mplhep
import pickle

import graph_data
import train_end2end
import time

def collate(batch):
    print(batch)
    return batch
        
if __name__ == "__main__":
    device = torch.device("cuda")
   
    epoch = 40 
    model = "PFNet6_TTbar_14TeV_TuneCUETP8M1_cfi_gen__npar_425491__cfg_36ec608897__user_jpata__ntrain_100__lr_0.0002__1586359112"
    weights = torch.load("data/{}/epoch_{}/weights.pth".format(model, epoch))
    
    model = train_end2end.PFNet6(23, 128, 7, dropout_rate=0.2)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    for dataset, start, stop in [
        #("test/SinglePiFlatPt0p7To10_cfi", 900, 1000),
        #("test/SingleElectronFlatPt1To100_pythia8_cfi", 0, 100),
        #("test/SingleGammaFlatPt10To100_pythia8_cfi", 0, 100),
        ("test/TTbar_14TeV_TuneCUETP8M1_cfi", 9000, 10000),
        ]:
        print(dataset)    
        full_dataset = graph_data.PFGraphDataset(root=dataset)
        test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=start, stop=stop))
        
        loader = DataLoader(test_dataset, batch_size=None, batch_sampler=None, pin_memory=False, shuffle=False)
        loader.collate_fn = collate

        big_df = train_end2end.prepare_dataframe(model, loader)
   
        big_df.to_pickle("{}.pkl.bz2".format(dataset))
        print(big_df[big_df["cand_pid"]!=1].head())
