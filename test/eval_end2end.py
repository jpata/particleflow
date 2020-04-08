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

def collate(batch):
    return batch
        
if __name__ == "__main__":
    device = torch.device("cuda")
   
    epoch = 50 
    weights = torch.load("data/PFNet7_TTbar_14TeV_TuneCUETP8M1_cfi_gen__npar_2922514__cfg_0acd46a859__user_jpata__ntrain_100__lr_0.0005__1586303540/epoch_{}/weights.pth".format(epoch))
    
    model = train_end2end.PFNet7(23, 512, 7, dropout_rate=0.2)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    for dataset, start, stop in [
        #("test/SinglePiFlatPt0p7To10_cfi", 900, 1000),
        #("test/SingleElectronFlatPt1To100_pythia8_cfi", 0, 100),
        #("test/SingleGammaFlatPt10To100_pythia8_cfi", 0, 100),
        ("test/TTbar_14TeV_TuneCUETP8M1_cfi", 100, 110),
        ]:
        print(dataset)    
        full_dataset = graph_data.PFGraphDataset(root=dataset)
        test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=start, stop=stop))
        
        loader = DataLoader(test_dataset, batch_size=None, batch_sampler=None, pin_memory=False, shuffle=False)
        loader.collate_fn = collate

        big_df = train_end2end.prepare_dataframe(model, loader)
   
        big_df.to_pickle("{}.pkl.bz2".format(dataset))
        print(big_df[big_df["cand_pid"]!=0].head())
