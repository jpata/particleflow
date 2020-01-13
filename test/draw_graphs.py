import setGPU
import torch
from torch_geometric.data import Data, DataLoader
from graph_data import PFGraphDataset
from models import EdgeNet
import os
import os.path as osp
import math
import numpy as np
import tqdm
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from models import EdgeNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args): 

    full_dataset = PFGraphDataset(root='/storage/user/jduarte/particleflow/graph_data/')
    
    data = full_dataset.get(1)

    features = ["eta",
                "phi"]

    df = pd.DataFrame(data.x.cpu().detach().numpy()[:,2:4], columns=features)

    row, col = data.edge_index.cpu().detach().numpy()    
    y_truth = data.y.cpu().detach().numpy()
    
    
    input_dim = data.x.shape[1]
    hidden_dim = 32
    n_iters = 1
    model = EdgeNet(input_dim=input_dim,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    modpath = 'EdgeNet_13873_10e465f628_jduarte.best.pth'
    model.load_state_dict(torch.load(modpath))
    data = data.to(device)
    output = model(data)
    print(data.y)
    print(output)

    max_phi = 1.5
    max_eta = 1.5
    extra = 0.5
    for plot_type in [['input', 'truth'], 
                      ['input', 'output'], 
                      ['truth', 'output'],
                      ['input'],
                      ['output'],
                      ['truth']]:
        x = 'eta'
        y = 'phi'
        plt.figure()       
        k = 0
        for i, j in tqdm.tqdm(zip(row, col),total=len(y_truth)):
            x1 = df[x][i]
            x2 = df[x][j]
            y1 = df[y][i]
            y2 = df[y][j]
            if np.abs(x1) > max_eta+extra or np.abs(x2) > max_eta+extra: continue
            if np.abs(y1) > max_phi+extra or np.abs(y2) > max_phi+extra: continue
            if 'input' in plot_type:
                seg_args = dict(c='b',alpha=0.1,zorder=1)
                plt.plot([df[x][i], df[x][j]],
                         [df[y][i], df[y][j]], '-', **seg_args)
            if 'truth' in plot_type and y_truth[k]:
                seg_args = dict(c='r',alpha=0.5,zorder=2)
                plt.plot([df[x][i], df[x][j]],
                         [df[y][i], df[y][j]], '-', **seg_args)
            if 'output' in plot_type:
                seg_args = dict(c='g',alpha=output[k].item(),zorder=3)
                plt.plot([df[x][i], df[x][j]],
                         [df[y][i], df[y][j]], '-', **seg_args)
            k+=1
        plt.scatter(df[x][(np.abs(df[x])<max_eta+extra) & (np.abs(df[y])<max_eta+extra)], df[y][(np.abs(df[x])<max_eta+extra) & (np.abs(df[y])<max_phi+extra)],c='k',marker='o',s=4,zorder=4,alpha=1)
        plt.xlabel("%s"%x)
        plt.ylabel("%s"%y)
        plt.xlim(-1*max_eta, max_eta)
        plt.ylim(-1*max_phi, max_phi)
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig('graph_%s_%s_%s.pdf'%(x,y,'_'.join(plot_type)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
