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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args): 

    full_dataset = PFGraphDataset(root='graph_data/')
    
    data = full_dataset.get(0)

    features = ["clusters_energy",
                "clusers_eta",
                "clusters_phi"]
    feature_scale = np.array([1., 1., 1.])
    df = pd.DataFrame(data.x.cpu().detach().numpy()*feature_scale, columns=features)

    mask = np.abs(df['clusters_energy'])>0
    df = df[mask]
    row, col = data.edge_index.cpu().detach().numpy()    
    y_truth = data.y.cpu().detach().numpy()
    print(sum(y_truth))

    for x,y in [('clusers_eta', 'clusters_phi')]:
        plt.figure()       
        k = 0
        for i, j in tqdm.tqdm(zip(row, col)):
            seg_args = dict(c='b',alpha=0.1,zorder=1)
            plt.plot([df[x][i], df[x][j]],
                     [df[y][i], df[y][j]], '-', **seg_args)
            if y_truth[k]:
                seg_args = dict(c='r',alpha=1,zorder=2)
                plt.plot([df[x][i], df[x][j]],
                         [df[y][i], df[y][j]], '-', **seg_args)
            k+=1
        plt.scatter(df[x], df[y],c='k',marker='o',s=4,zorder=3,alpha=1)
        plt.xlabel("%s"%x)
        plt.ylabel("%s"%y)
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig('graph_%s_%s.pdf'%(x,y))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
