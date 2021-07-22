from glob import glob
import sys, os
import os.path as osp
import pickle, math, time, numba, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib, mplhep
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Check if the GPU configuration has been provided
import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
        if multi_gpu:
            print('Will use multi_gpu..')
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print('Will use single_gpu..')
except Exception as e:
    print("Could not import setGPU, running CPU-only")

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch_geometric.nn import GravNetConv
from torch.utils.data import random_split
import torch_cluster

sys.path.insert(1, '../')
sys.path.insert(1, '../../../plotting/')
sys.path.insert(1, '../../../mlpf/plotting/')
import args
from args import parse_args
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd
import evaluate
from evaluate import make_plots, Evaluate
from plot_utils import plot_confusion_matrix
from model_LRP import PFNet7

from LRP import LRP
from model_io import model_io
import torch
import torch.nn as nn

activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = input[0]
    return hook

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)



  def forward(self, x):
    x = self.relu(self.conv(x))
    x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated
                                                                  #conv outside less than 0

    # print whether there is any negative grad
    s=x.register_hook(lambda grad: torch.zeros(grad.shape))
    return self.fc1(self.flatten(x))


net = myNet()
print(net)

for name, param in net.named_parameters():
  # if the param is from a linear and is a bias
  if "fc" in name and "bias" in name:
    param.register_hook(lambda grad: torch.zeros(grad.shape))


out = net(torch.randn(1,3,8,8))

(1 - out).mean().backward()

print("The biases are", net.fc1.bias.grad)     #bias grads are zero


print(s)
