import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
import model_io
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy
import pickle, math, time
import _pickle as cPickle

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class LRP:
    EPSILON=1e-9

    def __init__(self,model:model_io):
        self.model=model

    def register_model(model:model_io):
        self.model=model

    """
    LRP rules
    """
    @staticmethod
    def eps_rule(layer, input, R, index, output_layer, activation_layer):

        EPSILON=1e-9
        a=copy_tensor(input)
        a.retain_grad()
        z = layer.forward(a)
        # basically layer.forward does this: output=(torch.matmul(a,torch.transpose(w,0,1))+b) , assuming the following w & b are retrieved

        if activation_layer:
            w = torch.eye(a.shape[1])
        else:
            w = layer.weight
            b = layer.bias

        wt = torch.transpose(w,0,1)

        if output_layer:
            R_list = [None]*R.shape[1]
            Wt = [None]*R.shape[1]
            for output_node in range(R.shape[1]):
                R_list[output_node]=(R[:,output_node].reshape(-1,1).clone())
                Wt[output_node]=(wt[:,output_node].reshape(-1,1))
        else:
            R_list = R
            Wt = [wt]*len(R_list)

        R_previous=[None]*len(R_list)
        for output_node in range(len(R_list)):
            # rep stands for repeated
            a_rep = a.reshape(a.shape[0],a.shape[1],1).expand(-1,-1,R_list[output_node].shape[1])
            wt_rep = Wt[output_node].reshape(1,Wt[output_node].shape[0],Wt[output_node].shape[1]).expand(a.shape[0],-1,-1)

            H = a_rep*wt_rep
            deno = H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).expand(-1,a.shape[1],-1).float()

            G = H/deno

            R_previous[output_node] = (torch.matmul(G, R_list[output_node].reshape(R_list[output_node].shape[0],R_list[output_node].shape[1],1).float()))
            R_previous[output_node] = R_previous[output_node].reshape(R_previous[output_node].shape[0], R_previous[output_node].shape[1])

            print('- Finished computing R-scores for output neuron #: ', output_node+1)

        print(f'- Completed layer: {layer}')
        if (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1))):
            print('- R score is conserved up to relative tolerance 1e-5')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-4)):
            print('- R score is conserved up to relative tolerance 1e-4')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-3)):
            print('- R score is conserved up to relative tolerance 1e-3')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-2)):
            print('- R score is conserved up to relative tolerance 1e-2')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-1)):
            print('- R score is conserved up to relative tolerance 1e-1')

        return R_previous

    """
    explanation functions
    """

    def explain(self,
                to_explain:dict,
                save:bool=True,
                save_to:str="./relevance.pt",
                sort_nodes_by:int=0,
                signal=torch.tensor([1,0,0,0,0,0],dtype=torch.float32).to(device),
                return_result:bool=False):

        start_index = self.model.n_layers                  ##########################
        print('Total number of layers (including activation layers):', start_index)

        ### loop over each single layer
        for index in range(start_index+1, 1, -1):
            print(f"Explaining layer {1+start_index+1-index}/{start_index+1-1}")
            if index==start_index+1:
                R = self.explain_single_layer(to_explain["pred"], to_explain, start_index+1, index)
            else:
                R = self.explain_single_layer(R, to_explain, start_index+1, index)

            with open(to_explain["outpath"]+'/'+to_explain["load_model"]+f'/R_score_layer{index}.pkl', 'wb') as f:
                cPickle.dump(R, f, protocol=4)

        print("Finished explaining all layers.")

    def explain_single_layer(self, R, to_explain, output_layer_index, index=None,name=None):

        # preparing variables required for computing LRP
        layer=self.model.get_layer(index=index,name=name)

        if name is None:
            name=self.model.index2name(index)
        if index is None:
            index=self.model.name2index(name)

        input=to_explain['A'][name]

        if index==output_layer_index:
            output_layer_bool=True
        else:
            output_layer_bool=False

        # backward pass with specified LRP rule
        if 'Linear' in str(layer):
            R = self.eps_rule(layer, input, R, index, output_layer_bool, activation_layer=False)
        elif 'LeakyReLU' or 'ELU' in str(layer):
            R = self.eps_rule(layer, input, R, index, output_layer_bool, activation_layer=True)

        return R

def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)


##-----------------------------------------------------------------------------
#
# arep=torch.transpose(a[0].repeat(6, 1),0,1)   # repeat it 6 times
# H=arep*wt
#
# G = H/H.sum(axis=0).float()
#
# Num = torch.matmul(G, R[0].float())
#
# print('Num.sum()', Num.sum())
#
# print(R[0].sum())
