import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy
import pickle, math, time
import _pickle as cPickle
from sys import getsizeof
from tqdm import tqdm

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj

import lrp

class lrp_clf:
    EPSILON=1e-9

    def __init__(self, device, model:lrp.model_io):
        self.device=device
        self.model=model

    def register_model(model:lrp.model_io):
        self.model=model

    """
    lrp rules
    """

    # this rule is wrong.. it is just here because it is much quicker for experimentation and gives the correct dimensions needed for debugging (if you haven't hit the message passing step)
    @staticmethod
    def easy_rule(self, layer, input, R, index,output_layer, activation_layer, print_statement):
        EPSILON=1e-9
        # input.retain_grad()
        # z = layer.forward(input)
        # basically layer.forward does this: output=(torch.matmul(input,torch.transpose(w,0,1))+b) , assuming the following w & b are retrieved

        if activation_layer:
            w = torch.eye(input.shape[1]).to(self.device)
        else:
            w = layer.weight.detach().to(self.device)

        if output_layer: # for the output layer
            T, W, r = [], [], []

            for i in range(R.shape[1]):
                T.append(R[:,i].reshape(-1,1).to(self.device))
                W.append(w[i,:].reshape(1,-1).to(self.device))
                I = torch.ones_like(R[:,i]).reshape(-1,1).to(self.device)

                Numerator = (input*torch.matmul(T[i],W[i]))
                Denominator = (input*torch.matmul(I,W[i])).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])
                r.append(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

            print('- Finished computing R-scores')
            return r
        else:
            for i in range(len(R)):
                I = torch.ones_like(R[i])

                Numerator = (input*torch.matmul(R[i],w))
                Denominator = (input*torch.matmul(I,w)).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])
                R[i]=(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

            print('- Finished computing R-scores')
            return R


    @staticmethod
    def eps_rule(self, layer, input, R, index, output_layer, activation_layer, print_statement, adjacency_matrix=None, message_passing=False):

        if activation_layer:
            w = torch.eye(input.shape[1]).detach().to(self.device)
        elif message_passing: # message passing hack
            w = adjacency_matrix.detach().to(self.device)
        else:
            w = layer.weight.detach().to(self.device)

        wt = torch.transpose(w,0,1)

        if output_layer:
            R_list = [None]*R.shape[1]
            Wt = [None]*R.shape[1]
            for output_neuron in range(R.shape[1]):
                R_list[output_neuron] = (R[:,output_neuron].reshape(-1,1).clone())
                Wt[output_neuron] = (wt[:,output_neuron].reshape(-1,1))
        else:
            R_list = R
            Wt = [wt]*len(R_list)

        R_previous=[None]*len(R_list)

        for output_neuron in range(len(R_list)):

            if message_passing: # message passing hack
                R_list[output_neuron] = torch.transpose(R_list[output_neuron],0,1)

            # rep stands for repeated/expanded
            a_rep = input.reshape(input.shape[0],input.shape[1],1).expand(-1,-1,R_list[output_neuron].shape[1]).to(self.device)
            wt_rep = Wt[output_neuron].reshape(1,Wt[output_neuron].shape[0],Wt[output_neuron].shape[1]).expand(input.shape[0],-1,-1).to(self.device)

            H = a_rep*wt_rep
            deno = H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).expand(-1,input.shape[1],-1)

            G = H/deno

            R_previous[output_neuron] = (torch.matmul(G, R_list[output_neuron].reshape(R_list[output_neuron].shape[0],R_list[output_neuron].shape[1],1).to(self.device)))
            R_previous[output_neuron] = R_previous[output_neuron].reshape(R_previous[output_neuron].shape[0], R_previous[output_neuron].shape[1]).to('cpu')

            if message_passing: # message passing hack
                R_previous[output_neuron] = torch.transpose(R_previous[output_neuron],0,1)

        if print_statement:
            print('- Finished computing R-scores')
            if message_passing:
                if (torch.allclose(torch.transpose(R_previous[output_neuron],0,1).sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1))):
                    print('- R score is conserved up to relative tolerance 1e-5')
                elif (torch.allclose(torch.transpose(R_previous[output_neuron],0,1).sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-4)):
                    print('- R score is conserved up to relative tolerance 1e-4')
                elif (torch.allclose(torch.transpose(R_previous[output_neuron],0,1).sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-3)):
                    print('- R score is conserved up to relative tolerance 1e-3')
                elif (torch.allclose(torch.transpose(R_previous[output_neuron],0,1).sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-2)):
                    print('- R score is conserved up to relative tolerance 1e-2')
                elif (torch.allclose(torch.transpose(R_previous[output_neuron],0,1).sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-1)):
                    print('- R score is conserved up to relative tolerance 1e-1')
            else:
                if (torch.allclose(R_previous[output_neuron].sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1))):
                    print('- R score is conserved up to relative tolerance 1e-5')
                elif (torch.allclose(R_previous[output_neuron].sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-4)):
                    print('- R score is conserved up to relative tolerance 1e-4')
                elif (torch.allclose(R_previous[output_neuron].sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-3)):
                    print('- R score is conserved up to relative tolerance 1e-3')
                elif (torch.allclose(R_previous[output_neuron].sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-2)):
                    print('- R score is conserved up to relative tolerance 1e-2')
                elif (torch.allclose(R_previous[output_neuron].sum(axis=1), R_list[output_neuron].to('cpu').sum(axis=1), rtol=1e-1)):
                    print('- R score is conserved up to relative tolerance 1e-1')

        return R_previous

    @staticmethod
    def message_passing_rule(self, layer, input, R, big_list, edge_index, edge_weight, after_message, before_message, index, outpath, load_model):

        # first time you hit message passing: construct and start filling the big tensor from scratch
        if len(big_list)==0:
            big_list = [[torch.zeros(R[0].shape[0],R[0].shape[1]) for i in range(len(R))] for i in range(R[0].shape[0])]
            print('- Finished allocating memory for the big tensor of R-scores for all nodes')

            for node_i in range(len(big_list)):
                for output_neuron in range(len(big_list[0])):
                    big_list[node_i][output_neuron][node_i] = R[output_neuron][node_i]
            print('- Finished initializing the big tensor')

        # build the adjacency matrix
        A = to_dense_adj(edge_index, edge_attr=edge_weight)[0] # adjacency matrix

        if torch.allclose(torch.matmul(A, before_message), after_message, rtol=1e-3):
            print("- Adjacency matrix is correctly computed")

        # # the following saves a version of the R-scores before the message passing
        # torch.save(big_list, outpath + '/lrp/R_score_layer_before_msg_passing.pt')

        # modify the big tensor based on message passing rule
        for node_i in tqdm(range(len(big_list))):
            big_list[node_i] = self.eps_rule(self, layer, torch.transpose(before_message,0,1), big_list[node_i], index, output_layer=False, activation_layer=False, print_statement=True, adjacency_matrix=A, message_passing=True)
            print(f'- Finished computing R-score for node {node_i+1}/{len(big_list)} for the message passing..')
        print('- Finished computing R-scores for the message passing layer')
        return big_list


    """
    explanation functions
    """

    def explain(self, to_explain):

        start_index = self.model.n_layers                  ##########################
        outpath = to_explain["outpath"]+'/'+to_explain["load_model"]

        print('Total number of layers (including activation layers):', start_index)

        # store the R-scores for the output layer (they are basically the model predictions)
        torch.save(to_explain["pred_id"].detach(), outpath + f'/lrp/R_score_layer{start_index+1}.pt')

        ### loop over each single layer
        big_list = []
        output_layer_index = start_index+1
        for index in range(start_index+1, 1,-1):
            if index==start_index+1:
                R, big_list, output_layer_index  = self.explain_single_layer(to_explain["pred_id"].detach(), to_explain, big_list, outpath, output_layer_index, index)
            else:
                R, big_list, output_layer_index  = self.explain_single_layer(R, to_explain, big_list, outpath, output_layer_index, index)
        print("Finished explaining all layers.")
        return big_list      # returns the heatmaps for layer0 (i.e. input features)


    def explain_single_layer(self, R, to_explain, big_list, outpath, output_layer_index, index=None, name=None):

        layer = self.model.get_layer(index=index,name=name)

        if name is None:
            name = self.model.index2name(index)
        if index is None:
            index = self.model.name2index(name)

        input = to_explain['A'][name].detach()

        # skip the last DNN because this is only for classification
        if 'nn3' in str(name):
            print('Skipping a DNN3 regression layer..')
            output_layer_index = output_layer_index-1
            return R, big_list, output_layer_index

        if index==output_layer_index:
            output_layer_bool = True
        else:
            output_layer_bool = False

        # it works out of the box that the conv1.lin_s layer which we don't care about is in the same place of the message passing.. so we can just replace its action
        if 'conv1.lin_s' in str(name):
            print(f"Explaining layer {output_layer_index+1-index}/{output_layer_index-1}: Message Passing")
            big_list = self.message_passing_rule(self, layer, input, R, big_list, to_explain["edge_index"].detach(), to_explain["edge_weight"].detach(), to_explain["after_message"].detach(), to_explain["before_message"].detach(), index, outpath, to_explain["load_model"])
            return R, big_list, output_layer_index

        print(f"Explaining layer {output_layer_index+1-index}/{output_layer_index-1}: {layer}")

        if len(big_list)==0:  # if you haven't hit the message passing step yet
            if 'Linear' in str(layer):
                R = self.eps_rule(self, layer, input, R, index, output_layer_bool, activation_layer=False, print_statement=True)
            elif 'LeakyReLU' or 'ELU' in str(layer):
                R = self.eps_rule(self, layer, input, R, index, output_layer_bool, activation_layer=True, print_statement=True)
        else:
            for node_i in tqdm(range(len(big_list))):
                if 'Linear' in str(layer):
                    big_list[node_i] = self.eps_rule(self, layer, input, big_list[node_i], index, output_layer_bool, activation_layer=False, print_statement=False)
                elif 'LeakyReLU' or 'ELU' in str(layer):
                    big_list[node_i] =  self.eps_rule(self, layer, input, big_list[node_i], index, output_layer_bool, activation_layer=True, print_statement=False)
        return R, big_list, output_layer_index

##-----------------------------------------------------------------------------
# # big_list is a list of length 5k
# # each element is another list of length 6 (corresponding to each of the output pid probability prediction)
# # each element of that second list is a tensor of shape (5k,x) where x is the dimension of the latent space
