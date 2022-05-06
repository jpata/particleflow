import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU


class LRP_MLPF():

    """
    A class that act on graph datasets and GNNs based on the Gravnet layer (e.g. the MLPF model)
    The main trick is to realize that the ".lin_s" layers in Gravnet are irrelevant for explanations so shall be skipped
    The hack, however, is to substitute them precisely with the message_passing step

    Differences from standard LRP
        - Rscores become tensors/graphs of input features per output neuron instead of vectors
        - accomodates message passing steps by using the adjacency matrix as the weight matrix in standard LRP,
          and redistributing Rscores over the other dimension (over nodes instead of features)
    """

    def __init__(self, device, model, epsilon):

        self.device = device
        self.model = model.to(device)
        self.epsilon = epsilon  # for stability reasons in the lrp-epsilon rule (by default: a very small number)

        # check if the model has any skip connections to accomodate them
        self.skip_connections = self.find_skip_connections()
        self.msg_passing_layers = self.find_msg_passing_layers()

    """
    explanation functions
    """

    def explain(self, input, neuron_to_explain):
        """
        Primary function to call on an LRP instance to start explaining predictions.
        First, it registers hooks and runs a forward pass on the input.
        Then, it attempts to explain the whole model by looping over the layers in the model and invoking the explain_single_layer function.

        Args:
            input: tensor containing the input sample you wish to explain
            neuron_to_explain: the index for a particular neuron in the output layer you wish to explain

        Returns:
            R_tensor: a tensor/graph containing the relevance scores of the input graph for a particular output neuron
            preds: the model predictions of the input (for further plotting/processing purposes only)
            input: the input that was explained (for further plotting/processing purposes only)
        """

        # register forward hooks to retrieve intermediate activations
        # in simple words, when the forward pass is called, the following dict() will be filled with (key, value) = ("layer_name", activations)
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0]
            return hook

        for name, module in self.model.named_modules():
            # unfold any containers so as to register hooks only for their child modules (equivalently we are demanding type(module) != nn.Sequential))
            if ('Linear' in str(type(module))) or ('activation' in str(type(module))) or ('BatchNorm1d' in str(type(module))):
                module.register_forward_hook(get_activation(name))

        # run a forward pass
        self.model.eval()
        preds, self.A, self.msg_activations = self.model(input.to(self.device))

        # get the activations
        self.activations = activations
        self.num_layers = len(activations.keys())
        self.in_features_dim = self.name2layer(list(activations.keys())[0]).in_features

        print(f'Total number of layers: {self.num_layers}')

        # initialize Rscores for skip connections (in case there are any)
        if len(self.skip_connections) != 0:
            self.skip_connections_relevance = 0

        # initialize the Rscores tensor using the output predictions
        Rscores = preds[:, neuron_to_explain].reshape(-1, 1).detach()

        # build the Rtensor which is going to be a whole graph of Rscores per node
        R_tensor = torch.zeros([Rscores.shape[0], Rscores.shape[0], Rscores.shape[1]]).to(self.device)
        for node in range(R_tensor.shape[0]):
            R_tensor[node][node] = Rscores[node]

        # loop over layers in the model to propagate Rscores backward
        for layer_index in range(self.num_layers, 0, -1):
            R_tensor = self.explain_single_layer(R_tensor, layer_index, neuron_to_explain)

        print("Finished explaining all layers.")

        if len(self.skip_connections) != 0:
            return R_tensor + self.skip_connections_relevance, preds, input

        return R_tensor, preds, input

    def explain_single_layer(self, R_tensor_old, layer_index, neuron_to_explain):
        """
        Attempts to explain a single layer in the model by propagating Rscores backwards using the lrp-epsilon rule.

        Args:
            R_tensor_old: a tensor/graph containing the Rscores, of the current layer, to be propagated backwards
            layer_index: index that corresponds to the position of the layer in the model (see helper functions)
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            R_tensor_new: a tensor/graph containing the computed Rscores of the previous layer
        """

        # get layer information
        layer_name = self.index2name(layer_index)
        layer = self.name2layer(layer_name)

        # get layer activations (depends wether it's a message passing step)
        if layer_name in self.msg_passing_layers.keys():
            print(f"Explaining layer {self.num_layers+1-layer_index}/{self.num_layers}: MessagePassing layer")
            input = self.msg_activations[layer_name[:-6]].to(self.device).detach()
            msg_passing_layer = True
        else:
            print(f"Explaining layer {self.num_layers+1-layer_index}/{self.num_layers}: {layer}")
            input = self.activations[layer_name].to(self.device).detach()
            msg_passing_layer = False

        # run lrp
        if 'Linear' in str(layer):
            R_tensor_new = self.eps_rule(self, layer, layer_name, input, R_tensor_old, neuron_to_explain, msg_passing_layer)
            print('- Finished computing Rscores')
            return R_tensor_new
        else:
            if 'activation' in str(layer):
                print(f"- skipping layer because it's an activation layer")
            elif 'BatchNorm1d' in str(layer):
                print(f"- skipping layer because it's a BatchNorm layer")
            print(f"- Rscores do not need to be computed")
            return R_tensor_old

    """
    lrp-epsilon rule
    """

    @staticmethod
    def eps_rule(self, layer, layer_name, x, R_tensor_old, neuron_to_explain, msg_passing_layer):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.

        Can accomodate message_passing layers if the adjacency matrix and the activations before the message_passing are provided.
        The trick (or as we like to call it, the message_passing hack) is in
            (1) using the adjacency matrix as the weight matrix in the standard lrp rule
            (2) transposing the activations to distribute the Rscores over the other dimension (over nodes instead of features)

        Args:
            layer: a torch.nn module with a corresponding weight matrix W
            x: vector containing the activations of the previous layer
            R_tensor_old: a tensor/graph containing the Rscores, of the current layer, to be propagated backwards
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            R_tensor_new: a tensor/graph containing the computed Rscores of the previous layer
        """

        torch.cuda.empty_cache()

        if msg_passing_layer:   # message_passing hack
            x = torch.transpose(x, 0, 1)               # transpose the activations to distribute the Rscores over the other dimension (over nodes instead of features)
            W = self.A[layer_name[:-6]].detach().to(self.device)       # use the adjacency matrix as the weight matrix
        else:
            W = layer.weight.detach()  # get weight matrix
            W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

        # for the output layer, pick the part of the weight matrix connecting only to the neuron you're attempting to explain
        if layer == list(self.model.modules())[-1]:
            W = W[:, neuron_to_explain].reshape(-1, 1)

        # (1) compute the denominator
        denominator = torch.matmul(x, W) + self.epsilon
        # (2) scale the Rscores
        if msg_passing_layer:  # message_passing hack
            R_tensor_old = torch.transpose(R_tensor_old, 1, 2)
        scaledR = R_tensor_old / denominator
        # (3) compute the new Rscores
        R_tensor_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * x

        # checking conservation of Rscores for a given random node (# 17)
        rtol = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for tol in rtol:
            if (torch.allclose(R_tensor_new[17].sum(), R_tensor_old[17].sum(), rtol=tol)):
                print(f'- Rscores are conserved up to relative tolerance {str(tol)}')
                break

        if layer in self.skip_connections:
            # set aside the relevance of the input_features in the skip connection
            # recall: it is assumed that the skip connections are defined in the following order torch.cat[(input_features, ...)] )
            self.skip_connections_relevance = self.skip_connections_relevance + R_tensor_new[:, :, :self.in_features_dim]
            return R_tensor_new[:, :, self.in_features_dim:]

        if msg_passing_layer:  # message_passing hack
            return torch.transpose(R_tensor_new, 1, 2)

        return R_tensor_new

    """
    helper functions
    """

    def index2name(self, layer_index):
        """
        Given the index of a layer (e.g. 3) returns the name of the layer (e.g. .nn1.3)
        """
        layer_name = list(self.activations.keys())[layer_index - 1]
        return layer_name

    def name2layer(self, layer_name):
        """
        Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
        """
        for name, module in self.model.named_modules():
            if layer_name == name:
                return module

    def find_skip_connections(self):
        """
        Given a torch model, retuns a list of layers with skip connections... the elements are torch modules (e.g. Linear(...))
        """
        explainable_layers = []
        for name, module in self.model.named_modules():
            if 'lin_s' in name:     # for models that are based on Gravnet, skip the lin_s layers
                continue
            if ('Linear' in str(type(module))):
                explainable_layers.append(module)

        skip_connections = []
        for layer_index in range(len(explainable_layers) - 1):
            if explainable_layers[layer_index].out_features != explainable_layers[layer_index + 1].in_features:
                skip_connections.append(explainable_layers[layer_index + 1])

        return skip_connections

    def find_msg_passing_layers(self):
        """
        Returns a list of ".lin_s" layers from model.named_modules() that shall be substituted with message passing
        """
        msg_passing_layers = {}
        for name, module in self.model.named_modules():
            if 'lin_s' in name:     # for models that are based on Gravnet, replace the .lin_s layers with message_passing
                msg_passing_layers[name] = {}

        return msg_passing_layers
