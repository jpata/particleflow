from glob import glob
import sys, os
import os.path as osp
import pickle as pkl
import _pickle as cPickle
import math, time, numba, tqdm
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
    print("GPU model:", torch.cuda.get_device_name(0))
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
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj

sys.path.insert(1, '../')
sys.path.insert(1, '../../../plotting/')
sys.path.insert(1, '../../../mlpf/plotting/')

import args
from args import parse_args
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd

from model_LRP_reg import PFNet7
from LRP_clf_gpu import LRP_clf
from LRP_reg_gpu import LRP_reg

from model_io import model_io

import networkx as nx
from torch_geometric.utils.convert import to_networkx

# NOTE: this script works by loading an already trained model

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size, task, title):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr,
        task,
        title)
    return model_fname

def map_index_to_pid(id):
    if id==0:
        return 'null'
    if id==1:
        return 'charged hadron'
    if id==2:
        return 'neutral hadron'
    if id==3:
        return 'photon'
    if id==4:
        return 'electron'
    if id==5:
        return 'muon'

def map_index_to_p4(index):
    if index==0:
        return 'charge'
    if index==1:
        return 'pt'
    if index==2:
        return 'eta'
    if index==3:
        return 'sin phi'
    if index==4:
        return 'cos phi'
    if index==5:
        return 'energy'

def make_heatmaps(big_list, to_explain, task):

    print(f'Making heatmaps for {task}..')

    X = to_explain["inputs"]
    gen_ids_one_hot = to_explain["gen_id"]
    pred_ids_one_hot = to_explain["pred_id"]

    gen_ids = gen_ids_one_hot.argmax(axis=1)
    pred_ids = pred_ids_one_hot.argmax(axis=1)

    # make directories to hold the heatmaps
    for i in range(6):
        if not osp.isdir(outpath + f'/class{str(i)}'):
            os.makedirs(outpath + f'/class{str(i)}')
        for j in range(6):
            if task=='regression':
                if not osp.isdir(outpath + f'/class{str(i)}'+f'/p4_elem{str(j)}'):
                    os.makedirs(outpath + f'/class{str(i)}'+f'/p4_elem{str(j)}')
            elif task=='classification':
                if not osp.isdir(outpath + f'/class{str(i)}'+f'/pid{str(j)}'):
                    os.makedirs(outpath + f'/class{str(i)}'+f'/pid{str(j)}')

    # attempt to break down big_list onto 6 smaller lists, 1 for each pid
    list0, list1, list2, list3, list4, list5 = [], [], [], [], [], []
    dist0, dist1, dist2, dist3, dist4, dist5 = [], [], [], [], [], []

    for node_i in range(len(big_list)):  # iterate over the nodes

        if gen_ids[node_i]==0:  # if it's a null then add it to the null list
            list0.append(big_list[node_i])
            dist0.append(node_i)
        if gen_ids[node_i]==1:  # if it's a chhadron then add it to the chhadron list
            list1.append(big_list[node_i])
            dist1.append(node_i)
        if gen_ids[node_i]==2:  # if it's a nhadron then add it to the nhadron list
            list2.append(big_list[node_i])
            dist2.append(node_i)
        if gen_ids[node_i]==3:  # if it's a photon then add it to the photon list
            list3.append(big_list[node_i])
            dist3.append(node_i)
        if gen_ids[node_i]==4:  # if it's a electron then add it to the electron list
            list4.append(big_list[node_i])
            dist4.append(node_i)
        if gen_ids[node_i]==5:  # if it's a muon then add it to the muon list
            list5.append(big_list[node_i])
            dist5.append(node_i)

    list = [list0,list1,list2,list3,list4,list5]
    dist = [dist0,dist1,dist2,dist3,dist4,dist5]

    if task=='regression':
        output_dim = output_dim_p4
    elif task=='classification':
        output_dim = output_dim_id

    for pid in range(output_dim_id):
        for node_i in range(len(list[pid])): # iterate over the nodes in each list
            print('- making heatmap for', map_index_to_pid(pid), 'node #:', node_i+1, '/', len(list[pid]))
            for output_neuron in range(output_dim):
                R_cat_feat = torch.cat([list[pid][node_i][output_neuron].to(device), X['x'].to(device), torch.arange(start=0, end=X['x'].shape[0], step=1).float().reshape(-1,1).to(device)], dim=1)

                non_empty_mask = R_cat_feat[:,:12].abs().sum(dim=1).bool()
                R_cat_feat_msk = R_cat_feat[non_empty_mask,:]   # R_cat_feat masked (non-zero)
                pos = dist[pid][node_i]
                probability = pred_ids_one_hot[pos]

                def get_type(t):
                    l = []
                    for elem in t:
                        if elem==1:
                            l.append('cluster')
                        if elem==2:
                            l.append('track')
                    return l

                node_types = get_type(R_cat_feat_msk[:,12])

                fig, ax = plt.subplots()
                fig.tight_layout()

                if task=='regression':
                    if (torch.argmax(probability)==pid):
                        ax.set_title('Heatmap for the "'+map_index_to_p4(output_neuron)+'" prediction of a correctly classified ' + map_index_to_pid(pid))
                    else:
                        ax.set_title('Heatmap for the "'+map_index_to_p4(output_neuron)+'" prediction of an incorrectly classified ' + map_index_to_pid(pid))

                elif task=='classification':
                    if (torch.argmax(probability)==pid):
                        ax.set_title('Heatmap for the "'+map_index_to_pid(output_neuron)+'" prediction of a correctly classified ' + map_index_to_pid(pid))
                    else:
                        ax.set_title('Heatmap for the "'+map_index_to_pid(output_neuron)+'" prediction of an incorrectly classified ' + map_index_to_pid(pid))

                ### TODO: Not the best way to do it.. I am assuming here that only charged hadrons are connected to all tracks
                if pid==1:
                    features = ["type", " pt", "eta",
                           "sphi", "cphi", "E", "eta_out", "sphi_out", "cphi_out", "charge", "is_gen_mu", "is_gen_el"]
                else:
                    features = ["type", "Et", "eta", "sphi", "cphi", "E", "Eem", "Ehad", "pad", "pad", "pad", "pad"]

                ax.set_xticks(np.arange(len(features)))
                ax.set_yticks(np.arange(len(node_types)))
                for col in range(len(features)):
                    for row in range(len(node_types)):
                        text = ax.text(col, row, round(R_cat_feat_msk[row,12+col].item(),2),
                                       ha="center", va="center", color="w")
                # ... and label them with the respective list entries
                ax.set_xticklabels(features)
                ax.set_yticklabels(node_types)
                plt.xlabel("\nposition of node is row # {pos} from the top \n class prediction: {R} \n where prob = [null, chhadron, nhadron, photon, electron, muon]".format(R=[round(num,2) for num in probability.detach().tolist()], pos=((R_cat_feat_msk[:,-1] == pos).nonzero(as_tuple=True)[0].item()+1)))
                plt.imshow(torch.abs(R_cat_feat_msk[:,:12]).detach().cpu().numpy(), interpolation="nearest", cmap='copper', aspect='auto')
                plt.colorbar()
                fig.set_size_inches(12, 12)
                if task=='regression':
                    plt.savefig(outpath + f'/class{str(pid)}'+f'/p4_elem{str(output_neuron)}'+f'/sample{str(node_i)}.jpg')
                elif task=='classification':
                    plt.savefig(outpath + f'/class{str(pid)}'+f'/pid{str(output_neuron)}'+f'/sample{str(node_i)}.jpg')
                plt.close(fig)

if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 2, 'patience': 100, 'hidden_dim':256, 'input_encoding': 12, 'encoding_dim': 64,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'LRP_dataset': '../../../test_tmp_delphes/data/pythia8_ttbar', 'LRP_dataset_qcd': '../../../test_tmp_delphes/data/pythia8_qcd',
    # 'LRP_outpath': '../../../prp/models/LRP/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0,
    # 'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
    # 'LRP_load_epoch': 9, 'LRP_load_model': 'LRP_reg_PFNet7_gen_ntrain_1_nepochs_10_batch_size_1_lr_0.001_alpha_0.0002_both_noembeddingsnoskip_nn1_nn3',
    # 'explain': False, 'make_heatmaps_clf': True,'make_heatmaps_reg': True,
    # 'clf': True, 'reg': True})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Processing the data..')
    full_dataset_ttbar = PFGraphDataset(args.LRP_dataset)
    full_dataset_qcd = PFGraphDataset(args.LRP_dataset_qcd)

    # constructs a loader from the data to iterate over batches
    print('Constructing data loaders..')
    train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
    test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    patience = args.patience

    model_classes = {"PFNet7": PFNet7}

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'input_encoding': args.input_encoding,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'space_dim': args.space_dim,
                    'propagate_dimensions': args.propagate_dimensions,
                    'nearest': args.nearest}

    print('Loading a previously trained model..')
    model = model_class(**model_kwargs)
    outpath = args.LRP_outpath + args.LRP_load_model
    PATH = outpath + '/epoch_' + str(args.LRP_load_epoch) + '_weights.pth'

    state_dict = torch.load(PATH, map_location=device)

    # if model was trained using DataParallel then we have to load it differently
    if "DataParallel" in args.LRP_load_model:
        state_dict = torch.load(PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
            # print('name is:', name)
        state_dict=new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)

    if args.explain:
        model.eval()
        print(model)

        # create some hooks to retrieve intermediate activations
        activation = {}
        hooks={}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0]
            return hook

        for name, module in model.named_modules():
            if (type(module)==nn.Linear) or (type(module)==nn.LeakyReLU) or (type(module)==nn.ELU):
                hooks[name] = module.register_forward_hook(get_activation("." + name))

        for i, batch in enumerate(train_loader):

            if multi_gpu:
                X = batch
            else:
                X = batch.to(device)

            if i==0:
                # code can be written better
                # basically i run at least one forward pass to get the activations to use their shape in defining the LRP layers
                pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4, edge_index, edge_weight, after_message, before_message = model(X)
                model = model_io(model,state_dict,dict(),activation)
                explainer_reg = LRP_reg(model)
                explainer_clf = LRP_clf(model)

            else:
                pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4, edge_index, edge_weight, after_message, before_message = model.model(X)

            if args.LRP_reg:
                print('Explaining the p4 predictions:')
                to_explain_reg = {"A": activation, "inputs": dict(x=X.x,batch=X.batch),
                                 "gen_p4": gen_p4.detach(), "gen_id": gen_ids_one_hot.detach(),
                                 "pred_p4": pred_p4.detach(), "pred_id": pred_ids_one_hot.detach(),
                                 "edge_index": edge_index.detach(), "edge_weight": edge_weight.detach(), "after_message": after_message.detach(), "before_message": before_message.detach(),
                                 "outpath": args.LRP_outpath, "load_model": args.LRP_load_model}

                model.set_dest(to_explain_reg["A"])

                big_list_reg = explainer_reg.explain(to_explain_reg)
                torch.save(big_list_reg, outpath + f'/big_list_reg.pt')
                torch.save(to_explain_reg, outpath + f'/to_explain_reg.pt')

            if args.LRP_clf:
                print('Explaining the pid predictions:')
                to_explain_clf = {"A": activation, "inputs": dict(x=X.x,batch=X.batch),
                                 "gen_p4": gen_p4.detach(), "gen_id": gen_ids_one_hot.detach(),
                                 "pred_p4": pred_p4.detach(), "pred_id": pred_ids_one_hot.detach(),
                                 "edge_index": edge_index.detach(), "edge_weight": edge_weight.detach(), "after_message": after_message.detach(), "before_message": before_message.detach(),
                                 "outpath": args.LRP_outpath, "load_model": args.LRP_load_model}

                model.set_dest(to_explain_clf["A"])

                big_list_clf = explainer_clf.explain(to_explain_clf)
                torch.save(big_list_clf, outpath + f'/big_list_clf.pt')
                torch.save(to_explain_clf, outpath + f'/to_explain_clf.pt')

            break # explain only one single event

    if args.make_heatmaps_reg:
        # load the necessary R-scores
        big_list_reg = torch.load(outpath + f'/big_list_reg.pt', map_location=device)
        to_explain_reg = torch.load(outpath + f'/to_explain_reg.pt', map_location=device)

        make_heatmaps(big_list_reg, to_explain_reg, 'regression')

    if args.make_heatmaps_clf:
        # load the necessary R-scores
        big_list_clf = torch.load(outpath + f'/big_list_clf.pt', map_location=device)
        to_explain_clf = torch.load(outpath + f'/to_explain_clf.pt', map_location=device)

        make_heatmaps(big_list_clf, to_explain_clf, 'classification')

# # ------------------------------------------------------------------------------------------------
# # if you got all the intermediate R-score heatmaps stored then you can check if these are equal as a check of conservation across all layers:
# print(R16[0].sum(axis=1)[0])
# print(R15[0].sum(axis=1)[0])
# print(R14[0].sum(axis=1)[0])
# print(R13[0].sum(axis=1)[0])
# print(R13[0].sum(axis=1)[0])
# print(R12[0].sum(axis=1)[0])
# print(R11[0].sum(axis=1)[0])
# print(R10[0].sum(axis=1)[0])
# print(R9[0].sum(axis=1)[0])
# print(R8[0].sum(axis=1)[0])
# print(R_score_layer_before_msg_passing[0][0].sum(axis=0).sum())
# print(R7[0][0].sum(axis=0).sum())
# print(R6[0][0].sum(axis=1).sum())
# print(R5[0][0].sum(axis=1).sum())
# print(R4[0][0].sum(axis=1).sum())
# print(R3[0][0].sum(axis=1).sum())
# print(R2[0][0].sum(axis=1).sum())
# print(R1[0][0].sum(axis=1).sum())
