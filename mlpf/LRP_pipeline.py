from glob import glob
import sys, os
import os.path as osp
import pickle as pkl
import _pickle as cPickle
import math, time, tqdm
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

from pytorch_delphes import PFGraphDataset, data_to_loader_ttbar, data_to_loader_qcd
from LRP import parse_args, make_heatmaps, model_io, PFNet7, LRP_clf, LRP_reg

# NOTE: this script works by loading an already trained model with very specefic specs

if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'n_test': 2, 'batch_size': 1,' hidden_dim':256, 'hidden_dim_nn1': 64,
    # 'input_encoding': 12, 'encoding_dim': 64, 'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16,
    # 'LRP_dataset': '../test_tmp_delphes/data/pythia8_ttbar', 'LRP_dataset_qcd': '../test_tmp_delphes/data/pythia8_qcd',
    # 'LRP_outpath': '../test_tmp_delphes/experiments/LRP/',
    # 'LRP_load_epoch': 9, 'LRP_load_model': 'LRP_reg_PFNet7_gen_ntrain_1_nepochs_10_batch_size_1_lr_0.001_alpha_0.0002_both_noembeddingsnoskip_nn1_nn3',
    # 'explain': False, 'LRP_clf': False, 'LRP_reg': False,
    # 'make_heatmaps_clf': True,'make_heatmaps_reg': True})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Processing the data..')
    full_dataset_qcd = PFGraphDataset(args.LRP_dataset_qcd)

    # constructs a loader from the data to iterate over batches
    print('Constructing data loader..')
    test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    outpath = args.LRP_outpath + args.LRP_load_model
    PATH = outpath + '/epoch_' + str(args.LRP_load_epoch) + '_weights.pth'

    # loading the model
    print('Loading a previously trained model..')
    with open(outpath + '/model_kwargs.pkl', 'rb') as f:
        model_kwargs = pkl.load(f)

    model = PFNet7(**model_kwargs)

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

        for i, batch in enumerate(test_loader):

            if multi_gpu:
                X = batch
            else:
                X = batch.to(device)

            if i==0:
                # code can be written better
                # basically i run at least one forward pass to get the activations to use their shape in defining the LRP layers
                pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4, edge_index, edge_weight, after_message, before_message = model(X)
                model = model_io(device, model, state_dict, dict(), activation)
                explainer_reg = LRP_reg(device, model)
                explainer_clf = LRP_clf(device, model)

            else:
                pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4, edge_index, edge_weight, after_message, before_message = model.model(X)

            if not osp.isdir(outpath + '/LRP'):
                os.makedirs(outpath + '/LRP')

            if args.LRP_reg:
                print('Explaining the p4 predictions:')
                to_explain_reg = {"A": activation, "inputs": dict(x=X.x,batch=X.batch),
                                 "gen_p4": gen_p4.detach(), "gen_id": gen_ids_one_hot.detach(),
                                 "pred_p4": pred_p4.detach(), "pred_id": pred_ids_one_hot.detach(),
                                 "edge_index": edge_index.detach(), "edge_weight": edge_weight.detach(), "after_message": after_message.detach(), "before_message": before_message.detach(),
                                 "outpath": args.LRP_outpath, "load_model": args.LRP_load_model}

                model.set_dest(to_explain_reg["A"])

                big_list_reg = explainer_reg.explain(to_explain_reg)
                torch.save(big_list_reg, outpath + '/LRP/big_list_reg.pt')
                torch.save(to_explain_reg, outpath + '/LRP/to_explain_reg.pt')

            if args.LRP_clf:
                print('Explaining the pid predictions:')
                to_explain_clf = {"A": activation, "inputs": dict(x=X.x,batch=X.batch),
                                 "gen_p4": gen_p4.detach(), "gen_id": gen_ids_one_hot.detach(),
                                 "pred_p4": pred_p4.detach(), "pred_id": pred_ids_one_hot.detach(),
                                 "edge_index": edge_index.detach(), "edge_weight": edge_weight.detach(), "after_message": after_message.detach(), "before_message": before_message.detach(),
                                 "outpath": args.LRP_outpath, "load_model": args.LRP_load_model}

                model.set_dest(to_explain_clf["A"])

                big_list_clf = explainer_clf.explain(to_explain_clf)

                torch.save(big_list_clf, outpath + '/LRP/big_list_clf.pt')
                torch.save(to_explain_clf, outpath + '/LRP/to_explain_clf.pt')

            break # explain only one single event

    if args.make_heatmaps_reg:
        # load the necessary R-scores
        big_list_reg = torch.load(outpath + '/LRP/big_list_reg.pt', map_location=device)
        to_explain_reg = torch.load(outpath + '/LRP/to_explain_reg.pt', map_location=device)

        make_heatmaps(big_list_reg, to_explain_reg, device, outpath, output_dim_id, output_dim_p4, 'regression')

    if args.make_heatmaps_clf:
        # load the necessary R-scores
        big_list_clf = torch.load(outpath + '/LRP/big_list_clf.pt', map_location=device)
        to_explain_clf = torch.load(outpath + '/LRP/to_explain_clf.pt', map_location=device)

        make_heatmaps(big_list_clf, to_explain_clf, device, outpath, output_dim_id, output_dim_p4, 'classification')

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
