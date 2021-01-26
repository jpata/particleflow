#import setGPU
import torch
import torch_geometric
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
import pandas
import mplhep
import pickle

import graph_data_delphes
from graph_data_delphes import PFGraphDataset
from data_preprocessing import from_data_to_loader
import train_end2end_delphes
import time
import math

import sys
sys.path.insert(1, '../plotting/')
sys.path.insert(1, '../mlpf/plotting/')

import plots_delphes
from plots_delphes import make_plots

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)

def prepare_test_data(full_dataset, start, stop, batch_size):

    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=start, stop=stop))

    # preprocessing the test_dataset in a good format for passing correct batches of events to the GNN
    test_dataset_batched=[]
    for i in range(len(test_dataset)):
        test_dataset_batched += test_dataset[i]
    test_dataset_batched = [[i] for i in test_dataset_batched]

    #hack for multi-gpu training
    if not multi_gpu:
        def collate(items):
            l = sum(items, [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum(items, [])
            return l

    test_loader = DataListLoader(test_dataset_batched, batch_size, pin_memory=True, shuffle=True)
    test_loader.collate_fn = collate

    return test_loader

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=sorted(train_end2end_delphes.model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--path", type=str, help="path to model", default="data/PFNet7_TTbar_14TeV_TuneCUETP8M1_cfi_gen__npar_221073__cfg_ee19d91068__user_jovyan__ntrain_400__lr_0.0001__1588215695")
    parser.add_argument("--epoch", type=str, default=0, help="Epoch to use")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--start", type=int, default=3800, help="first file index to evaluate")
    parser.add_argument("--stop", type=int, default=4000, help="last file index to evaluate")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], default="cand", help="type of data the model trained on (cand or gen)")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'model': 'PFNet7', 'dataset': '../../test_tmp_delphes/data/delphes_cfi', 'epoch' : 1, 'target': 'cand', 'start':1, 'stop':2,
    # 'path': '../../test_tmp_delphes/data/PFNet7_delphes_cfi_gen__npar_41414__cfg_fca529f313__user_fmokhtar__ntrain_1__lr_0.0001__1611654293'})

    epoch = args.epoch
    model = args.model
    path = args.path
    weights = torch.load("{}/epoch_{}_weights.pth".format(path, epoch), map_location=device)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}

    with open('{}/model_kwargs.pkl'.format(path),'rb') as f:
        model_kwargs = pickle.load(f)

    model_class = train_end2end_delphes.model_classes[args.model]
    model = model_class(**model_kwargs)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    # prepare some test_data
    print('Creating the test data and feeding it to the model..')
    full_dataset = PFGraphDataset(root=args.dataset)
    loader = prepare_test_data(full_dataset, start=args.start, stop=args.stop, batch_size=10)

    for batch in loader:
        pred_id, pred_p4, new_edges_ = model(batch)
        break

    print('Making plots for evaluation..')

    if args.target=='cand':
        make_plots(batch.ycand_id, batch.ycand, pred_id, pred_p4, out=path +'/')
    elif args.target=='gen':
        make_plots(batch.ygen_id, batch.ygen, pred_id, pred_p4, out=path +'/')

# def prepare_dataframe(model, loader, multi_gpu, device, target_type="cand"):
#     model.eval()
#     dfs = []
#     dfs_edges = []
#     eval_time = 0
#
#     for i, data in enumerate(loader):
#         if not multi_gpu:
#             data = data.to(device)
#         pred_id_onehot, pred_momentum, new_edges = model(data)
#         _, pred_id = torch.max(pred_id_onehot, -1)
#         pred_momentum[pred_id==0] = 0
#         data = [data]
#
#         x = torch.cat([d.x.to("cpu") for d in data])
#         gen_id = torch.cat([d.ygen_id.to("cpu") for d in data])
#         gen_p4 = torch.cat([d.ygen[:, :].to("cpu") for d in data])
#         cand_id = torch.cat([d.ycand_id.to("cpu") for d in data])
#         cand_p4 = torch.cat([d.ycand[:, :].to("cpu") for d in data])
#
#         # reverting the one_hot_embedding
#         gen_id_flat = torch.max(gen_id, -1)[1]
#         cand_id_flat = torch.max(cand_id, -1)[1]
#
#         df = pandas.DataFrame()
#         gen_p4.shape
#         gen_id.shape
#
#         # Recall:
#         # [pid] takes from 1 to 6
#         # [charge, pt (GeV), eta, sin phi, cos phi, E (GeV)]
#
#         df["elem_type"] = [int(elem_labels[i]) for i in torch.argmax(x[:, :len(elem_labels)], axis=-1).numpy()]
#
#         if target_type == "gen":
#             df["gen_pid"] = [int(class_labels[i]) for i in gen_id_flat.numpy()]
#             df["gen_charge"] = gen_p4[:, 0].numpy()
#             df["gen_eta"] = gen_p4[:, 2].numpy()
#             df["gen_sphi"] = gen_p4[:, 3].numpy()
#             df["gen_cphi"] = gen_p4[:, 4].numpy()
#             df["gen_e"] = gen_p4[:, 5].numpy()
#
#         elif target_type == "cand":
#             df["cand_pid"] = [int(class_labels[i]) for i in cand_id_flat.numpy()]
#             df["cand_charge"] = cand_p4[:, 0].numpy()
#             df["cand_eta"] = cand_p4[:, 2].numpy()
#             df["cand_sphi"] = cand_p4[:, 3].numpy()
#             df["cand_cphi"] = cand_p4[:, 4].numpy()
#             df["cand_e"] = cand_p4[:, 5].numpy()
#
#         df["pred_pid"] = [int(class_labels[i]) for i in pred_id.detach().cpu().numpy()]
#         df["pred_charge"] = pred_momentum[:, 0].detach().cpu().numpy()
#         df["pred_eta"] = pred_momentum[:, 2].detach().cpu().numpy()
#         df["pred_sphi"] = pred_momentum[:, 3].detach().cpu().numpy()
#         df["pred_cphi"] = pred_momentum[:, 4].detach().cpu().numpy()
#         df["pred_e"] = pred_momentum[:, 5].detach().cpu().numpy()
#
#         dfs.append(df)
#
#     df = pandas.concat(dfs, ignore_index=True)
#     return df
