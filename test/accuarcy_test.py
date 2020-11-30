#!/use/bin/env python3

import pickle
import argparse
import time
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels
from train_end2end import PFNet7
from sklearn.metrics import accuracy_score
import math




def compute_weights(target_ids, device):
    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(len(class_to_id)).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0/math.sqrt(float(v))
    return weights

def LoadModelCPU(kwargs_path,weights_path):
    try:
        kwargs_dict = torch.load(kwargs_path)
    except RuntimeError:
        pickle_in = open(kwargs_path,"rb")
        kwargs_dict = pickle.load(pickle_in)

    device = torch.device('cpu')
    predet_weights = torch.load(weights_path,map_location = device)


    model = PFNet7(**kwargs_dict)
    missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
    model.load_state_dict(missing_key_fix)
    model.eval()

    return model, device


def LoadModelGPU(kwargs_path, weights_path):
    try:
        kwargs_dict = torch.load(kwargs_path)
    except RuntimeError:
        pickle_in = open(kwargs_path,"rb")
        kwargs_dict = pickle.load(pickle_in)
    device = torch.device("cuda")
    predet_weights = torch.load(weights_path)

    model = PFNet7(**kwargs_dict)
    missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
    model.load_state_dict(missing_key_fix)
    model.to(device)
    model.eval()

    return model, device


def MainCPU(kwargs_path,weights_path,test_bin,l1m,l2m,target_type):
    
    model, device = LoadModelCPU(kwargs_path,weights_path)
    test_bin = PFGraphDataset(test_bin)
    bin_length = len(test_bin)


    accuracies_batch = np.zeros(bin_length+1)
    losses = np.zeros((bin_length, 2))
    
    progress = 0
    for i, data in enumerate(test_bin):
        t0 =time.time()
        print(progress)
        progress = progress + 1
 
        #modelusage
        cand_id_onehot, cand_momentum = model(data)

        #reference prediction of ID
        _dev = cand_id_onehot.device
        _, indices = torch.max(cand_id_onehot, -1)
        #Multi-GPUs will not be used
        data = [data]

        #using gen or cand  as targets 
        if args.target_type == "gen":
            target_ids = torch.cat([d.y_gen_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ygen[:, :4] for d in data]).to(_dev)
        elif args.target_type == "cand":
            target_ids = torch.cat([d.y_candidates_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ycand[:, :4] for d in data]).to(_dev)
        

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        msk = ((indices != 0) & (target_ids != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids))

        accuracies_batch[i] = accuracy_score(target_ids[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())
        
        #Loss for output candidate id (multiclass)
        weights = compute_weights(target_ids, _dev)
       


        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, target_ids, weight=weights)
        else:
            l1 = torch.tensor(0.0).to(device=_dev)

        #Loss for candidate p4 properties (regression)
        l2 = torch.tensor(0.0).to(device=_dev)
        
        if l2m > 0.0:
            l2 = l2m*torch.nn.functional.mse_loss(cand_momentum[msk2], target_p4[msk2])
        else:
            l2 = torch.tensor(0.0).to(device=_dev)


        batch_loss = l1 + l2
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()
        batch_loss_item = batch_loss.item()
        t1 = time.time()

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(test_bin), batch_loss_item, t1-t0), end='\r', flush=True)

        if i == bin_length - 1:
            print( f'Avrage accurcy for this data is:{np.mean(accuracies_batch)}')
            losses = losses.sum(axis=0)
            print(losses)
            quit()






def MainGPU(kwargs_path,weights_path,test_bin,l1m,l2m,target_type):

    model, device = LoadModelGPU(kwargs_path,weights_path)
    test_bin = PFGraphDataset(test_bin)
    bin_length = len(test_bin)


    accuracies_batch = np.zeros(bin_length+1)
    losses = np.zeros((bin_length, 2))

    progress = 0


    for i, data in enumerate(test_bin):
        t0 =time.time()
        print(progress)
        progress = progress + 1

        #modelusage
        data.to(device)
        cand_id_onehot, cand_momentum = model(data)

        #reference prediction of ID
        _dev = cand_id_onehot.device
        _, indices = torch.max(cand_id_onehot, -1)
        #Multi-GPUs will not be used
        data = [data]

        #using gen or cand  as targets 
        if args.target_type == "gen":
            target_ids = torch.cat([d.y_gen_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ygen[:, :4] for d in data]).to(_dev)
        elif args.target_type == "cand":
            target_ids = torch.cat([d.y_candidates_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ycand[:, :4] for d in data]).to(_dev)


        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        msk = ((indices != 0) & (target_ids != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids))

        accuracies_batch[i] = accuracy_score(target_ids[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        #Loss for output candidate id (multiclass)
        weights = compute_weights(target_ids, _dev)



        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, target_ids, weight=weights)
        else:
            l1 = torch.tensor(0.0).to(device=_dev)

        #Loss for candidate p4 properties (regression)
        l2 = torch.tensor(0.0).to(device=_dev)

        if l2m > 0.0:
            l2 = l2m*torch.nn.functional.mse_loss(cand_momentum[msk2], target_p4[msk2])
        else:
            l2 = torch.tensor(0.0).to(device=_dev)


        batch_loss = l1 + l2
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()
        batch_loss_item = batch_loss.item()
        t1 = time.time()

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(test_bin), batch_loss_item, t1-t0), end='\r', flush=True)

        if i == bin_length - 1:
            print( f'Avrage accuracy for this data is:{np.mean(accuracies_batch)}')
            losses = losses.sum(axis=0)
            print(f'losses sum on axis=0{losses}')
            
            quit()






 
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs_path",help="Path were the kwargs file is located as a pkl file for the model ",type = str )
    parser.add_argument("--weights_path", help="Path where the weights.pth file is located for the model ", type =str)
    parser.add_argument("--test_bin", help="Path to the directory where data resides ", type =str)

    parser.add_argument("--l1m", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2m", type=float, default=1.0, help="Loss multiplier for momentum regression")
    parser.add_argument("--target_type", help = "Target type either cand or gen", type = str)
    parser.add_argument("--device", help = "The device where the program will be ran on. Either cpu or gpu",type =str)

    args = parser.parse_args()
 
    if args.device == "cpu":
        print("Running on cpu")
        MainCPU(args.kwargs_path,args.weights_path,args.test_bin,args.l1m,args.l2m,args.target_type)

    if args.device == "gpu":
        print("Running on cpu")
        MainGPU(args.kwargs_path,args.weights_path,args.test_bin,args.l1m,args.l2m,args.target_type)
 
  

