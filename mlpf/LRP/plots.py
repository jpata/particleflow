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

import torch

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

def make_heatmaps(big_list, to_explain, device, outpath, output_dim_id, output_dim_p4, task):

    print(f'Making heatmaps for {task}..')

    X = to_explain["inputs"]
    gen_ids_one_hot = to_explain["gen_id"]
    pred_ids_one_hot = to_explain["pred_id"]

    gen_ids = gen_ids_one_hot.argmax(axis=1)
    pred_ids = pred_ids_one_hot.argmax(axis=1)

    # make directories to hold the heatmaps
    for i in range(6):
        if not osp.isdir(outpath + '/LRP'):
            os.makedirs(outpath + '/LRP')
        if not osp.isdir(outpath + f'/LRP/class{str(i)}'):
            os.makedirs(outpath + f'/LRP/class{str(i)}')
        for j in range(6):
            if task=='regression':
                if not osp.isdir(outpath + f'/LRP/class{str(i)}'+f'/p4_elem{str(j)}'):
                    os.makedirs(outpath + f'/LRP/class{str(i)}'+f'/p4_elem{str(j)}')
            elif task=='classification':
                if not osp.isdir(outpath + f'/LRP/class{str(i)}'+f'/pid{str(j)}'):
                    os.makedirs(outpath + f'/LRP/class{str(i)}'+f'/pid{str(j)}')

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
        if pid!=1:
            continue
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
                    plt.savefig(outpath + f'/LRP/class{str(pid)}'+f'/p4_elem{str(output_neuron)}'+f'/sample{str(node_i)}.jpg')
                elif task=='classification':
                    plt.savefig(outpath + f'/LRP/class{str(pid)}'+f'/pid{str(output_neuron)}'+f'/sample{str(node_i)}.jpg')
                plt.close(fig)
