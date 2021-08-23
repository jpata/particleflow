import pickle as pkl
import math, time, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

import torch

import pytorch_delphes

def make_predictions(model, multi_gpu, test_loader, outpath, target, device, epoch, which_data):

    print('Making predictions on ' + which_data)
    t0=time.time()

    gen_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}
    pred_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}
    cand_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}

    for i, batch in enumerate(test_loader):
        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4 = model(X)

        _, gen_ids = torch.max(gen_ids_one_hot.detach(), -1)
        _, pred_ids = torch.max(pred_ids_one_hot.detach(), -1)
        _, cand_ids = torch.max(cand_ids_one_hot.detach(), -1)

        # to make "num_gen vs num_pred" plots
        gen_list["null"].append((gen_ids==0).sum().item())
        gen_list["chhadron"].append((gen_ids==1).sum().item())
        gen_list["nhadron"].append((gen_ids==2).sum().item())
        gen_list["photon"].append((gen_ids==3).sum().item())
        gen_list["electron"].append((gen_ids==4).sum().item())
        gen_list["muon"].append((gen_ids==5).sum().item())

        pred_list["null"].append((pred_ids==0).sum().item())
        pred_list["chhadron"].append((pred_ids==1).sum().item())
        pred_list["nhadron"].append((pred_ids==2).sum().item())
        pred_list["photon"].append((pred_ids==3).sum().item())
        pred_list["electron"].append((pred_ids==4).sum().item())
        pred_list["muon"].append((pred_ids==5).sum().item())

        cand_list["null"].append((cand_ids==0).sum().item())
        cand_list["chhadron"].append((cand_ids==1).sum().item())
        cand_list["nhadron"].append((cand_ids==2).sum().item())
        cand_list["photon"].append((cand_ids==3).sum().item())
        cand_list["electron"].append((cand_ids==4).sum().item())
        cand_list["muon"].append((cand_ids==5).sum().item())

        gen_p4 = gen_p4.detach()
        pred_p4 = pred_p4.detach()
        cand_p4 = cand_p4.detach()

        if i==0:
            gen_ids_all = gen_ids
            gen_p4_all = gen_p4

            pred_ids_all = pred_ids
            pred_p4_all = pred_p4

            cand_ids_all = cand_ids
            cand_p4_all = cand_p4
        else:
            gen_ids_all = torch.cat([gen_ids_all,gen_ids])
            gen_p4_all = torch.cat([gen_p4_all,gen_p4])

            pred_ids_all = torch.cat([pred_ids_all,pred_ids])
            pred_p4_all = torch.cat([pred_p4_all,pred_p4])

            cand_ids_all = torch.cat([cand_ids_all,cand_ids])
            cand_p4_all = torch.cat([cand_p4_all,cand_p4])

        if len(test_loader)<5000:
            print(f'event #: {i+1}/{len(test_loader)}')
        else:
            print(f'event #: {i+1}/{5000}')

        if i==4999:
            break

    t1=time.time()

    print('Time taken to make predictions is:', round(((t1-t0)/60),2), 'min')

    # store the 3 list dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list = [pred_list, gen_list, cand_list]

    torch.save(list, outpath + '/list_for_multiplicities.pt')

    torch.save(gen_ids_all, outpath + '/gen_ids.pt')
    torch.save(gen_p4_all, outpath + '/gen_p4.pt')
    torch.save(pred_ids_all, outpath + '/pred_ids.pt')
    torch.save(pred_p4_all, outpath + '/pred_p4.pt')
    torch.save(cand_ids_all, outpath + '/cand_ids.pt')
    torch.save(cand_p4_all, outpath + '/cand_p4.pt')

    ygen = torch.cat([gen_ids_all.reshape(-1,1).float(),gen_p4_all], axis=1)
    ypred = torch.cat([pred_ids_all.reshape(-1,1).float(),pred_p4_all], axis=1)
    ycand = torch.cat([cand_ids_all.reshape(-1,1).float(),cand_p4_all], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {"ygen":ygen.reshape(1,-1,7).detach().cpu().numpy(), "ycand":ycand.reshape(1,-1,7).detach().cpu().numpy(), "ypred":ypred.detach().reshape(1,-1,7).cpu().numpy()}

    torch.save(predictions, outpath + '/predictions.pt')
