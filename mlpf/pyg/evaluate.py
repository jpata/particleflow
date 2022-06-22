from pyg.utils_plots import plot_confusion_matrix
from pyg.utils_plots import plot_distributions_pid, plot_distributions_all, plot_particle_multiplicity
from pyg.utils_plots import draw_efficiency_fakerate, plot_reso
from pyg.utils_plots import pid_to_name_delphes, name_to_pid_delphes, pid_to_name_cms, name_to_pid_cms
from pyg.utils import define_regions, batch_event_into_regions
from pyg.utils import one_hot_embedding

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import os
import os.path as osp
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")


def make_predictions(device, data, model, multi_gpu, loader, batch_size, num_classes, outpath):
    """
    Runs inference on the qcd test dataset to evaluate performance. Saves the predictions as .pt files.

    Args
        data: data specification ('cms' or 'delphes')
        model: pytorch model
        multi_gpu: boolean for multi_gpu training (if multigpus are available)
        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
    """
    if device == 'cpu':
        print(f"Running inference on cpu")
    else:
        torch.cuda.empty_cache()
        for rank in range(torch.cuda.device_count()):
            print(f"Running inference on rank {rank}: {torch.cuda.get_device_name(rank)}")

    if not osp.isdir(f'{outpath}/predictions'):
        os.makedirs(f'{outpath}/predictions')

    tt0 = time.time()

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms

    pfcands = list(name_to_pid.keys())

    gen_list, cand_list, pred_list = {}, {}, {}
    for pfcand in pfcands:
        gen_list[pfcand] = []
        cand_list[pfcand] = []
        pred_list[pfcand] = []

    t0, tff = time.time(), 0

    t = 0
    for i, batch in enumerate(loader):

        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        ti = time.time()
        pred, target = model(X)
        tf = time.time()
        print(f'batch {i}/{len(loader)}, forward pass = {round(tf - ti, 3)}s')
        t = t + (tf - ti)

        # retrieve predictions
        pred_p4 = pred[:, num_classes:].detach().to('cpu')
        pred_ids_one_hot = pred[:, :num_classes].detach().to('cpu')
        pred_ids = torch.argmax(pred_ids_one_hot, axis=1)

        # retrieve target
        gen_p4 = target['ygen'].detach().to('cpu')
        gen_ids = target['ygen_id'].detach().to('cpu')
        cand_p4 = target['ycand'].detach().to('cpu')
        cand_ids = target['ycand_id'].detach().to('cpu')

        # one hot encode the target
        gen_ids_one_hot = one_hot_embedding(gen_ids, num_classes).to('cpu')
        cand_ids_one_hot = one_hot_embedding(cand_ids, num_classes).to('cpu')

        # to make "num_gen vs num_pred" plots
        for key, value in name_to_pid.items():
            gen_list[key].append((gen_ids == value).sum().item())
            pred_list[key].append((pred_ids == value).sum().item())
            cand_list[key].append((cand_ids == value).sum().item())

        if i == 0:
            gen_ids_all = gen_ids
            gen_p4_all = gen_p4

            pred_ids_all = pred_ids
            pred_p4_all = pred_p4

            cand_ids_all = cand_ids
            cand_p4_all = cand_p4
        else:
            gen_ids_all = torch.cat([gen_ids_all, gen_ids])
            gen_p4_all = torch.cat([gen_p4_all, gen_p4])

            pred_ids_all = torch.cat([pred_ids_all, pred_ids])
            pred_p4_all = torch.cat([pred_p4_all, pred_p4])

            cand_ids_all = torch.cat([cand_ids_all, cand_ids])
            cand_p4_all = torch.cat([cand_p4_all, cand_p4])

        if i == 2:
            break

    print(f'Average inference time per batch is {round((t / (len(loader))), 3)}s')

    print('Time taken to make predictions is:', round(((time.time() - tt0) / 60), 2), 'min')

    # store the 3 dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list_dict = [pred_list, gen_list, cand_list]
    torch.save(list_dict, outpath + 'predictions/list_for_multiplicities.pt')

    torch.save(gen_ids_all, outpath + 'predictions/gen_ids.pt')
    torch.save(gen_p4_all, outpath + 'predictions/gen_p4.pt')
    torch.save(pred_ids_all, outpath + 'predictions/pred_ids.pt')
    torch.save(pred_p4_all, outpath + 'predictions/pred_p4.pt')
    torch.save(cand_ids_all, outpath + 'predictions/cand_ids.pt')
    torch.save(cand_p4_all, outpath + 'predictions/cand_p4.pt')

    ygen = torch.cat([gen_ids_all.reshape(-1, 1).float(), gen_p4_all], axis=1)
    ypred = torch.cat([pred_ids_all.reshape(-1, 1).float(), pred_p4_all], axis=1)
    ycand = torch.cat([cand_ids_all.reshape(-1, 1).float(), cand_p4_all], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {"ygen": ygen.reshape(1, -1, 7).cpu().numpy(),
                   "ycand": ycand.reshape(1, -1, 7).cpu().numpy(),
                   "ypred": ypred.reshape(1, -1, 7).cpu().numpy()}

    torch.save(predictions, outpath + 'predictions/predictions.pt')


def make_plots(data, num_classes, outpath, target, epoch, tag):

    print('Making plots...')

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms

    pfcands = list(name_to_pid.keys())

    t0 = time.time()

    # load the necessary predictions to make the plots
    gen_ids = torch.load(outpath + f'predictions/gen_ids.pt', map_location='cpu')
    gen_p4 = torch.load(outpath + f'predictions/gen_p4.pt', map_location='cpu')
    pred_ids = torch.load(outpath + f'predictions/pred_ids.pt', map_location='cpu')
    pred_p4 = torch.load(outpath + f'predictions/pred_p4.pt', map_location='cpu')
    cand_ids = torch.load(outpath + f'predictions/cand_ids.pt', map_location='cpu')
    cand_p4 = torch.load(outpath + f'predictions/cand_p4.pt', map_location='cpu')

    predictions = torch.load(outpath + f'predictions/predictions.pt', map_location='cpu')

    # reformat a bit
    ygen = predictions["ygen"].reshape(-1, 7)
    ypred = predictions["ypred"].reshape(-1, 7)
    ycand = predictions["ycand"].reshape(-1, 7)

    # make confusion matrix for mlpf
    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        pred_ids.cpu(),
                                                        labels=range(num_classes),
                                                        normalize="true")

    plot_confusion_matrix(conf_matrix_mlpf, pfcands, epoch + 1, outpath + 'plots/confusion_matrix_plots/', f'cm_mlpf_epoch_{str(epoch)}')

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        cand_ids.cpu(),
                                                        labels=range(num_classes),
                                                        normalize="true")

    plot_confusion_matrix(conf_matrix_cand, pfcands, epoch + 1, outpath + 'plots/confusion_matrix_plots/', 'cm_cand', target="rule-based")

    # making all the other plots
    if 'QCD' in tag:
        sample = "QCD, 14 TeV, PU200"
    else:
        sample = "$t\\bar{t}$, 14 TeV, PU200"

    # make distribution plots
    for key, value in name_to_pid.items():
        if key != 'null':
            plot_distributions_pid(data, value, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,
                                   target, epoch, outpath + 'plots/', legend_title=sample + "\n")

    plot_distributions_all(data, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots combining all classes together
                           target, epoch, outpath + 'plots/', legend_title=sample + "\n")

    # plot particle multiplicity plots
    list_for_multiplicities = torch.load(outpath + f'predictions/list_for_multiplicities.pt', map_location='cpu')

    for pfcand in pfcands:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_null = plot_particle_multiplicity(data, list_for_multiplicities, pfcand, ax)
        plt.savefig(outpath + f"plots/multiplicity_plots/num_{pfcand}.pdf", bbox_inches="tight")
        plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons and neutral hadrons
    for pfcand in pfcands:
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "pt", np.linspace(0, 3, 61), outpath + f"plots/efficiency_plots/eff_fake_{pfcand}_pt.pdf", both=True, legend_title=sample + "\n")
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "eta", np.linspace(-3, 3, 61), outpath + f"plots/efficiency_plots/eff_fake_{pfcand}_eta.pdf", both=True, legend_title=sample + "\n")
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "energy", np.linspace(0, 50, 75), outpath + f"plots/efficiency_plots/eff_fake_{pfcand}_energy.pdf", both=True, legend_title=sample + "\n")

    # make pt, eta, and energy resolution plots
    for var in ['pt', 'eta', 'energy']:
        for pfcand in pfcands:
            plot_reso(data, ygen, ypred, ycand, pfcand, var, outpath + 'plots/', legend_title=sample + "\n")

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
