from pyg.utils_plots import plot_confusion_matrix
from pyg.utils_plots import plot_distributions_pid, plot_distributions_all, plot_particle_multiplicity
from pyg.utils_plots import draw_efficiency_fakerate, plot_reso
from pyg.utils_plots import pid_to_name_delphes, name_to_pid_delphes, pid_to_name_cms, name_to_pid_cms
from pyg.utils import define_regions, batch_event_into_regions
from pyg.utils import one_hot_embedding, target_p4

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, DataListLoader

import glob
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


def make_predictions(device, data, model, multi_gpu, file_loader, batch_size, num_classes, outpath):
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

    ibatch = 0
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} is {round(time.time() - t0, 3)}s')
        tff = tff + (time.time() - t0)

        file = [x for t in file for x in t]     # unpack the list of tuples to a list

        if multi_gpu:
            loader = DataListLoader(file, batch_size=batch_size)
        else:
            loader = DataLoader(file, batch_size=batch_size)

        t = 0

        outs = {}
        for i, batch in enumerate(loader):
            np_outfile = f"{outpath}/predictions/pred_batch{ibatch}.npz"

            if multi_gpu:
                X = batch   # a list (not torch) instance so can't be passed to device
            else:
                X = batch.to(device)

            ti = time.time()
            pred, target = model(X)
            tf = time.time()
            print(f'batch {i}/{len(loader)}, forward pass = {round(tf - ti, 3)}s')
            t = t + (tf - ti)

            # zero pad the events to use the same plotting scripts as the tf pipeline
            padded_num_elem_size = 6400

            vars = {'X': batch.x,
                    'ygen': target['ygen'].detach().to('cpu'),
                    'ycand': target['ycand'].detach().to('cpu'),
                    'pred_p4': pred[:, 9:].detach().to('cpu'),
                    'gen_ids_one_hot': one_hot_embedding(target['ygen_id'].detach().to('cpu'), num_classes).to('cpu'),
                    'cand_ids_one_hot': one_hot_embedding(target['ycand_id'].detach().to('cpu'), num_classes).to('cpu'),
                    'pred_ids_one_hot': pred[:, :9].detach().to('cpu')
                    }

            vars_padded = {}
            for key, var in vars.items():
                var = var[:padded_num_elem_size]
                var = np.pad(var, [(0, padded_num_elem_size - var.shape[0]), (0, 0)])
                var = np.expand_dims(var, 0)

                vars_padded[key] = var

            outs = {}
            outs['gen_cls'] = vars_padded['gen_ids_one_hot']
            outs['cand_cls'] = vars_padded['cand_ids_one_hot']
            outs['pred_cls'] = vars_padded['pred_ids_one_hot']

            for feat, key in enumerate(target_p4):
                outs[f'gen_{key}'] = vars_padded['ygen'][:, :, feat].reshape(-1, padded_num_elem_size, 1)
                outs[f'cand_{key}'] = vars_padded['ycand'][:, :, feat].reshape(-1, padded_num_elem_size, 1)
                outs[f'pred_{key}'] = vars_padded['pred_p4'][:, :, feat].reshape(-1, padded_num_elem_size, 1)

            np.savez(
                np_outfile,
                X=vars_padded['X'],
                **outs
            )
            ibatch += 1

        #     if i == 3:
        #         break
        #
        # if num == 2:
        #     break

        print(f'Average inference time per batch is {round((t / (len(loader))), 3)}s')

        t0 = time.time()

    print(f'Average time to load a file {round((tff / len(file_loader)), 3)}s')

    print('Time taken to make predictions is:', round(((time.time() - tt0) / 60), 2), 'min')


def make_plots(data, num_classes, outpath, target, epoch, tag):

    print('Making plots...')

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms

    pfcands = list(name_to_pid.keys())

    t0 = time.time()

    # load the necessary predictions to make the plots

    Xs = []
    yvals = {}
    for fi in list(glob.glob(outpath + "/predictions/pred_batch*.npz")):
        dd = np.load(fi)
        Xs.append(dd["X"])

        keys_in_file = list(dd.keys())
        for k in keys_in_file:
            if k == "X":
                continue
            if not (k in yvals):
                yvals[k] = []
            yvals[k].append(dd[k])

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
    list_for_multiplicities = torch.load(outpath + f'list_for_multiplicities.pt', map_location='cpu')

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
