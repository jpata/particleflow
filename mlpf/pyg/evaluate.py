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

    print('Making predictions...')
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
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} is {round(time.time() - t0, 3)}s')
        tff = tff + (time.time() - t0)

        file = [x for t in file for x in t]     # unpack the list of tuples to a list

        if multi_gpu:
            loader = DataListLoader(file, batch_size=batch_size)
        else:
            loader = DataLoader(file, batch_size=batch_size)

        t = 0
        for i, batch in enumerate(loader):

            if multi_gpu:
                X = batch   # a list (not torch) instance so can't be passed to device
            else:
                X = batch.to(device)

            ti = time.time()
            pred, target = model(X)
            tf = time.time()
            t = t + (tf - ti)

            # retrieve target
            gen_ids_one_hot = one_hot_embedding(target['ygen_id'].detach(), num_classes).to(device)
            gen_p4 = target['ygen'].detach()
            cand_ids_one_hot = one_hot_embedding(target['ycand_id'].detach(), num_classes).to(device)
            cand_p4 = target['ycand'].detach()

            # retrieve predictions
            pred_ids_one_hot = pred[:, :num_classes].detach()
            pred_p4 = pred[:, num_classes:].detach()

            # revert the one-hot encodings
            _, gen_ids = torch.max(gen_ids_one_hot, -1)
            _, pred_ids = torch.max(pred_ids_one_hot, -1)
            _, cand_ids = torch.max(cand_ids_one_hot, -1)

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

            print(f'batch #: {i+1}/{len(loader)}')

        print(f'Average inference time per batch is {round((t / (len(loader))), 3)}s')
        if num == 1:
            break
        t0 = time.time()

    print(f'Average time to load a file {round((tff / len(file_loader)), 3)}s')

    print('Time taken to make predictions is:', round(((time.time() - tt0) / 60), 2), 'min')

    # store the 3 dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list_dict = [pred_list, gen_list, cand_list]
    torch.save(list_dict, outpath + '/list_for_multiplicities.pt')

    torch.save(gen_ids_all, outpath + '/gen_ids.pt')
    torch.save(gen_p4_all, outpath + '/gen_p4.pt')
    torch.save(pred_ids_all, outpath + '/pred_ids.pt')
    torch.save(pred_p4_all, outpath + '/pred_p4.pt')
    torch.save(cand_ids_all, outpath + '/cand_ids.pt')
    torch.save(cand_p4_all, outpath + '/cand_p4.pt')

    ygen = torch.cat([gen_ids_all.reshape(-1, 1).float(), gen_p4_all], axis=1)
    ypred = torch.cat([pred_ids_all.reshape(-1, 1).float(), pred_p4_all], axis=1)
    ycand = torch.cat([cand_ids_all.reshape(-1, 1).float(), cand_p4_all], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {"ygen": ygen.reshape(1, -1, 7).cpu().numpy(),
                   "ycand": ycand.reshape(1, -1, 7).cpu().numpy(),
                   "ypred": ypred.reshape(1, -1, 7).cpu().numpy()}

    torch.save(predictions, outpath + '/predictions.pt')


def make_plots(device, data, num_classes, outpath, target, epoch, tag):

    print('Making plots...')

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms

    pfcands = list(name_to_pid.keys())

    t0 = time.time()

    # load the necessary predictions to make the plots
    gen_ids = torch.load(outpath + f'/gen_ids.pt', map_location=device)
    gen_p4 = torch.load(outpath + f'/gen_p4.pt', map_location=device)
    pred_ids = torch.load(outpath + f'/pred_ids.pt', map_location=device)
    pred_p4 = torch.load(outpath + f'/pred_p4.pt', map_location=device)
    cand_ids = torch.load(outpath + f'/cand_ids.pt', map_location=device)
    cand_p4 = torch.load(outpath + f'/cand_p4.pt', map_location=device)

    predictions = torch.load(outpath + f'/predictions.pt', map_location=device)

    # reformat a bit
    ygen = predictions["ygen"].reshape(-1, 7)
    ypred = predictions["ypred"].reshape(-1, 7)
    ycand = predictions["ycand"].reshape(-1, 7)

    # make confusion matrix for mlpf
    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        pred_ids.cpu(),
                                                        labels=range(num_classes),
                                                        normalize="true")

    plot_confusion_matrix(conf_matrix_mlpf, pfcands, epoch + 1, outpath + '/confusion_matrix_plots/', f'cm_mlpf_epoch_{str(epoch)}')

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        cand_ids.cpu(),
                                                        labels=range(num_classes),
                                                        normalize="true")

    plot_confusion_matrix(conf_matrix_cand, pfcands, epoch + 1, outpath + '/confusion_matrix_plots/', 'cm_cand', target="rule-based")

    # making all the other plots
    if 'QCD' in tag:
        sample = "QCD, 14 TeV, PU200"
    else:
        sample = "$t\\bar{t}$, 14 TeV, PU200"

    # make distribution plots
    for key, value in name_to_pid.items():
        if key != 'null':
            plot_distributions_pid(data, value, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,
                                   target, epoch, outpath, legend_title=sample + "\n")

    plot_distributions_all(data, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots combining all classes together
                           target, epoch, outpath, legend_title=sample + "\n")

    # plot particle multiplicity plots
    list_for_multiplicities = torch.load(outpath + f'/list_for_multiplicities.pt', map_location=device)

    for pfcand in pfcands:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_null = plot_particle_multiplicity(data, list_for_multiplicities, pfcand, ax)
        plt.savefig(outpath + f"/multiplicity_plots/num_{pfcand}.png", bbox_inches="tight")
        plt.savefig(outpath + f"/multiplicity_plots/num_{pfcand}.pdf", bbox_inches="tight")
        plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons and neutral hadrons
    for pfcand in pfcands:
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "pt", np.linspace(0, 3, 61), outpath + f"/efficiency_plots/eff_fake_{pfcand}_pt.png", both=True, legend_title=sample + "\n")
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "eta", np.linspace(-3, 3, 61), outpath + f"/efficiency_plots/eff_fake_{pfcand}_eta.png", both=True, legend_title=sample + "\n")
        ax, _ = draw_efficiency_fakerate(data, ygen, ypred, ycand, name_to_pid[pfcand], "energy", np.linspace(0, 50, 75), outpath + f"/efficiency_plots/eff_fake_{pfcand}_energy.png", both=True, legend_title=sample + "\n")

    # make pt, eta, and energy resolution plots
    for var in ['pt', 'eta', 'energy']:
        for pfcand in pfcands:
            plot_reso(data, ygen, ypred, ycand, pfcand, var, outpath, legend_title=sample + "\n")

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
