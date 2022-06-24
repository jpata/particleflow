from pyg.utils_plots import plot_confusion_matrix
from pyg.utils_plots import plot_distributions_pid, plot_distributions_all, plot_particle_multiplicity
from pyg.utils_plots import draw_efficiency_fakerate, plot_reso
from pyg.utils_plots import pid_to_name_delphes, name_to_pid_delphes, pid_to_name_cms, name_to_pid_cms
from pyg.utils import define_regions, batch_event_into_regions
from pyg.utils import one_hot_embedding, target_p4
from pyg.cms_utils import CLASS_NAMES_CMS
from pyg.cms_plots import plot_numPFelements, plot_met, plot_sum_energy, plot_sum_pt, plot_energy_res, plot_eta_res, plot_multiplicity
from pyg.cms_plots import plot_dist, plot_cm, plot_eff_and_fake_rate, distribution_icls

import torch
import torch_geometric
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


def make_predictions(rank, model, file_loader, batch_size, num_classes, PATH):
    """
    Runs inference on the qcd test dataset to evaluate performance. Saves the predictions as .pt files.

    Args
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        model: pytorch model
        file_loader:  a pytorch Dataloader that loads .pt files for training when you invoke the get() method
    """

    ti = time.time()

    yvals = {}

    t0, tf = time.time(), 0
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - t0, 3)}s')
        tf = tf + (time.time() - t0)

        file = [x for t in file for x in t]     # unpack the list of tuples to a list

        loader = torch_geometric.loader.DataLoader(file, batch_size=batch_size)

        t = 0
        for i, batch in enumerate(loader):

            t0 = time.time()
            pred_ids_one_hot, pred_p4 = model(batch.to(rank))
            t1 = time.time()
            print(f'batch {i}/{len(loader)}, forward pass on rank {rank} = {round(t1 - t0, 3)}s, for batch with {batch.num_nodes} nodes')
            t = t + (t1 - t0)

            # zero pad the events to use the same plotting scripts as the tf pipeline
            padded_num_elem_size = 6400

            pred_ids_one_hot_list = []
            pred_p4_list = []
            for z in range(batch_size):
                pred_ids_one_hot_list.append(pred_ids_one_hot[batch.batch == z])
                pred_p4_list.append(pred_p4[batch.batch == z])

            batch_list = batch.to_data_list()
            for j, event in enumerate(batch_list):
                vars = {'X': event.x.detach().to('cpu'),
                        'ygen': event.ygen.detach().to('cpu'),
                        'ycand': event.ycand.detach().to('cpu'),
                        'pred_p4': pred_p4_list[j].detach().to('cpu'),
                        'gen_ids_one_hot': one_hot_embedding(event.ygen_id.detach().to('cpu'), num_classes),
                        'cand_ids_one_hot': one_hot_embedding(event.ycand_id.detach().to('cpu'), num_classes),
                        'pred_ids_one_hot': pred_ids_one_hot_list[j].detach().to('cpu')
                        }

                vars_padded = {}
                for key, var in vars.items():
                    var = var[:padded_num_elem_size]
                    var = np.pad(var, [(0, padded_num_elem_size - var.shape[0]), (0, 0)])
                    var = np.expand_dims(var, 0)

                    vars_padded[key] = var

                if not bool(yvals):
                    X = vars_padded['X']
                    yvals[f'gen_cls'] = vars_padded['gen_ids_one_hot']
                    yvals[f'cand_cls'] = vars_padded['cand_ids_one_hot']
                    yvals[f'pred_cls'] = vars_padded['pred_ids_one_hot']
                    Y_gen = vars_padded['ygen'].reshape(1, padded_num_elem_size, -1)
                    Y_cand = vars_padded['ycand'].reshape(1, padded_num_elem_size, -1)
                    Y_pred = vars_padded['pred_p4'].reshape(1, padded_num_elem_size, -1)

                else:
                    X = np.concatenate([X, vars_padded['X']])
                    yvals[f'gen_cls'] = np.concatenate([yvals[f'gen_cls'], vars_padded['gen_ids_one_hot']])
                    yvals[f'cand_cls'] = np.concatenate([yvals[f'cand_cls'], vars_padded['cand_ids_one_hot']])
                    yvals[f'pred_cls'] = np.concatenate([yvals[f'pred_cls'], vars_padded['pred_ids_one_hot']])
                    Y_gen = np.concatenate([Y_gen, vars_padded['ygen'].reshape(1, padded_num_elem_size, -1)])
                    Y_cand = np.concatenate([Y_cand, vars_padded['ycand'].reshape(1, padded_num_elem_size, -1)])
                    Y_pred = np.concatenate([Y_pred, vars_padded['pred_p4'].reshape(1, padded_num_elem_size, -1)])

            if i == 2:
                break
        if num == 2:
            break

        print(f'Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s')

        t0 = time.time()

    print(f'Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s')

    print(f'Time taken to make predictions on rank {rank} is: {round(((time.time() - ti) / 60), 2)} min')

    for feat, key in enumerate(target_p4):
        yvals[f'gen_{key}'] = Y_gen[:, :, feat].reshape(-1, padded_num_elem_size, 1)
        yvals[f'cand_{key}'] = Y_cand[:, :, feat].reshape(-1, padded_num_elem_size, 1)
        yvals[f'pred_{key}'] = Y_pred[:, :, feat].reshape(-1, padded_num_elem_size, 1)

    for val in ["gen", "cand", "pred"]:
        yvals[f"{val}_phi"] = np.arctan2(yvals[f"{val}_sin_phi"], yvals[f"{val}_cos_phi"])
        yvals[f"{val}_cls_id"] = np.expand_dims(np.argmax(yvals[f"{val}_cls"], axis=-1), axis=-1)

        yvals[f"{val}_px"] = np.sin(yvals[f"{val}_phi"]) * yvals[f"{val}_pt"]
        yvals[f"{val}_py"] = np.cos(yvals[f"{val}_phi"]) * yvals[f"{val}_pt"]

    print('--> Saving predictions...')
    np.save(f'{PATH}/predictions/predictions_X_{rank}.npy', X)

    with open(f'{PATH}/predictions/predictions_yvals_{rank}.pkl', 'wb') as f:
        pkl.dump(yvals, f)


def load_predictions(pred_path):

    print('--> Loading predictions...')
    t0 = time.time()

    X = []
    for fi in list(glob.glob(f'{pred_path}/predictions_X_*')):
        if len(X) == 0:
            X = np.load(fi, allow_pickle=True)
        else:
            X = np.concatenate([X, np.load(fi, allow_pickle=True)])

    yvals = {}
    for fi in list(glob.glob(f'{pred_path}/predictions_yvals_*')):
        if not bool(yvals):
            with open(fi, 'rb') as f:
                yvals = pkl.load(f)
        else:
            with open(fi, 'rb') as f:
                int = pkl.load(f)
            for key, value in yvals.items():
                yvals[key] = np.concatenate([yvals[key], int[key]])

    print('Further processing for convenient plotting')

    def flatten(arr):
        # return arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
        return arr.reshape(-1, arr.shape[-1])

    X_f = flatten(X)

    msk_X_f = X_f[:, 0] != 0

    yvals_f = {k: flatten(v) for k, v in yvals.items()}

    # remove the last dim
    for k in yvals_f.keys():
        if yvals_f[k].shape[-1] == 1:
            yvals_f[k] = yvals_f[k][..., -1]

    print(f'Time taken to load and process predictions is: {round(((time.time() - t0) / 60), 2)} min')

    return X, X_f, msk_X_f, yvals, yvals_f


def make_plots_cms(pred_path, plot_path, sample):

    t0 = time.time()

    X, X_f, msk_X_f, yvals, yvals_f = load_predictions(pred_path)

    print('Making plots...')

    # plot distributions
    print('plot_dist...')
    plot_dist(yvals_f, 'pt', np.linspace(0, 200, 61), r'$p_T$', plot_path, sample)
    plot_dist(yvals_f, 'energy', np.linspace(0, 2000, 61), r'$E$', plot_path, sample)
    plot_dist(yvals_f, 'eta', np.linspace(-6, 6, 61), r'$\eta$', plot_path, sample)

    # plot cm
    print('plot_cm...')
    plot_cm(yvals_f, msk_X_f, 'pred_cls_id', 'MLPF', plot_path)
    plot_cm(yvals_f, msk_X_f, 'cand_cls_id', 'PF', plot_path)

    # plot eff_and_fake_rate
    print('plot_eff_and_fake_rate...')
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=1, ivar=4, ielem=1, bins=np.logspace(-1, 3, 41), log=True)
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=1, ivar=3, ielem=1, bins=np.linspace(-4, 4, 41), log=False, xlabel="PFElement $\eta$")
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=2, ivar=4, ielem=5, bins=np.logspace(-1, 3, 41), log=True)
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=2, ivar=3, ielem=5, bins=np.linspace(-5, 5, 41), log=False, xlabel="PFElement $\eta$")
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=5, ivar=4, ielem=4, bins=np.logspace(-1, 2, 41), log=True)
    plot_eff_and_fake_rate(X_f, yvals_f, plot_path, sample, icls=5, ivar=3, ielem=4, bins=np.linspace(-5, 5, 41), log=False, xlabel="PFElement $\eta$")

    # distribution_icls
    print('distribution_icls...')
    distribution_icls(yvals_f, plot_path)

    print('plot_numPFelements...')
    plot_numPFelements(X, plot_path, sample)
    print('plot_met...')
    plot_met(X, yvals, plot_path, sample)
    print('plot_sum_energy...')
    plot_sum_energy(X, yvals, plot_path, sample)
    print('plot_sum_pt...')
    plot_sum_pt(X, yvals, plot_path, sample)
    print('plot_multiplicity...')
    plot_multiplicity(X, yvals, plot_path, sample)

    # for energy resolution plotting purposes, initialize pid -> (ylim, bins) dictionary
    print('plot_energy_res...')
    dic = {1: (1e9, np.linspace(-2, 15, 100)),
           2: (1e7, np.linspace(-2, 15, 100)),
           3: (1e7, np.linspace(-2, 40, 100)),
           4: (1e7, np.linspace(-2, 30, 100)),
           5: (1e7, np.linspace(-2, 10, 100)),
           6: (1e4, np.linspace(-1, 1, 100)),
           7: (1e4, np.linspace(-0.1, 0.1, 100))
           }
    for pid, tuple in dic.items():
        plot_energy_res(X, yvals_f, pid, tuple[1], tuple[0], plot_path, sample)

    # for eta resolution plotting purposes, initialize pid -> (ylim) dictionary
    print('plot_eta_res...')
    dic = {1: 1e10,
           2: 1e8}
    for pid, ylim in dic.items():
        plot_eta_res(X, yvals_f, pid, ylim, plot_path, sample)

    print(f'Time taken to make plots is: {round(((time.time() - t0) / 60), 2)} min')


def make_predictions_delphes(model, multi_gpu, test_loader, outpath, device, epoch):

    print('Making predictions...')
    t0 = time.time()

    gen_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}
    pred_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}
    cand_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}

    t = []

    for i, batch in enumerate(test_loader):
        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        ti = time.time()

        pred_ids_one_hot, pred_p4 = model(X)

        gen_ids_one_hot = one_hot_embedding(X.ygen_id.detach().to('cpu'), num_classes)
        gen_p4 = X.ygen.detach().to('cpu')
        cand_ids_one_hot = one_hot_embedding(X.ycand_id.detach().to('cpu'), num_classes)
        cand_p4 = X.ycand.detach().to('cpu')

        tf = time.time()
        if i != 0:
            t.append(round((tf - ti), 2))

        _, gen_ids = torch.max(gen_ids_one_hot.detach(), -1)
        _, pred_ids = torch.max(pred_ids_one_hot.detach(), -1)
        _, cand_ids = torch.max(cand_ids_one_hot.detach(), -1)

        # to make "num_gen vs num_pred" plots
        gen_list["null"].append((gen_ids == 0).sum().item())
        gen_list["chhadron"].append((gen_ids == 1).sum().item())
        gen_list["nhadron"].append((gen_ids == 2).sum().item())
        gen_list["photon"].append((gen_ids == 3).sum().item())
        gen_list["electron"].append((gen_ids == 4).sum().item())
        gen_list["muon"].append((gen_ids == 5).sum().item())

        pred_list["null"].append((pred_ids == 0).sum().item())
        pred_list["chhadron"].append((pred_ids == 1).sum().item())
        pred_list["nhadron"].append((pred_ids == 2).sum().item())
        pred_list["photon"].append((pred_ids == 3).sum().item())
        pred_list["electron"].append((pred_ids == 4).sum().item())
        pred_list["muon"].append((pred_ids == 5).sum().item())

        cand_list["null"].append((cand_ids == 0).sum().item())
        cand_list["chhadron"].append((cand_ids == 1).sum().item())
        cand_list["nhadron"].append((cand_ids == 2).sum().item())
        cand_list["photon"].append((cand_ids == 3).sum().item())
        cand_list["electron"].append((cand_ids == 4).sum().item())
        cand_list["muon"].append((cand_ids == 5).sum().item())

        gen_p4 = gen_p4.detach()
        pred_p4 = pred_p4.detach()
        cand_p4 = cand_p4.detach()

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

        if len(test_loader) < 5000:
            print(f'event #: {i+1}/{len(test_loader)}')
        else:
            print(f'event #: {i+1}/{5000}')

        if i == 4999:
            break

    print("Average Inference time per event is: ", round((sum(t) / len(t)), 2), 's')

    t1 = time.time()

    print('Time taken to make predictions is:', round(((t1 - t0) / 60), 2), 'min')

    # store the 3 list dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list = [pred_list, gen_list, cand_list]

    torch.save(list, outpath + '/list_for_multiplicities.pt')

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
    predictions = {"ygen": ygen.reshape(1, -1, 7).detach().cpu().numpy(),
                   "ycand": ycand.reshape(1, -1, 7).detach().cpu().numpy(),
                   "ypred": ypred.detach().reshape(1, -1, 7).cpu().numpy()}

    torch.save(predictions, outpath + '/predictions.pt')


def make_plots_delphes(model, test_loader, outpath, target, device, epoch, tag):

    print('Making plots...')
    t0 = time.time()

    # load the necessary predictions to make the plots
    gen_ids = torch.load(outpath + f'/gen_ids.pt', map_location=device)
    gen_p4 = torch.load(outpath + f'/gen_p4.pt', map_location=device)
    pred_ids = torch.load(outpath + f'/pred_ids.pt', map_location=device)
    pred_p4 = torch.load(outpath + f'/pred_p4.pt', map_location=device)
    cand_ids = torch.load(outpath + f'/cand_ids.pt', map_location=device)
    cand_p4 = torch.load(outpath + f'/cand_p4.pt', map_location=device)

    list_for_multiplicities = torch.load(outpath + f'/list_for_multiplicities.pt', map_location=device)

    predictions = torch.load(outpath + f'/predictions.pt', map_location=device)

    # reformat a bit
    ygen = predictions["ygen"].reshape(-1, 7)
    ypred = predictions["ypred"].reshape(-1, 7)
    ycand = predictions["ycand"].reshape(-1, 7)

    # make confusion matrix for MLPF
    target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        pred_ids.cpu(),
                                                        labels=range(6), normalize="true")

    plot_confusion_matrix(conf_matrix_mlpf, target_names, epoch, outpath + '/confusion_matrix_plots/', f'cm_mlpf_epoch_{str(epoch)}')

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        cand_ids.cpu(),
                                                        labels=range(6), normalize="true")

    plot_confusion_matrix(conf_matrix_cand, target_names, epoch, outpath + '/confusion_matrix_plots/', 'cm_cand', target="rule-based")

    # making all the other plots
    if 'QCD' in tag:
        sample = "QCD, 14 TeV, PU200"
    else:
        sample = "$t\\bar{t}$, 14 TeV, PU200"

    # make distribution plots
    plot_distributions_pid(1, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for chhadrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(2, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for nhadrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(3, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for photons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(4, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for electrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(5, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for muons
                           target, epoch, outpath, legend_title=sample + "\n")

    plot_distributions_all(gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for all together
                           target, epoch, outpath, legend_title=sample + "\n")

    # plot particle multiplicity plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_null = plot_particle_multiplicity(list_for_multiplicities, "null", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_null.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_null.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_chhad = plot_particle_multiplicity(list_for_multiplicities, "chhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_chhadron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_chhadron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_nhad = plot_particle_multiplicity(list_for_multiplicities, "nhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_nhadron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_nhadron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_photon = plot_particle_multiplicity(list_for_multiplicities, "photon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_photon.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_photon.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_electron = plot_particle_multiplicity(list_for_multiplicities, "electron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_electron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_electron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_muon = plot_particle_multiplicity(list_for_multiplicities, "muon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_muon.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_muon.pdf", bbox_inches="tight")
    plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "pt", np.linspace(0, 3, 61), outpath + "/efficiency_plots/eff_fake_pid1_pt.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "eta", np.linspace(-3, 3, 61), outpath + "/efficiency_plots/eff_fake_pid1_eta.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "energy", np.linspace(0, 50, 75), outpath + "/efficiency_plots/eff_fake_pid1_energy.png", both=True, legend_title=sample + "\n")

    # make efficiency and fake rate plots for neutral hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "pt", np.linspace(0, 3, 61), outpath + "/efficiency_plots/eff_fake_pid2_pt.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "eta", np.linspace(-3, 3, 61), outpath + "/efficiency_plots/eff_fake_pid2_eta.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "energy", np.linspace(0, 50, 75), outpath + "/efficiency_plots/eff_fake_pid2_energy.png", both=True, legend_title=sample + "\n")

    # make resolution plots for chhadrons: pid=1
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_pt = plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_E = plot_reso(ygen, ypred, ycand, 1, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for nhadrons: pid=2
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_pt = plot_reso(ygen, ypred, ycand, 2, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_eta = plot_reso(ygen, ypred, ycand, 2, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_E = plot_reso(ygen, ypred, ycand, 2, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for photons: pid=3
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_pt = plot_reso(ygen, ypred, ycand, 3, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_eta = plot_reso(ygen, ypred, ycand, 3, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_E = plot_reso(ygen, ypred, ycand, 3, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for electrons: pid=4
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_pt = plot_reso(ygen, ypred, ycand, 4, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_eta = plot_reso(ygen, ypred, ycand, 4, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_E = plot_reso(ygen, ypred, ycand, 4, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for muons: pid=5
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_pt = plot_reso(ygen, ypred, ycand, 5, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_eta = plot_reso(ygen, ypred, ycand, 5, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_E = plot_reso(ygen, ypred, ycand, 5, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
