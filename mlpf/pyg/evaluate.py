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


def make_predictions(rank, data, model, file_loader, batch_size, num_classes, outpath, epoch):
    """
    Runs inference on the qcd test dataset to evaluate performance. Saves the predictions as .pt files.

    Args
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        data: data specification ('cms' or 'delphes')
        model: pytorch model
        file_loader:  a pytorch Dataloader that loads .pt files for training when you invoke the get() method

        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
    """

    ti = time.time()

    # conf_matrix_mlpf = np.zeros((num_classes, num_classes))
    # conf_matrix_pf = np.zeros((num_classes, num_classes))

    yvals = {}
    Xs, yvals[f'gen_cls'], yvals[f'cand_cls'], yvals[f'pred_cls'] = [], [], [], []
    for feat, key in enumerate(target_p4):
        yvals[f'gen_{key}'], yvals[f'cand_{key}'], yvals[f'pred_{key}'] = [], [], []

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
            # print(f'batch {i}/{len(loader)}, forward pass on rank {rank} = {round(t1 - t0, 3)}s, for batch with {batch.num_nodes} nodes')
            t = t + (t1 - t0)

            # conf_matrix_mlpf += sklearn.metrics.confusion_matrix(batch.ygen_id.detach().cpu(), torch.argmax(pred_ids_one_hot, axis=1).detach().cpu(), labels=range(num_classes))
            # conf_matrix_pf += sklearn.metrics.confusion_matrix(batch.ygen_id.detach().cpu(), batch.ycand_id.detach().to('cpu'), labels=range(num_classes))

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

                Xs.append(vars_padded['X'])
                yvals[f'gen_cls'].append(vars_padded['gen_ids_one_hot'])
                yvals[f'cand_cls'].append(vars_padded['cand_ids_one_hot'])
                yvals[f'pred_cls'].append(vars_padded['pred_ids_one_hot'])

                for feat, key in enumerate(target_p4):
                    yvals[f'gen_{key}'].append(vars_padded['ygen'][:, :, feat].reshape(-1, padded_num_elem_size, 1))
                    yvals[f'cand_{key}'].append(vars_padded['ycand'][:, :, feat].reshape(-1, padded_num_elem_size, 1))
                    yvals[f'pred_{key}'].append(vars_padded['pred_p4'][:, :, feat].reshape(-1, padded_num_elem_size, 1))
        #
        #     if i == 2:
        #         break
        #
        # if num == 2:
        #     break

        print(f'Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s')

        t0 = time.time()

    print(f'Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s')

    print(f'Time taken to make predictions on rank {rank} is: {round(((time.time() - ti) / 60), 2)} min')

    print('--> Concatenating all the predictions into giant arrays and dictionaries')

    X = np.concatenate(Xs)

    def flatten(arr):
        # return arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
        return arr.reshape(-1, arr.shape[-1])

    X_f = flatten(X)

    msk_X_f = X_f[:, 0] != 0

    print('further processing to save predictions for convenient plotting')
    yvals = {k: np.concatenate(v) for k, v in yvals.items()}

    for val in ["gen", "cand", "pred"]:
        yvals["{}_phi".format(val)] = np.arctan2(yvals["{}_sin_phi".format(val)], yvals["{}_cos_phi".format(val)])
        yvals["{}_cls_id".format(val)] = np.expand_dims(np.argmax(yvals["{}_cls".format(val)], axis=-1), axis=-1)

        yvals["{}_px".format(val)] = np.sin(yvals["{}_phi".format(val)]) * yvals["{}_pt".format(val)]
        yvals["{}_py".format(val)] = np.cos(yvals["{}_phi".format(val)]) * yvals["{}_pt".format(val)]

    yvals_f = {k: flatten(v) for k, v in yvals.items()}

    # remove the last dim
    for k in yvals_f.keys():
        if yvals_f[k].shape[-1] == 1:
            yvals_f[k] = yvals_f[k][..., -1]

    print(f'saving predictions')
    np.savez(
        f'{outpath}/testing_epoch_{epoch}/predictions/predictions_X_{rank}.npz',
        X=X, X_f=X_f, msk_X_f=msk_X_f
    )

    with open(f'{outpath}/testing_epoch_{epoch}/predictions/predictions_yvals_{rank}.pkl', 'wb') as f:
        pkl.dump(yvals, f)
    with open(f'{outpath}/testing_epoch_{epoch}/predictions/predictions_yvals_f_{rank}.pkl', 'wb') as f:
        pkl.dump(yvals_f, f)

    #
    # # make confusion_matrix plots
    # conf_matrix_mlpf = conf_matrix_mlpf / conf_matrix_mlpf.sum(axis=1)[:, np.newaxis]
    # conf_matrix_pf = conf_matrix_pf / conf_matrix_pf.sum(axis=1)[:, np.newaxis]
    #
    # if data == 'delphes':
    #     target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
    # elif data == 'cms':
    #     target_names = CLASS_NAMES_CMS
    #
    # plot_confusion_matrix(conf_matrix_mlpf, target_names, epoch + 1, f'{outpath}/testing_epoch_{epoch}/plots/', f'confusion_matrix_MLPF')
    # plot_confusion_matrix(conf_matrix_pf, target_names, epoch + 1, f'{outpath}/testing_epoch_{epoch}/plots/', f'confusion_matrix_PF')


def load_predictions(path):

    Xs = []
    yvals = {}
    for i, fi in enumerate(list(glob.glob(path + "/pred_batch*.npz"))):
        print(f'loading prediction # {i+1}/{len(list(glob.glob(path + "/pred_batch*.npz")))}')
        dd = np.load(fi)
        Xs.append(dd["X"])

        keys_in_file = list(dd.keys())
        for k in keys_in_file:
            if k == "X":
                continue
            if not (k in yvals):
                yvals[k] = []
            yvals[k].append(dd[k])

    print('--> Concatenating all the predictions into one big numpy array')
    X = np.concatenate(Xs)

    def flatten(arr):
        # return arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
        return arr.reshape(-1, arr.shape[-1])

    X_f = flatten(X)

    msk_X_f = X_f[:, 0] != 0

    print('further processing to make plots')
    yvals = {k: np.concatenate(v) for k, v in yvals.items()}

    for val in ["gen", "cand", "pred"]:
        yvals["{}_phi".format(val)] = np.arctan2(yvals["{}_sin_phi".format(val)], yvals["{}_cos_phi".format(val)])
        yvals["{}_cls_id".format(val)] = np.expand_dims(np.argmax(yvals["{}_cls".format(val)], axis=-1), axis=-1)

        yvals["{}_px".format(val)] = np.sin(yvals["{}_phi".format(val)]) * yvals["{}_pt".format(val)]
        yvals["{}_py".format(val)] = np.cos(yvals["{}_phi".format(val)]) * yvals["{}_pt".format(val)]

    yvals_f = {k: flatten(v) for k, v in yvals.items()}

    # remove the last dim
    for k in yvals_f.keys():
        if yvals_f[k].shape[-1] == 1:
            yvals_f[k] = yvals_f[k][..., -1]

    return X, yvals_f, yvals, X_f, msk_X_f


def make_plots(data, num_classes, pred_path, plot_path, target, epoch, sample):

    print('Loading predictions...')

    t0 = time.time()

    # load the predictions to make the plots
    # X, yvals_f, yvals, X_f, msk_X_f = load_predictions(pred_path)

    d = np.load(f'{pred_path}/predictions_X.npz', allow_pickle=True)
    X = d['X']
    X_f = d['X_f']
    msk_X_f = d['msk_X_f']

    with open(f'{pred_path}/predictions_yvals_f.pkl', 'rb') as f:
        yvals_f = pkl.load(f)
    with open(f'{pred_path}/predictions_yvals.pkl', 'rb') as f:
        yvals = pkl.load(f)

    print('Making plots...')

    # plot distributions
    print('plot_dist...')
    plot_dist(yvals_f, 'pt', np.linspace(0, 200, 61), r'$p_T$', plot_path, sample)
    plot_dist(yvals_f, 'energy', np.linspace(0, 2000, 61), r'$E$', plot_path, sample)
    plot_dist(yvals_f, 'eta', np.linspace(-6, 6, 61), r'$\eta$', plot_path, sample)

    # plot cm
    print('plot_cm...')
    plot_cm(yvals_f, msk_X_f, 'pred_cls_id', 'MLPF', plot_path)
    plot_cm(yvals_f, msk_X_f, 'pred_cls_id', 'PF', plot_path)

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

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
