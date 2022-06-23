from pyg.utils_plots import plot_confusion_matrix
from pyg.utils_plots import plot_distributions_pid, plot_distributions_all, plot_particle_multiplicity
from pyg.utils_plots import draw_efficiency_fakerate, plot_reso
from pyg.utils_plots import pid_to_name_delphes, name_to_pid_delphes, pid_to_name_cms, name_to_pid_cms
from pyg.utils import define_regions, batch_event_into_regions
from pyg.utils import one_hot_embedding, target_p4

from pyg.cms_plots import plot_numPFelements, plot_met, plot_sum_energy, plot_sum_pt, plot_energy_res, plot_eta_res, plot_multiplicity

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


def make_predictions(rank, data, model, file_loader, batch_size, num_classes, outpath):
    """
    Runs inference on the qcd test dataset to evaluate performance. Saves the predictions as .pt files.

    Args
        data: data specification ('cms' or 'delphes')
        model: pytorch model
        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
    """

    tt0 = time.time()

    t0, tff = time.time(), 0

    ibatch = 0
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - t0, 3)}s')
        tff = tff + (time.time() - t0)

        file = [x for t in file for x in t]     # unpack the list of tuples to a list

        loader = torch_geometric.loader.DataLoader(file, batch_size=batch_size)

        t = 0
        for i, batch in enumerate(loader):
            np_outfile = f"{outpath}/pred_batch{ibatch}_rank{rank}.npz"

            ti = time.time()
            pred_ids_one_hot, pred_p4 = model(batch.to(rank))

            tf = time.time()
            # print(f'batch {i}/{len(loader)}, forward pass on rank {rank} = {round(tf - ti, 3)}s, for batch with {batch.num_nodes} nodes')
            t = t + (tf - ti)

            # zero pad the events to use the same plotting scripts as the tf pipeline
            padded_num_elem_size = 6400

            pred_ids_one_hot_list = []
            pred_p4_list = []
            for z in range(batch_size):
                pred_ids_one_hot_list.append(pred_ids_one_hot[batch.batch == z])
                pred_p4_list.append(pred_p4[batch.batch == z])

            batch_list = batch.to_data_list()

            outs = {}
            outs[f'X'], outs[f'gen_cls'], outs[f'cand_cls'], outs[f'pred_cls'] = [], [], [], []
            for feat, key in enumerate(target_p4):
                outs[f'gen_{key}'], outs[f'cand_{key}'], outs[f'pred_{key}'] = [], [], []

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

                outs[f'X'].append(vars_padded['X'])
                outs[f'gen_cls'].append(vars_padded['gen_ids_one_hot'])
                outs[f'cand_cls'].append(vars_padded['cand_ids_one_hot'])
                outs[f'pred_cls'].append(vars_padded['pred_ids_one_hot'])

                for feat, key in enumerate(target_p4):
                    outs[f'gen_{key}'].append(vars_padded['ygen'][:, :, feat].reshape(-1, padded_num_elem_size, 1))
                    outs[f'cand_{key}'].append(vars_padded['ycand'][:, :, feat].reshape(-1, padded_num_elem_size, 1))
                    outs[f'pred_{key}'].append(vars_padded['pred_p4'][:, :, feat].reshape(-1, padded_num_elem_size, 1))

            print(f'saving predictions at {np_outfile}')

            out = {}
            for key, value in outs.items():
                out[key] = np.concatenate(value)

            np.savez(
                np_outfile,
                **out
            )
            ibatch += 1

        #     if i == 3:
        #         break
        #
        # if num == 2:
        #     break

        print(f'Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s')

        t0 = time.time()

    print(f'Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s')

    print(f'Time taken to make predictions on rank {rank} is: {round(((time.time() - tt0) / 60), 2)} min')


def load_predictions(path):

    Xs = []
    yvals = {}
    for i, fi in enumerate(list(glob.glob(path + "/pred_batch*.npz"))):
        print(f'loading prediction # {i}/{len(list(glob.glob(path + "/pred_batch*.npz")))}')
        dd = np.load(fi)
        Xs.append(dd["X"])

        keys_in_file = list(dd.keys())
        for k in keys_in_file:
            if k == "X":
                continue
            if not (k in yvals):
                yvals[k] = []
            yvals[k].append(dd[k])

    X = np.concatenate(Xs)

    def flatten(arr):
        # return arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
        return arr.reshape(-1, arr.shape[-1])

    X_f = flatten(X)

    msk_X_f = X_f[:, 0] != 0

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

    return X, yvals_f, yvals


def make_plots(data, num_classes, pred_path, plot_path, target, epoch, sample):

    print('Making plots...')

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms

    pfcands = list(name_to_pid.keys())

    t0 = time.time()

    # load the necessary predictions to make the plots
    X, yvals_f, yvals = load_predictions(pred_path)

    plot_numPFelements(X, plot_path, sample)
    plot_met(X, yvals, plot_path, sample)
    plot_sum_energy(X, yvals, plot_path, sample)
    plot_sum_pt(X, yvals, plot_path, sample)

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

    dic = {1: 1e10,
           2: 1e8}
    for pid, ylim in dic.items():
        plot_eta_res(X, yvals_f, pid, ylim, plot_path, sample)

    plot_multiplicity(X, yvals, plot_path, sample)

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
