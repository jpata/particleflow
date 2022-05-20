from pyg import make_plot
from pyg.utils_plots import plot_confusion_matrix
from pyg.utils import define_regions, batch_event_into_regions

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

import mplhep as hep
import matplotlib.pyplot as plt
import os
import pickle as pkl
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use("Agg")


# Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')


def compute_weights(device, target_ids, output_dim_id):
    """
    computes necessary weights to accomodate class imbalance in the loss function
    """

    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(output_dim_id).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    # weights[2] = weights[2] * 3
    return weights


@torch.no_grad()
def validation_run(device, model, multi_gpu, batch_events, loader,
                   alpha, target_type, output_dim_id, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(device, model, multi_gpu, batch_events, loader, optimizer, alpha, target_type, output_dim_id, outpath)
    return ret


def train(device, model, multi_gpu, batch_events, loader, optimizer,
          alpha, target_type, output_dim_id, outpath):
    """
    A training/validation block over a given epoch. If optimizer is set to None, it freezes the model for a validation_run.

    Args:
        device: 'cpu' or cuda
        model: pytorch model
        multi_gpu: boolean for multi_gpu training (if multigpus are available)
        batch_events: boolean to batch the event into eta,phi regions so that the graphs are only built within the regions
        loader: pytorch geometric dataloader which is an iterator of Batch() objects where each Batch() is a single event
        optimizer: optimizer to use for training (by default: Adam). If set to None, it freezes the model for a validation_run.
        alpha: the hyperparameter controlling the classification vs regression task balance (alpha=0 means pure regression, and greater positive values emphasize regression)
        target_type: 'gen' or 'cand' training
        output_dim_id: number of particle candidate classes to predict (6 for delphes, 9 for cms)
        outpath: path to store the model weights and training plots
    """

    if batch_events:
        regions = define_regions(num_eta_regions=10, num_phi_regions=10)

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    # initialize loss and accuracy
    losses_clf, losses_reg, losses_tot, accuracies = 0, 0, 0, 0

    # setup confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    # to compute average inference time
    t = 0

    for i, batch in enumerate(loader):

        if batch_events:    # batch events into eta,phi regions to build graphs only within regions
            batch = batch_event_into_regions(batch, regions)

        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        # run forward pass
        t0 = time.time()
        pred, target, _, _ = model(X)
        t1 = time.time()
        t = t + (t1 - t0)

        pred_ids_one_hot = pred[:, :output_dim_id]
        pred_p4 = pred[:, output_dim_id:]

        # define target
        if target_type == 'gen':
            target_ids_one_hot = target['ygen_id']
            target_p4 = target['ygen']
        elif target_type == 'cand':
            target_ids_one_hot = target['ycand_id']
            target_p4 = target['ycand']

        # revert one hot encoding
        _, target_ids = torch.max(target_ids_one_hot, -1)
        _, pred_ids = torch.max(pred_ids_one_hot, -1)

        # define some useful masks
        msk = ((pred_ids != 0) & (target_ids != 0))
        msk2 = ((pred_ids != 0) & (pred_ids == target_ids))

        # computing loss
        weights = compute_weights(device, target_ids, output_dim_id)    # to accomodate class imbalance
        loss_clf = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids, weight=weights)  # for classifying PID
        loss_reg = torch.nn.functional.mse_loss(pred_p4[msk2], target_p4[msk2])  # for regressing p4

        loss_tot = loss_clf + alpha * loss_reg

        if is_train:
            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

        losses_clf = losses_clf + loss_clf
        losses_reg = losses_reg + loss_reg
        losses_tot = losses_tot + loss_tot

        accuracies = accuracies + sklearn.metrics.accuracy_score(target_ids[msk].detach().cpu().numpy(),
                                                                 pred_ids[msk].detach().cpu().numpy())

        conf_matrix += sklearn.metrics.confusion_matrix(target_ids.detach().cpu().numpy(),
                                                        pred_ids.detach().cpu().numpy(),
                                                        labels=range(output_dim_id))

    losses_clf = (losses_clf / len(loader)).item()
    losses_reg = (losses_reg / len(loader)).item()
    losses_tot = (losses_tot / len(loader)).item()

    accuracies = (accuracies / len(loader)).item()

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    print(f'Average inference time per event is {round((t / len(loader)),3)}s')

    return losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_norm


def training_loop(data, device, model, multi_gpu, batch_events,
                  train_loader, valid_loader, n_epochs, patience,
                  optimizer, alpha, target, output_dim_id, outpath):
    """
    Main function for training a model. Will call the train() and validation_run() functions every epoch.

    Args:
        n_epochs: number of epochs for a full training
        patience: number of stale epochs allowed before stopping the training
    """

    t0_initial = time.time()

    losses_clf_train, losses_reg_train, losses_tot_train = [], [], []
    losses_clf_valid, losses_reg_valid, losses_tot_valid = [], [], []

    accuracies_train, accuracies_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(n_epochs))
    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        model.train()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_train = train(device, model, multi_gpu, batch_events, train_loader, optimizer, alpha, target, output_dim_id, outpath)

        losses_clf_train.append(losses_clf)
        losses_reg_train.append(losses_reg)
        losses_tot_train.append(losses_tot)

        accuracies_train.append(accuracies)

        # validation step
        model.eval()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_val = validation_run(device, model, multi_gpu, batch_events, valid_loader, alpha, target, output_dim_id, outpath)

        losses_clf_valid.append(losses_clf)
        losses_reg_valid.append(losses_reg)
        losses_tot_valid.append(losses_tot)

        accuracies_valid.append(accuracies)

        # early-stopping
        if losses_tot < best_val_loss:
            best_val_loss = losses_tot
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(f"epoch={epoch + 1} / {n_epochs} train_loss={round(losses_tot_train[epoch], 4)} valid_loss={round(losses_tot_valid[epoch], 4)} train_acc={round(accuracies_train[epoch], 4)} valid_acc={round(accuracies_valid[epoch], 4)} stale={stale_epochs} eta={round(eta, 1)}m")

        # save the model's weights
        torch.save(model.state_dict(), f'{outpath}/epoch_{epoch}_weights.pth')

        # create directory to hold training plots
        if not os.path.exists(outpath + '/training_plots/'):
            os.makedirs(outpath + '/training_plots/')

        # make confusion matrix plots
        cm_path = outpath + '/training_plots/confusion_matrix_plots/'
        if not os.path.exists(cm_path):
            os.makedirs(cm_path)

        if data == 'delphes':
            target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
        elif data == 'cms':
            target_names = ["none", "HFEM", "HFHAD", "el", "mu", "g", "n.had", "ch.had"]

        plot_confusion_matrix(conf_matrix_train, target_names, epoch + 1, cm_path, f'cmT_epoch_{str(epoch)}')
        plot_confusion_matrix(conf_matrix_val, target_names, epoch + 1, cm_path, f'cmV_epoch_{str(epoch)}')

    # make loss plots
    make_plot(losses_clf_train, 'train loss_clf', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_clf_train')
    make_plot(losses_reg_train, 'train loss_reg', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_reg_train')
    make_plot(losses_tot_train, 'train loss_tot', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_tot_train')

    make_plot(losses_clf_valid, 'valid loss_clf', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_clf_valid')
    make_plot(losses_reg_valid, 'valid loss_reg', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_reg_valid')
    make_plot(losses_tot_valid, 'valid loss_tot', 'Epochs', 'Loss', outpath + '/training_plots/losses/', 'losses_tot_valid')

    # make accuracy plots
    make_plot(accuracies_train, 'train accuracy', 'Epochs', 'Accuracy', outpath + '/training_plots/accuracies/', 'accuracies_train')
    make_plot(accuracies_valid, 'valid accuracy', 'Epochs', 'Accuracy', outpath + '/training_plots/accuracies/', 'accuracies_valid')

    print('Done with training.')
