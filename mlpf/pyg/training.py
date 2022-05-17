from pyg import make_plot
from pyg.utils_plots import plot_confusion_matrix

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


def define_regions(num_eta_regions=10, num_phi_regions=10, max_eta=5, min_eta=-5, max_phi=1.5, min_phi=-1.5):
    """
    Defines regions in (eta,phi) space to make bins within an event and build graphs within these bins.

    Returns
        regions: a list of tuples ~ (eta_tuples, phi_tuples) where eta_tuples is a tuple ~ (eta_min, eta_max) that defines the limits of a region and equivalenelty phi
    """
    eta_step = (max_eta - min_eta) / num_eta_regions
    phi_step = (max_phi - min_phi) / num_phi_regions

    tuples_eta = []
    for j in range(num_eta_regions):
        tuple = (min_eta + eta_step * (j), min_eta + eta_step * (j + 1))
        tuples_eta.append(tuple)

    tuples_phi = []
    for i in range(num_phi_regions):
        tuple = (min_phi + phi_step * (i), min_phi + phi_step * (i + 1))
        tuples_phi.append(tuple)

    # make regions
    regions = []
    for i in range(len(tuples_eta)):
        for j in range(len(tuples_phi)):
            regions.append((tuples_eta[i], tuples_phi[j]))

    return regions


def batch_event_into_regions(data, regions):
    """
    Given an event and a set of regions in (eta,phi) space, returns a binned version of the event.

    Args
        data: a Batch() object containing the event and its information
        regions: a tuple of tuples containing the defined regions to bin an event (see define_regions)

    Returns
        data: a modified Batch() object of based on data, where data.batch seperates the events in the different bins
    """

    x = None
    for region in range(len(regions)):
        in_region_msk = (data.x[:, 2] > regions[region][0][0]) & (data.x[:, 2] < regions[region][0][1]) & (torch.arcsin(data.x[:, 3]) > regions[region][1][0]) & (torch.arcsin(data.x[:, 3]) < regions[region][1][1])

        if in_region_msk.sum() != 0:  # if region is not empty
            if x == None:   # first iteration
                x = data.x[in_region_msk]
                ygen = data.ygen[in_region_msk]
                ygen_id = data.ygen_id[in_region_msk]
                ycand = data.ycand[in_region_msk]
                ycand_id = data.ycand_id[in_region_msk]
                batch = region + torch.zeros([len(data.x[in_region_msk])])    # assumes events were already fed one at a time (i.e. batch_size=1)
            else:
                x = torch.cat([x, data.x[in_region_msk]])
                ygen = torch.cat([ygen, data.ygen[in_region_msk]])
                ygen_id = torch.cat([ygen_id, data.ygen_id[in_region_msk]])
                ycand = torch.cat([ycand, data.ycand[in_region_msk]])
                ycand_id = torch.cat([ycand_id, data.ycand_id[in_region_msk]])
                batch = torch.cat([batch, region + torch.zeros([len(data.x[in_region_msk])])])    # assumes events were already fed one at a time (i.e. batch_size=1)

    data = Batch(x=x,
                 ygen=ygen,
                 ygen_id=ygen_id,
                 ycand=ycand,
                 ycand_id=ycand_id,
                 batch=batch.long(),
                 )
    return data


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
def validation_run(device, model, multi_gpu, loader, epoch, alpha, target_type, output_dim_id, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(device, model, multi_gpu, loader, epoch, optimizer, alpha, target_type, output_dim_id, outpath)
    return ret


def train(device, model, multi_gpu, loader, epoch, optimizer, alpha, target_type, output_dim_id, outpath):
    """
    a training block over a given epoch...
    if optimizer is set to None, it freezes the model for a validation_run
    """

    regions = define_regions(num_eta_regions=1, num_phi_regions=1)

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    # initialize placeholders for loss and accuracy
    losses_clf, losses_reg, losses_tot, accuracies = [], [], [], []

    # setup confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    # to compute average inference time
    t = []

    for i, batch in enumerate(loader):

        # batch = batch_event_into_regions(batch, regions)

        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
            # data = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)
            # data = batch.to(device)  # a list (not torch) instance so can't be passed to device
        #
        # for region in range(len(regions)):
        #     in_region_msk = (data.x[:, 2] > regions[region][0][0]) & (data.x[:, 2] < regions[region][0][1]) & (torch.arcsin(data.x[:, 3]) > regions[region][1][0]) & (torch.arcsin(data.x[:, 3]) < regions[region][1][1])
        #
        #     if in_region_msk.sum() != 0:  # if region is not empty
        #         X = Batch(x=data.x[in_region_msk],
        #                   ygen=data.ygen[in_region_msk],
        #                   ygen_id=data.ygen_id[in_region_msk],
        #                   ycand=data.ycand[in_region_msk],
        #                   ycand_id=data.ycand[in_region_msk],
        #                   batch=torch.zeros([len(data.x[in_region_msk])]).long(),    # assumes events were already fed one at a time
        #                   )

        # run forward pass
        t0 = time.time()

        pred, target, _, _ = model(X)
        t1 = time.time()
        t.append(t1 - t0)

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

        losses_clf.append(loss_clf.detach().cpu().item())
        losses_reg.append(loss_reg.detach().cpu().item())
        losses_tot.append(loss_tot.detach().cpu().item())

        accuracies.append(sklearn.metrics.accuracy_score(target_ids[msk].detach().cpu().numpy(), pred_ids[msk].detach().cpu().numpy()))

        conf_matrix += sklearn.metrics.confusion_matrix(target_ids.detach().cpu().numpy(),
                                                        pred_ids.detach().cpu().numpy(),
                                                        labels=range(output_dim_id))

    losses_clf = np.mean(losses_clf)
    losses_reg = np.mean(losses_reg)
    losses_tot = np.mean(losses_tot)

    accuracies = np.mean(accuracies)

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    avg_inference_time = sum(t) / len(t)
    print(f'Average inference time per event is {round(avg_inference_time,3)}s')

    return losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_norm


def training_loop(data, device, model, multi_gpu, train_loader, valid_loader, n_epochs, patience, optimizer, alpha, target, output_dim_id, outpath):
    """
    Main function for training a model
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
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_train = train(device, model, multi_gpu, train_loader, epoch, optimizer, alpha, target, output_dim_id, outpath)

        losses_clf_train.append(losses_clf)
        losses_reg_train.append(losses_reg)
        losses_tot_train.append(losses_tot)

        accuracies_train.append(accuracies)

        # validation step
        model.eval()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_val = validation_run(device, model, multi_gpu, valid_loader, epoch, alpha, target, output_dim_id, outpath)

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
    return
