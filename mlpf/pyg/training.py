from pyg import make_plot
from pyg.utils_plots import plot_confusion_matrix
from pyg.utils import define_regions, batch_event_into_regions
from pyg.dataset import one_hot_embedding

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, DataListLoader

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


def compute_weights(device, target_ids, num_classes):
    """
    computes necessary weights to accomodate class imbalance in the loss function
    """

    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(num_classes).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    # weights[2] = weights[2] * 3  # ephasize nhadrons
    return weights


@torch.no_grad()
def validation_run(device, model, multi_gpu, train_loader, valid_loader, batch_events,
                   alpha, target_type, num_classes, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(device, model, multi_gpu, train_loader, valid_loader, batch_events,
                    optimizer, alpha, target_type, num_classes, outpath)
    return ret


def train(device, model, multi_gpu, train_loader, valid_loader, batch_events,
          optimizer, alpha, target_type, num_classes, outpath):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    # batch events into eta,phi regions to build graphs only within regions
    if batch_events:
        regions = define_regions(num_eta_regions=5, num_phi_regions=5)

    is_train = not (optimizer is None)

    if is_train:
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = valid_loader

    # initialize loss and accuracy and time
    losses_clf, losses_reg, losses_tot, accuracies = 0, 0, 0, 0

    # setup confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    t, num_forward_passes = 0, 0
    t0 = time.time()
    for num, batches_list in enumerate(loader):
        print(f'time to load file {num}/{len(loader)} is {round(time.time() - t0, 3)}s')

        batches_to_loop_over = []
        if multi_gpu:
            num_gpus = 2    # TODO: will fail for more gpus
            for i in range(0, len(l), num_gpus):
                batch_to_loop_over.append([batches_list[i], batches_list[i + 1]])
        else:
            batches_to_loop_over = batches_list

        for i, batch in enumerate(batches_to_loop_over):

            if multi_gpu:   # batch will be a list of Batch() objects so that each element is forwarded to a different gpu
                if batch_events:
                    for i in range(len(batch)):
                        batch[i] = batch_event_into_regions(batch[i], regions)
                X = batch   # a list (not torch) instance so can't be passed to device
            else:
                if batch_events:
                    batch = batch_event_into_regions(batch, regions)
                X = batch.to(device)

            # run forward pass
            t0 = time.time()
            pred, target = model(X)
            t1 = time.time()
            print(f'batch {i}/{len(batches_to_loop_over)}, forward pass = {round(t1 - t0, 3)}s')
            t = t + (t1 - t0)
            num_forward_passes = num_forward_passes + 1

            pred_ids_one_hot = pred[:, :num_classes]
            pred_p4 = pred[:, num_classes:]

            # define target
            if target_type == 'gen':
                target_p4 = target['ygen']
                target_ids = target['ygen_id']
            elif target_type == 'cand':
                target_p4 = target['ycand']
                target_ids = target['ycand_id']

            # one hot encode the target
            target_ids_one_hot = one_hot_embedding(target_ids, num_classes).to(device)

            # revert one hot encoding for the predictions
            pred_ids = torch.argmax(pred_ids_one_hot, axis=1)

            # define some useful masks
            msk = ((pred_ids != 0) & (target_ids != 0))
            msk2 = ((pred_ids != 0) & (pred_ids == target_ids))

            # compute the loss
            weights = compute_weights(device, target_ids, num_classes)    # to accomodate class imbalance
            loss_clf = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids, weight=weights)  # for classifying PID
            loss_reg = torch.nn.functional.mse_loss(pred_p4[msk2], target_p4[msk2])  # for regressing p4

            loss_tot = loss_clf + alpha * loss_reg

            if is_train:
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

            losses_clf = losses_clf + loss_clf.detach().cpu()
            losses_reg = losses_reg + loss_reg.detach().cpu()
            losses_tot = losses_tot + loss_tot.detach().cpu()

            accuracies = accuracies + sklearn.metrics.accuracy_score(target_ids[msk].detach().cpu().numpy(),
                                                                     pred_ids[msk].detach().cpu().numpy())

            conf_matrix += sklearn.metrics.confusion_matrix(target_ids.detach().cpu().numpy(),
                                                            pred_ids.detach().cpu().numpy(),
                                                            labels=range(num_classes))
            if i == 3:
                break
        t0 = 0
    print(f'Average inference time per event is {round((t / num_forward_passes), 3)}s')

    losses_clf = (losses_clf / (len(loader) * len(batches_to_loop_over))).item()
    losses_reg = (losses_reg / (len(loader) * len(batches_to_loop_over))).item()
    losses_tot = (losses_tot / (len(loader) * len(batches_to_loop_over))).item()

    accuracies = (accuracies / (len(loader) * len(batches_to_loop_over))).item()

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_norm


def training_loop(device, data, model, multi_gpu, train_loader, valid_loader,
                  batch_events, n_epochs, patience,
                  optimizer, alpha, target, num_classes, outpath):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        device: 'cpu' or cuda
        data: data sepecification ('cms' or 'delphes')
        model: pytorch model
        multi_gpu: boolean for multi_gpu training (if multigpus are available)
        dataset: a PFGraphDataset object
        n_train: number of files to use for training
        n_train: number of files to use for validation
        batch_size: how many events to use for the forward pass at a time
        batch_events: boolean to batch the event into eta,phi regions so that the graphs are only built within the regions
        loader: pytorch geometric dataloader which is an iterator of Batch() objects where each Batch() is a single event
        n_epochs: number of epochs for a full training
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        alpha: the hyperparameter controlling the classification vs regression task balance (alpha=0 means pure regression, and greater positive values emphasize regression)
        target: 'gen' or 'cand' training
        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
        outpath: path to store the model weights and training plots
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
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_train = train(device, model, multi_gpu, train_loader, valid_loader, batch_events, optimizer, alpha, target, num_classes, outpath)

        losses_clf_train.append(losses_clf)
        losses_reg_train.append(losses_reg)
        losses_tot_train.append(losses_tot)

        accuracies_train.append(accuracies)

        # validation step
        model.eval()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_val = validation_run(device, model, multi_gpu, train_loader, valid_loader, batch_events, alpha, target, num_classes, outpath)

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
            target_names = ["none", "HFEM", "HFHAD", "el", "mu", "g", "n.had", "ch.had", "tau"]

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
