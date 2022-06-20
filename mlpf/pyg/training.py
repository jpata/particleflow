from pyg import make_plot_from_lists
from pyg.utils_plots import plot_confusion_matrix
from pyg.utils import define_regions, batch_event_into_regions, one_hot_embedding

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


def compute_weights(rank, target_ids, num_classes):
    """
    computes necessary weights to accomodate class imbalance in the loss function
    """

    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(num_classes).to(device=rank)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    # weights[2] = weights[2] * 3  # emphasize nhadrons
    return weights


@torch.no_grad()
def validation_run(rank, model, train_loader, valid_loader, batch_size,
                   alpha, target_type, num_classes, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(rank, model, train_loader, valid_loader, batch_size,
                    optimizer, alpha, target_type, num_classes, outpath)
    return ret


def train(rank, model, train_loader, valid_loader, batch_size,
          optimizer, alpha, target_type, num_classes, outpath):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print(f'---->Initiating a training run on rank {rank}')
        model.train()
        file_loader = train_loader
    else:
        print(f'---->Initiating a validation run rank {rank}')
        model.eval()
        file_loader = valid_loader

    # initialize loss and accuracy and time
    losses_clf, losses_reg, losses_tot, accuracies, t, tf = 0, 0, 0, 0, 0, 0

    # setup confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    t0 = time.time()
    # p = 0
    for num, file in enumerate(file_loader):
        print(f'Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - t0, 3)}s')
        tf = tf + (time.time() - t0)
        file = [x for t in file for x in t]     # unpack the list of tuples to a list

        loader = DataLoader(file, batch_size=batch_size)

        t = 0
        p = p + len(loader)
        for i, X in enumerate(loader):

            # run forward pass
            t0 = time.time()
            pred, target = model(X.to(rank))
            t1 = time.time()
            # print(f'batch {i}/{len(loader)}, forward pass on rank {rank} = {round(t1 - t0, 3)}s, for batch with {X.num_nodes} nodes')
            t = t + (t1 - t0)

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
            target_ids_one_hot = one_hot_embedding(target_ids, num_classes).to(rank)

            # revert one hot encoding for the predictions
            pred_ids = torch.argmax(pred_ids_one_hot, axis=1)

            # define some useful masks
            msk = ((pred_ids != 0) & (target_ids != 0))
            msk2 = ((pred_ids != 0) & (pred_ids == target_ids))

            # compute the loss
            weights = compute_weights(rank, target_ids, num_classes)    # to accomodate class imbalance
            loss_clf = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids, weight=weights)  # for classifying PID
            loss_reg = torch.nn.functional.mse_loss(pred_p4[msk2], target_p4[msk2])  # for regressing p4

            loss_tot = loss_clf + (alpha * loss_reg)

            if is_train:
                for param in model.parameters():    # better than calling optimizer.zero_grad() according to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                    param.grad = None
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

            # if i == 1:
            #     break

        print(f'Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s')
        # if num == 1:
        #     break
    # print(f'Average inference time per batch on rank {rank} is {round((t / p), 3)}s')
    print(f'Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s')

    t0 = time.time()

    losses_clf = (losses_clf / (len(loader) * len(file_loader))).item()
    losses_reg = (losses_reg / (len(loader) * len(file_loader))).item()
    losses_tot = (losses_tot / (len(loader) * len(file_loader))).item()

    accuracies = (accuracies / (len(loader) * len(file_loader))).item()

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_norm


def training_loop(rank, data, model, train_loader, valid_loader,
                  batch_size, n_epochs, patience,
                  optimizer, alpha, target, num_classes, outpath):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        data: data sepecification ('cms' or 'delphes')
        model: a pytorch model wrapped by DistributedDataParallel (DDP)
        dataset: a PFGraphDataset object
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method
        batch_size: how many events to use for the forward pass at a time
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

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        model.train()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_train = train(rank, model, train_loader, valid_loader,
                                                                                  batch_size, optimizer, alpha, target, num_classes, outpath)

        losses_clf_train.append(losses_clf)
        losses_reg_train.append(losses_reg)
        losses_tot_train.append(losses_tot)

        accuracies_train.append(accuracies)

        # validation step
        model.eval()
        losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_val = validation_run(rank, model, train_loader, valid_loader,
                                                                                         batch_size, alpha, target, num_classes, outpath)

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
        time_per_epoch = (t1 - t0) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(f"Rank {rank}: epoch={epoch + 1} / {n_epochs} train_loss={round(losses_tot_train[epoch], 4)} valid_loss={round(losses_tot_valid[epoch], 4)} train_acc={round(accuracies_train[epoch], 4)} valid_acc={round(accuracies_valid[epoch], 4)} stale={stale_epochs} time={round((t1-t0)/60, 2)}m eta={round(eta, 1)}m")

        # save the model's weights
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, f'{outpath}/epoch_{epoch}_weights.pth')

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

        plot_confusion_matrix(conf_matrix_train, target_names, epoch + 1, cm_path, f'epoch_{str(epoch)}_cmTrain')
        plot_confusion_matrix(conf_matrix_val, target_names, epoch + 1, cm_path, f'epoch_{str(epoch)}_cmValid')

        # make loss plots
        make_plot_from_lists('Classification loss',
                             'Epochs', 'Loss', 'loss_clf',
                             [losses_clf_train, losses_clf_valid],
                             ['training', 'validation'],
                             ['clf_losses_train', 'clf_losses_valid'],
                             outpath + '/training_plots/losses/'
                             )
        make_plot_from_lists('Regression loss',
                             'Epochs', 'Loss', 'loss_reg',
                             [losses_reg_train, losses_reg_valid],
                             ['training', 'validation'],
                             ['reg_losses_train', 'reg_losses_valid'],
                             outpath + '/training_plots/losses/'
                             )
        make_plot_from_lists('Total loss',
                             'Epochs', 'Loss', 'loss_tot',
                             [losses_tot_train, losses_tot_valid],
                             ['training', 'validation'],
                             ['tot_losses_train', 'tot_losses_valid'],
                             outpath + '/training_plots/losses/'
                             )

        # make accuracy plots
        make_plot_from_lists('Accuracy',
                             'Epochs', 'Accuracy', 'acc',
                             [accuracies_train, accuracies_valid],
                             ['training', 'validation'],
                             ['acc_train', 'acc_valid'],
                             outpath + '/training_plots/accuracies/'
                             )
        print('----------------------------------------------------------')
    print(f'Done with training. Total training time on rank {rank} is {round((time.time() - t0_initial)/60,3)}min')
