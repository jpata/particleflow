from pytorch_delphes import make_plot
from pytorch_delphes.utils_plots import plot_confusion_matrix

import torch
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

        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        # run forward pass
        t0 = time.time()
        pred, target = model(X)
        t1 = time.time()
        t.append(t1 - t0)

        pred_ids_one_hot = pred[:, :6]
        pred_p4 = pred[:, 6:]

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
                                                        labels=range(6))

    losses_clf = np.mean(losses_clf)
    losses_reg = np.mean(losses_reg)
    losses_tot = np.mean(losses_tot)

    accuracies = np.mean(accuracies)

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    avg_inference_time = sum(t) / len(t)
    print(f'Average inference time per event is {round(avg_inference_time,3)}s')

    return losses_clf, losses_reg, losses_tot, accuracies, conf_matrix_norm


def training_loop(device, model, multi_gpu, train_loader, valid_loader, n_epochs, patience, optimizer, alpha, target, output_dim_id, outpath):
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
        target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
        plot_confusion_matrix(conf_matrix_train, target_names, epoch, cm_path, f'cmT_epoch_{str(epoch)}')
        plot_confusion_matrix(conf_matrix_val, target_names, epoch, cm_path, f'cmV_epoch_{str(epoch)}')

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
