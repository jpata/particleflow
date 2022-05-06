import sklearn
import sklearn.metrics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import mplhep
import math

import sys
import os.path as osp

from plot_utils import plot_confusion_matrix, cms_label, particle_label, sample_label
from plot_utils import plot_E_reso, plot_eta_reso, plot_phi_reso, bins
import torch

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 3, 4, 5]

# map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}
# map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}


def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi


def mse_unreduced(true, pred):
    return torch.square(true - pred)

# computes accuracy of PID predictions given a one_hot_embedding: truth & pred


def accuracy(true_id, pred_id):
    # revert one_hot_embedding
    _, true_id = torch.max(true_id, -1)
    _, pred_id = torch.max(pred_id, -1)

    is_true = (true_id != 0)
    is_same = (true_id == pred_id)

    acc = (is_same & is_true).sum() / is_true.sum()
    return acc

# computes the resolution given a one_hot_embedding truth & pred + p4 of truth & pred


def energy_resolution(true_id, true_p4, pred_id, pred_p4):
    # revert one_hot_embedding
    _, true_id = torch.max(true_id, -1)
    _, pred_id = torch.max(pred_id, -1)

    msk = (true_id != 0)

    return mse_unreduced(true_p4[msk], pred_p4[msk])


def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Reconstructed PID (normed to gen)")
    plt.xlabel("MLPF PID")
    plt.ylabel("Gen PID")
    plt.xticks(range(6), ["none", "ch.had", "n.had", "g", "el", "mu"])
    plt.yticks(range(6), ["none", "ch.had", "n.had", "g", "el", "mu"])
    plt.colorbar()
    plt.tight_layout()
    return fig


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    import imageio

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig('name', format='png')
    plt.close(figure)

    image = imageio.imread('name')
    return image


def plot_regression(val_x, val_y, var_name, rng, fname):
    fig = plt.figure(figsize=(5, 5))
    plt.hist2d(
        val_x,
        val_y,
        bins=(rng, rng),
        cmap="Blues",
        # norm=matplotlib.colors.LogNorm()
    )
    plt.xlabel("Gen {}".format(var_name))
    plt.ylabel("MLPF {}".format(var_name))

    plt.savefig(fname + '.png')
    return fig


def plot_distributions(val_x, val_y, var_name, rng, fname):
    fig = plt.figure(figsize=(5, 5))
    plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="gen")
    plt.hist(val_y, bins=rng, density=True, histtype="step", lw=2, label="MLPF")
    plt.xlabel(var_name)
    plt.legend(loc="best", frameon=False)
    plt.ylim(0, 1.5)

    plt.savefig(fname + '.png')
    return fig


def plot_particles(fname, true_id, true_p4, pred_id, pred_p4, pid=1):
    # Ground truth vs model prediction particles
    fig = plt.figure(figsize=(10, 10))

    true_p4 = true_p4.detach().numpy()
    pred_p4 = pred_p4.detach().numpy()

    msk = (true_id == pid)
    plt.scatter(true_p4[msk, 2], np.arctan2(true_p4[msk, 3], true_p4[msk, 4]), s=2 * true_p4[msk, 2], marker="o", alpha=0.5)

    msk = (pred_id == pid)
    plt.scatter(pred_p4[msk, 2], np.arctan2(pred_p4[msk, 3], pred_p4[msk, 4]), s=2 * pred_p4[msk, 2], marker="o", alpha=0.5)

    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.xlim(-5, 5)
    plt.ylim(-4, 4)

    plt.savefig(fname + '.png')
    return fig


def make_plots(true_id, true_p4, pred_id, pred_p4, out):

    num_output_classes = len(class_labels)

    _, true_id = torch.max(true_id, -1)
    _, pred_id = torch.max(pred_id, -1)

    cm = sklearn.metrics.confusion_matrix(
        true_id,
        pred_id, labels=list(range(num_output_classes)))
    cm_normed = sklearn.metrics.confusion_matrix(
        true_id,
        pred_id, labels=list(range(num_output_classes)), normalize="true")

    figure = plot_confusion_matrix(cm)

    figure = plot_confusion_matrix(cm_normed)

    msk = (pred_id != 0) & (true_id != 0)

    ch_true = true_p4[msk, 0].flatten().detach().numpy()
    ch_pred = pred_p4[msk, 0].flatten().detach().numpy()

    pt_true = true_p4[msk, 1].flatten().detach().numpy()
    pt_pred = pred_p4[msk, 1].flatten().detach().numpy()

    e_true = true_p4[msk, 5].flatten().detach().numpy()
    e_pred = pred_p4[msk, 5].flatten().detach().numpy()

    eta_true = true_p4[msk, 2].flatten().detach().numpy()
    eta_pred = pred_p4[msk, 2].flatten().detach().numpy()

    sphi_true = true_p4[msk, 3].flatten().detach().numpy()
    sphi_pred = pred_p4[msk, 3].flatten().detach().numpy()

    cphi_true = true_p4[msk, 4].flatten().detach().numpy()
    cphi_pred = pred_p4[msk, 4].flatten().detach().numpy()

    figure = plot_regression(ch_true, ch_pred, "charge", np.linspace(-2, 2, 100), fname=out + 'charge_regression')

    figure = plot_regression(pt_true, pt_pred, "pt", np.linspace(0, 5, 100), fname=out + 'pt_regression')

    figure = plot_distributions(pt_true, pt_pred, "pt", np.linspace(0, 5, 100), fname=out + 'pt_distribution')

    figure = plot_regression(e_true, e_pred, "E", np.linspace(-1, 5, 100), fname=out + 'energy_regression')

    figure = plot_distributions(e_true, e_pred, "E", np.linspace(-1, 5, 100), fname=out + 'energy_distribution')

    figure = plot_regression(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), fname=out + 'eta_regression')

    figure = plot_distributions(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), fname=out + 'eta_distribution')

    figure = plot_regression(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), fname=out + 'sphi_regression')

    figure = plot_distributions(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), fname=out + 'sphi_distribution')

    figure = plot_regression(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), fname=out + 'cphi_regression')

    figure = plot_distributions(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), fname=out + 'cphi_distribution')

    figure = plot_particles(out + 'particleID1', true_id, true_p4, pred_id, pred_p4, pid=1)

    figure = plot_particles(out + 'particleID2', true_id, true_p4, pred_id, pred_p4, pid=2)
