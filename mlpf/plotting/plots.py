import sklearn
import sklearn.metrics
import numpy as np
import pandas
import mplhep
import pickle as pkl
import time
import math

import sys
import os.path as osp

import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch.utils.data import random_split
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import mplhep as hep

import plotting

plt.style.use(hep.style.ROOT)

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 3, 4, 5]

# map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}
# map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}

sample_title_qcd = "QCD, 14 TeV, PU200"
sample_title_ttbar = "$t\\bar{t}$, 14 TeV, PU200"

ranges = {
    "pt": np.linspace(0, 10, 61),
    "eta": np.linspace(-5, 5, 61),
    "sphi": np.linspace(-1, 1, 61),
    "cphi": np.linspace(-1, 1, 61),
    "energy": np.linspace(0, 100, 61)
}
pid_names = {
    0: "Null",
    1: "Charged hadrons",
    2: "Neutral hadrons",
    3: "Photons",
    4: "Electrons",
    5: "Muons",
}
key_to_pid = {
    "null": 0,
    "chhadron": 1,
    "nhadron": 2,
    "photon": 3,
    "electron": 4,
    "muon": 5,
}
var_names = {
    "pt": r"$p_\mathrm{T}$ [GeV]",
    "eta": r"$\eta$",
    "sphi": r"$\mathrm{sin} \phi$",
    "cphi": r"$\mathrm{cos} \phi$",
    "energy": r"$E$ [GeV]"
}
var_names_nounit = {
    "pt": r"$p_\mathrm{T}$",
    "eta": r"$\eta$",
    "sphi": r"$\mathrm{sin} \phi$",
    "cphi": r"$\mathrm{cos} \phi$",
    "energy": r"$E$"
}
var_names_bare = {
    "pt": "p_\mathrm{T}",
    "eta": "\eta",
    "energy": "E",
}
var_indices = {
    "pt": 2,
    "eta": 3,
    "sphi": 4,
    "cphi": 5,
    "energy": 6
}


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


def plot_regression(val_x, val_y, var_name, rng, target, fname):
    fig = plt.figure(figsize=(5, 5))
    plt.hist2d(
        val_x,
        val_y,
        bins=(rng, rng),
        cmap="Blues",
        # norm=matplotlib.colors.LogNorm()
    )

    if target == 'cand':
        plt.xlabel("Cand {}".format(var_name))
    elif target == 'gen':
        plt.xlabel("Gen {}".format(var_name))

    plt.ylabel("MLPF {}".format(var_name))

    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig


def plot_particles(fname, true_id, true_p4, pred_id, pred_p4, pid=1):
    # Ground truth vs model prediction particles
    fig = plt.figure(figsize=(10, 10))

    true_p4 = true_p4.detach().cpu().numpy()
    pred_p4 = pred_p4.detach().cpu().numpy()

    msk = (true_id == pid)
    plt.scatter(true_p4[msk, 2], np.arctan2(true_p4[msk, 3], true_p4[msk, 4]), s=2 * true_p4[msk, 2], marker="o", alpha=0.5)

    msk = (pred_id == pid)
    plt.scatter(pred_p4[msk, 2], np.arctan2(pred_p4[msk, 3], pred_p4[msk, 4]), s=2 * pred_p4[msk, 2], marker="o", alpha=0.5)

    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.xlim(-5, 5)
    plt.ylim(-4, 4)

    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig


def plot_distribution(pid, val_x, val_y, var_name, rng, target, fname, legend_title=""):
    plt.style.use(mplhep.style.CMS)

    fig = plt.figure(figsize=(10, 10))

    if target == 'cand':
        plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="cand")
    elif target == 'gen':
        plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="gen")

    plt.hist(val_y, bins=rng, density=True, histtype="step", lw=2, label="MLPF")
    plt.xlabel(var_name)

    if pid != -1:
        plt.legend(frameon=False, title=legend_title + pid_names[pid])
    else:
        plt.legend(frameon=False, title=legend_title)

    plt.ylim(0, 1.5)
    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig


def plot_distributions_pid(pid, true_id, true_p4, pred_id, pred_p4, pf_id, cand_p4, target, epoch, outpath, legend_title=""):
    plt.style.use("default")

    ch_true = true_p4[true_id == pid, 0].flatten().detach().cpu().numpy()
    ch_pred = pred_p4[pred_id == pid, 0].flatten().detach().cpu().numpy()

    pt_true = true_p4[true_id == pid, 1].flatten().detach().cpu().numpy()
    pt_pred = pred_p4[pred_id == pid, 1].flatten().detach().cpu().numpy()

    eta_true = true_p4[true_id == pid, 2].flatten().detach().cpu().numpy()
    eta_pred = pred_p4[pred_id == pid, 2].flatten().detach().cpu().numpy()

    sphi_true = true_p4[true_id == pid, 3].flatten().detach().cpu().numpy()
    sphi_pred = pred_p4[pred_id == pid, 3].flatten().detach().cpu().numpy()

    cphi_true = true_p4[true_id == pid, 4].flatten().detach().cpu().numpy()
    cphi_pred = pred_p4[pred_id == pid, 4].flatten().detach().cpu().numpy()

    e_true = true_p4[true_id == pid, 5].flatten().detach().cpu().numpy()
    e_pred = pred_p4[pred_id == pid, 5].flatten().detach().cpu().numpy()

    figure = plot_distribution(pid, ch_true, ch_pred, "charge", np.linspace(0, 5, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_charge_distribution', legend_title=legend_title)
    figure = plot_distribution(pid, pt_true, pt_pred, "pt", np.linspace(0, 5, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_pt_distribution', legend_title=legend_title)
    figure = plot_distribution(pid, e_true, e_pred, "E", np.linspace(-1, 5, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_energy_distribution', legend_title=legend_title)
    figure = plot_distribution(pid, eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_eta_distribution', legend_title=legend_title)
    figure = plot_distribution(pid, sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_sphi_distribution', legend_title=legend_title)
    figure = plot_distribution(pid, cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), target, fname=outpath + '/distribution_plots/' + pid_names[pid] + '_cphi_distribution', legend_title=legend_title)


def plot_distributions_all(true_id, true_p4, pred_id, pred_p4, pf_id, cand_p4, target, epoch, outpath, legend_title=""):
    plt.style.use("default")

    msk = (pred_id != 0) & (true_id != 0)

    ch_true = true_p4[msk, 0].flatten().detach().cpu().numpy()
    ch_pred = pred_p4[msk, 0].flatten().detach().cpu().numpy()

    pt_true = true_p4[msk, 1].flatten().detach().cpu().numpy()
    pt_pred = pred_p4[msk, 1].flatten().detach().cpu().numpy()

    eta_true = true_p4[msk, 2].flatten().detach().cpu().numpy()
    eta_pred = pred_p4[msk, 2].flatten().detach().cpu().numpy()

    sphi_true = true_p4[msk, 3].flatten().detach().cpu().numpy()
    sphi_pred = pred_p4[msk, 3].flatten().detach().cpu().numpy()

    cphi_true = true_p4[msk, 4].flatten().detach().cpu().numpy()
    cphi_pred = pred_p4[msk, 4].flatten().detach().cpu().numpy()

    e_true = true_p4[msk, 5].flatten().detach().cpu().numpy()
    e_pred = pred_p4[msk, 5].flatten().detach().cpu().numpy()

    figure = plot_distribution(-1, ch_true, ch_pred, "charge", np.linspace(0, 5, 100), target, fname=outpath + '/distribution_plots/all_charge_distribution', legend_title=legend_title)
    figure = plot_distribution(-1, pt_true, pt_pred, "pt", np.linspace(0, 5, 100), target, fname=outpath + '/distribution_plots/all_pt_distribution', legend_title=legend_title)
    figure = plot_distribution(-1, e_true, e_pred, "E", np.linspace(-1, 5, 100), target, fname=outpath + '/distribution_plots/all_energy_distribution', legend_title=legend_title)
    figure = plot_distribution(-1, eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), target, fname=outpath + '/distribution_plots/all_eta_distribution', legend_title=legend_title)
    figure = plot_distribution(-1, sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), target, fname=outpath + '/distribution_plots/all_sphi_distribution', legend_title=legend_title)
    figure = plot_distribution(-1, cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), target, fname=outpath + '/distribution_plots/all_cphi_distribution', legend_title=legend_title)


def midpoints(x):
    return x[:-1] + np.diff(x) / 2


def mask_empty(hist):
    h0 = hist[0].astype(np.float64)
    h0[h0 < 50] = 0
    return (h0, hist[1])


def divide_zero(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    out = np.zeros_like(a)
    np.divide(a, b, where=b > 0, out=out)
    return out


def plot_pt_eta(ygen, legend_title=""):
    plt.style.use(hep.style.ROOT)

    b = np.linspace(0, 100, 41)

    msk_pid1 = (ygen[:, 0] == 1)
    msk_pid2 = (ygen[:, 0] == 2)
    msk_pid3 = (ygen[:, 0] == 3)
    msk_pid4 = (ygen[:, 0] == 4)
    msk_pid5 = (ygen[:, 0] == 5)

    h1 = np.histogram(ygen[msk_pid1, 2], bins=b)
    h2 = np.histogram(ygen[msk_pid2, 2], bins=b)
    h3 = np.histogram(ygen[msk_pid3, 2], bins=b)
    h4 = np.histogram(ygen[msk_pid4, 2], bins=b)
    h5 = np.histogram(ygen[msk_pid5, 2], bins=b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2 * 8))

    xs = midpoints(h1[1])
    width = np.diff(h1[1])

    hep.histplot([h5[0], h4[0], h3[0], h2[0], h1[0]], bins=h1[1], ax=ax1, stack=True, histtype="fill",
                 label=["Muons", "Electrons", "Photons", "Neutral hadrons", "Charged hadrons"])

    ax1.legend(loc="best", frameon=False, title=legend_title)
    ax1.set_yscale("log")
    ax1.set_ylim(1e1, 1e9)
    ax1.set_xlabel(r"Truth particle $p_\mathrm{T}$ [GeV]")
    ax1.set_ylabel("Truth particles")

    b = np.linspace(-8, 8, 41)
    h1 = np.histogram(ygen[msk_pid1, 3], bins=b)
    h2 = np.histogram(ygen[msk_pid2, 3], bins=b)
    h3 = np.histogram(ygen[msk_pid3, 3], bins=b)
    h4 = np.histogram(ygen[msk_pid4, 3], bins=b)
    h5 = np.histogram(ygen[msk_pid5, 3], bins=b)
    xs = midpoints(h1[1])
    width = np.diff(h1[1])

    hep.histplot([h5[0], h4[0], h3[0], h2[0], h1[0]], bins=h1[1], ax=ax2, stack=True, histtype="fill",
                 label=["Muons", "Electrons", "Photons", "Neutral hadrons", "Charged hadrons"])
    leg = ax2.legend(loc="best", frameon=False, ncol=2, title=legend_title)
    leg._legend_box.align = "left"
    ax2.set_yscale("log")
    ax2.set_ylim(1e1, 1e9)
    ax2.set_xlabel("Truth particle $\eta$")
    ax2.set_ylabel("Truth particles")
    return ax1, ax2


def plot_num_particles_pid(list, key, ax=None, legend_title=""):
    plt.style.use(hep.style.ROOT)

    pid = key_to_pid[key]
    if not ax:
        plt.figure(figsize=(4, 4))
        ax = plt.axes()

    cand_list = list[0]
    target_list = list[1]
    pf_list = list[2]

    a = np.array(pf_list[key])
    b = np.array(target_list[key])

    ratio_dpf = (a - b) / b
    ratio_dpf[ratio_dpf > 10] = 10
    ratio_dpf[ratio_dpf < -10] = -10
    mu_dpf = np.mean(ratio_dpf)
    sigma_dpf = np.std(ratio_dpf)

    ax.scatter(
        target_list[key],
        cand_list[key],
        marker="o",
        label="Rule-based PF, $r={0:.3f}$\n$\mu={1:.3f}\\ \sigma={2:.3f}$".format(
            np.corrcoef(a, b)[0, 1], mu_dpf, sigma_dpf
        ),
        alpha=0.5
    )

    c = np.array(cand_list[key])
    b = np.array(target_list[key])

    ratio_mlpf = (c - b) / b
    ratio_mlpf[ratio_mlpf > 10] = 10
    ratio_mlpf[ratio_mlpf < -10] = -10
    mu_mlpf = np.mean(ratio_mlpf)
    sigma_mlpf = np.std(ratio_mlpf)

    ax.scatter(
        target_list[key],
        cand_list[key],
        marker="^",
        label="MLPF, $r={0:.3f}$\n$\mu={1:.3f}\\ \sigma={2:.3f}$".format(
            np.corrcoef(a, b)[0, 1], mu_mlpf, sigma_mlpf
        ),
        alpha=0.5
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.plot(lims, lims, '--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    ax.legend(frameon=False, title=legend_title + pid_names[pid])
    ax.set_xlabel("Truth particles / event")
    ax.set_ylabel("Reconstructed particles / event")
    plt.title("Particle multiplicity")


def draw_efficiency_fakerate(ygen, ypred, ycand, pid, var, bins, outpath, both=True, legend_title=""):
    var_idx = var_indices[var]

    msk_gen = ygen[:, 0] == pid
    msk_pred = ypred[:, 0] == pid
    msk_cand = ycand[:, 0] == pid

    hist_gen = np.histogram(ygen[msk_gen, var_idx], bins=bins)
    hist_cand = np.histogram(ygen[msk_gen & msk_cand, var_idx], bins=bins)
    hist_pred = np.histogram(ygen[msk_gen & msk_pred, var_idx], bins=bins)

    hist_gen = mask_empty(hist_gen)
    hist_cand = mask_empty(hist_cand)
    hist_pred = mask_empty(hist_pred)

    # efficiency plot
    if both:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2 * 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 1 * 8))
        ax2 = None

    #ax1.set_title("reco efficiency for {}".format(pid_names[pid]))
    ax1.errorbar(
        midpoints(hist_gen[1]),
        divide_zero(hist_cand[0], hist_gen[0]),
        divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_cand[0], hist_gen[0]),
        lw=0, label="Rule-based PF", elinewidth=2, marker=".", markersize=10)
    ax1.errorbar(
        midpoints(hist_gen[1]),
        divide_zero(hist_pred[0], hist_gen[0]),
        divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_pred[0], hist_gen[0]),
        lw=0, label="MLPF", elinewidth=2, marker=".", markersize=10)
    ax1.legend(frameon=False, loc=0, title=legend_title + pid_names[pid])
    ax1.set_ylim(0, 1.2)
    # if var=="energy":
    #     ax1.set_xlim(0,30)
    ax1.set_xlabel(var_names[var])
    ax1.set_ylabel("Efficiency")

    hist_cand2 = np.histogram(ygen[msk_cand & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_pred2 = np.histogram(ygen[msk_pred & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_cand_gen2 = np.histogram(ygen[msk_cand & ~msk_gen & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_pred_gen2 = np.histogram(ygen[msk_pred & ~msk_gen & (ygen[:, 0] != 0), var_idx], bins=bins)

    hist_cand2 = mask_empty(hist_cand2)
    hist_cand_gen2 = mask_empty(hist_cand_gen2)
    hist_pred2 = mask_empty(hist_pred2)
    hist_pred_gen2 = mask_empty(hist_pred_gen2)

    if both:
        # fake rate plot
        #ax2.set_title("reco fake rate for {}".format(pid_names[pid]))
        ax2.errorbar(
            midpoints(hist_cand2[1]),
            divide_zero(hist_cand_gen2[0], hist_cand2[0]),
            divide_zero(np.sqrt(hist_cand_gen2[0]), hist_cand2[0]),
            lw=0, label="Rule-based PF", elinewidth=2, marker=".", markersize=10)
        ax2.errorbar(
            midpoints(hist_pred2[1]),
            divide_zero(hist_pred_gen2[0], hist_pred2[0]),
            divide_zero(np.sqrt(hist_pred_gen2[0]), hist_pred2[0]),
            lw=0, label="MLPF", elinewidth=2, marker=".", markersize=10)
        ax2.legend(frameon=False, loc=0, title=legend_title + pid_names[pid])
        ax2.set_ylim(0, 1.0)
        # plt.yscale("log")
        ax2.set_xlabel(var_names[var])
        ax2.set_ylabel("Fake rate")

    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return ax1, ax2


def get_eff(ygen, ypred, ycand):
    msk_gen = (ygen[:, 0] == pid) & (ygen[:, var_indices["pt"]] > 5.0)
    msk_pred = ypred[:, 0] == pid
    msk_cand = ycand[:, 0] == pid

    hist_gen = np.histogram(ygen[msk_gen, var_idx], bins=bins)
    hist_cand = np.histogram(ygen[msk_gen & msk_cand, var_idx], bins=bins)
    hist_pred = np.histogram(ygen[msk_gen & msk_pred, var_idx], bins=bins)

    hist_gen = mask_empty(hist_gen)
    hist_cand = mask_empty(hist_cand)
    hist_pred = mask_empty(hist_pred)

    return {
        "x": midpoints(hist_gen[1]),
        "y": divide_zero(hist_pred[0], hist_gen[0]),
        "yerr": divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_pred[0], hist_gen[0])
    }


def get_fake(ygen, ypred, ycand):
    msk_gen = ygen[:, 0] == pid
    msk_pred = ypred[:, 0] == pid
    msk_cand = ycand[:, 0] == pid

    hist_cand2 = np.histogram(ygen[msk_cand & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_pred2 = np.histogram(ygen[msk_pred & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_cand_gen2 = np.histogram(ygen[msk_cand & ~msk_gen & (ygen[:, 0] != 0), var_idx], bins=bins)
    hist_pred_gen2 = np.histogram(ygen[msk_pred & ~msk_gen & (ygen[:, 0] != 0), var_idx], bins=bins)

    hist_cand2 = mask_empty(hist_cand2)
    hist_cand_gen2 = mask_empty(hist_cand_gen2)
    hist_pred2 = mask_empty(hist_pred2)
    hist_pred_gen2 = mask_empty(hist_pred_gen2)

    return {
        "x": midpoints(hist_pred2[1]),
        "y": divide_zero(hist_pred_gen2[0], hist_pred2[0]),
        "yerr": divide_zero(np.sqrt(hist_pred_gen2[0]), hist_pred2[0])
    }


def plot_reso(ygen, ypred, ycand, pid, var, rng, ax=None, legend_title=""):
    plt.style.use(hep.style.ROOT)

    var_idx = var_indices[var]
    msk = (ygen[:, 0] == pid) & (ycand[:, 0] == pid)
    bins = np.linspace(-rng, rng, 100)
    yg = ygen[msk, var_idx]
    yp = ypred[msk, var_idx]

    yc = ycand[msk, var_idx]
    ratio_mlpf = (yp - yg) / yg
    ratio_dpf = (yc - yg) / yg

    # remove outliers for std value computation
    outlier = 10
    ratio_mlpf[ratio_mlpf < -outlier] = -outlier
    ratio_mlpf[ratio_mlpf > outlier] = outlier
    ratio_dpf[ratio_dpf < -outlier] = -outlier
    ratio_dpf[ratio_dpf > outlier] = outlier

    res_dpf = np.mean(ratio_dpf), np.std(ratio_dpf)
    res_mlpf = np.mean(ratio_mlpf), np.std(ratio_mlpf)

    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.axes()

    #plt.title("{} resolution for {}".format(var_names_nounit[var], pid_names[pid]))
    ax.hist(ratio_dpf, bins=bins, histtype="step", lw=2, label="Rule-based PF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_dpf))
    ax.hist(ratio_mlpf, bins=bins, histtype="step", lw=2, label="MLPF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_mlpf))
    ax.legend(frameon=False, title=legend_title + pid_names[pid])
    ax.set_xlabel("{nounit} resolution, $({bare}^\prime - {bare})/{bare}$".format(nounit=var_names_nounit[var], bare=var_names_bare[var]))
    ax.set_ylabel("Particles")
    #plt.ylim(0, ax.get_ylim()[1]*2)
    ax.set_ylim(1, 1e10)
    ax.set_yscale("log")

    return {"dpf": res_dpf, "mlpf": res_mlpf}


def make_plots(model, test_loader, outpath, target, device, epoch, which_data):

    print('Making plots on ' + which_data)
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
    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        pred_ids.cpu(), labels=range(6), normalize="true")

    plotting.plot_confusion_matrix(conf_matrix_mlpf, ["none", "ch.had", "n.had", "g", "el", "mu"], fname=outpath + '/conf_matrix_mlpf' + str(epoch), epoch=epoch)
    torch.save(conf_matrix_mlpf, outpath + '/conf_matrix_mlpf' + str(epoch) + '.pt')

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        cand_ids.cpu(), labels=range(6), normalize="true")

    plotting.plot_confusion_matrix(conf_matrix_cand, ["none", "ch.had", "n.had", "g", "el", "mu"], fname=outpath + '/conf_matrix_cand' + str(epoch), epoch=epoch)
    torch.save(conf_matrix_cand, outpath + '/conf_matrix_cand' + str(epoch) + '.pt')

    # making all the other plots
    if 'test' in which_data:
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

    # make pt, eta plots to visualize dataset
    ax, _ = plot_pt_eta(ygen)
    plt.savefig(outpath + "/gen_pt_eta.png", bbox_inches="tight")

    # plot particle multiplicity plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_null = plot_num_particles_pid(list_for_multiplicities, "null", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_null.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_chhad = plot_num_particles_pid(list_for_multiplicities, "chhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_chhadron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_nhad = plot_num_particles_pid(list_for_multiplicities, "nhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_nhadron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_photon = plot_num_particles_pid(list_for_multiplicities, "photon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_photon.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_electron = plot_num_particles_pid(list_for_multiplicities, "electron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_electron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    ret_num_particles_muon = plot_num_particles_pid(list_for_multiplicities, "muon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_muon.png", bbox_inches="tight")
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
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_E = plot_reso(ygen, ypred, ycand, 1, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for nhadrons: pid=2
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_pt = plot_reso(ygen, ypred, ycand, 2, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_eta = plot_reso(ygen, ypred, ycand, 2, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_E = plotting.plot_reso(ygen, ypred, ycand, 2, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for photons: pid=3
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_pt = plot_reso(ygen, ypred, ycand, 3, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_eta = plot_reso(ygen, ypred, ycand, 3, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_E = plot_reso(ygen, ypred, ycand, 3, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for electrons: pid=4
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_pt = plot_reso(ygen, ypred, ycand, 4, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_eta = plot_reso(ygen, ypred, ycand, 4, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_E = plot_reso(ygen, ypred, ycand, 4, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for muons: pid=5
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_pt = plot_reso(ygen, ypred, ycand, 5, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_eta = plot_reso(ygen, ypred, ycand, 5, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_E = plot_reso(ygen, ypred, ycand, 5, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
