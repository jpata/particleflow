import numpy as np
import pandas
import torch
import os.path as osp
import itertools
import pickle as pkl

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)


pid_to_name_delphes = {
    0: "Null",
    1: "Charged hadrons",
    2: "Neutral hadrons",
    3: "Photons",
    4: "Electrons",
    5: "Muons",
}

pid_to_name_cms = {0: 'null',
                   1: 'HFEM',
                   2: 'HFHAD',
                   3: 'ele',
                   4: 'mu',
                   5: 'photon',
                   6: 'nhadron',
                   7: 'chhadron',
                   8: 'tau',
                   }

name_to_pid_delphes = {'null': 0,
                       'chhadron': 1,
                       'nhadron': 2,
                       'photon': 3,
                       'ele': 4,
                       'mu': 5,
                       }

name_to_pid_cms = {'null': 0,
                   'HFEM': 1,
                   'HFHAD': 2,
                   'ele': 3,
                   'mu': 4,
                   'photon': 5,
                   'nhadron': 6,
                   'chhadron': 7,
                   'tau': 8,
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
    "charge": 0,
    "pt": 1,
    "eta": 2,
    "sphi": 3,
    "cphi": 4,
    "energy": 5,
}

bins = {'charge': np.linspace(0, 5, 100),
        'pt': np.linspace(0, 5, 100),
        'E': np.linspace(-1, 5, 100),
        'eta': np.linspace(-5, 5, 100),
        'sin phi': np.linspace(-2, 2, 100),
        'cos phi': np.linspace(-2, 2, 100),
        }


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


def plot_distribution(data, pid, target, mlpf, var_name, rng, target_type, fname, legend_title=""):
    """
    plot distributions for the target and mlpf of a given feature for a given PID
    """
    plt.style.use(hep.style.CMS)

    if data == 'delphes':
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        pid_to_name = pid_to_name_cms

    fig = plt.figure(figsize=(10, 10))

    if target_type == 'cand':
        plt.hist(target, bins=rng, density=True, histtype="step", lw=2, label="cand")
    elif target_type == 'gen':
        plt.hist(target, bins=rng, density=True, histtype="step", lw=2, label="gen")

    plt.hist(mlpf, bins=rng, density=True, histtype="step", lw=2, label="MLPF")
    plt.xlabel(var_name)

    if pid != -1:
        plt.legend(frameon=False, title=legend_title + pid_to_name[pid])
    else:
        plt.legend(frameon=False, title=legend_title)

    plt.ylim(0, 1.5)
    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig


def plot_distributions_pid(data, pid, true_id, true_p4, pred_id, pred_p4, pf_id, cand_p4, target, epoch, outpath, legend_title=""):
    """
    plot distributions for the target and mlpf of the regressed features for a given PID
    """
    plt.style.use("default")

    if data == 'delphes':
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        pid_to_name = pid_to_name_cms

    for i, bin_dict in enumerate(bins.items()):
        true = true_p4[true_id == pid, i].flatten().detach().cpu().numpy()
        pred = pred_p4[pred_id == pid, i].flatten().detach().cpu().numpy()
        figure = plot_distribution(data, pid, true, pred, bin_dict[0], bin_dict[1], target, fname=outpath + '/distribution_plots/' + pid_to_name[pid] + f'_{bin_dict[0]}_distribution', legend_title=legend_title)


def plot_distributions_all(data, true_id, true_p4, pred_id, pred_p4, pf_id, cand_p4, target, epoch, outpath, legend_title=""):
    """
    plot distributions for the target and mlpf of a all features, merging all PIDs
    """
    plt.style.use("default")

    msk = (pred_id != 0) & (true_id != 0)
    if data == 'delphes':
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        pid_to_name = pid_to_name_cms

    for i, bin_dict in enumerate(bins.items()):
        true = true_p4[msk, i].flatten().detach().cpu().numpy()
        pred = pred_p4[msk, i].flatten().detach().cpu().numpy()
        figure = plot_distribution(data, -1, true, pred, bin_dict[0], bin_dict[1], target, fname=outpath + f'/distribution_plots/all_{bin_dict[0]}_distribution', legend_title=legend_title)


def plot_particle_multiplicity(data, list, key, ax=None, legend_title=""):
    """
    plot particle multiplicity for PF and mlpf
    """
    plt.style.use(hep.style.ROOT)

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms
        pid_to_name = pid_to_name_cms

    pid = name_to_pid[key]
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
    ax.legend(frameon=False, title=legend_title + pid_to_name[pid])
    ax.set_xlabel("Truth particles / event")
    ax.set_ylabel("Reconstructed particles / event")
    plt.title("Particle multiplicity")


def draw_efficiency_fakerate(data, ygen, ypred, ycand, pid, var, bins, outpath, both=True, legend_title=""):
    if data == 'delphes':
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        pid_to_name = pid_to_name_cms

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

    # ax1.set_title("reco efficiency for {}".format(pid_to_name_delphes[pid]))
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
    ax1.legend(frameon=False, loc=0, title=legend_title + pid_to_name[pid])
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
        # ax2.set_title("reco fake rate for {}".format(pid_to_name_delphes[pid]))
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
        ax2.legend(frameon=False, loc=0, title=legend_title + pid_to_name[pid])
        ax2.set_ylim(0, 1.0)
        # plt.yscale("log")
        ax2.set_xlabel(var_names[var])
        ax2.set_ylabel("Fake rate")

    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return ax1, ax2


def plot_reso(data, ygen, ypred, ycand, pfcand, var, outpath, legend_title=""):
    plt.style.use(hep.style.ROOT)

    if data == 'delphes':
        name_to_pid = name_to_pid_delphes
        pid_to_name = pid_to_name_delphes
    elif data == 'cms':
        name_to_pid = name_to_pid_cms
        pid_to_name = pid_to_name_cms
    pid = name_to_pid[pfcand]

    var_idx = var_indices[var]
    msk = (ygen[:, 0] == pid) & (ycand[:, 0] == pid)
    bins = np.linspace(-2, 2, 100)
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(ratio_dpf, bins=bins, histtype="step", lw=2, label="Rule-based PF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_dpf))
    ax.hist(ratio_mlpf, bins=bins, histtype="step", lw=2, label="MLPF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_mlpf))
    ax.legend(frameon=False, title=legend_title + pfcand)
    ax.set_xlabel("{nounit} resolution, $({bare}^\prime - {bare})/{bare}$".format(nounit=var_names_nounit[var], bare=var_names_bare[var]))
    ax.set_ylabel("Particles")
    ax.set_ylim(1, 1e10)
    ax.set_yscale("log")
    plt.savefig(outpath + f"/resolution_plots/res_{pfcand}_{var}.png", bbox_inches="tight")
    plt.savefig(outpath + f"/resolution_plots/res_{pfcand}_{var}.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)


def plot_confusion_matrix(cm, target_names,
                          epoch, outpath, save_as,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, target=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    plt.style.use('default')

    # # only true if it weren't normalized:
    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    if len(target_names) > 6:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(5, 4))

    ax = plt.axes()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if target == "rule-based":
        plt.title(title + ' for rule-based PF')
    else:
        plt.title(title + ' for MLPF at epoch ' + str(epoch))

    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlim(-1, len(target_names))
    plt.ylim(-1, len(target_names))
    plt.xlabel('Predicted label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.savefig(outpath + save_as + '.pdf')
    plt.close(fig)

    # torch.save(cm, outpath + save_as + '.pt')

    return fig, ax
