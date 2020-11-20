import matplotlib.pyplot as plt
import numpy as np
import mplhep
import os.path as osp

pid_to_text = {
    211: r"charged hadrons ($\pi^\pm$, ...)",
    130: r"neutral hadrons (K, ...)",
    1: r"HF hadron (EM)",
    2: r"HF-HAD hadron (HAD)",
    11: r"$e^{\pm}$",
    13: r"$\mu^{\pm}$",
    22: r"$\gamma$",
}

bins = {
    211: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 61),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    -211: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    130: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    22: {
        "E_val": np.linspace(0, 2, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-2, 2, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    11: {
        "E_val": np.linspace(0, 10, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    13: {
        "E_val": np.linspace(0, 10, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    1: {
        "E_val": np.linspace(0, 100, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-6, 6, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    2: {
        "E_val": np.linspace(0, 50, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-6, 6, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "Energy [GeV]",
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    }
}

def get_eff(df, pid):
    v0 = np.sum(df==pid)
    return v0 / len(df), np.sqrt(v0)/len(df)

def get_fake(df, pid):
    v0 = np.sum(df!=pid)
    return v0 / len(df), np.sqrt(v0)/len(df)

def cms_label(x0=0.12, x1=0.23, x2=0.67, y=0.90):
    plt.figtext(x0, y,'CMS',fontweight='bold', wrap=True, horizontalalignment='left', fontsize=12)
    plt.figtext(x1, y,'Simulation Preliminary', style='italic', wrap=True, horizontalalignment='left', fontsize=10)
    plt.figtext(x2, y,'Run 3 (14 TeV)',  wrap=True, horizontalalignment='left', fontsize=10)

def sample_label(ax, y=0.98):
    plt.text(0.03, y, "$\mathrm{t}\overline{\mathrm{t}}$ events", va="top", ha="left", size=10, transform=ax.transAxes)

def particle_label(ax, pid):
    plt.text(0.03, 0.92, pid_to_text[pid], va="top", ha="left", size=10, transform=ax.transAxes)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
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
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    return fig, ax


def plot_E_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins, target='target', outpath='./'):
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    hist = np.histogram2d(v0[msk_both, 0], v0[msk_both, 1], bins=(bins["E_val"], bins["E_val"]))
    mplhep.hist2dplot(hist[0], hist[1], hist[2], cmap="Blues", cbar=False);
    plt.xlabel(bins["true_val"] + " " + bins["E_xlabel"])
    plt.ylabel(bins["pred_val"]+ " " + bins["E_xlabel"])
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.plot(
        [bins["E_val"][0], bins["E_val"][-1]],
        [bins["E_val"][0], bins["E_val"][-1]],
        color="black", ls="--", lw=0.5)
    plt.savefig(osp.join(outpath,"energy_2d_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    plt.hist(v0[msk_true, 0], bins=bins["E_val"], density=1.0, histtype="step", lw=2, label=bins["true_val"]);
    plt.hist(v0[msk_pred, 1], bins=bins["E_val"], density=1.0, histtype="step", lw=2, label=bins["pred_val"]);
    plt.xlabel(bins["E_xlabel"])
    plt.ylabel("number of particles\n(normalized, a.u.)")
    plt.legend(frameon=False)
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    ax.set_ylim(ax.get_ylim()[0], 1.5*ax.get_ylim()[1])
    plt.savefig(osp.join(outpath,"energy_hist_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])

    res = (v0[msk_both, 1] - v0[msk_both, 0])/v0[msk_both, 0]
    res[np.isnan(res)] = -1

    plt.figure(figsize=(4,4))
    ax = plt.axes()
    ax.text(0.98, 0.98, "avg. $\Delta E / E$\n$%.2f \pm %.2f$"%(np.mean(res), np.std(res)), transform=ax.transAxes, ha="right", va="top")
    plt.hist(res, bins=bins["E_res"], density=1.0);
    plt.xlabel("$\Delta E / E$")
    plt.ylabel("number of particles\n(normalized, a.u.)")
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.savefig(osp.join(outpath,"energy_ratio_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    #efficiency vs fake rate
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    big_df["bins_{}_e".format(target)] = np.searchsorted(bins["E_val"], big_df["{}_e".format(target)])
    big_df["bins_pred_e"] = np.searchsorted(bins["E_val"], big_df["pred_e"])

    vals_eff = big_df[(big_df["{}_pid".format(target)]==pid)].groupby("bins_{}_e".format(target))["pred_pid"].apply(get_eff, pid)
    vals_fake = big_df[(big_df["pred_pid"]==pid)].groupby("bins_pred_e")["{}_pid".format(target)].apply(get_fake, pid)

    out_eff = np.zeros((len(bins["E_val"]), 2))
    out_fake = np.zeros((len(bins["E_val"]), 2))
    for ib in range(len(bins["E_val"])):
        if ib in vals_eff.keys():
            out_eff[ib, 0] = vals_eff[ib][0]
            out_eff[ib, 1] = vals_eff[ib][1]
        if ib in vals_fake.keys():
            out_fake[ib, 0] = vals_fake[ib][0]
            out_fake[ib, 1] = vals_fake[ib][1]

    cms_label()
    sample_label(ax)
    particle_label(ax, pid)

    plt.errorbar(bins["E_val"], out_eff[:, 0], out_eff[:, 1], marker=".", lw=0, elinewidth=1.0, color="green", label="efficiency")
    plt.ylabel("efficiency\nN(pred|true) / N(true)")
    ax.set_ylim(0, 1.5)
    plt.xlabel(bins["E_xlabel"])

    ax2 = ax.twinx()
    col = "red"
    plt.errorbar(bins["E_val"], out_fake[:, 0], out_fake[:, 1], marker=".", lw=0, elinewidth=1.0, color=col, label="fake rate")
    plt.ylabel("fake rate\nN(true|pred) / N(pred)")
    plt.xlabel(bins["E_xlabel"])
    ax2.set_ylim(0, 1.5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, frameon=False)
    plt.savefig(osp.join(outpath,"energy_eff_fake_pid{}.pdf".format(pid)), bbox_inches="tight")

def plot_eta_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins, target='target', outpath='./'):
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    hist = np.histogram2d(v0[msk_both, 0], v0[msk_both, 1], bins=(bins["eta_val"], bins["eta_val"]))
    mplhep.hist2dplot(hist[0], hist[1], hist[2], cmap="Blues", cbar=False);
    plt.xlabel(bins["true_val"] + " " + bins["eta_xlabel"])
    plt.ylabel(bins["pred_val"]+ " " + bins["eta_xlabel"])
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.plot(
        [bins["eta_val"][0], bins["eta_val"][-1]],
        [bins["eta_val"][0], bins["eta_val"][-1]],
        color="black", ls="--", lw=0.5)
    plt.savefig(osp.join(outpath,"eta_2d_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    plt.hist(v0[msk_true, 0], bins=bins["eta_val"], density=1.0, histtype="step", lw=2, label=bins["true_val"]);
    plt.hist(v0[msk_pred, 1], bins=bins["eta_val"], density=1.0, histtype="step", lw=2, label=bins["pred_val"]);
    plt.xlabel(bins["eta_xlabel"])
    plt.ylabel("number of particles\n(normalized, a.u.)")
    plt.legend(frameon=False)
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    ax.set_ylim(ax.get_ylim()[0], 1.5*ax.get_ylim()[1])
    plt.savefig(osp.join(outpath,"eta_hist_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])

    res = (v0[msk_both, 1] - v0[msk_both, 0])
    res[np.isnan(res)] = -1

    plt.figure(figsize=(4,4))
    ax = plt.axes()
    ax.text(0.98, 0.98, "avg. $\Delta \eta$\n$%.2f \pm %.2f$"%(np.mean(res), np.std(res)), transform=ax.transAxes, ha="right", va="top")
    plt.hist(res, bins=bins["eta_res"], density=1.0);
    plt.xlabel("$\Delta \eta$")
    plt.ylabel("number of particles\n(normalized, a.u.)")
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.savefig(osp.join(outpath,"eta_ratio_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    #efficiency vs fake rate
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    big_df["bins_{}_eta".format(target)] = np.searchsorted(bins["eta_val"], big_df["{}_eta".format(target)])
    big_df["bins_pred_eta"] = np.searchsorted(bins["eta_val"], big_df["pred_eta"])

    vals_eff = big_df[(big_df["{}_pid".format(target)]==pid)].groupby("bins_{}_eta".format(target))["pred_pid"].apply(get_eff, pid)
    vals_fake = big_df[(big_df["pred_pid"]==pid)].groupby("bins_pred_eta")["{}_pid".format(target)].apply(get_fake, pid)

    out_eff = np.zeros((len(bins["eta_val"]), 2))
    out_fake = np.zeros((len(bins["eta_val"]), 2))
    for ib in range(len(bins["eta_val"])):
        if ib in vals_eff.keys():
            out_eff[ib, 0] = vals_eff[ib][0]
            out_eff[ib, 1] = vals_eff[ib][1]
        if ib in vals_fake.keys():
            out_fake[ib, 0] = vals_fake[ib][0]
            out_fake[ib, 1] = vals_fake[ib][1]

    cms_label()
    sample_label(ax)
    particle_label(ax, pid)

    plt.errorbar(bins["eta_val"], out_eff[:, 0], out_eff[:, 1], marker=".", lw=0, elinewidth=1.0, color="green", label="efficiency")
    plt.ylabel("efficiency\nN(pred|true) / N(true)")
    ax.set_ylim(0, 1.5)
    plt.xlabel(bins["eta_xlabel"])

    ax2 = ax.twinx()
    col = "red"
    plt.errorbar(bins["eta_val"], out_fake[:, 0], out_fake[:, 1], marker=".", lw=0, elinewidth=1.0, color=col, label="fake rate")
    plt.ylabel("fake rate\nN(true|pred) / N(pred)")
    plt.xlabel(bins["eta_xlabel"])
    ax2.set_ylim(0, 1.5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, frameon=False)
    plt.savefig(osp.join(outpath,"eta_eff_fake_pid{}.pdf".format(pid)), bbox_inches="tight")

def plot_phi_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins, target='target', outpath='./'):
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    hist = np.histogram2d(v0[msk_both, 0], v0[msk_both, 1], bins=(bins["phi_val"], bins["phi_val"]))
    mplhep.hist2dplot(hist[0], hist[1], hist[2], cmap="Blues", cbar=False);
    plt.xlabel(bins["true_val"] + " " + bins["phi_xlabel"])
    plt.ylabel(bins["pred_val"]+ " " + bins["phi_xlabel"])
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.plot(
        [bins["phi_val"][0], bins["phi_val"][-1]],
        [bins["phi_val"][0], bins["phi_val"][-1]],
        color="black", ls="--", lw=0.5)
    plt.savefig(osp.join(outpath,"phi_2d_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    plt.hist(v0[msk_true, 0], bins=bins["phi_val"], density=1.0, histtype="step", lw=2, label=bins["true_val"]);
    plt.hist(v0[msk_pred, 1], bins=bins["phi_val"], density=1.0, histtype="step", lw=2, label=bins["pred_val"]);
    plt.xlabel(bins["phi_xlabel"])
    plt.ylabel("number of particles\n(normalized, a.u.)")
    plt.legend(frameon=False)
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.savefig(osp.join(outpath,"phi_hist_pid{}.pdf".format(pid)), bbox_inches="tight")
    ax.set_ylim(ax.get_ylim()[0], 1.5*ax.get_ylim()[1])

    res = (v0[msk_both, 1] - v0[msk_both, 0])
    res[np.isnan(res)] = -1

    plt.figure(figsize=(4,4))
    ax = plt.axes()
    ax.text(0.98, 0.98, "avg. $\Delta \phi$\n$%.2f \pm %.2f$"%(np.mean(res), np.std(res)), transform=ax.transAxes, ha="right", va="top")
    plt.hist(res, bins=bins["phi_res"], density=1.0);
    plt.xlabel("$\Delta \phi$")
    plt.ylabel("number of particles\n(normalized, a.u.)")
    cms_label()
    sample_label(ax)
    particle_label(ax, pid)
    plt.savefig(osp.join(outpath,"phi_ratio_pid{}.pdf".format(pid)), bbox_inches="tight")
    
    #efficiency vs fake rate
    plt.figure(figsize=(4,4))
    ax = plt.axes()
    big_df["bins_{}_phi".format(target)] = np.searchsorted(bins["phi_val"], big_df["{}_phi".format(target)])
    big_df["bins_pred_phi"] = np.searchsorted(bins["phi_val"], big_df["pred_phi"])

    vals_eff = big_df[(big_df["{}_pid".format(target)]==pid)].groupby("bins_{}_phi".format(target))["pred_pid"].apply(get_eff, pid)
    vals_fake = big_df[(big_df["pred_pid"]==pid)].groupby("bins_pred_phi")["{}_pid".format(target)].apply(get_fake, pid)

    out_eff = np.zeros((len(bins["phi_val"]), 2))
    out_fake = np.zeros((len(bins["phi_val"]), 2))
    for ib in range(len(bins["phi_val"])):
        if ib in vals_eff.keys():
            out_eff[ib, 0] = vals_eff[ib][0]
            out_eff[ib, 1] = vals_eff[ib][1]
        if ib in vals_fake.keys():
            out_fake[ib, 0] = vals_fake[ib][0]
            out_fake[ib, 1] = vals_fake[ib][1]

    cms_label()
    sample_label(ax)
    particle_label(ax, pid)

    plt.errorbar(bins["phi_val"], out_eff[:, 0], out_eff[:, 1], marker=".", lw=0, elinewidth=1.0, color="green", label="efficiency")
    plt.ylabel("efficiency\nN(pred|true) / N(true)")
    ax.set_ylim(0, 1.5)
    plt.xlabel(bins["phi_xlabel"])

    ax2 = ax.twinx()
    col = "red"
    plt.errorbar(bins["phi_val"], out_fake[:, 0], out_fake[:, 1], marker=".", lw=0, elinewidth=1.0, color=col, label="fake rate")
    plt.ylabel("fake rate\nN(true|pred) / N(pred)")
    plt.xlabel(bins["phi_xlabel"])
    ax2.set_ylim(0, 1.5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, frameon=False)
    plt.savefig(osp.join(outpath,"phi_eff_fake_pid{}.pdf".format(pid)), bbox_inches="tight")
