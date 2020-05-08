import sklearn
import sklearn.metrics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import mplhep

import sys

from plot_utils import plot_confusion_matrix, cms_label, particle_label, sample_label
from plot_utils import plot_E_reso, plot_eta_reso, plot_phi_reso, bins

from tf_model import class_labels

def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

def prepare_resolution_plots(big_df, pid, bins):
    msk_true = (big_df["target_pid"]==pid)
    msk_pred = (big_df["pred_pid"]==pid)
    msk_both = msk_true&msk_pred
    v0 = big_df[["target_e", "pred_e"]].values
    v1 = big_df[["target_eta", "pred_eta"]].values
    v2 = big_df[["target_phi", "pred_phi"]].values
    
    plot_E_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins)
    plot_eta_reso(big_df, pid, v1, msk_true, msk_pred, msk_both, bins)
    plot_phi_reso(big_df, pid, v2, msk_true, msk_pred, msk_both, bins)

def prepare_confusion_matrix(big_df):
    fig, ax = plot_confusion_matrix(
        cm=confusion2, target_names=[int(x) for x in class_labels], normalize=True
    )

    acc = sklearn.metrics.accuracy_score(big_df["target_pid"][msk], big_df["pred_pid"][msk])
    plt.title("")
    #plt.title("ML-PF, accuracy={:.2f}".format(acc))
    plt.ylabel("reco PF candidate PID\nassociated to input PFElement")
    plt.xlabel("predicted PID\nML-PF candidate,\naccuracy: {:.2f}".format(acc))
    cms_label(x0=0.20, x1=0.26, y=0.95)
    sample_label(ax, y=0.995)
    plt.savefig("confusion_mlpf.pdf", bbox_inches="tight")

if __name__ == "__main__":
    big_df = pandas.read_pickle("df_1.pkl.bz2")
    big_df["pred_phi"] = np.arctan2(np.sin(big_df["pred_phi"]), np.cos(big_df["pred_phi"]))

    msk = np.ones(len(big_df), dtype=np.bool)
    confusion2 = sklearn.metrics.confusion_matrix(
        big_df["target_pid"][msk], big_df["pred_pid"][msk],
        labels=class_labels
    )

    prepare_confusion_matrix(big_df)
    prepare_resolution_plots(big_df, 211, bins[211])
    prepare_resolution_plots(big_df, 130, bins[130])
    prepare_resolution_plots(big_df, 11, bins[11])
    prepare_resolution_plots(big_df, 13, bins[13])
    prepare_resolution_plots(big_df, 22, bins[22])
