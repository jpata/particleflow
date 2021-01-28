import sklearn
import sklearn.metrics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import mplhep

import sys
import os.path as osp

from plot_utils import plot_confusion_matrix, cms_label, particle_label, sample_label
from plot_utils import plot_E_reso, plot_eta_reso, plot_phi_reso, bins

#from tf_model import class_labels
class_labels = [0, 1, 2, 11, 13, 22, 130, 211]

def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

def prepare_resolution_plots(big_df, pid, bins, target='cand', outpath='./'):
    msk_true = (big_df["{}_pid".format(target)]==pid)
    msk_pred = (big_df["pred_pid"]==pid)
    msk_both = msk_true&msk_pred
    v0 = big_df[["{}_e".format(target), "pred_e"]].values
    v1 = big_df[["{}_eta".format(target), "pred_eta"]].values
    v2 = big_df[["{}_phi".format(target), "pred_phi"]].values
    
    plot_E_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)
    plot_eta_reso(big_df, pid, v1, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)
    plot_phi_reso(big_df, pid, v2, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--pkl", type=str, default = 'data/test.pkl.bz2', help="Dataframe pkl")
    args = parser.parse_args()

        
    big_df = pandas.read_pickle(args.pkl)

    big_df["pred_phi"] = np.arctan2(np.sin(big_df["pred_phi"]), np.cos(big_df["pred_phi"]))

    #msk = (big_df["{}_pid".format(args.target)] != 0) & ((big_df["pred_pid"] != 0))
    msk = np.ones(len(big_df), dtype=np.bool)
    
    plt.figure()
    plt.hist(big_df["{}_pid".format(args.target)][msk])
    plt.savefig(osp.join(osp.dirname(args.pkl),"{}_pid.pdf".format(args.target)), bbox_inches="tight")

    
    confusion2 = sklearn.metrics.confusion_matrix(
        big_df["{}_pid".format(args.target)][msk], big_df["pred_pid"][msk],
        labels=class_labels
    )

    fig, ax = plot_confusion_matrix(
        cm=confusion2, target_names=[int(x) for x in class_labels], normalize=True
    )

    acc = sklearn.metrics.accuracy_score(big_df["{}_pid".format(args.target)][msk], big_df["pred_pid"][msk])
    plt.title("")
    #plt.title("ML-PF, accuracy={:.2f}".format(acc))
    plt.ylabel("{} PF candidate PID\nassociated to input PFElement".format(args.target))
    plt.xlabel("predicted PID\nML-PF candidate,\naccuracy: {:.2f}".format(acc))
    cms_label(x0=0.20, x1=0.26, y=0.95)
    sample_label(ax, y=0.995)
    plt.savefig(osp.join(osp.dirname(args.pkl),"confusion_mlpf.pdf"), bbox_inches="tight")

    prepare_resolution_plots(big_df, 211, bins[211], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 130, bins[130], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 11, bins[11], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 13, bins[13], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 22, bins[22], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 1, bins[1], target=args.target, outpath=osp.dirname(args.pkl))
    prepare_resolution_plots(big_df, 2, bins[2], target=args.target, outpath=osp.dirname(args.pkl))
