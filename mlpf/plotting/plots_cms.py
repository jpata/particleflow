import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from plot_utils import plot_confusion_matrix
from plot_utils import plot_E_reso
from plot_utils import plot_eta_reso
from plot_utils import plot_phi_reso

class_labels = list(range(8))


def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi


def prepare_resolution_plots(big_df, pid, bins, target="cand", outpath="./"):
    msk_true = big_df["{}_pid".format(target)] == pid
    msk_pred = big_df["pred_pid"] == pid
    msk_both = msk_true & msk_pred
    v0 = big_df[["{}_e".format(target), "pred_e"]].values
    v1 = big_df[["{}_eta".format(target), "pred_eta"]].values
    v2 = big_df[["{}_phi".format(target), "pred_phi"]].values

    plot_E_reso(big_df, pid, v0, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)
    plot_eta_reso(big_df, pid, v1, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)
    plot_phi_reso(big_df, pid, v2, msk_true, msk_pred, msk_both, bins, target=target, outpath=outpath)


def load_np(npfile):
    X = np.load(npfile)["X"]
    ycand = np.load(npfile)["ycand"]
    ypred = np.load(npfile)["ypred"]
    return X, ycand, ypred


def flatten(arr):
    return arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand"
    )
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    X, ycand, ypred = load_np(args.input)

    X_flat = flatten(X)
    ycand_flat = flatten(ycand)
    ypred_flat = flatten(ypred)
    msk = X_flat[:, 0] != 0

    confusion = sklearn.metrics.confusion_matrix(ycand_flat[msk, 0], ypred_flat[msk, 0], labels=range(8))

    fig, ax = plot_confusion_matrix(cm=confusion, target_names=[int(x) for x in class_labels], normalize=True)

    plt.savefig(osp.join(osp.dirname(args.input), "confusion_mlpf.pdf"), bbox_inches="tight")

#    prepare_resolution_plots(big_df, 211, bins[211], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 130, bins[130], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 11, bins[11], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 13, bins[13], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 22, bins[22], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 1, bins[1], target=args.target, outpath=osp.dirname(args.input))
#    prepare_resolution_plots(big_df, 2, bins[2], target=args.target, outpath=osp.dirname(args.input))
