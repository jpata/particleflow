import argparse

import matplotlib.pyplot as plt
import pandas as pd
import setGPU  # noqa F401
import torch
import tqdm
from graph_data import PFGraphDataset

from models import EdgeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device %s" % device)


def main(args):

    full_dataset = PFGraphDataset(root="/storage/user/jduarte/particleflow/graph_data/")

    data = full_dataset.get(1)

    features = ["eta", "phi"]

    x_data = data.x.cpu().detach().numpy()

    mask = (x_data[:, 4] == 0) & (x_data[:, 5] == 0) & (x_data[:, 6] == 0) & (x_data[:, 7] == 0)
    # good_index = np.zeros((x_data.shape[0], 1, 2), dtype=int)

    good_x = x_data[:, 2:4].copy()
    # good_x[~mask] = x_data[~mask,6:8].copy()
    good_x[~mask] = x_data[~mask, 2:4].copy()

    df = pd.DataFrame(good_x, columns=features)

    df["isTrack"] = ~mask

    row, col = data.edge_index.cpu().detach().numpy()
    y_truth = data.y.cpu().detach().numpy()

    input_dim = data.x.shape[1]
    hidden_dim = 32
    edge_dim = 1
    n_iters = 1
    model = EdgeNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        n_iters=n_iters,
    ).to(device)
    modpath = "data/EdgeNet_14001_ca9bbfb3bb_jduarte.best.pth"
    model.load_state_dict(torch.load(modpath))
    data = data.to(device)
    output = model(data)

    min_phi = -1.5
    max_phi = 1.5
    min_eta = -1.5
    max_eta = 1.5
    extra = 1.0
    for plot_type in [  # ['input', 'truth'],
        # ['input', 'output'],
        # ['truth', 'output'],
        ["input"],
        ["output"],
        ["truth"],
    ]:
        x = "eta"
        y = "phi"
        plt.figure()
        k = 0
        for i, j in tqdm.tqdm(zip(row, col), total=len(y_truth)):
            x1 = df[x][i]
            x2 = df[x][j]
            y1 = df[y][i]
            y2 = df[y][j]
            if (x1 < min_eta - extra or x1 > max_eta + extra) or (x2 < min_eta - extra or x2 > max_eta + extra):
                continue
            if (y1 < min_phi - extra or y1 > max_phi + extra) or (y2 < min_phi - extra or y2 > max_phi + extra):
                continue
            if "input" in plot_type:
                seg_args = dict(c="b", alpha=0.1, zorder=1)
                plt.plot([df[x][i], df[x][j]], [df[y][i], df[y][j]], "-", **seg_args)
            if "truth" in plot_type and y_truth[k]:
                seg_args = dict(c="r", alpha=0.5, zorder=2)
                plt.plot([df[x][i], df[x][j]], [df[y][i], df[y][j]], "-", **seg_args)
            if "output" in plot_type:
                seg_args = dict(
                    c="g",
                    alpha=output[k].item() * (output[k].item() > 0.9),
                    zorder=3,
                )
                plt.plot([df[x][i], df[x][j]], [df[y][i], df[y][j]], "-", **seg_args)
            k += 1

        cut_mask = (
            (df[x] > min_eta - extra) & (df[x] < max_eta + extra) & (df[y] > min_phi - extra) & (df[y] < max_phi + extra)
        )
        cluster_mask = cut_mask & ~df["isTrack"]
        track_mask = cut_mask & df["isTrack"]
        plt.scatter(
            df[x][cluster_mask],
            df[y][cluster_mask],
            c="g",
            marker="o",
            s=50,
            zorder=4,
            alpha=1,
        )
        plt.scatter(
            df[x][track_mask],
            df[y][track_mask],
            c="b",
            marker="p",
            s=50,
            zorder=5,
            alpha=1,
        )
        plt.xlabel(r"Track or Cluster $\eta$", fontsize=14)
        plt.ylabel(r"Track or Cluster $\phi$", fontsize=14)
        plt.xlim(min_eta, max_eta)
        plt.ylim(min_phi, max_phi)
        plt.figtext(
            0.12,
            0.90,
            "CMS",
            fontweight="bold",
            wrap=True,
            horizontalalignment="left",
            fontsize=16,
        )
        plt.figtext(
            0.22,
            0.90,
            "Simulation Preliminary",
            style="italic",
            wrap=True,
            horizontalalignment="left",
            fontsize=14,
        )
        plt.figtext(
            0.67,
            0.90,
            "Run 3 (14 TeV)",
            wrap=True,
            horizontalalignment="left",
            fontsize=14,
        )
        plt.savefig("graph_%s_%s_%s.pdf" % (x, y, "_".join(plot_type)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
