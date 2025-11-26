import click
import json
import glob
import pandas
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mplhep
from pathlib import Path
from matplotlib.lines import Line2D


def load_history(path, min_epoch=None, max_epoch=None):
    """
    Load training history from a directory of json files.
    The json files are expected to be named epoch_*.json.
    """
    ret = {}
    files = list(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No history files found at {path}")

    for fi in files:
        data = json.load(open(fi))
        data2 = {}
        for k1 in ["train", "valid"]:
            if k1 in data:
                for k2 in data[k1].keys():
                    data2[f"{k1}_{k2}"] = data[k1][k2]
        epoch = int(fi.split("_")[-1].split(".")[0])
        ret[epoch] = data2

    if not ret:
        return pandas.DataFrame()

    if not max_epoch:
        max_epoch = max(ret.keys())
    if not min_epoch:
        min_epoch = min(ret.keys())

    ret2 = []
    # fill in missing epochs with NaNs
    for i in range(min_epoch, max_epoch + 1):
        if i in ret:
            ret2.append(ret[i])
        else:
            dummy = {}
            for k in ret[min_epoch].keys():
                dummy[k] = np.nan
            ret2.append(dummy)

    return pandas.DataFrame(ret2)


def loss_plot(epochs, train, test, margin=0.05, smoothing=False, ylabel="", title=""):
    """
    Plots training and validation loss curves.
    """
    fig = plt.figure()
    ax = plt.axes()

    alpha = 0.2 if smoothing else 1.0
    l0 = "Training"
    l1 = "Validation"

    p0 = plt.plot(epochs, train, alpha=alpha, label=l0, marker="o", ls="--")
    p1 = plt.plot(epochs, test, alpha=alpha, label=l1, marker="x")

    if smoothing:
        train_smooth = np.convolve(train[~np.isnan(train)], np.ones(5) / 5, mode="valid")
        epochs_smooth = epochs[len(epochs) - len(train_smooth) :]
        plt.plot(epochs_smooth, train_smooth, color=p0[0].get_color(), lw=2)

        test_smooth = np.convolve(test[~np.isnan(test)], np.ones(5) / 5, mode="valid")
        plt.plot(epochs_smooth, test_smooth, color=p1[0].get_color(), lw=2)

    # get last valid value
    last_valid_loss = test[~np.isnan(test)][-1]

    if last_valid_loss is not np.nan:
        plt.ylim(last_valid_loss * (1.0 - margin), last_valid_loss * (1.0 + margin))

    plt.legend(loc=3, frameon=False, fontsize=30)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    mplhep.cms.label("", data=False, rlabel="Run 3 configuration")
    return fig, ax


@click.command()
@click.option("--input-dirs", "-i", multiple=True, required=True, help="Input directories containing history/epoch_*.json files.")
@click.option("--labels", "-l", multiple=True, help="Labels for each input directory. Must be the same number as input-dirs.")
@click.option("--output-dir", "-o", required=True, type=str, help="Output directory for plots.")
def main(input_dirs, labels, output_dir):
    """
    Generates loss curve plots from training history files.
    """
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    if labels and len(input_dirs) != len(labels):
        raise ValueError("Number of input directories and labels must be the same.")

    histories = []
    for d in input_dirs:
        history = load_history(f"{d}/history/epoch_*.json")
        histories.append(history)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot individual loss curves for each history
    for i, history in enumerate(histories):
        history["train_Total"] = history["train_Total"] - history["train_ispu"].values
        history["valid_Total"] = history["valid_Total"] - history["valid_ispu"].values
        if history.empty:
            print(f"No history found for {input_dirs[i]}, skipping.")
            continue

        label = labels[i] if labels else Path(input_dirs[i]).name

        # Total loss
        fig, ax = loss_plot(
            history.index + 1, history["train_Total"].values, history["valid_Total"].values, margin=0.1, ylabel="Total loss", title=label
        )
        plt.xticks(range(1, len(history) + 1))
        plt.savefig(output_path / f"{label}_loss.pdf")
        plt.close(fig)

        # Classification loss
        if "train_Classification" in history.columns:
            fig, ax = loss_plot(
                history.index + 1,
                history["train_Classification"].values * 100,
                history["valid_Classification"].values * 100,
                margin=0.05,
                ylabel="Particle ID loss x100",
                title=label,
            )
            plt.xticks(range(1, len(history) + 1))
            plt.savefig(output_path / f"{label}_cls_loss.pdf")
            plt.close(fig)

        # Binary classification loss
        if "train_Classification_binary" in history.columns:
            fig, ax = loss_plot(
                history.index + 1,
                history["train_Classification_binary"].values,
                history["valid_Classification_binary"].values,
                margin=0.1,
                ylabel="Binary classification loss",
                title=label,
            )
            plt.xticks(range(1, len(history) + 1))
            plt.savefig(output_path / f"{label}_cls_bin_loss.pdf")
            plt.close(fig)

        # Regression loss
        reg_losses = [f"Regression_{_loss}" for _loss in ["energy", "pt", "eta", "sin_phi", "cos_phi"]]
        if all([f"train_{loss}" in history.columns for loss in reg_losses]):
            reg_loss = sum([history[f"train_{loss}"].values for loss in reg_losses])
            val_reg_loss = sum([history[f"valid_{loss}"].values for loss in reg_losses])
            fig, ax = loss_plot(history.index + 1, reg_loss, val_reg_loss, margin=0.2, ylabel="Regression loss", title=label)
            plt.xticks(range(1, len(history) + 1))
            plt.savefig(output_path / f"{label}_reg_loss.pdf")
            plt.close(fig)

        # PU classification loss
        if "train_ispu" in history.columns:
            fig, ax = loss_plot(
                history.index + 1,
                history["train_ispu"].values,
                history["valid_ispu"].values,
                margin=0.5,
                ylabel="PU classification loss",
                title=label,
            )
            plt.xticks(range(1, len(history) + 1))
            plt.savefig(output_path / f"{label}_ispu_loss.pdf")
            plt.close(fig)

        # Combined loss plot with 4 axes
        fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        # fig.suptitle(label, fontsize=35)

        loss_info = [
            {"name": "Total", "label": "Total Loss"},
            {"name": "Classification", "label": "PID Classification Loss"},
            {"name": "Classification_binary", "label": "Binary Classification Loss"},
            {"name": "Regression", "label": "Regression Loss"},
        ]

        for i, (ax, info) in enumerate(zip(axs, loss_info)):
            loss_name = info["name"]
            ylabel = info["label"]

            train_loss_col = f"train_{loss_name}"
            valid_loss_col = f"valid_{loss_name}"

            if loss_name == "Regression":
                reg_losses = [f"Regression_{_loss}" for _loss in ["energy", "pt", "eta", "sin_phi", "cos_phi"]]
                if all([f"train_{loss}" in history.columns for loss in reg_losses]):
                    train_loss = sum([history[f"train_{loss}"].values for loss in reg_losses])
                    valid_loss = sum([history[f"valid_{loss}"].values for loss in reg_losses])
                else:
                    continue
            else:
                if train_loss_col in history.columns and valid_loss_col in history.columns:
                    train_loss = history[train_loss_col].values
                    valid_loss = history[valid_loss_col].values
                else:
                    continue

            ax.plot(history.index + 1, train_loss, ls="--", color="black", marker="o")
            ax.plot(history.index + 1, valid_loss, ls="-", color="black", marker="s")
            ax.set_title(ylabel, y=0.8, x=0.45)
            ax.tick_params(axis="y", labelsize=25)

        # Add a single legend for the figure
        custom_lines = [Line2D([0], [0], color="black", lw=2, ls="-"), Line2D([0], [0], color="black", lw=2, ls="--")]
        custom_labels = ["Validation", "Training"]
        ax.legend(handles=custom_lines, labels=custom_labels, loc="upper right", fontsize=30)

        plt.xlabel("Epoch")
        fig.subplots_adjust(hspace=0.1)
        plt.savefig(output_path / f"{label}_all_losses_subplots.pdf")
        plt.tight_layout()
        plt.close(fig)

    # Plot comparison if multiple histories
    if len(histories) > 1:
        plt.figure()
        ax = plt.axes()
        for i, history in enumerate(histories):
            if history.empty:
                continue
            label = labels[i] if labels else Path(input_dirs[i]).name
            plt.plot(history.index + 1, history["valid_Total"], marker="o", label=label)
        plt.ylabel("Total valid. loss")
        plt.xlabel("epoch")
        plt.legend(loc="best")
        mplhep.cms.label("", data=False, rlabel="Run 3 configuration")
        plt.savefig(output_path / "loss_comparison.pdf")
        plt.close()

    print(f"Generated plots in {output_dir}")


if __name__ == "__main__":
    main()
