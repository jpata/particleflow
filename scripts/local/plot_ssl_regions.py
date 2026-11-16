"""Plot the saved region view without loading the model or particle targets."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    with np.load(args.run / "region_view.npz", allow_pickle=False) as d:
        x = d["X"][0]
        mapping = d["hit_to_query"][0]
        counts, energy, valid, kinds = [d[k][0] for k in ["counts", "energy", "valid", "types"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for col, kind in enumerate([1, 2]):
        for hidden, color, label in [(False, "#bbbbbb", "Visible"), (True, "#c84432", "Hidden targets")]:
            use = (x[:, 0] == kind) & ((mapping >= 0) == hidden)
            axes[0, col].scatter(x[use, 2], np.arctan2(x[use, 3], x[use, 4]), s=4, alpha=.65, color=color, label=label)
        axes[0, col].set(xlabel="Hit eta", ylabel="Hit phi", title="Tracker" if kind == 1 else "Calorimeter")
        axes[0, col].legend()
    for col, (values, title) in enumerate([(counts, "Measured hit counts per query cell"), (energy, "Calorimeter deposited energy targets")]):
        supported = valid if col == 0 else valid & (kinds == 2)[:, None]
        display = np.ma.array(np.log1p(values), mask=~supported)
        image = axes[1, col].imshow(display, aspect="auto", interpolation="nearest")
        axes[1, col].set(xlabel="Fixed subcell index", ylabel="Query index (tracker then calo)", title=title)
        fig.colorbar(image, ax=axes[1, col], label="log(1 + value)")
    fig.suptitle("Masked detector regions: full event projected over layers\nQueries use fixed geometry and visible other-layer context; target values are not student inputs")
    for extension in ["png", "pdf"]:
        fig.savefig(args.run / f"region_view.{extension}", dpi=160)
    plt.close(fig)
    print(args.run / "region_view.png")


if __name__ == "__main__":
    main()
