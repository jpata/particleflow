import os
import argparse
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input parquet file(s) from evaluator")
    parser.add_argument("--outdir", type=str, default="plots_eval", help="Output directory for plots")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print(f"Loading results from {len(args.input)} files")
    datasets = []
    for fpath in args.input:
        print(f"Reading {fpath}")
        datasets.append(ak.from_parquet(fpath))

    ds = ak.concatenate(datasets)
    print(f"Total events: {len(ds)}")

    # Particle types from mlpf/conf.py
    pdg_to_name = {211: "Charged Hadron", 130: "Neutral Hadron", 22: "Photon", 11: "Electron", 13: "Muon"}

    # Flatten across events
    true_pdg = ak.flatten(ds.true_pdg).to_numpy()
    true_pt = ak.flatten(ds.true_pt).to_numpy()

    pred_pdg = ak.flatten(ds.pred_pdg).to_numpy()
    pred_pt = ak.flatten(ds.pred_pt).to_numpy()

    for pdg, name in pdg_to_name.items():
        mask_true = true_pdg == pdg
        mask_pred = pred_pdg == pdg

        vals_true = true_pt[mask_true]
        vals_pred = pred_pt[mask_pred]

        if len(vals_true) == 0 and len(vals_pred) == 0:
            print(f"Skipping {name} (PDG {pdg}) - no particles found.")
            continue

        plt.figure(figsize=(8, 6))

        # Determine bins automatically based on data
        all_vals = np.concatenate([vals_true, vals_pred])
        if len(all_vals) == 0:
            continue

        vmin = max(1e-1, np.min(all_vals))
        vmax = np.max(all_vals)
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)

        plt.hist(vals_true, bins=bins, alpha=0.7, label=f"True (N={len(vals_true)})", histtype="step", lw=2)
        plt.hist(vals_pred, bins=bins, alpha=0.7, label=f"Predicted (N={len(vals_pred)})", histtype="step", lw=2)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Particle $p_T$ [GeV]")
        plt.ylabel("Number of particles")
        plt.title(f"$p_T$ distribution for {name}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        fname = name.lower().replace(" ", "_")
        out_path = os.path.join(args.outdir, f"pt_{fname}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Created plot for {name}: {out_path}")

    # Plot number of true vs predicted particles per event
    n_true = ak.num(ds.true_pdg).to_numpy()
    n_pred = ak.num(ds.pred_pdg).to_numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(n_true, n_pred, alpha=0.5)

    # Add diagonal line
    max_val = max(np.max(n_true), np.max(n_pred))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="y=x")

    plt.xlabel("Number of True Particles")
    plt.ylabel("Number of Predicted Particles")
    plt.title("Number of Particles per Event: True vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.2)

    out_path = os.path.join(args.outdir, "n_particles_scatter.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Created scatterplot: {out_path}")

    # Plot sum-pt per event (true vs predicted)
    sum_pt_true = ak.sum(ds.true_pt, axis=1).to_numpy()
    sum_pt_pred = ak.sum(ds.pred_pt, axis=1).to_numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(sum_pt_true, sum_pt_pred, alpha=0.5)

    max_val = max(np.max(sum_pt_true), np.max(sum_pt_pred))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="y=x")

    plt.xlabel("True $\sum p_T$ [GeV]")
    plt.ylabel("Predicted $\sum p_T$ [GeV]")
    plt.title("Sum $p_T$ per Event: True vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.2)

    out_path = os.path.join(args.outdir, "sum_pt_scatter.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Created sum-pt scatterplot: {out_path}")

    # Plot sum-pt response
    valid_mask = sum_pt_true > 0
    response = sum_pt_pred[valid_mask] / sum_pt_true[valid_mask]
    plt.figure(figsize=(8, 6))
    plt.hist(response, bins=np.linspace(0, 2, 101), histtype="step", lw=2)
    plt.xlabel("$\sum p_T^{pred} / \sum p_T^{true}$")
    plt.ylabel("Number of events")
    plt.title("Sum $p_T$ Response")
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(args.outdir, "sum_pt_response.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Created sum-pt response plot: {out_path}")


if __name__ == "__main__":
    main()
