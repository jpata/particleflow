import os
import argparse
import json
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import fastjet
import vector
from mlpf.jet_utils import match_two_jet_collections
from mlpf.conf import JET_CONFIG, Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input parquet file(s) from evaluator")
    parser.add_argument("--outdir", type=str, default="plots_eval", help="Output directory for plots")
    parser.add_argument("--detector", type=str, default="clic", choices=["clic", "cld", "clic_hits", "cld_hits"], help="Detector type")
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

    if len(response) > 0:
        sum_pt_response_median = float(np.median(response))
        sum_pt_response_p25 = np.percentile(response, 25)
        sum_pt_response_p75 = np.percentile(response, 75)
        sum_pt_response_iqr = float(sum_pt_response_p75 - sum_pt_response_p25)
    else:
        sum_pt_response_median = None
        sum_pt_response_iqr = None

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

    # Jet clustering
    print("Clustering jets...")
    jet_config = JET_CONFIG[Dataset(args.detector).value]
    algo = getattr(fastjet, jet_config["algo"])
    if "p" in jet_config:
        jetdef = fastjet.JetDefinition(algo, jet_config["r"], jet_config["p"])
    else:
        jetdef = fastjet.JetDefinition(algo, jet_config["r"])

    jet_ptcut = jet_config["ptcut"]
    jet_match_dr = jet_config["match_dr"]

    # Prep vectors for clustering
    true_p4 = vector.awk(ak.zip({"pt": ds.true_pt, "eta": ds.true_eta, "phi": ds.true_phi, "energy": ds.true_energy}))
    pred_p4 = vector.awk(ak.zip({"pt": ds.pred_pt, "eta": ds.pred_eta, "phi": ds.pred_phi, "energy": ds.pred_energy}))

    # Filter particles (must have pt > 0 for fastjet)
    true_p4 = true_p4[true_p4.pt > 1e-3]
    pred_p4 = pred_p4[pred_p4.pt > 1e-3]

    # True jets
    true_cluster = fastjet.ClusterSequence(true_p4.to_xyzt(), jetdef)
    true_jets = true_cluster.inclusive_jets(min_pt=jet_ptcut)
    true_jets_p4 = vector.awk(ak.zip({"px": true_jets.px, "py": true_jets.py, "pz": true_jets.pz, "E": true_jets.e}))

    # Pred jets
    pred_cluster = fastjet.ClusterSequence(pred_p4.to_xyzt(), jetdef)
    pred_jets = pred_cluster.inclusive_jets(min_pt=jet_ptcut)
    pred_jets_p4 = vector.awk(ak.zip({"px": pred_jets.px, "py": pred_jets.py, "pz": pred_jets.pz, "E": pred_jets.e}))

    # Matching
    jets_coll = {"gen": true_jets_p4, "pred": pred_jets_p4}
    matched_jets = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)

    # Plot jet pt
    plt.figure(figsize=(8, 6))
    bins = np.logspace(np.log10(jet_ptcut), np.log10(500), 50)
    plt.hist(ak.flatten(true_jets_p4.pt), bins=bins, histtype="step", lw=2, label="True")
    plt.hist(ak.flatten(pred_jets_p4.pt), bins=bins, histtype="step", lw=2, label="Pred")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.ylabel("Number of jets")
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "jet_pt.png"), bbox_inches="tight")
    plt.close()
    print(f"Created jet pT plot: {os.path.join(args.outdir, 'jet_pt.png')}")

    # Plot jet eta
    plt.figure(figsize=(8, 6))
    bins = np.linspace(-5, 5, 50)
    plt.hist(ak.flatten(true_jets_p4.eta), bins=bins, histtype="step", lw=2, label="True")
    plt.hist(ak.flatten(pred_jets_p4.eta), bins=bins, histtype="step", lw=2, label="Pred")
    plt.xlabel("Jet $\eta$")
    plt.ylabel("Number of jets")
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "jet_eta.png"), bbox_inches="tight")
    plt.close()
    print(f"Created jet eta plot: {os.path.join(args.outdir, 'jet_eta.png')}")

    # Plot jet response
    matched_true_pt = ak.flatten(true_jets_p4.pt[matched_jets.gen])
    matched_pred_pt = ak.flatten(pred_jets_p4.pt[matched_jets.pred])

    n_true_jets = len(ak.flatten(true_jets_p4.pt))
    n_matched_jets = len(matched_true_pt)
    matching_fraction = float(n_matched_jets / n_true_jets if n_true_jets > 0 else 0)

    jet_response_median = None
    jet_response_iqr = None

    if len(matched_true_pt) > 0:
        response = matched_pred_pt / matched_true_pt
        plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 2, 101)

        # Calculate median and IQR for the label
        p25 = np.percentile(response, 25)
        p50 = np.percentile(response, 50)
        p75 = np.percentile(response, 75)
        iqr = p75 - p25

        jet_response_median = float(p50)
        jet_response_iqr = float(iqr)

        plt.hist(response, bins=bins, histtype="step", lw=2, label="MLPF, median={:.2f}, IQR={:.2f}".format(p50, iqr))
        plt.axvline(1.0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Jet $p_T^{pred} / p_T^{true}$")
        plt.ylabel("Number of matched jets")
        plt.legend()
        plt.savefig(os.path.join(args.outdir, "jet_response.png"), bbox_inches="tight")
        plt.close()
        print(f"Created jet response plot: {os.path.join(args.outdir, 'jet_response.png')}")
    else:
        print("No matched jets found, skipping response plot.")

    # Save metrics to JSON
    metrics = {
        "matching_fraction": matching_fraction,
        "jet_response_median": jet_response_median,
        "jet_response_iqr": jet_response_iqr,
        "sum_pt_response_median": sum_pt_response_median,
        "sum_pt_response_iqr": sum_pt_response_iqr,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {os.path.join(args.outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()
