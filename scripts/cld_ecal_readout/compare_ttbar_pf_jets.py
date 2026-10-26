"""Compare CLD Pandora jet response for paired ECAL readout productions.

Jets use the Key4hep postprocessing ee-genkt definition (R=0.4, p=-1,
inclusive pT>5 GeV). Truth uses its propagated status-1 plus surrogate-parent
selection; reconstructed jets are clustered directly from PandoraPFOs.
"""

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
import vector

from mlpf.data.key4hep.postprocessing import compute_jets, find_surrogate_ancestors_from_record
from mlpf.jet_utils import _match_jets_event


SAMPLE = "p8_ee_ttbar_ecm365"
MC_FIELDS = ("PDG", "generatorStatus", "simulatorStatus", "momentum.x", "momentum.y", "momentum.z", "mass", "daughters_begin", "daughters_end")
PFO_FIELDS = ("PDG", "mass", "momentum.x", "momentum.y", "momentum.z")
BRANCHES = [
    *(f"MCParticles.{f}" for f in MC_FIELDS),
    "_MCParticles_daughters/_MCParticles_daughters.index",
    *(f"PandoraPFOs.{f}" for f in PFO_FIELDS),
]
PT_BINS = np.array([5, 10, 20, 30, 50, 75, 100, 150, 250], dtype=float)
ETA_BINS = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=float)


def files(workspace):
    root = workspace / "gen" / SAMPLE / "root"
    prefix = f"reco_{SAMPLE}_"
    found = {int(p.stem.removeprefix(prefix)): p for p in root.glob(f"{prefix}*.root")}
    if not found:
        raise FileNotFoundError(root)
    return found


def particles_and_jets(path):
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
        tree = source["events"]
        data = tree.arrays(BRANCHES, library="ak")
    status = data["MCParticles.generatorStatus"]
    pdg = data["MCParticles.PDG"]
    status1 = status == 1
    status1_pdg = pdg[status1]
    status1_momentum = tuple(data[f"MCParticles.momentum.{axis}"][status1] for axis in "xyz")
    propagated = (data["MCParticles.simulatorStatus"] & 0x0F000000) != 0
    mask = status1 & (abs(pdg) != 12) & (abs(pdg) != 14) & (abs(pdg) != 16) & propagated
    mask = mask | find_surrogate_ancestors_from_record(data, "MCParticles")
    truth = vector.awk(
        ak.zip(
            {
                "px": data["MCParticles.momentum.x"][mask],
                "py": data["MCParticles.momentum.y"][mask],
                "pz": data["MCParticles.momentum.z"][mask],
                "mass": data["MCParticles.mass"][mask],
            }
        )
    )
    pfo_mask = data["PandoraPFOs.PDG"] != 0
    pf = vector.awk(
        ak.zip(
            {
                "px": data["PandoraPFOs.momentum.x"][pfo_mask],
                "py": data["PandoraPFOs.momentum.y"][pfo_mask],
                "pz": data["PandoraPFOs.momentum.z"][pfo_mask],
                "mass": data["PandoraPFOs.mass"][pfo_mask],
            }
        )
    )
    return status1_pdg, status1_momentum, compute_jets(truth), compute_jets(pf), ak.to_numpy(ak.num(pf))


def jet_matrix(jets):
    if len(jets["pt"]) == 0:
        return np.empty((0, 4), dtype=float)
    mat = np.column_stack([ak.to_numpy(jets[field]) for field in ("pt", "eta", "phi", "energy")])
    return mat[np.isfinite(mat).all(axis=1) & (mat[:, 0] > 0)]


def summarize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0}
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(p50),
        "p16": float(p16),
        "p84": float(p84),
        "half_width_68": float((p84 - p16) / 2),
        "rms": float(values.std()),
    }


def binned(pt_or_eta, responses, edges, absolute=False):
    coord = np.abs(pt_or_eta) if absolute else pt_or_eta
    return [summarize(responses[(coord >= lo) & (coord < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"5.1 mm": files(args.baseline), "2 mm": files(args.fine)}
    seeds = sorted(paths["2 mm"])
    if not set(seeds).issubset(paths["5.1 mm"]):
        raise ValueError("A 2 mm seed is missing from the baseline")
    records = {
        arm: {
            key: []
            for key in ("response", "gen_pt", "gen_eta", "all_gen_pt", "all_gen_eta", "matched_gen_pt", "pfo_count", "truth_count", "reco_count")
        }
        for arm in paths
    }
    pair_responses = {arm: [] for arm in paths}
    paired_by_event = {arm: [] for arm in paths}
    truth_pt_differences = []
    checks = {
        "events": 0,
        "status1_pdg_mismatch_events": 0,
        "status1_momentum_max_abs_GeV": 0.0,
        "status1_momentum_nonidentical_events": 0,
        "truth_jets_baseline": 0,
        "truth_jets_fine": 0,
        "truth_jets_matched_dr001": 0,
        "truth_jet_pt_max_abs_GeV": 0.0,
    }
    for seed in seeds:
        pair = {arm: particles_and_jets(path[seed]) for arm, path in paths.items()}
        n = len(pair["5.1 mm"][2])
        if n != len(pair["2 mm"][2]):
            raise ValueError(f"Seed {seed}: different event counts")
        print(f"seed {seed}: {n} events", flush=True)
        for i in range(n):
            checks["events"] += 1
            pdg0, mom0 = pair["5.1 mm"][0][i], [m[i] for m in pair["5.1 mm"][1]]
            pdg1, mom1 = pair["2 mm"][0][i], [m[i] for m in pair["2 mm"][1]]
            if not ak.to_list(pdg0) == ak.to_list(pdg1):
                checks["status1_pdg_mismatch_events"] += 1
            elif len(pdg0):
                maximum = max(float(np.max(np.abs(ak.to_numpy(a) - ak.to_numpy(b)))) for a, b in zip(mom0, mom1))
                checks["status1_momentum_max_abs_GeV"] = max(checks["status1_momentum_max_abs_GeV"], maximum)
                checks["status1_momentum_nonidentical_events"] += maximum > 0
            jets = {arm: (jet_matrix(value[2][i]), jet_matrix(value[3][i])) for arm, value in pair.items()}
            g0, _ = jets["5.1 mm"]
            g1, _ = jets["2 mm"]
            ti0, ti1 = _match_jets_event(g0[:, 1], g0[:, 2], g1[:, 1], g1[:, 2], 0.01)
            checks["truth_jets_baseline"] += len(g0)
            checks["truth_jets_fine"] += len(g1)
            checks["truth_jets_matched_dr001"] += len(ti0)
            if len(ti0):
                truth_pt_differences.extend((g1[ti1, 0] - g0[ti0, 0]).tolist())
                checks["truth_jet_pt_max_abs_GeV"] = max(checks["truth_jet_pt_max_abs_GeV"], float(np.max(np.abs(g0[ti0, 0] - g1[ti1, 0]))))
            matches = {}
            for arm, (gen, reco) in jets.items():
                gi, ri = _match_jets_event(gen[:, 1], gen[:, 2], reco[:, 1], reco[:, 2], 0.1)
                matches[arm] = dict(zip(gi.tolist(), reco[ri, 0].tolist()))
                rec = records[arm]
                rec["response"].extend((reco[ri, 0] / gen[gi, 0]).tolist())
                rec["gen_pt"].extend(gen[gi, 0].tolist())
                rec["gen_eta"].extend(gen[gi, 1].tolist())
                rec["all_gen_pt"].extend(gen[:, 0].tolist())
                rec["all_gen_eta"].extend(gen[:, 1].tolist())
                rec["matched_gen_pt"].extend(gen[gi, 0].tolist())
                rec["pfo_count"].append(int(pair[arm][4][i]))
                rec["truth_count"].append(len(gen))
                rec["reco_count"].append(len(reco))
            this_paired = {arm: [] for arm in paths}
            for a, b in zip(ti0, ti1):
                if a in matches["5.1 mm"] and b in matches["2 mm"]:
                    # One truth denominator removes small Geant4 record changes
                    # from the paired reconstruction comparison.
                    base_response = matches["5.1 mm"][a] / g0[a, 0]
                    fine_response = matches["2 mm"][b] / g0[a, 0]
                    pair_responses["5.1 mm"].append(base_response)
                    pair_responses["2 mm"].append(fine_response)
                    this_paired["5.1 mm"].append(base_response)
                    this_paired["2 mm"].append(fine_response)
            for arm in paths:
                paired_by_event[arm].append(np.asarray(this_paired[arm]))
    if checks["status1_pdg_mismatch_events"]:
        raise ValueError("Generator particle identities differ; see checks")
    if checks["truth_jets_matched_dr001"] < 0.99 * min(checks["truth_jets_baseline"], checks["truth_jets_fine"]):
        raise ValueError("Truth jets differ too much for paired comparison")

    data = {arm: {k: np.asarray(v, dtype=float) for k, v in rec.items()} for arm, rec in records.items()}
    report = {
        "sample": SAMPLE,
        "seeds": seeds,
        "checks": checks,
        "definition": {
            "jet_algorithm": "ee_genkt R=0.4 p=-1; inclusive pT>5 GeV",
            "truth": "propagated status-1 visible particles plus surrogate ancestors, as in Key4hep postprocessing",
            "reco": "jets reclustered from PandoraPFOs (PDG != 0)",
            "matching": "maximum-cardinality one-to-one deltaR<0.1, no response cut",
            "response": "reco jet pT / truth jet pT",
            "resolution": "(response p84 - response p16)/2",
            "paired_denominator": "baseline truth jet pT for both Pandora jets",
        },
        "arms": {},
        "common_truth_jets": {},
    }
    for arm, rec in data.items():
        report["arms"][arm] = {
            "response": summarize(rec["response"]),
            "matched_fraction": float(len(rec["response"]) / len(rec["all_gen_pt"])),
            "pfo_count_per_event": summarize(rec["pfo_count"]),
            "truth_jets_per_event": summarize(rec["truth_count"]),
            "reco_jets_per_event": summarize(rec["reco_count"]),
            "pt_bins": PT_BINS.tolist(),
            "eta_bins": ETA_BINS.tolist(),
            "response_by_gen_pt": binned(rec["gen_pt"], rec["response"], PT_BINS),
            "response_by_abs_gen_eta": binned(rec["gen_eta"], rec["response"], ETA_BINS, True),
        }
        report["common_truth_jets"][arm] = summarize(pair_responses[arm])
    diff = np.asarray(pair_responses["2 mm"]) - np.asarray(pair_responses["5.1 mm"])
    report["common_truth_jets"]["paired_response_difference_2mm_minus_5p1mm"] = summarize(diff)
    report["common_truth_jets"]["truth_jet_pt_difference_2mm_minus_5p1mm_GeV"] = summarize(truth_pt_differences)
    report["common_truth_jets"]["n"] = int(len(diff))
    rng = np.random.default_rng(12345)
    bootstrap_width_diff = []
    bootstrap_median_diff = []
    for _ in range(300):
        sampled = rng.integers(0, checks["events"], checks["events"])
        b = np.concatenate([paired_by_event["5.1 mm"][j] for j in sampled])
        f = np.concatenate([paired_by_event["2 mm"][j] for j in sampled])
        bootstrap_width_diff.append(summarize(f)["half_width_68"] - summarize(b)["half_width_68"])
        bootstrap_median_diff.append(float(np.median(f - b)))
    report["common_truth_jets"]["bootstrap_events_300"] = {
        "resolution_half_width_difference_2mm_minus_5p1mm": {
            "point": report["common_truth_jets"]["2 mm"]["half_width_68"] - report["common_truth_jets"]["5.1 mm"]["half_width_68"],
            "ci95": np.percentile(bootstrap_width_diff, [2.5, 97.5]).tolist(),
        },
        "median_paired_response_difference_2mm_minus_5p1mm": {
            "point": float(np.median(diff)),
            "ci95": np.percentile(bootstrap_median_diff, [2.5, 97.5]).tolist(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"5.1 mm": "#2563eb", "2 mm": "#e76f51"}
    for arm, rec in data.items():
        r = rec["response"]
        axes[0, 0].hist(
            r,
            bins=np.linspace(0.4, 1.6, 61),
            histtype="step",
            linewidth=1.8,
            density=True,
            color=colors[arm],
            label=f"{arm}: N={len(r)}, 68% half-width={report['arms'][arm]['response']['half_width_68']:.3f}",
        )
        for ax, edges, coord in ((axes[0, 1], PT_BINS, rec["gen_pt"]), (axes[1, 0], ETA_BINS, np.abs(rec["gen_eta"]))):
            centers = (edges[:-1] + edges[1:]) / 2
            summaries = binned(coord, r, edges)
            valid = np.array([s.get("n", 0) >= 20 for s in summaries])
            median = np.array([s.get("median", np.nan) for s in summaries])
            width = np.array([s.get("half_width_68", np.nan) for s in summaries])
            ax.errorbar(centers[valid], median[valid], yerr=width[valid], fmt="o-", capsize=3, color=colors[arm], label=arm)
        all_pt = rec["all_gen_pt"]
        matched_pt = rec["matched_gen_pt"]
        den, _ = np.histogram(all_pt, PT_BINS)
        num, _ = np.histogram(matched_pt, PT_BINS)
        centers = (PT_BINS[:-1] + PT_BINS[1:]) / 2
        axes[1, 1].plot(centers[den > 0], (num / np.maximum(den, 1))[den > 0], "o-", color=colors[arm], label=arm)
    axes[0, 0].set(xlabel="Pandora jet pT / truth jet pT", ylabel="Density", title="Matched jet response (tails outside plot retained in summary)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="Truth jet pT [GeV]", ylabel="Median response ± 68% half-width", title="Response and spread vs truth pT")
    axes[1, 0].set(xlabel="|Truth jet η|", ylabel="Median response ± 68% half-width", title="Response and spread vs |η|")
    axes[1, 1].set(xlabel="Truth jet pT [GeV]", ylabel="Matched truth jet fraction", ylim=(0, 1.05), title="ΔR < 0.1 matching efficiency")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=8)
    fig.suptitle(f"CLD Pandora jets: {checks['events']} paired ttbar events, 5.1 mm vs 2 mm ECAL")
    fig.savefig(args.output_dir / "pandora_jet_resolution.png", dpi=160)
    fig2, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm, rec in data.items():
        for ax, edges, coord in ((axs[0], PT_BINS, rec["gen_pt"]), (axs[1], ETA_BINS, np.abs(rec["gen_eta"]))):
            centers = (edges[:-1] + edges[1:]) / 2
            summaries = binned(coord, rec["response"], edges)
            width = np.array([s.get("half_width_68", np.nan) if s.get("n", 0) >= 30 else np.nan for s in summaries])
            ax.plot(centers, width, "o-", color=colors[arm], label=arm)
    axs[0].set(xlabel="Truth jet pT [GeV]", ylabel="Response 68% half-width", title="Pandora jet pT resolution vs pT")
    axs[1].set(xlabel="|Truth jet η|", ylabel="Response 68% half-width", title="Pandora jet pT resolution vs |η|")
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend()
    fig2.savefig(args.output_dir / "pandora_jet_resolution_width.png", dpi=160)
    print(args.output_dir / "summary.json")
    print(args.output_dir / "pandora_jet_resolution.png")
    print(args.output_dir / "pandora_jet_resolution_width.png")


if __name__ == "__main__":
    main()
