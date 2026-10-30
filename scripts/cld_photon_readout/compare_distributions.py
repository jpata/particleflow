"""Compare paired CLD photon productions at hit and Pandora cluster level."""

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from validate_pair import SAMPLE, files_by_seed


FIELDS = {
    "pdg": ("MCParticles", "PDG"),
    "status": ("MCParticles", "generatorStatus"),
    "px": ("MCParticles", "momentum.x"),
    "py": ("MCParticles", "momentum.y"),
    "pz": ("MCParticles", "momentum.z"),
    "sim_b": ("ECalBarrelCollection", "energy"),
    "sim_e": ("ECalEndcapCollection", "energy"),
    "digi_b": ("ECALBarrel", "energy"),
    "digi_e": ("ECALEndcap", "energy"),
    "cluster_energy": ("PandoraClusters", "energy"),
    "cluster_begin": ("PandoraClusters", "hits_begin"),
    "cluster_end": ("PandoraClusters", "hits_end"),
    "pfo_pdg": ("PandoraPFOs", "PDG"),
    "pfo_energy": ("PandoraPFOs", "energy"),
}


def read(path):
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
        tree = source["events"]
        data = {key: tree[f"{coll}/{coll}.{field}"].array(library="ak") for key, (coll, field) in FIELDS.items()}
    primary = (data["pdg"] == 22) & (data["status"] == 1)
    if not np.all(ak.to_numpy(ak.sum(primary, axis=1)) == 1):
        raise ValueError(f"Expected exactly one primary photon per event in {path}")
    momentum = np.column_stack([ak.to_numpy(ak.flatten(data[axis][primary])) for axis in ("px", "py", "pz")])
    sim = ak.concatenate((data["sim_b"], data["sim_e"]), axis=1)
    digi = ak.concatenate((data["digi_b"], data["digi_e"]), axis=1)
    cluster = data["cluster_energy"]
    cluster_size = data["cluster_end"] - data["cluster_begin"]
    pfo = np.asarray(
        [
            max((float(e) for pdg, e in zip(pdgs, energies) if pdg == 22), default=0)
            for pdgs, energies in zip(ak.to_list(data["pfo_pdg"]), ak.to_list(data["pfo_energy"]))
        ]
    )
    return {
        "momentum": momentum,
        "sim_hits": ak.to_numpy(ak.num(sim)),
        "sim_energy": ak.to_numpy(ak.sum(sim, axis=1)),
        "digi_hits": ak.to_numpy(ak.num(digi)),
        "digi_energy": ak.to_numpy(ak.sum(digi, axis=1)),
        "clusters": ak.to_numpy(ak.num(cluster)),
        "cluster_energy": ak.to_numpy(ak.sum(cluster, axis=1)),
        "leading_cluster_energy": ak.to_numpy(ak.fill_none(ak.max(cluster, axis=1), 0)),
        "leading_cluster_hits": ak.to_numpy(ak.fill_none(ak.firsts(cluster_size[ak.argsort(cluster, axis=1, ascending=False)]), 0)),
        "pfo_energy": pfo,
        "hit_energy_values": ak.to_numpy(ak.flatten(digi)),
        "cluster_energy_values": ak.to_numpy(ak.flatten(cluster)),
        "cluster_hit_values": ak.to_numpy(ak.flatten(cluster_size)),
    }


def stats(value):
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p10": float(np.percentile(value, 10)),
        "p90": float(np.percentile(value, 90)),
        "zero_fraction": float(np.mean(value == 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"5.1 mm": files_by_seed(args.baseline), "2 mm": files_by_seed(args.fine)}
    if paths["5.1 mm"].keys() != paths["2 mm"].keys():
        raise ValueError("Seed sets differ")
    chunks = {arm: [] for arm in paths}
    event_ids = []
    for seed in sorted(paths["5.1 mm"]):
        pair = {arm: read(files[seed]) for arm, files in paths.items()}
        if not np.array_equal(pair["5.1 mm"]["momentum"], pair["2 mm"]["momentum"]):
            raise ValueError(f"Primary momentum mismatch for seed {seed}")
        chunks["5.1 mm"].append(pair["5.1 mm"])
        chunks["2 mm"].append(pair["2 mm"])
        event_ids.extend({"seed": seed, "event": i} for i in range(len(pair["5.1 mm"]["momentum"])))
    data = {arm: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]} for arm, parts in chunks.items()}
    keys = [
        "sim_hits",
        "sim_energy",
        "digi_hits",
        "digi_energy",
        "clusters",
        "cluster_energy",
        "leading_cluster_energy",
        "leading_cluster_hits",
        "pfo_energy",
    ]
    report = {"sample": SAMPLE, "events": len(event_ids), "arms": {}, "paired_differences_2mm_minus_5p1mm": {}}
    for arm in paths:
        report["arms"][arm] = {key: stats(data[arm][key]) for key in keys}
        report["arms"][arm]["per_digi_hit_energy_GeV"] = stats(data[arm]["hit_energy_values"])
        report["arms"][arm]["per_cluster_energy_GeV"] = stats(data[arm]["cluster_energy_values"])
        report["arms"][arm]["per_cluster_hits"] = stats(data[arm]["cluster_hit_values"])
        report["arms"][arm]["events_with_multiple_clusters_fraction"] = float(np.mean(data[arm]["clusters"] > 1))
    for key in keys:
        report["paired_differences_2mm_minus_5p1mm"][key] = stats(data["2 mm"][key] - data["5.1 mm"][key])
    delta = data["2 mm"]["leading_cluster_energy"] - data["5.1 mm"]["leading_cluster_energy"]
    report["largest_leading_cluster_energy_differences"] = [
        {
            **event_ids[i],
            "baseline_GeV": float(data["5.1 mm"]["leading_cluster_energy"][i]),
            "fine_GeV": float(data["2 mm"]["leading_cluster_energy"][i]),
        }
        for i in np.argsort(np.abs(delta))[-10:][::-1]
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plots = [
        ("sim_hits", "Simulated ECAL hits / event", (0, 900)),
        ("digi_hits", "Digitized ECAL hits / event", (0, 900)),
        ("hit_energy_values", "Digitized ECAL hit energy [GeV]", None),
        ("clusters", "Pandora clusters / event", (0, 12)),
        ("leading_cluster_energy", "Leading cluster energy [GeV]", (0, 14)),
        ("leading_cluster_hits", "Leading cluster hits", (0, 900)),
        ("cluster_energy", "Total cluster energy / event [GeV]", (0, 16)),
        ("cluster_energy_values", "Individual cluster energy [GeV]", (0, 14)),
        ("pfo_energy", "Leading photon PFO energy [GeV]", (0, 14)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for ax, (key, title, span) in zip(axes.flat, plots):
        if key == "hit_energy_values":
            bins = np.geomspace(1e-4, 1, 65)
            ax.set_xscale("log")
        elif key == "clusters":
            bins = np.arange(-0.5, 12.5, 1)
        else:
            bins = np.linspace(*span, 61)
        for arm, color in (("5.1 mm", "#2563eb"), ("2 mm", "#e76f51")):
            ax.hist(data[arm][key], bins=bins, histtype="step", linewidth=1.7, density=True, label=arm, color=color)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"CLD 10 GeV photon gun: {len(event_ids)} paired events")
    fig.savefig(args.output_dir / "hit_cluster_distributions.png", dpi=160)
    print(args.output_dir / "summary.json")
    print(args.output_dir / "hit_cluster_distributions.png")


if __name__ == "__main__":
    main()
