"""Validate CLD granularity scans with per-event hit and reconstruction counts.

Analyze each point with --point, then draw both scan summaries with --plot.
Only collection lengths and PFO PDG codes are read from the EDM4hep files.
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

BASE = Path("/local/joosep/mlpf/cld/granularity_scan_2025-05-29")
OUT = Path("reports/cld_granularity_scan/validation")
PITCHES = {"ecal": ["5p1", "4", "3", "2p5", "2", "1p5"], "hcal": ["30", "25", "20", "15", "12", "10"]}
SAMPLES = {"ecal": "p8_ee_Z_tautau_ecm91", "hcal": "p8_ee_ttbar_ecm365"}
COLLECTIONS = {
    "sim_ecal": ("ECalBarrelCollection", "ECalEndcapCollection"),
    "reco_ecal": ("ECALBarrel", "ECALEndcap"),
    "sim_hcal": ("HCalBarrelCollection", "HCalEndcapCollection", "HCalRingCollection"),
    "reco_hcal": ("HCALBarrel", "HCALEndcap", "HCALOther"),
    "tracks": ("SiTracks_Refitted",),
    "clusters": ("PandoraClusters",),
    "pfos": ("PandoraPFOs",),
}
FIELD = {
    name: "type" if name == "SiTracks_Refitted" else "energy" if name in ("PandoraClusters", "PandoraPFOs") else "cellID"
    for names in COLLECTIONS.values()
    for name in names
}
BRANCHES = [f"{name}/{name}.{FIELD[name]}" for name in FIELD]
PFO_PDG = "PandoraPFOs/PandoraPFOs.PDG"
BRANCHES.append(PFO_PDG)
ORDER = ("sim_ecal", "reco_ecal", "sim_hcal", "reco_hcal", "tracks", "clusters", "pfos", "photon_pfos")


def point_files(point):
    group = point.split("_", 1)[0]
    sample = SAMPLES[group]
    root = BASE / f"{point}mm" / "gen" / sample / "root"
    files = sorted(root.glob(f"reco_{sample}_*.root"))
    if len(files) != 100:
        raise ValueError(f"Expected 100 files for {point}, found {len(files)}")
    return files


def summarize(x):
    x = np.asarray(x)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "max": int(x.max()),
    }


def analyze(point, out):
    buffers = {key: [] for key in ORDER}
    sizes = []
    for file in point_files(point):
        sizes.append(file.stat().st_size / 1e6)
        with uproot.open(file, handler=uproot.source.file.MemmapSource) as source:
            tree = source["events"]
            if tree.num_entries != 100:
                raise ValueError(f"Expected 100 events in {file}")
            data = tree.arrays(BRANCHES, library="ak")
        counts = {name: ak.to_numpy(ak.num(data[f"{name}/{name}.{FIELD[name]}"])) for name in FIELD}
        for key, names in COLLECTIONS.items():
            buffers[key].append(sum(counts[name] for name in names))
        buffers["photon_pfos"].append(ak.to_numpy(ak.sum(data[PFO_PDG] == 22, axis=1)))
    values = {key: np.concatenate(chunks).astype(np.int32) for key, chunks in buffers.items()}
    if any(len(v) != 10000 for v in values.values()):
        raise ValueError(f"Incomplete event arrays for {point}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{point}_counts.npz", **values)
    report = {
        "point": point,
        "files": len(sizes),
        "events": 10000,
        "mean_file_MB": float(np.mean(sizes)),
        "counts": {k: summarize(v) for k, v in values.items()},
        "definitions": {
            "sim_ecal": "ECalBarrelCollection + ECalEndcapCollection SimCalorimeterHits",
            "reco_ecal": "ECALBarrel + ECALEndcap reconstructed hits",
            "sim_hcal": "HCalBarrelCollection + HCalEndcapCollection + HCalRingCollection SimCalorimeterHits",
            "reco_hcal": "HCALBarrel + HCALEndcap + HCALOther reconstructed hits",
            "tracks": "SiTracks_Refitted",
            "clusters": "PandoraClusters",
            "pfos": "PandoraPFOs",
            "photon_pfos": "PandoraPFOs with PDG 22",
        },
    }
    target = out / f"{point}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(point, {key: round(report["counts"][key]["mean"], 2) for key in ORDER}, flush=True)


def plot_group(group, out):
    points = [f"{group}_{pitch}" for pitch in PITCHES[group]]
    reports = [json.loads((out / f"{point}.json").read_text()) for point in points]
    if any(r["events"] != 10000 or r["files"] != 100 for r in reports):
        raise ValueError(f"Incomplete {group} point")
    x = np.array([float(p.replace("p", ".")) for p in PITCHES[group]])
    target_sim, target_reco = ("sim_ecal", "reco_ecal") if group == "ecal" else ("sim_hcal", "reco_hcal")
    control_sim, control_reco = ("sim_hcal", "reco_hcal") if group == "ecal" else ("sim_ecal", "reco_ecal")
    panels = [
        (target_sim, "Changed calorimeter: simulated hits"),
        (target_reco, "Changed calorimeter: reconstructed hits"),
        (control_sim, "Other calorimeter: simulated hits"),
        (control_reco, "Other calorimeter: reconstructed hits"),
        ("tracks", "Refitted tracks"),
        ("clusters", "Pandora clusters"),
        ("pfos", "Pandora PFOs"),
        ("photon_pfos", "Pandora photon PFOs"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, (key, title) in zip(axs.flat, panels):
        mean = [r["counts"][key]["mean"] for r in reports]
        sem = [r["counts"][key]["sem"] for r in reports]
        ax.errorbar(x, mean, yerr=sem, fmt="o-", capsize=2)
        ax.set(xlabel=f"{group.upper()} cell pitch [mm]", ylabel="Mean / event", title=title)
        ax.grid(alpha=0.25)
    fig.suptitle(f"CLD {group.upper()} granularity validation · 10,000 events per point")
    fig.savefig(out / f"{group}_count_trends.png", dpi=160)
    plt.close(fig)

    first, last = points[0], points[-1]
    arrays = {point: np.load(out / f"{point}_counts.npz") for point in (first, last)}
    keys = (target_sim, target_reco, "tracks", "clusters")
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, key in zip(axs.flat, keys):
        xmax = max(np.percentile(arrays[p][key], 99.5) for p in (first, last))
        bins = np.linspace(0, xmax, 50)
        for point, label in ((first, PITCHES[group][0].replace("p", ".")), (last, PITCHES[group][-1].replace("p", "."))):
            ax.hist(arrays[point][key], bins=bins, histtype="step", density=True, linewidth=1.8, label=f"{label} mm")
        ax.set(xlabel=f"{key.replace('_', ' ')} / event", ylabel="Density")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(f"CLD {group.upper()}: baseline and finest pitch event-count distributions")
    fig.savefig(out / f"{group}_count_distributions.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", choices=[f"{g}_{p}" for g in PITCHES for p in PITCHES[g]])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.plot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for group in PITCHES:
            plot_group(group, args.output_dir)
    elif args.point:
        analyze(args.point, args.output_dir)
    else:
        parser.error("Specify --point or --plot")


if __name__ == "__main__":
    main()
