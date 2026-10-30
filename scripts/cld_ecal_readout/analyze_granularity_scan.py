"""Measure Pandora PF performance across the CLD granularity scan.

Run one point at a time with --point, then make the scan plots with --plot.
The photon and jet definitions follow compare_ttbar_pf_photons.py and
compare_ttbar_pf_jets.py.  The ECAL sample additionally measures whether both
direct daughters of a generator pi0 are reconstructed as separate photons.
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

from mlpf.jet_utils import _match_jets_event
from compare_ttbar_pf_jets import jet_matrix, particles_and_jets, summarize
from compare_ttbar_pf_photons import kinematics

BASE = Path("/local/joosep/mlpf/cld/granularity_scan_2025-05-29")
OUT = Path("reports/cld_granularity_scan/performance")
ECAL = ["5p1", "4", "3", "2p5", "2", "1p5"]
HCAL = ["30", "25", "20", "15", "12", "10"]
PI0_ANGLE_BINS = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.0])
PHOTON_FIELDS = [
    *(
        f"MCParticles.{x}"
        for x in ("PDG", "generatorStatus", "simulatorStatus", "momentum.x", "momentum.y", "momentum.z", "daughters_begin", "daughters_end")
    ),
    "_MCParticles_daughters/_MCParticles_daughters.index",
    *(f"PandoraPFOs.{x}" for x in ("PDG", "energy", "momentum.x", "momentum.y", "momentum.z")),
]


def sample_files(point, max_seeds=None):
    sample = "p8_ee_Z_tautau_ecm91" if point.startswith("ecal") else "p8_ee_ttbar_ecm365"
    paths = sorted((BASE / f"{point}mm" / "gen" / sample / "root").glob(f"reco_{sample}_*.root"))
    if not paths:
        raise FileNotFoundError(f"No ROOT files for {point}")
    if max_seeds is not None:
        paths = paths[:max_seeds]
    return paths


def summary(values):
    return summarize(np.asarray(values, dtype=float))


def photon_event(data, i):
    pdg = ak.to_numpy(data["MCParticles.PDG"][i])
    status = ak.to_numpy(data["MCParticles.generatorStatus"][i])
    sim = ak.to_numpy(data["MCParticles.simulatorStatus"][i])
    px, py, pz = (ak.to_numpy(data[f"MCParticles.momentum.{axis}"][i]) for axis in "xyz")
    e, eta, phi, costh = kinematics(px, py, pz)
    selected = (pdg == 22) & (status == 1) & ((sim & 0x0F000000) != 0) & (e >= 0.1) & (np.abs(costh) < 0.98) & np.isfinite(eta)
    ids = np.where(selected)[0]
    truth = np.column_stack((e[ids], eta[ids], phi[ids]))
    rpdg = ak.to_numpy(data["PandoraPFOs.PDG"][i])
    rpx, rpy, rpz = (ak.to_numpy(data[f"PandoraPFOs.momentum.{axis}"][i]) for axis in "xyz")
    _, reta, rphi, rcos = kinematics(rpx, rpy, rpz)
    re = ak.to_numpy(data["PandoraPFOs.energy"][i])
    rm = (rpdg == 22) & (re >= 0.5) & (np.abs(rcos) < 0.98) & np.isfinite(reta)
    reco = np.column_stack((re[rm], reta[rm], rphi[rm]))
    gi, ri = _match_jets_event(truth[:, 1], truth[:, 2], reco[:, 1], reco[:, 2], 0.03)
    db = ak.to_numpy(data["MCParticles.daughters_begin"][i])
    de = ak.to_numpy(data["MCParticles.daughters_end"][i])
    links = ak.to_numpy(data["_MCParticles_daughters/_MCParticles_daughters.index"][i])
    id_to_truth = {int(k): j for j, k in enumerate(ids)}
    matched = dict(zip(gi.tolist(), ri.tolist()))
    pairs = []
    for parent in np.where(pdg == 111)[0]:
        daughters = links[db[parent] : de[parent]]
        if len(daughters) != 2 or not np.all(pdg[daughters] == 22):
            continue
        if not all(int(d) in id_to_truth for d in daughters):
            continue
        a, b = (id_to_truth[int(d)] for d in daughters)
        if min(truth[a, 0], truth[b, 0]) < 1:
            continue
        # The opening angle is three-dimensional; the matcher remains in delta R.
        va = np.array([px[daughters[0]], py[daughters[0]], pz[daughters[0]]])
        vb = np.array([px[daughters[1]], py[daughters[1]], pz[daughters[1]]])
        cosang = np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)
        pairs.append((float(np.arccos(cosang)), a in matched and b in matched))
    good = truth[gi, 0] >= 1
    responses = reco[ri[good], 0] / truth[gi[good], 0]
    unmatched = np.ones(len(reco), dtype=bool)
    unmatched[ri] = False
    return truth, reco, gi, ri, responses, unmatched, pairs


def analyze_ecal(point, paths):
    all_e, matched_e, reco_e, unmatched_e, response, angles, resolved = ([] for _ in range(7))
    events = 0
    for path in paths:
        print(f"{point}: {path.name}", flush=True)
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
            data = f["events"].arrays(PHOTON_FIELDS, library="ak")
        events += len(data)
        for i in range(len(data)):
            truth, reco, gi, ri, rr, um, pairs = photon_event(data, i)
            all_e.extend(truth[truth[:, 0] >= 1, 0])
            matched_e.extend(truth[gi[truth[gi, 0] >= 1], 0])
            reco_e.extend(reco[reco[:, 0] >= 1, 0])
            unmatched_e.extend(reco[um & (reco[:, 0] >= 1), 0])
            response.extend(rr)
            angles.extend(p[0] for p in pairs)
            resolved.extend(p[1] for p in pairs)
    den = np.histogram(angles, PI0_ANGLE_BINS)[0]
    num = np.histogram(np.asarray(angles)[np.asarray(resolved, dtype=bool)], PI0_ANGLE_BINS)[0]
    return {
        "point": point,
        "files": len(paths),
        "events": events,
        "truth_photons_e_ge_1": len(all_e),
        "matched_photons_e_ge_1": len(matched_e),
        "photon_efficiency": len(matched_e) / len(all_e),
        "reco_photons_e_ge_1": len(reco_e),
        "unmatched_reco_photons_e_ge_1": len(unmatched_e),
        "unmatched_candidate_fraction": len(unmatched_e) / len(reco_e),
        "photon_energy_response": summary(response),
        "pi0_pairs": len(angles),
        "resolved_pi0_pairs": sum(resolved),
        "pi0_pair_efficiency": float(np.mean(resolved)) if resolved else None,
        "pi0_angle_bins_rad": PI0_ANGLE_BINS.tolist(),
        "pi0_angle_denominator": den.tolist(),
        "pi0_angle_resolved": num.tolist(),
    }


def analyze_hcal(point, paths):
    response, truth_pt, matched_pt, pfo_count = ([] for _ in range(4))
    ntruth = events = 0
    for path in paths:
        print(f"{point}: {path.name}", flush=True)
        _, _, truth_jets, reco_jets, counts = particles_and_jets(path)
        events += len(truth_jets)
        pfo_count.extend(ak.to_numpy(counts))
        for i in range(len(truth_jets)):
            truth = jet_matrix(truth_jets[i])
            reco = jet_matrix(reco_jets[i])
            gi, ri = _match_jets_event(truth[:, 1], truth[:, 2], reco[:, 1], reco[:, 2], 0.1)
            ntruth += len(truth)
            truth_pt.extend(truth[:, 0])
            matched_pt.extend(truth[gi, 0])
            response.extend(reco[ri, 0] / truth[gi, 0])
    return {
        "point": point,
        "files": len(paths),
        "events": events,
        "truth_jets": ntruth,
        "matched_jets": len(response),
        "jet_matching_efficiency": len(response) / ntruth,
        "jet_pt_response": summary(response),
        "pfo_count_per_event": summary(pfo_count),
    }


def plot_scan(out):
    def load(group, pitches):
        rows = []
        for pitch in pitches:
            path = out / f"{group}_{pitch}.json"
            rows.append(json.loads(path.read_text()))
            if rows[-1]["files"] != 100 or rows[-1]["events"] != 10000:
                raise ValueError(f"Incomplete point: {path}")
        return rows

    e = load("ecal", ECAL)
    h = load("hcal", HCAL)
    x = np.array([5.1, 4, 3, 2.5, 2, 1.5])
    y = np.array([30, 25, 20, 15, 12, 10])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for a, key, label in zip(
        ax.flat,
        ("photon_efficiency", "pi0_pair_efficiency", "unmatched_candidate_fraction", "photon_energy_response"),
        ("Truth photon matching efficiency", "Resolved π⁰→γγ fraction", "Unmatched photon PFO fraction", "Photon energy response"),
    ):
        if key == "photon_energy_response":
            a.plot(x, [r[key]["median"] for r in e], "o-", label="Median")
            a.plot(x, [r[key]["half_width_68"] for r in e], "s-", label="68% half-width")
            a.legend()
        else:
            den_key = {
                "photon_efficiency": "truth_photons_e_ge_1",
                "pi0_pair_efficiency": "pi0_pairs",
                "unmatched_candidate_fraction": "reco_photons_e_ge_1",
            }[key]
            values = np.array([r[key] for r in e])
            errors = np.sqrt(values * (1 - values) / np.array([r[den_key] for r in e]))
            a.errorbar(x, values, yerr=errors, fmt="o-", capsize=2)
        a.set(xlabel="ECAL cell pitch [mm]", ylabel=label)
        a.grid(alpha=0.25)
    fig.suptitle("CLD Pandora photons in Z→ττ, 10,000 events per point")
    fig.savefig(out / "ecal_photon_performance.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for pitch, row in zip(ECAL, e):
        den = np.asarray(row["pi0_angle_denominator"])
        num = np.asarray(row["pi0_angle_resolved"])
        centers = np.sqrt(PI0_ANGLE_BINS[:-1] * PI0_ANGLE_BINS[1:])
        centers[0] = PI0_ANGLE_BINS[1] / 2
        ax.plot(centers[den >= 20], (num / np.maximum(den, 1))[den >= 20], "o-", label=pitch.replace("p", ".") + " mm")
    ax.set(xscale="log", xlabel="Truth γγ opening angle [rad]", ylabel="Resolved π⁰→γγ fraction", ylim=(0, 1.05))
    ax.legend(title="ECAL pitch", ncol=2)
    ax.grid(alpha=0.25)
    fig.savefig(out / "ecal_pi0_opening_angle.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for a, key, label in zip(
        ax.flat,
        ("jet_matching_efficiency", "jet_pt_response", "jet_pt_response", "pfo_count_per_event"),
        ("Truth jet matching efficiency", "Median jet pT response", "Jet response 68% half-width", "Pandora PFOs per event"),
    ):
        value = [
            (
                r[key]["median" if a is ax[0, 1] else "half_width_68"]
                if key == "jet_pt_response"
                else r[key]["mean"] if key == "pfo_count_per_event" else r[key]
            )
            for r in h
        ]
        if key == "jet_matching_efficiency":
            values = np.asarray(value)
            error = np.sqrt(values * (1 - values) / np.array([r["truth_jets"] for r in h]))
            a.errorbar(y, values, yerr=error, fmt="o-", capsize=2)
        else:
            a.plot(y, value, "o-")
        a.set(xlabel="HCAL cell pitch [mm]", ylabel=label)
        a.grid(alpha=0.25)
    fig.suptitle("CLD Pandora PF jets in ttbar, 10,000 events per point")
    fig.savefig(out / "hcal_jet_performance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", choices=[*(f"ecal_{x}" for x in ECAL), *(f"hcal_{x}" for x in HCAL)])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_scan(args.output_dir)
        return
    if not args.point:
        parser.error("Specify --point or --plot")
    paths = sample_files(args.point, args.max_seeds)
    result = analyze_ecal(args.point, paths) if args.point.startswith("ecal") else analyze_hcal(args.point, paths)
    result["definitions"] = {
        "photon": "Previous ttbar photon study: status-1 propagated truth photon E>=1 GeV; Pandora PDG 22 E>=0.5 GeV; one-to-one ΔR<0.03",
        "pi0": "Direct pi0→γγ daughters, both propagated in |cosθ|<0.98 and E>=1 GeV; resolved when each has a distinct matched Pandora photon",
        "jet": "Previous ttbar jet study: ee-genkt R=0.4 p=-1, pT>5 GeV; propagated visible truth vs Pandora PFO jets; one-to-one ΔR<0.1; response=pT reco/truth",
        "width": "(p84-p16)/2; no response cut; unchanged Pandora calibration",
    }
    target = args.output_dir / f"{args.point}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        target,
        json.dumps(
            {k: v for k, v in result.items() if k not in ("definitions", "pi0_angle_bins_rad", "pi0_angle_denominator", "pi0_angle_resolved")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
