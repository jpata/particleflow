"""Compare truth-matched Pandora photons in paired CLD ECAL productions."""

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


SAMPLE = "p8_ee_ttbar_ecm365"
FIELDS = [
    *(f"MCParticles.{field}" for field in ("PDG", "generatorStatus", "simulatorStatus", "momentum.x", "momentum.y", "momentum.z")),
    *(f"PandoraPFOs.{field}" for field in ("PDG", "energy", "momentum.x", "momentum.y", "momentum.z")),
]
E_BINS = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 120, 250], dtype=float)
COLORS = {"5.1 mm": "#2563eb", "2 mm": "#e76f51"}


def files(workspace):
    directory = workspace / "gen" / SAMPLE / "root"
    prefix = f"reco_{SAMPLE}_"
    result = {int(path.stem.removeprefix(prefix)): path for path in directory.glob(f"{prefix}*.root")}
    if not result:
        raise FileNotFoundError(directory)
    return result


def kinematics(px, py, pz):
    pt = np.hypot(px, py)
    p = np.hypot(pt, pz)
    eta = np.arcsinh(np.divide(pz, pt, out=np.full_like(pt, np.nan), where=pt > 0))
    phi = np.arctan2(py, px)
    return p, eta, phi, np.divide(pz, p, out=np.full_like(p, np.nan), where=p > 0)


def event_particles(data, prefix, index, truth):
    pdg = ak.to_numpy(data[f"{prefix}.PDG"][index])
    px, py, pz = (ak.to_numpy(data[f"{prefix}.momentum.{axis}"][index]) for axis in "xyz")
    momentum, eta, phi, costheta = kinematics(px, py, pz)
    energy = momentum if truth else ak.to_numpy(data[f"{prefix}.energy"][index])
    mask = (pdg == 22) & np.isfinite(eta) & (np.abs(costheta) < 0.98)
    if truth:
        status = ak.to_numpy(data[f"{prefix}.generatorStatus"][index])
        simulator = ak.to_numpy(data[f"{prefix}.simulatorStatus"][index])
        mask &= (status == 1) & ((simulator & 0x0F000000) != 0) & (energy >= 0.1)
    else:
        mask &= np.isfinite(energy) & (energy >= 0.5)
    return np.column_stack((energy[mask], eta[mask], phi[mask]))


def summarize_response(values):
    if not len(values):
        return {"n": 0}
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    return {"n": int(len(values)), "median": float(p50), "p16": float(p16), "p84": float(p84), "half_width_68": float((p84 - p16) / 2)}


def binned_fraction(denominator, numerator, edges):
    total = np.histogram(denominator, edges)[0]
    passed = np.histogram(numerator, edges)[0]
    fraction = np.divide(passed, total, out=np.full(len(total), np.nan), where=total > 0)
    error = np.sqrt(np.divide(fraction * (1 - fraction), total, out=np.full(len(total), np.nan), where=total > 0))
    return total, passed, fraction, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-events-per-seed", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    args = parser.parse_args()
    paths = {"5.1 mm": files(args.baseline), "2 mm": files(args.fine)}
    seeds = sorted(paths["2 mm"])
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    if not set(seeds).issubset(paths["5.1 mm"]):
        raise ValueError("A fine-readout seed is missing from the baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        arm: {
            key: []
            for key in ("gen_e", "matched_gen_e", "reco_e", "unmatched_reco_e", "response", "response_gen_e", "isolated_response", "isolated_gen_e")
        }
        for arm in paths
    }
    checks = {"events": 0, "truth_photon_count_mismatch_events": 0, "truth_photon_energy_max_abs_GeV": 0.0}
    for seed in seeds:
        arrays = {}
        for arm, arm_paths in paths.items():
            with uproot.open(arm_paths[seed], handler=uproot.source.file.MemmapSource) as source:
                arrays[arm] = source["events"].arrays(FIELDS, entry_stop=args.max_events_per_seed, library="ak")
        n = len(arrays["5.1 mm"])
        if n != len(arrays["2 mm"]):
            raise ValueError(f"Seed {seed}: event count mismatch")
        print(f"seed {seed}: {n} events", flush=True)
        for event in range(n):
            checks["events"] += 1
            gen = {arm: event_particles(data, "MCParticles", event, True) for arm, data in arrays.items()}
            reco = {arm: event_particles(data, "PandoraPFOs", event, False) for arm, data in arrays.items()}
            if len(gen["5.1 mm"]) != len(gen["2 mm"]):
                checks["truth_photon_count_mismatch_events"] += 1
            elif len(gen["5.1 mm"]):
                checks["truth_photon_energy_max_abs_GeV"] = max(
                    checks["truth_photon_energy_max_abs_GeV"], float(np.max(np.abs(gen["5.1 mm"][:, 0] - gen["2 mm"][:, 0])))
                )
            # Compare both reconstructions to exactly the same generated photons.
            g = gen["5.1 mm"]
            for arm in paths:
                r = reco[arm]
                gi, ri = _match_jets_event(g[:, 1], g[:, 2], r[:, 1], r[:, 2], 0.03)
                rec = records[arm]
                rec["gen_e"].extend(g[g[:, 0] >= 1, 0].tolist())
                rec["matched_gen_e"].extend(g[gi[g[gi, 0] >= 1], 0].tolist())
                rec["reco_e"].extend(r[r[:, 0] >= 1, 0].tolist())
                unmatched = np.ones(len(r), dtype=bool)
                unmatched[ri] = False
                rec["unmatched_reco_e"].extend(r[unmatched & (r[:, 0] >= 1), 0].tolist())
                selected = g[gi, 0] >= 1
                response = r[ri[selected], 0] / g[gi[selected], 0]
                rec["response"].extend(response.tolist())
                rec["response_gen_e"].extend(g[gi[selected], 0].tolist())
                # Isolation uses the nearest other selected truth photon.
                if len(g) > 1 and np.any(selected):
                    de = g[:, 1, None] - g[None, :, 1]
                    dp = np.arctan2(np.sin(g[:, 2, None] - g[None, :, 2]), np.cos(g[:, 2, None] - g[None, :, 2]))
                    dr = np.hypot(de, dp)
                    np.fill_diagonal(dr, np.inf)
                    isolated = np.min(dr, axis=1)[gi[selected]] > 0.1
                    rec["isolated_response"].extend(response[isolated].tolist())
                    rec["isolated_gen_e"].extend(g[gi[selected][isolated], 0].tolist())
    data = {arm: {key: np.asarray(value, dtype=float) for key, value in rec.items()} for arm, rec in records.items()}
    report = {
        "sample": SAMPLE,
        "seeds": seeds,
        "checks": checks,
        "definition": {
            "truth": "MCParticles PDG 22, generatorStatus 1, propagated bit 0x0F000000, |cos(theta)|<0.98, E>=0.1 GeV for matching",
            "reco": "PandoraPFOs PDG 22, |cos(theta)|<0.98, E>=0.5 GeV for matching",
            "matching": "maximum-cardinality one-to-one deltaR<0.03; no energy-response cut",
            "efficiency_denominator": "baseline truth photons E>=1 GeV for both reconstructions",
            "unmatched_candidate_fraction": "unmatched Pandora photon PFOs / all photon PFOs, E>=1 GeV",
            "response": "Pandora PFO energy / generator photon momentum magnitude; truth E>=1 GeV",
            "isolation": "nearest other selected generator photon deltaR>0.1",
        },
        "arms": {},
    }
    for arm, rec in data.items():
        report["arms"][arm] = {
            "n_gen_e_ge_1": len(rec["gen_e"]),
            "n_matched_gen_e_ge_1": len(rec["matched_gen_e"]),
            "efficiency_e_ge_1": len(rec["matched_gen_e"]) / len(rec["gen_e"]),
            "n_reco_e_ge_1": len(rec["reco_e"]),
            "n_unmatched_reco_e_ge_1": len(rec["unmatched_reco_e"]),
            "unmatched_candidate_fraction_e_ge_1": len(rec["unmatched_reco_e"]) / len(rec["reco_e"]),
            "response": summarize_response(rec["response"]),
            "isolated_response": summarize_response(rec["isolated_response"]),
        }
    (args.output_dir / "photon_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    centers = np.sqrt(E_BINS[:-1] * E_BINS[1:])
    for arm, rec in data.items():
        color = COLORS[arm]
        den, _, fraction, error = binned_fraction(rec["gen_e"], rec["matched_gen_e"], E_BINS)
        valid = den >= 30
        axes[0, 0].errorbar(
            centers[valid],
            fraction[valid],
            yerr=error[valid],
            fmt="o-",
            capsize=2,
            color=color,
            label=f"{arm} ({report['arms'][arm]['efficiency_e_ge_1']:.1%} overall)",
        )
        den, _, fraction, error = binned_fraction(rec["reco_e"], rec["unmatched_reco_e"], E_BINS)
        valid = den >= 30
        axes[0, 1].errorbar(
            centers[valid],
            fraction[valid],
            yerr=error[valid],
            fmt="o-",
            capsize=2,
            color=color,
            label=f"{arm} ({report['arms'][arm]['unmatched_candidate_fraction_e_ge_1']:.1%} overall)",
        )
        axes[1, 0].hist(
            rec["response"],
            bins=np.linspace(0, 2, 81),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{arm}: median {report['arms'][arm]['response']['median']:.3f}",
        )
        widths = []
        for lo, hi in zip(E_BINS[:-1], E_BINS[1:]):
            values = rec["response"][(rec["response_gen_e"] >= lo) & (rec["response_gen_e"] < hi)]
            summary = summarize_response(values)
            widths.append(summary.get("half_width_68", np.nan) if len(values) >= 100 else np.nan)
        axes[1, 1].plot(centers, widths, "o-", color=color, label=arm)
    axes[0, 0].set(title="Photon matching efficiency", ylabel="Matched truth photon fraction", xlabel="Generator photon energy [GeV]", ylim=(0, 1.05))
    axes[0, 1].set(
        title="Unmatched Pandora photon candidates", ylabel="Unmatched candidate fraction", xlabel="Pandora photon energy [GeV]", ylim=(0, 1.05)
    )
    axes[1, 0].set(title="Matched photon energy response", xlabel="Pandora energy / generator energy", ylabel="Density")
    axes[1, 1].set(title="Energy response spread", xlabel="Generator photon energy [GeV]", ylabel="(p84 − p16) / 2")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xscale("log")
    fig.suptitle(f"CLD Pandora photons: {checks['events']} paired ttbar events, ΔR < 0.03")
    fig.savefig(args.output_dir / "pandora_photons.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
