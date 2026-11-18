"""Paired numerical, particle and clustered-jet closure for retained CMS outputs.

Physics agreement is reported with paired event-bootstrap intervals; no physics
acceptance margin is inferred from the observed sample. Raw-head acceptance
continues to use atol=1e-3 and rtol=3e-3 independently of these distributions.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

from mlpf.standalone_eval.cms_evidence import input_manifest_path
from types import SimpleNamespace

import numpy as np

HEADS = ("bid", "id", "momentum", "pu")
BINS = {
    "particle_pt": np.geomspace(0.01, 1000, 65),
    "particle_energy": np.geomspace(0.01, 5000, 65),
    "particle_eta": np.linspace(-6, 6, 61),
    "particle_phi": np.linspace(-np.pi, np.pi, 65),
    "particle_multiplicity": np.linspace(0, 6000, 61),
    "particle_pid": np.arange(-0.5, 9.5, 1),
    "jet_pt": np.geomspace(3, 2000, 65),
    "jet_energy": np.geomspace(3, 10000, 65),
    "jet_eta": np.linspace(-6, 6, 61),
    "jet_phi": np.linspace(-np.pi, np.pi, 65),
    "jet_mass": np.linspace(-20, 300, 65),
    "jet_multiplicity": np.linspace(-0.5, 200.5, 68),
    "jet_leading_pt": np.geomspace(3, 2000, 65),
    "jet_subleading_pt": np.geomspace(3, 2000, 65),
    "met": np.linspace(0, 1000, 65),
    "sumpt": np.linspace(0, 6000, 65),
    "jet_nopu_pt": np.geomspace(3, 2000, 65),
    "jet_nopu_multiplicity": np.linspace(-0.5, 200.5, 68),
    "jet30_pt": np.geomspace(30, 2000, 65),
    "jet30_mass": np.linspace(-20, 300, 65),
    "jet30_multiplicity": np.arange(-0.5, 30.5, 1),
}
for pid in range(1, 9):
    BINS[f"particle_pt_pid{pid}"] = BINS["particle_pt"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def columnar_events(rows):
    """Build jagged columns without boxing every particle into Python objects."""
    import awkward as ak

    columns = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if isinstance(values[0], dict):
            columns[key] = columnar_events(values)
        elif np.asarray(values[0]).ndim == 0:
            columns[key] = np.asarray(values)
        else:
            counts = np.array([len(value) for value in values])
            columns[key] = ak.unflatten(np.concatenate(values), counts)
    return ak.zip(columns, depth_limit=1)


def decode(raw, features):
    """Invoke production decoding without creating another network or copying it."""
    import torch
    from mlpf.conf import OutputMode
    from mlpf.model.mlpf import MLPF

    proxy = SimpleNamespace(
        output_mode=OutputMode.ELEMENTWISE, forward=lambda x, mask, source_id=None: tuple(torch.from_numpy(raw[k].copy())[None] for k in HEADS)
    )
    x = torch.from_numpy(features.copy())[None]
    with torch.inference_mode():
        particles = MLPF.predict_particles(proxy, x, torch.ones(x.shape[:2], dtype=torch.bool))
    return {key: particles[key][0].numpy() for key in ["cls_id", "pt", "eta", "phi", "energy", "ispu"]}


def cluster(particles, no_pu=False):
    import awkward as ak
    import fastjet
    import vector

    active = particles["cls_id"] != 0
    if no_pu:
        active &= particles["ispu"] < 0.8
    pt, eta, phi, energy = [np.asarray(particles[k][active], dtype=np.float64) for k in ["pt", "eta", "phi", "energy"]]
    if not len(pt):
        return {k: np.empty(0) for k in ["pt", "eta", "phi", "energy", "mass"]}
    if not all(np.isfinite(a).all() for a in [pt, eta, phi, energy]):
        raise ValueError("Nonfinite decoded four-vectors")
    vectors = ak.zip({"px": pt * np.cos(phi), "py": pt * np.sin(phi), "pz": pt * np.sinh(eta), "E": energy})
    sequence = fastjet.ClusterSequence(vectors, fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4))
    jets = vector.awk(sequence.inclusive_jets(min_pt=3.0))
    order = np.argsort(-ak.to_numpy(jets.pt))
    return {k: ak.to_numpy(getattr(jets, prop))[order] for k, prop in [("pt", "pt"), ("eta", "eta"), ("phi", "phi"), ("energy", "E"), ("mass", "m")]}


def decode_target(values):
    import torch
    from mlpf.model.utils import unpack_target

    particles = unpack_target(torch.from_numpy(values.copy()), None)
    return {key: particles[key].numpy() for key in ["cls_id", "pt", "eta", "phi", "energy", "ispu"]}


def distributions(particles, jets, jets_nopu):
    active = particles["cls_id"] != 0
    pt, phi = particles["pt"][active], particles["phi"][active]
    values = {"particle_" + k: particles[k][active] for k in ["pt", "energy", "eta", "phi"]}
    values.update(
        particle_multiplicity=np.array([active.sum()]),
        particle_pid=particles["cls_id"][active],
        met=np.array([np.hypot(np.sum(pt * np.cos(phi)), np.sum(pt * np.sin(phi)))]),
        sumpt=np.array([pt.sum()]),
    )
    hard = jets["pt"] >= 30
    values.update(jet30_pt=jets["pt"][hard], jet30_mass=jets["mass"][hard], jet30_multiplicity=np.array([hard.sum()]))
    values.update({"jet_" + k: v for k, v in jets.items()})
    values.update(
        jet_multiplicity=np.array([len(jets["pt"])]),
        jet_leading_pt=jets["pt"][:1],
        jet_subleading_pt=jets["pt"][1:2],
        jet_nopu_pt=jets_nopu["pt"],
        jet_nopu_multiplicity=np.array([len(jets_nopu["pt"])]),
    )
    for pid in range(1, 9):
        values[f"particle_pt_pid{pid}"] = particles["pt"][particles["cls_id"] == pid]
    return values


def histograms(values):
    # Retain under/overflow separately rather than silently dropping tails.
    return {
        key: np.r_[np.count_nonzero(values[key] < edges[0]), np.histogram(values[key], edges)[0], np.count_nonzero(values[key] > edges[-1])]
        for key, edges in BINS.items()
    }


def jet_residuals(reference, candidate):
    from mlpf.jet_utils import _match_jets_event

    r, c = _match_jets_event(reference["eta"], reference["phi"], candidate["eta"], candidate["phi"], 0.1)
    dphi = np.arctan2(np.sin(candidate["phi"][c] - reference["phi"][r]), np.cos(candidate["phi"][c] - reference["phi"][r]))
    return {
        "pt_response": candidate["pt"][c] / reference["pt"][r],
        "energy_response": candidate["energy"][c] / reference["energy"][r],
        "delta_mass": candidate["mass"][c] - reference["mass"][r],
        "delta_r": np.hypot(candidate["eta"][c] - reference["eta"][r], dphi),
        "unmatched_reference": len(reference["pt"]) - len(r),
        "unmatched_candidate": len(candidate["pt"]) - len(c),
        "reference_count": len(reference["pt"]),
        "candidate_count": len(candidate["pt"]),
    }


def retained_cases(directory, cases, root, model):
    """Require successful workers and traceable outputs before plotting closure."""
    directory = directory.resolve()
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = json.loads((root / "release.json").read_text())["models"][model]["checkpoint_sha256"]
        assert manifest["checkpoint_sha256"] == expected
        return {c["case"]: c for c in manifest["cases"]}
    records = {}
    for slots in sorted({c["slots"] for c in cases}):
        manifest = json.loads((directory / f"bucket_{slots}.json").read_text())
        static = directory / f"bucket_{slots}.onnx"
        assert hashlib.sha256(static.read_bytes()).hexdigest() == manifest["static_sha256"]
        if manifest["provider"] != "cpu":
            assert manifest["placement"]["full_provider_placement"]
        for case in manifest["cases"]:
            assert case["finite"] and case["padding_zero"]
            assert case["case"] not in records
            records[case["case"]] = case
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=["attention", "heptv2"], required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-label", default="PyTorch CPU FP32 reference")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--allow-small-sample", action="store_true", help="Explicitly label a smoke test, never a full qualification")
    args = parser.parse_args()
    import awkward as ak
    import torch

    torch.set_num_threads(1)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = json.loads(input_manifest_path(args.root).read_text())["cases"]
    if args.limit:
        cases = cases[: args.limit]
    if len(cases) < 1000 and not args.allow_small_sample:
        parser.error("Full physics qualification requires at least 1000 unique events")
    reference = args.reference or args.root / "reference" / args.model
    reference_records = retained_cases(reference, cases, args.root, args.model)
    candidate_records = retained_cases(args.candidate, cases, args.root, args.model)
    args.outdir.mkdir(parents=True, exist_ok=True)
    numeric = {
        head: {"max_abs": 0.0, "sum_abs": 0.0, "sum_squared": 0.0, "target_squared": 0.0, "elements": 0, "violations": 0, "argmax_changes": 0}
        for head in HEADS
    }
    refs = {key: [] for key in BINS}
    candidates = {key: [] for key in BINS}
    contexts = {name: {key: [] for key in BINS} for name in ["target", "cand"]}
    response = []
    hard_response = []
    event_shifts = {key: [] for key in ["particle_count", "jet_count", "jet30_count", "met", "sumpt"]}
    events = []
    presence_counts = []
    migrations = np.zeros((9, 9), dtype=np.int64)
    for i, case in enumerate(cases):
        path = args.root / "inputs" / case["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        for directory, records in [(reference, reference_records), (args.candidate, candidate_records)]:
            assert records[case["file"]]["input_sha256"] == case["sha256"]
            assert hashlib.sha256((directory / case["file"]).read_bytes()).hexdigest() == records[case["file"]]["output_sha256"]
        with np.load(path) as data, np.load(reference / case["file"]) as ref, np.load(args.candidate / case["file"]) as cand:
            x = data["X"].copy()
            for head in HEADS:
                a, b = cand[head].astype(np.float64), ref[head].astype(np.float64)
                assert a.shape == b.shape and len(a) == case["length"] and np.isfinite(a).all() and np.isfinite(b).all()
                delta = a - b
                row = numeric[head]
                row["max_abs"] = max(row["max_abs"], float(abs(delta).max()))
                row["sum_abs"] += float(abs(delta).sum())
                row["sum_squared"] += float((delta**2).sum())
                row["target_squared"] += float((b**2).sum())
                row["elements"] += a.size
                row["violations"] += int(np.count_nonzero(abs(delta) > 1e-3 + 3e-3 * abs(b)))
                if head != "momentum":
                    row["argmax_changes"] += int(np.count_nonzero(a.argmax(-1) != b.argmax(-1)))
            pr, pc = decode(ref, x), decode(cand, x)
            reference_present = pr["cls_id"] != 0
            candidate_present = pc["cls_id"] != 0
            presence_counts.append(
                (
                    int(reference_present.sum()),
                    int(candidate_present.sum()),
                    int((~reference_present & candidate_present).sum()),
                    int((reference_present & ~candidate_present).sum()),
                )
            )
            jr, jc = cluster(pr), cluster(pc)
            nr, nc = cluster(pr, True), cluster(pc, True)
            context_particles = {name: decode_target(data[name]) for name in contexts}
            context_jets = {name: cluster(particles) for name, particles in context_particles.items()}
            for name in contexts:
                hh = histograms(distributions(context_particles[name], context_jets[name], cluster(context_particles[name], True)))
                for key in BINS:
                    contexts[name][key].append(hh[key])
            vr, vc = distributions(pr, jr, nr), distributions(pc, jc, nc)
            hr, hc = histograms(vr), histograms(vc)
            for key, observable in [
                ("particle_count", "particle_multiplicity"),
                ("jet_count", "jet_multiplicity"),
                ("jet30_count", "jet30_multiplicity"),
                ("met", "met"),
                ("sumpt", "sumpt"),
            ]:
                event_shifts[key].append(float(vc[observable][0] - vr[observable][0]))
            for key in BINS:
                refs[key].append(hr[key])
                candidates[key].append(hc[key])
            np.add.at(migrations, (pr["cls_id"], pc["cls_id"]), 1)
            response.append(jet_residuals(jr, jc))
            hard_response.append(jet_residuals({k: v[jr["pt"] >= 30] for k, v in jr.items()}, {k: v[jc["pt"] >= 30] for k, v in jc.items()}))
            events.append(
                {
                    "index": case["index"],
                    "particles_reference": pr,
                    "particles_candidate": pc,
                    "particles_target": context_particles["target"],
                    "particles_cand": context_particles["cand"],
                    "jets_target": context_jets["target"],
                    "jets_cand": context_jets["cand"],
                    "genjets": data["genjets"].copy(),
                    "genmet": data["genmet"].copy(),
                    "jets_reference": jr,
                    "jets_candidate": jc,
                    "jet_residuals": response[-1],
                    "jet30_residuals": hard_response[-1],
                }
            )
        if (i + 1) % 32 == 0:
            print("Analyzed", i + 1, "events", flush=True)
    for head, row in numeric.items():
        row["mae"] = row["sum_abs"] / row["elements"]
        row["rmse"] = np.sqrt(row["sum_squared"] / row["elements"])
        row["relative_l2"] = np.sqrt(row["sum_squared"] / max(row["target_squared"], 1e-24))
        row["violating_fraction"] = row["violations"] / row["elements"]
    weights = np.random.default_rng(20261030).multinomial(len(cases), np.full(len(cases), 1 / len(cases)), size=args.bootstrap)
    histogram_arrays = {}
    plots = []
    for key, edges in BINS.items():
        r, c = np.array(refs[key]), np.array(candidates[key])
        delta = c - r
        lo, hi = np.quantile(weights @ delta, [0.025, 0.975], axis=0)
        histogram_arrays.update(
            {
                key + "_edges": edges,
                key + "_reference": r.sum(0),
                key + "_candidate": c.sum(0),
                key + "_difference_ci_low": lo,
                key + "_difference_ci_high": hi,
                key + "_event_reference": r,
                key + "_event_candidate": c,
            }
        )
        fig, (ax, ratio) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]})
        rr, cc = r.sum(0)[1:-1], c.sum(0)[1:-1]
        centers = (edges[1:] + edges[:-1]) / 2
        ax.stairs(rr, edges, label=args.reference_label)
        ax.stairs(cc, edges, label="Candidate runtime", linestyle="--")
        for name in contexts:
            context = np.array(contexts[name][key]).sum(0)
            histogram_arrays[key + "_" + name] = context
            ax.stairs(context[1:-1], edges, label={"target": "Target particles", "cand": "CMS PF candidates"}[name], alpha=0.4, linewidth=0.8)
        ax.legend()
        ax.set_ylabel("Entries")
        ax.set_title(key)
        nonzero = rr > 0
        ratio.plot(centers[nonzero], (cc - rr)[nonzero] / rr[nonzero], ".")
        ratio.fill_between(centers[nonzero], lo[1:-1][nonzero] / rr[nonzero], hi[1:-1][nonzero] / rr[nonzero], alpha=0.25)
        ratio.axhline(0, color="black", lw=0.5)
        ratio.set_ylabel("(C−R)/R")
        ratio.set_xlabel(key)
        if "pt" in key or key in ["particle_energy", "jet_energy"]:
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(args.outdir / (key + ".png"))
        plt.close(fig)
        plots.append(key + ".png")
    np.savez_compressed(args.outdir / "histograms.npz", **histogram_arrays)
    presence = np.asarray(presence_counts, dtype=np.int64)
    np.savez_compressed(
        args.outdir / "particle-presence-counts.npz",
        cases=np.asarray([case["file"] for case in cases]),
        reference=presence[:, 0],
        candidate=presence[:, 1],
        gained=presence[:, 2],
        lost=presence[:, 3],
    )
    print("Writing paired particle and jet collections", flush=True)
    ak.to_parquet(columnar_events(events), args.outdir / "paired-events.parquet")
    matched = {}
    for key in ["pt_response", "energy_response", "delta_mass", "delta_r"]:
        values = np.concatenate([r[key] for r in response])
        matched[key] = {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "quantiles": np.quantile(values, [0, 0.01, 0.16, 0.5, 0.84, 0.99, 1]).tolist() if len(values) else [],
        }
    for key in ["unmatched_reference", "unmatched_candidate", "reference_count", "candidate_count"]:
        matched[key] = sum(r[key] for r in response)
    hard_matched = {}
    for key in ["pt_response", "energy_response", "delta_mass", "delta_r"]:
        values = np.concatenate([r[key] for r in hard_response])
        hard_matched[key] = {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "quantiles": np.quantile(values, [0, 0.01, 0.16, 0.5, 0.84, 0.99, 1]).tolist() if len(values) else [],
        }
    for key in ["unmatched_reference", "unmatched_candidate", "reference_count", "candidate_count"]:
        hard_matched[key] = sum(r[key] for r in hard_response)
    shifts = {}
    for key, values in event_shifts.items():
        values = np.array(values)
        shifts[key] = {
            "mean_candidate_minus_reference": float(values.mean()),
            "sum": float(values.sum()),
            "mean_ci95": np.quantile(weights @ values / len(cases), [0.025, 0.975]).tolist(),
            "max_abs_event_difference": float(abs(values).max()),
        }
    outliers = sorted(
        [
            {
                "case": case["file"],
                "dataset_index": case["index"],
                "max_abs_jet_pt_response_minus_one": float(np.max(abs(r["pt_response"] - 1))) if len(r["pt_response"]) else 0.0,
            }
            for case, r in zip(cases, response, strict=True)
        ],
        key=lambda row: row["max_abs_jet_pt_response_minus_one"],
        reverse=True,
    )[:20]
    summary = {
        "model": args.model,
        "events": len(cases),
        "full_sample": len(cases) >= 1000,
        "candidate": str(args.candidate),
        "reference": str(reference),
        "reference_label": args.reference_label,
        "numerical": numeric,
        "numerical_passed": all(r["violations"] == 0 for r in numeric.values()),
        "particle_pid_migration": migrations.tolist(),
        "jet_matching": matched,
        "jet30_matching": hard_matched,
        "paired_event_shifts": shifts,
        "largest_jet_response_outliers": outliers,
        "jet_definition": {"algorithm": "antikt", "R": 0.4, "pt_min": 3.0, "match_dr": 0.1, "nopu_cut": 0.8},
        "bootstrap_replicates": args.bootstrap,
        "physics_acceptance": "Report-only: no formal physics pass/fail requested",
    }
    write_json(args.outdir / "summary.json", summary)
    body = '<!doctype html><meta charset="utf-8"><title>CMS physics closure</title><h1>CMS physics closure</h1>'
    body += (
        "<p>"
        + html.escape(
            str(len(cases))
            + " paired test events. Numerical passed: "
            + str(summary["numerical_passed"])
            + ". Physics is reported without a formal pass/fail, as requested."
        )
        + "</p>"
    )
    body += "<p>Ratios omit empty reference bins. Intervals are paired event-bootstrap 95% intervals. Underflow and overflow counts are retained in histograms.npz.</p>"
    body += '<a href="summary.json">Metrics</a> · <a href="paired-events.parquet">Paired particles and jets</a>'
    body += "".join('<div><img style="max-width:750px" src="' + name + '"></div>' for name in plots)
    (args.outdir / "report.html").write_text(body)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
