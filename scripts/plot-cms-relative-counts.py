"""Relative CMS particle/jet counts and per-event distributions from retained evidence."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from mlpf.standalone_eval.cms_evidence import input_manifest_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [
    ("amd-on", "AMD ONNX fused", "#d35400", "-"),
    ("amd-off", "AMD ONNX unfused", "#d35400", "--"),
    ("cuda-on", "NVIDIA ONNX fused", "#2171b5", "-"),
    ("cuda-off", "NVIDIA ONNX unfused", "#2171b5", "--"),
    ("amd-pytorch-vs-nvidia", "AMD native PyTorch", "#666666", ":"),
]
OBSERVABLES = [
    ("particles", "particle_pt", "particle_count", "Particles"),
    ("jets3", "jet_pt", "jet_count", "Jets, pT ≥ 3 GeV"),
    ("jets30", "jet30_pt", "jet30_count", "Jets, pT ≥ 30 GeV"),
]


def counts(histograms, key, side, expected_events):
    # All bins, including underflow and overflow, count every selected object.
    values = histograms[key + "_event_" + side]
    assert values.ndim == 2 and values.shape[0] == expected_events
    result = values.sum(axis=1)
    assert np.all(result >= 0) and np.all(result == np.floor(result))
    return result.astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(input_manifest_path(a.root).read_text())["cases"]
    event_count = len(manifest)
    diagnostic_path = a.root / "particle-count-debug/summary.json"
    diagnostic = json.loads(diagnostic_path.read_text())["rows"] if diagnostic_path.exists() else {}
    rows, arrays, provenance = [], {}, {}
    for model in ["attention", "heptv2"]:
        for scenario, label, color, style in SCENARIOS:
            folder = a.root / "analysis" / f"{model}-{scenario}"
            summary = json.loads((folder / "summary.json").read_text())
            path = folder / "histograms.npz"
            provenance[str(path.relative_to(a.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as histograms:
                for observable, hist, shift, title in OBSERVABLES:
                    ref, cand = [counts(histograms, hist, side, event_count) for side in ["reference", "candidate"]]
                    delta = cand - ref
                    assert int(delta.sum()) == summary["paired_event_shifts"][shift]["sum"]
                    if observable.startswith("jets"):
                        matching = summary["jet30_matching" if observable == "jets30" else "jet_matching"]
                        assert ref.sum() == matching["reference_count"] and cand.sum() == matching["candidate_count"]
                    nonzero = ref > 0
                    relative = np.full(len(ref), np.nan)
                    relative[nonzero] = 100.0 * delta[nonzero] / ref[nonzero]
                    key = f"{model}__{scenario}__{observable}"
                    arrays[key + "__reference"] = ref
                    arrays[key + "__candidate"] = cand
                    arrays[key + "__delta_percent"] = relative
                    rows.append(
                        dict(
                            model=model,
                            scenario=scenario,
                            observable=observable,
                            events=len(ref),
                            reference_total=int(ref.sum()),
                            candidate_total=int(cand.sum()),
                            delta_total=int(delta.sum()),
                            delta_percent=float(100.0 * delta.sum() / ref.sum()),
                            mean_event_delta_percent=float(relative[nonzero].mean()),
                            median_event_delta_percent=float(np.median(relative[nonzero])),
                            p95_abs_event_delta_percent=float(np.quantile(abs(relative[nonzero]), 0.95)),
                            reference_zero_events=int((~nonzero).sum()),
                            reference_zero_candidate_nonzero_events=int(((~nonzero) & (cand > 0)).sum()),
                        )
                    )
            provider = "pytorch-amd" if scenario == "amd-pytorch-vs-nvidia" else ("migraphx" if scenario.startswith("amd") else "cuda")
            mode = "native" if provider == "pytorch-amd" else scenario.split("-")[1]
            presence_path = folder / "particle-presence-counts.npz"
            if presence_path.exists():
                with np.load(presence_path) as presence:
                    assert presence["cases"].tolist() == [c["file"] for c in manifest]
                    record = dict(
                        reference_particles=int(presence["reference"].sum()),
                        candidate_particles=int(presence["candidate"].sum()),
                        gained=int(presence["gained"].sum()),
                        lost=int(presence["lost"].sum()),
                        event_rows=[
                            dict(case=c["file"], gained=int(g), lost=int(l), net=int(g - l))
                            for c, g, l in zip(manifest, presence["gained"], presence["lost"])
                        ],
                    )
                    np.testing.assert_array_equal(presence["reference"], arrays[f"{model}__{scenario}__particles__reference"])
                    np.testing.assert_array_equal(presence["candidate"], arrays[f"{model}__{scenario}__particles__candidate"])
            else:
                record = diagnostic[f"{model}/{provider}/{mode}"]
            by_case = {r["case"]: r for r in record["event_rows"]}
            ref = arrays[f"{model}__{scenario}__particles__reference"]
            cand = arrays[f"{model}__{scenario}__particles__candidate"]
            assert ref.sum() == record["reference_particles"] and cand.sum() == record["candidate_particles"]
            for metric in ["gained", "lost", "net"]:
                values = np.array([by_case[c["file"]][metric] for c in manifest])
                arrays[f"{model}__{scenario}__particles__{metric}_percent"] = 100.0 * values / ref
                if metric == "net":
                    np.testing.assert_array_equal(values, cand - ref)
            particle_row = next(r for r in rows if r["model"] == model and r["scenario"] == scenario and r["observable"] == "particles")
            particle_row.update(
                gained_percent=100.0 * record["gained"] / ref.sum(),
                lost_percent=100.0 * record["lost"] / ref.sum(),
                gross_presence_changes_percent=100.0 * (record["gained"] + record["lost"]) / ref.sum(),
            )
    np.savez_compressed(a.outdir / "per-event-counts.npz", **arrays)
    description = {
        "overall_percent": "100 * sum(candidate-reference) / sum(reference)",
        "event_percent": "100 * (candidate_i-reference_i) / reference_i; undefined for reference_i=0",
        "histogram_normalization": "Each scenario histogram sums to 100% of included events; all recorded tails retained.",
        "reference": "NVIDIA PyTorch BF16, separately for each model",
        "gain_loss": "Particle gains/losses refer to elementwise presence changes; both normalized by reference particle count.",
        "rows": rows,
    }
    (a.outdir / "summary.json").write_text(json.dumps(description, indent=2) + "\n")
    columns = [
        "model",
        "scenario",
        "observable",
        "events",
        "reference_total",
        "candidate_total",
        "delta_total",
        "delta_percent",
        "mean_event_delta_percent",
        "median_event_delta_percent",
        "p95_abs_event_delta_percent",
        "reference_zero_events",
        "reference_zero_candidate_nonzero_events",
        "gained_percent",
        "lost_percent",
        "gross_presence_changes_percent",
    ]
    with (a.outdir / "relative-counts.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for kind in ["relative-deltas", "multiplicities", "particle-gains-losses"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for mi, model in enumerate(["attention", "heptv2"]):
            for oi in range(3):
                ax = axes[mi, oi]
                observable, _, _, title = OBSERVABLES[oi]
                if kind == "particle-gains-losses":
                    metric = ["net", "gained", "lost"][oi]
                    title = ["Net particle change", "Particles gained", "Particles lost"][oi]
                    values = [arrays[f"{model}__{s}__particles__{metric}_percent"] for s, _, _, _ in SCENARIOS]
                elif kind == "relative-deltas":
                    values = [arrays[f"{model}__{s}__{observable}__delta_percent"] for s, _, _, _ in SCENARIOS]
                else:
                    values = [arrays[f"{model}__{s}__{observable}__candidate"] for s, _, _, _ in SCENARIOS]
                    values.append(arrays[f"{model}__amd-on__{observable}__reference"])
                finite = np.concatenate([v[np.isfinite(v)] for v in values])
                lo, hi = float(finite.min()), float(finite.max())
                if kind == "multiplicities" and observable != "particles":
                    bins = np.arange(lo - 0.5, hi + 1.5, 1)
                else:
                    span = max(hi - lo, 1e-6)
                    bins = np.linspace(lo - 0.01 * span, hi + 0.01 * span, 81)
                for (scenario, label, color, style), v in zip(SCENARIOS, values):
                    v = v[np.isfinite(v)]
                    ax.hist(
                        v,
                        bins=bins,
                        weights=np.full(len(v), 100.0 / len(v)),
                        histtype="step",
                        label=label,
                        color=color,
                        linestyle=style,
                        linewidth=1.4,
                    )
                if kind == "multiplicities":
                    v = values[-1]
                    ax.hist(
                        v,
                        bins=bins,
                        weights=np.full(len(v), 100.0 / len(v)),
                        histtype="step",
                        label="NVIDIA native reference",
                        color="black",
                        linewidth=1.2,
                    )
                else:
                    ax.axvline(0, color="black", linewidth=0.6, alpha=0.6)
                ax.set_title(model.capitalize() + " · " + title)
                ax.set_xlabel("Objects per event" if kind == "multiplicities" else "Change / reference count per event (%)")
                ax.set_ylabel("Events in bin (%)")
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
        fig.suptitle(
            {
                "relative-deltas": "Per-event relative count changes",
                "multiplicities": "Particle and jet multiplicity distributions",
                "particle-gains-losses": "Net particle shifts versus gains and losses",
            }[kind]
            + f" · {event_count} CMS events"
        )
        fig.tight_layout(rect=[0, 0.09, 1, 0.95])
        for suffix in ["png", "pdf"]:
            fig.savefig(a.outdir / f"{kind}.{suffix}", dpi=170)
        plt.close(fig)
    table = "<table><tr><th>Model</th><th>Scenario</th><th>Particles Δ%</th><th>Jets ≥3 GeV Δ%</th><th>Jets ≥30 GeV Δ%</th></tr>"
    for model in ["attention", "heptv2"]:
        for scenario, label, _, _ in SCENARIOS:
            subset = [
                next(r for r in rows if r["model"] == model and r["scenario"] == scenario and r["observable"] == o) for o, _, _, _ in OBSERVABLES
            ]
            table += (
                "<tr><td>"
                + model
                + "</td><td>"
                + html.escape(label)
                + "</td>"
                + "".join(f'<td>{r["delta_percent"]:+.5f}%</td>' for r in subset)
                + "</tr>"
            )
    table += "</table>"
    body = '<!doctype html><meta charset="utf-8"><title>Relative CMS object counts</title><style>body{font:16px system-ui;margin:2rem;max-width:1550px}p{max-width:1100px;line-height:1.5}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse}img{width:100%}</style><h1>Relative particle and jet counts</h1>'
    body += (
        f"<p>All changes are relative to the native NVIDIA PyTorch BF16 reference for the same model, across the same {event_count} events. Table percentages are 100 × (total candidate − total reference) / total reference. They are not averages of event percentages. Positive means more objects; negative means fewer.</p>"
        + table
    )
    body += "<p>Distribution plots use per-event values and show the full recorded range. The logarithmic vertical axis exposes rare tails; histogram weights sum to 100% for each scenario. A zero-reference event has no defined percentage and is excluded from that relative histogram, with its count recorded in the CSV/JSON. Such events remain in multiplicity plots and overall totals. Gain/loss percentages use reference particle counts and are not truth-based efficiency or fake rates.</p>"
    for name, title in [
        ("relative-deltas", "Relative count-change distributions"),
        ("multiplicities", "Object multiplicity distributions"),
        ("particle-gains-losses", "Particle gains and losses"),
    ]:
        body += f'<h2>{title}</h2><a href="{name}.pdf">PDF</a><img src="{name}.png" alt="{title}">'
    body += '<p>These count plots do not measure whether jets correspond to the same objects; angular matching is a separate comparison. No inference or jet clustering was rerun, and no physics acceptance is assigned.</p><p><a href="relative-counts.csv">CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="per-event-counts.npz">Per-event arrays</a> · <a href="validation.json">Validation</a></p>'
    (a.outdir / "report.html").write_text(body)
    (a.outdir / "provenance.json").write_text(
        json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "histogram_sha256": provenance}, indent=2) + "\n"
    )
    (a.outdir / "validation.json").write_text(
        json.dumps(
            {
                "all_30_overall_count_deltas_match_original_reports": True,
                "all_particle_counts_match_independent_decoding": True,
                "all_per_event_particle_deltas_match_independent_decoding": True,
                "underflow_overflow_included": True,
                "events_per_scenario": event_count,
            },
            indent=2,
        )
        + "\n"
    )
    (a.outdir / "checksums.json").write_text(
        json.dumps(
            {
                str(f.relative_to(a.outdir)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(a.outdir.iterdir())
                if f.is_file() and f.name != "checksums.json"
            },
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(row["model"], row["scenario"], row["observable"], f"{row['delta_percent']:+.5f}%", "zero-reference", row["reference_zero_events"])


if __name__ == "__main__":
    main()
