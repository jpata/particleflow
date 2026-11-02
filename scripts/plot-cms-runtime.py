"""Plot retained event latency and size without rerunning inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


ENGINES = [
    ("native", "PyTorch BF16 native fused", "#555555"),
    ("on", "ONNX BF16 fusion on", "#2171b5"),
    ("off", "ONNX BF16 fusion off", "#d35400"),
]


def main(root, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def read(path):
        provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    cases = read(root / "input-manifest.json")["cases"]
    names = [c["file"] for c in cases]
    assert len(set(names)) == len(names)
    sizes = np.array([c["length"] for c in cases])
    slots = np.array([c["slots"] for c in cases])
    buckets = sorted(set(slots.tolist()))
    aggregate = read(root / "qualification-summary.json")
    size_edges = np.arange((sizes.min() // 250) * 250, (sizes.max() // 250 + 2) * 250, 250)
    summaries, values, csv_rows = [], {}, []
    for model in ["attention", "heptv2"]:
        for device in ["cuda", "amd"]:
            for mode, engine, _ in ENGINES:
                if mode == "native":
                    records = read(root / ("reference" if device == "cuda" else "reference-amd") / model / "manifest.json")["cases"]
                else:
                    provider, profile = ("cuda", "attention24-bf16") if device == "cuda" else ("migraphx", "decomposed-bf16")
                    folder = root / "results" / model / provider / profile / mode
                    records = []
                    for bucket in buckets:
                        data = read(folder / f"bucket_{bucket}.json")
                        assert data["placement"]["full_provider_placement"]
                        records.extend(data["cases"])
                by_case = {c["case"]: c for c in records}
                assert len(records) == len(by_case) == len(cases) and set(by_case) == set(names)
                for case in cases:
                    record = by_case[case["file"]]
                    assert record["input_sha256"] == case["sha256"] and record["slots"] == case["slots"]
                    csv_rows.append(
                        dict(
                            model=model,
                            device=device,
                            engine=engine,
                            case=case["file"],
                            dataset_index=case["index"],
                            valid_elements=case["length"],
                            padded_slots=case["slots"],
                            inference_ms=record["inference_ms"],
                        )
                    )
                latency = np.array([by_case[name]["inference_ms"] for name in names])
                assert np.isfinite(latency).all() and (latency > 0).all()
                expected = next(row for row in aggregate["runtimes"] if (row["model"], row["device"], row["engine"]) == (model, device, engine))
                stats = dict(
                    mean_ms=float(latency.mean()),
                    p50_ms=float(np.quantile(latency, 0.5)),
                    p95_ms=float(np.quantile(latency, 0.95)),
                    p99_ms=float(np.quantile(latency, 0.99)),
                    min_ms=float(latency.min()),
                    max_ms=float(latency.max()),
                )
                for key in ["mean_ms", "p50_ms", "p95_ms"]:
                    assert np.isclose(stats[key], expected[key], rtol=1e-12, atol=1e-12)
                bins = []
                for lo, hi in zip(size_edges[:-1], size_edges[1:]):
                    sample = latency[(sizes >= lo) & (sizes < hi)]
                    if len(sample):
                        bins.append(
                            dict(
                                low=int(lo),
                                high=int(hi),
                                events=len(sample),
                                p10_ms=float(np.quantile(sample, 0.1)),
                                p50_ms=float(np.median(sample)),
                                p90_ms=float(np.quantile(sample, 0.9)),
                            )
                        )
                assert sum(b["events"] for b in bins) == len(cases)
                summaries.append(dict(model=model, device=device, engine=engine, events=len(latency), **stats, size_bins=bins))
                values[model, device, mode] = latency

    all_latency = np.concatenate(list(values.values()))
    latency_edges = np.geomspace(all_latency.min() * 0.98, all_latency.max() * 1.02, 81)
    for kind in ["latency-distributions", "latency-vs-event-size"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for mi, model in enumerate(["attention", "heptv2"]):
            for di, device in enumerate(["cuda", "amd"]):
                ax = axes[mi, di]
                for mode, engine, color in ENGINES:
                    latency = values[model, device, mode]
                    if kind == "latency-distributions":
                        counts, _ = np.histogram(latency, latency_edges)
                        assert counts.sum() == len(cases)
                        ax.stairs(100 * counts / len(cases), latency_edges, color=color, label=engine, linewidth=1.5)
                    else:
                        ax.scatter(sizes, latency, s=4, alpha=0.12, color=color, rasterized=True, edgecolors="none")
                        row = next(r for r in summaries if (r["model"], r["device"], r["engine"]) == (model, device, engine))
                        bins = [b for b in row["size_bins"] if b["events"] >= 20]
                        x = [(b["low"] + b["high"]) / 2 for b in bins]
                        ax.plot(x, [b["p50_ms"] for b in bins], color=color, label=engine, linewidth=1.6)
                        ax.fill_between(x, [b["p10_ms"] for b in bins], [b["p90_ms"] for b in bins], color=color, alpha=0.12)
                ax.set_title(("Attention" if model == "attention" else "HEPTv2") + " · " + ("NVIDIA L40S" if device == "cuda" else "AMD W7900"))
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
                if kind == "latency-distributions":
                    ax.set_xscale("log")
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.set_xlabel("Event latency (ms; logarithmic scale)")
                    ax.set_ylabel("Events in bin (%)")
                else:
                    for bucket in buckets:
                        if sizes.min() < bucket < sizes.max():
                            ax.axvline(bucket, color="black", linewidth=0.6, linestyle=":", alpha=0.35)
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    ax.set_xlabel("Valid input elements in event")
                    ax.set_ylabel("Event latency (ms; logarithmic scale)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
        fig.suptitle(
            ("Event latency distributions" if kind == "latency-distributions" else "Latency versus event size")
            + f" · {len(cases):,} paired CMS test events"
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        for suffix in ["png", "pdf"]:
            fig.savefig(outdir / f"{kind}.{suffix}", dpi=170)
        plt.close(fig)

    with (outdir / "per-event-latency.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = dict(
        events=len(cases),
        scenarios=len(summaries),
        timed_records=len(csv_rows),
        all_aggregate_timings_match=True,
        all_histograms_include_all_events=True,
        event_size="Number of valid input elements; padded ONNX execution slots retained separately in CSV",
        latency_bins_ms=latency_edges.tolist(),
        rows=summaries,
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    body = '<!doctype html><meta charset="utf-8"><title>CMS runtime latency</title><style>body{font:16px system-ui;margin:2rem;max-width:1300px}p{max-width:1100px;line-height:1.5}img{width:100%}</style><h1>Runtime latency</h1>'
    body += f"<p>The same {len(cases):,} events per scenario, measured once after warmup. Synchronous batch-one CPU inputs to CPU FP32 outputs, including transfers and conversion. File I/O, compilation, profiling and physics are excluded. These plots reuse retained timings; inference was not rerun.</p>"
    body += '<h2>Latency distributions</h2><p>Each histogram sums to 100% of events, with shared logarithmic bins and logarithmic axes. All recorded latency tails are included.</p><a href="latency-distributions.pdf">PDF</a><img src="latency-distributions.png" alt="Latency distributions">'
    body += '<h2>Latency versus event size</h2><p>Dots show all event timings. Lines show median latency in 250-element bins; bands show the 10th–90th percentile event spread, not confidence intervals. Lines and bands require at least 20 events per bin; sparse bins still show every event as dots. Latency uses a logarithmic axis.</p><p>Event size counts valid input elements. Dotted vertical lines mark ONNX static bucket boundaries; ONNX executes padded shapes, whereas native PyTorch uses its qualified jagged/packed path. This plot describes the observed size dependence and does not isolate event-size effects from event composition or run order.</p><a href="latency-vs-event-size.pdf">PDF</a><img src="latency-vs-event-size.png" alt="Latency versus valid event size">'
    body += '<p><a href="per-event-latency.csv">Per-event CSV</a> · <a href="summary.json">Statistics</a> · <a href="provenance.json">Source hashes</a> · <a href="../report.html">Qualification report</a></p>'
    (outdir / "report.html").write_text(body)
    print(json.dumps(dict(events=len(cases), scenarios=len(summaries), timed_records=len(csv_rows), aggregate_timings_match=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.outdir)
