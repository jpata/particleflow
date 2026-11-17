"""Render throughput results from benchmark-cms-throughput.py without GPU access."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    rows, hashes = [], {}
    for path in sorted(root.glob("*/*-s*-b*.json")):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        assert len(row["samples"]) == row["timed_batches"]
        assert np.isclose(
            row["events_per_second"],
            1000 * row["timed_events"] / sum(s["ms"] for s in row["samples"]),
            rtol=1e-12,
            atol=1e-12,
        )
        if row["engine"] == "onnx":
            assert row["placement"]["full_provider_placement"]
        assert row["batch_one_comparison"]["all_baselines_available"]
        rows.append(row)
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = [(r["model"], r["provider"], r["engine"], r["slots"], r["batch"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate measurements"
    for slots in [4000, 5000]:
        subsets = [r for r in rows if r["slots"] == slots]
        assert all(r["events"] == subsets[0]["events"] for r in subsets), "Different input events"
    fields = [
        "model",
        "provider",
        "engine",
        "slots",
        "batch",
        "events_per_second",
        "mean_batch_ms",
        "median_batch_ms",
        "p95_batch_ms",
        "amortized_ms_per_event",
        "timed_batches",
        "timed_events",
        "session_and_first_run_seconds",
        "warmup_seconds",
    ]
    with (root / "throughput.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for metric, label, filename in [
        ("events_per_second", "Throughput (events/s)", "throughput"),
        ("median_batch_ms", "Median batch latency (ms)", "batch-latency"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
        for mi, model in enumerate(["attention", "heptv2"]):
            for si, slots in enumerate([4000, 5000]):
                ax = axes[mi, si]
                for provider, color, device in [
                    ("cuda", "#2171b5", "L40S"),
                    ("migraphx", "#d35400", "W7900"),
                ]:
                    for engine, style in [("onnx", "-"), ("native", "--")]:
                        group = sorted(
                            [r for r in rows if (r["model"], r["slots"], r["provider"], r["engine"]) == (model, slots, provider, engine)],
                            key=lambda r: r["batch"],
                        )
                        if not group:
                            continue
                        ax.plot(
                            [r["batch"] for r in group],
                            [r[metric] for r in group],
                            marker="o",
                            linestyle=style,
                            color=color,
                            label=f"{device} {'ONNX' if engine == 'onnx' else 'PyTorch'}",
                        )
                        if metric == "events_per_second":
                            ax.fill_between(
                                [r["batch"] for r in group],
                                [min(r["round_events_per_second"]) for r in group],
                                [max(r["round_events_per_second"]) for r in group],
                                color=color,
                                alpha=0.1,
                            )
                title = "Attention" if model == "attention" else "HEPTv2"
                ax.set_title(f"{title} · {slots:,} padded slots/event")
                ax.set_ylim(bottom=0)
                ax.set_xscale("log", base=2)
                ax.set_xticks([1, 2, 4, 8, 16])
                ax.set_xticklabels([1, 2, 4, 8, 16])
                ax.set_xlabel("Batch size")
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle("CMS BF16 · 64 real events per length · 8 rounds · synchronous CPU-to-CPU")
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        for suffix in ["png", "pdf"]:
            fig.savefig(root / f"{filename}.{suffix}", dpi=160)
        plt.close(fig)
    lines = [
        "# CMS batch throughput, 2026-11-03",
        "",
        "Attention and HEPTv2, native PyTorch BF16 and fusion-enabled ONNX BF16, NVIDIA L40S and AMD W7900.",
        "",
        (
            "Each length uses the same 64 distinct real CMS ttbar events across all batch sizes and runtimes. "
            "Each point measures eight passes (512 event executions), after six warmups, in a fresh process. "
            "Inputs and outputs are CPU FP32; transfers and output conversion are included. "
            "Input preparation, output validation, compilation, profiling and warmup are excluded. "
            "Throughput is total events divided by summed inference time; it excludes time between calls. "
            "The bands show the range across eight rounds, not confidence intervals. "
            "Workloads run serially on each GPU. "
            "This is a fixed-length, bucketed workload rather than the full 5,000-event mixture."
        ),
        "Native Attention packs valid elements into its jagged path; ONNX executes the padded static shape. Native HEPTv2 uses compiled FlexAttention; ONNX HEPTv2 retains its separate FP32 log-sum-exp branch.",
        "",
        "![Throughput](throughput.png)",
        "",
        "![Batch latency](batch-latency.png)",
        "",
        "| Model | GPU | Runtime | Slots | B1 | B2 | B4 | B8 | B16 | Best batch | Gain over B1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    findings_path = root / "findings.json"
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())
        lines[2:2] = [text for finding in findings for text in [finding, ""]]
    for model in ["attention", "heptv2"]:
        for provider, gpu in [("cuda", "L40S"), ("migraphx", "W7900")]:
            for engine in ["onnx", "native"]:
                for slots in [4000, 5000]:
                    group = {r["batch"]: r for r in rows if (r["model"], r["provider"], r["engine"], r["slots"]) == (model, provider, engine, slots)}
                    if not group:
                        continue
                    best = max(group, key=lambda b: group[b]["events_per_second"])
                    rates = [(f'{group[b]["events_per_second"]:.1f}' if b in group else "missing") for b in [1, 2, 4, 8, 16]]
                    gain = group[best]["events_per_second"] / group[1]["events_per_second"] if 1 in group else float("nan")
                    lines.append(f"| {model} | {gpu} | {engine} | {slots} | " + " | ".join(rates) + f" | {best} | {gain:.2f}× |")
    failures = []
    for path in root.glob("*/summary.json"):
        failures.extend(r for r in json.loads(path.read_text()) if r.get("status") != "ok")
    lines += [
        "",
        f"Completed points: {len(rows)}/80. Failed points: {len(failures)}. Table entries are events/s.",
        "",
        (
            "Batched outputs were checked for finiteness; ONNX padding outputs were checked for zero. "
            "For the first batch at each shape, per-head maximum absolute differences against the earlier batch-one outputs are retained in each result JSON. "
            "These are numerical diagnostics, not a renewed physics or equivalence qualification. "
            "ONNX requests Flash on CUDA and enables MLIR attention on MIGraphX; per-shape provider placement is verified. "
            "AMD fusion dispatch has not been independently requalified for every new batch shape."
        ),
        "",
        "Raw timings, source hashes, model/checkpoint hashes, selected input identities, warmup times and logs accompany this report.",
    ]
    proof_path = root / "cuda-fusion-verification.json"
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        lines += [
            "",
            f"CUDA BF16 Flash Attention dispatch was verified in separate untimed runs for all {proof['verified_shapes']} exact timed ONNX graphs; see cuda-fusion-verification.json.",
        ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    body = [
        '<!doctype html><meta charset="utf-8"><title>CMS batch throughput</title>',
        "<style>body{font:16px system-ui;margin:2rem;max-width:1300px}p{line-height:1.5}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:6px 10px;border:1px solid #ddd;text-align:right}</style>",
    ]
    in_table = False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = "th" if "Model" in line else "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(cell.strip())}</{tag}>" for cell in line.strip("|").split("|")) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if line.startswith("# "):
                body.append("<h1>" + html.escape(line[2:]) + "</h1>")
            elif line.startswith("!["):
                filename = line.split("(")[1].rstrip(")")
                body.append(f'<a href="{filename.replace(".png", ".pdf")}">PDF</a><img src="{filename}">')
            elif line:
                body.append("<p>" + html.escape(line) + "</p>")
    if in_table:
        body.append("</table>")
    body.append('<p><a href="throughput.csv">CSV</a> · <a href="validation.json">Validation</a></p>')
    (root / "report.html").write_text("\n".join(body) + "\n")
    (root / "validation.json").write_text(
        json.dumps(
            {
                "successful_points": len(rows),
                "failed_points": failures,
                "expected_points": 80,
                "source_hashes": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"successful_points": len(rows), "failed_points": len(failures)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
