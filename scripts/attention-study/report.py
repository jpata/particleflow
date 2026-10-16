"""Build figures and an evidence index from recorded attention-study artifacts."""

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import graphviz
import matplotlib.pyplot as plt
import onnx
from matplotlib.ticker import NullFormatter
from onnx import helper

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--docs", type=Path, required=True)
a = p.parse_args()
fig = a.docs / "figures"
fig.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 140, "svg.fonttype": "none"})
configs = {"cpu129": "CPU · ORT 1.29", "cuda129": "L40S · ORT 1.29", "rocm-default": "W7900 · default", "rocm-attention": "W7900 · MLIR attention"}
configs.update({"cpu123": "CPU · ORT 1.23.2", "cuda123": "L40S · ORT 1.23.2"})
styles = {
    "pytorch_math": ("PyTorch MATH", "#666666", "o"),
    "pytorch_flash_attention": ("PyTorch Flash", "#009E73", "*"),
    "sdpa_legacy": ("Legacy SDPA export", "#CC79A7", "^"),
    "explicit_safe": ("Explicit safe math", "#E69F00", "s"),
    "attention24_dense": ("Attention-24 dense", "#0072B2", "D"),
    "attention24_lengths": ("Attention-24 lengths", "#56B4E9", "v"),
    "sdpa_dynamo": ("Modern Attention-23", "#AA4499", "x"),
    "sdpa_dynamo_expanded": ("Attention-23 expanded", "#332288", "+"),
}
data = {
    k: json.loads((a.root / k / ("validated-results.json" if (a.root / k / "validated-results.json").exists() else "results.json")).read_text())
    for k in configs
    if (a.root / k / "results.json").exists()
}


def save(name):
    for ax in plt.gcf().axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ext in ["svg", "pdf", "png"]:
        plt.savefig(fig / f"{name}.{ext}", bbox_inches="tight")
    plt.close()


def ok(r):
    return r["status"] == "ok" and r.get("correct", True)


def static(rows):
    return [r for r in rows if not r["dynamic"]]


for dtype in ["fp32", "fp16"]:
    for mode in ["unmasked", "padding"]:
        f, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, (cfg, label) in zip(axs.flat, configs.items()):
            rs = [r for r in static(data.get(cfg, [])) if r["dtype"] == dtype and r["mode"] == mode and ok(r)]
            for kind, (name, color, marker) in styles.items():
                vals = sorted([r for r in rs if r["kind"] == kind], key=lambda r: r["seq"])
                if not vals:
                    continue
                xs = [r["seq"] for r in vals]
                ys = [r["median_ms"] for r in vals]
                lo = [min(r.get("trial_medians_ms", [r["median_ms"]])) for r in vals]
                hi = [max(r.get("trial_medians_ms", [r["median_ms"]])) for r in vals]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[np.maximum(0, np.array(ys) - lo), np.maximum(0, np.array(hi) - ys)],
                    color=color,
                    marker=marker,
                    label=name,
                    capsize=3,
                    linewidth=1.4,
                )
            ax.set(title=label, xlabel="Sequence length", ylabel="Median latency [ms]", xscale="log", yscale="log")
            ax.set_xticks([128, 512, 1280, 2560])
            ax.set_xticklabels(["128", "512", "1280", "2560"])
            ax.legend(fontsize=7)
        f.suptitle(
            f"{dtype.upper()}, {mode}: B=4, H=16, D=16 · ORT includes host I/O\nBars span medians of three timing trials; failed cases omitted here, retained in support matrix",
            fontsize=11,
        )
        f.tight_layout(rect=(0, 0, 1, 0.94))
        save(f"latency-{dtype}-{mode}")
# Throughput uses actual valid-token counts, not padded tokens.
f, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, (cfg, label) in zip(axs.flat, configs.items()):
    for kind, (name, color, marker) in styles.items():
        rr = sorted(
            [r for r in static(data.get(cfg, [])) if ok(r) and r["dtype"] == "fp16" and r["mode"] == "padding" and r["kind"] == kind],
            key=lambda r: r["seq"],
        )
        if not rr:
            continue

        def valid(r):
            return r.get("valid_tokens", sum(max(1, r["seq"] - int(r["seq"] * 0.5 * (i + 1) / r["batch"])) for i in range(r["batch"])))

        ax.plot([r["seq"] for r in rr], [1000 * valid(r) / r["median_ms"] for r in rr], color=color, marker=marker, label=name)
    ax.set(title=label, xlabel="Sequence length", ylabel="Valid tokens / second", yscale="log")
    ax.legend(fontsize=7)
f.suptitle("FP16 padded throughput · native PyTorch / ORT host-I/O boundaries")
f.tight_layout()
save("throughput")
# Batch effects use balanced padding, a separate workload from the sequence sweep.
batch_data = {
    k: json.loads((a.root / k / ("validated-results.json" if (a.root / k / "validated-results.json").exists() else "results.json")).read_text())
    for k in ["cpu-batch", "cuda-batch", "rocm-batch"]
    if (a.root / k / "results.json").exists()
}
if batch_data:
    f, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, seq in zip(axs, [512, 1280]):
        for cfg, rows in batch_data.items():
            for kind in ["pytorch_math", "explicit_safe", "attention24_lengths"]:
                rr = sorted([r for r in rows if ok(r) and r["seq"] == seq and r["kind"] == kind], key=lambda r: r["batch"])
                if rr:
                    ax.plot([r["batch"] for r in rr], [r["median_ms"] for r in rr], marker=styles[kind][2], label=cfg + " " + styles[kind][0])
        ax.set(title=f"S={seq}, FP16, balanced right padding", xlabel="Batch size", ylabel="Host/native latency [ms]", yscale="log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.legend(fontsize=6)
    f.tight_layout()
    save("batch-scaling")

# Support is measured, not inferred from provider lists.
kinds = list(styles)
cols = list(data)
f, ax = plt.subplots(figsize=(12, 9))
matrix = []
labels = []
for dtype in ["fp32", "fp16"]:
    for mode in ["unmasked", "padding"]:
        for kind in kinds:
            labels.append(f"{dtype} {mode} · {styles[kind][0]}")
            row = []
            for cfg in cols:
                rr = [r for r in static(data[cfg]) if r["dtype"] == dtype and r["mode"] == mode and r["kind"] == kind and r["seq"] == 512]
                row.append(1 if rr and ok(rr[0]) else 0.5 if rr and rr[0]["status"] == "incorrect" else 0 if rr else np.nan)
            matrix.append(row)
from matplotlib.colors import ListedColormap

ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=ListedColormap(["#E7A59B", "#F1D36C", "#80C5A0"]), vmin=0, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([configs[c] for c in cols], rotation=15, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_title(
    "Measured support at B=4, S=512 · green=correct, gold=tolerance fail, red=runtime fail, blank=not run / inapplicable\nGPU rows require CPU fallback disabled"
)
ax.grid(False)
save("support-matrix")
# Host vs device-resident APIs, kept separate from native PyTorch timings.
f, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, cfg in zip(axs, ["cuda129", "rocm-attention"]):
    for kind, (label, col, m) in styles.items():
        rr = sorted(
            [
                r
                for r in static(data.get(cfg, []))
                if ok(r) and r["dtype"] == "fp16" and r["mode"] == "unmasked" and r["kind"] == kind and "device_io" in r
            ],
            key=lambda r: r["seq"],
        )
        if not rr:
            continue
        ax.plot([r["seq"] for r in rr], [r["median_ms"] for r in rr], color=col, marker=m, label=label + " host")
        ax.plot([r["seq"] for r in rr], [r["device_io"]["median_ms"] for r in rr], color=col, marker=m, linestyle="--", label=label + " bound")
    ax.set(title=configs[cfg], xlabel="Sequence length", ylabel="FP16 latency [ms]", yscale="log")
    ax.legend(fontsize=7)
f.suptitle("API boundary matters: host arrays versus preallocated GPU I/O")
f.tight_layout()
save("io-boundary")
# Precision differences against the same FP32 reference inputs.
f, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, cfg in zip(axs, ["cuda129", "rocm-attention"]):
    for kind, (label, col, m) in styles.items():
        rr = sorted(
            [r for r in static(data.get(cfg, [])) if ok(r) and r["dtype"] == "fp16" and r["mode"] == "unmasked" and r["kind"] == kind],
            key=lambda r: r["seq"],
        )
        if rr:
            ax.plot([r["seq"] for r in rr], [max(r["mae"], 1e-12) for r in rr], color=col, marker=m, label=label)
    ax.set(title=configs[cfg], xlabel="Sequence length", ylabel="Mean absolute error", yscale="log")
    ax.legend(fontsize=7)
f.suptitle("Accuracy relative to FP32 MATH on the same quantized inputs")
f.tight_layout()
save("accuracy")
# Dynamic compilation timing series.
probes = list(a.root.glob("diagnostics/*dynamic*/probe.json"))
if probes:
    f, ax = plt.subplots(figsize=(10, 4))
    for probe in probes:
        rr = [r for r in json.loads(probe.read_text()) if "run_ms" in r]
        ax.plot(range(len(rr)), [r["run_ms"] for r in rr], marker="o", label=probe.parent.name)
    ax.set(
        xlabel="Call index (two sessions; sizes repeat)",
        ylabel="Wall time [ms]",
        yscale="log",
        title="Dynamic input sizes: compilation and cache reuse",
    )
    ax.legend(fontsize=8)
    save("dynamic-compilation")
# Actual memory deltas, when probes have completed. No PyTorch allocator claim for ORT.
mem = []
for path in a.root.glob("memory/*.json"):
    r = json.loads(path.read_text())
    r["label"] = path.stem
    mem.append(r)
if mem:
    f, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, provider in zip(axs, ["cpu", "cuda", "migraphx"]):
        for kind, color in [
            ("sdpa_legacy", "#CC79A7"),
            ("explicit_safe", "#E69F00"),
            ("attention24_dense", "#0072B2"),
            ("attention24_lengths", "#56B4E9"),
        ]:
            rr = [r for r in mem if r["provider"] == provider and kind in r["model"]]
            rr.sort(key=lambda r: int(r["model"].split("_s")[-1].split(".")[0]))
            if rr:
                ax.plot(
                    [int(r["model"].split("_s")[-1].split(".")[0]) for r in rr],
                    [r["inference_peak_delta_bytes"] / 2**20 for r in rr],
                    marker="o",
                    color=color,
                    label={
                        "sdpa_legacy": "Legacy (unmasked)",
                        "explicit_safe": "Safe math (padded)",
                        "attention24_dense": "Attention-24 (unmasked)",
                        "attention24_lengths": "Attention-24 lengths (padded)",
                    }[kind],
                )
        ax.set(
            title={"cpu": "CPU process RSS", "cuda": "L40S driver VRAM", "migraphx": "W7900 driver VRAM · fusion requested"}[provider],
            xlabel="Sequence length",
            ylabel="Delta above baseline [MiB]",
        )
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.set_xticks([512, 2560])
    f.suptitle("FP16, B=4, H=16, D=16 · separate memory probes\nIncludes provider/context caches; sampled peak is a lower bound")
    f.tight_layout()
    save("memory")
# Full model dumps and shapes; these are actual serialized artifacts, not a schematic.
modeldir = a.docs / "models"
modeldir.mkdir(exist_ok=True)
selected = [
    "sdpa_legacy_fp16_unmasked_b4_s512",
    "sdpa_legacy_fp16_padding_b4_s512",
    "explicit_safe_fp16_padding_b4_s512",
    "attention24_lengths_fp16_padding_b4_s512",
    "sdpa_dynamo_fp16_unmasked_b4_s512",
    "sdpa_dynamo_expanded_fp16_padding_b4_s512",
]
for stem in selected:
    path = a.root / "bundle" / (stem + ".onnx")
    if not path.exists():
        continue
    shutil.copy2(path, modeldir / path.name)
    m = onnx.load(path)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except onnx.shape_inference.InferenceError as exc:
        print(f"Shape inference unavailable for {path}: {exc}")
    (modeldir / (stem + ".txt")).write_text(onnx.printer.to_text(m))
    graph = graphviz.Digraph(stem, format="svg")
    graph.attr(rankdir="TB", fontsize="11", label=stem, labelloc="t")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
    producer = {}
    shapes = {v.name: helper.printable_type(v.type) for v in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output)}
    for i in m.graph.input:
        graph.node(i.name, i.name + "\n" + shapes.get(i.name, ""), fillcolor="#DDDDDD")
        producer[i.name] = i.name
    for idx, n in enumerate(m.graph.node):
        name = f"n{idx}"
        color = (
            "#E7A59B"
            if n.op_type == "IsNaN"
            else "#A6CEE3" if n.op_type == "Attention" else "#B2DF8A" if n.op_type in ["Softmax", "MatMul"] else "#EEEEEE"
        )
        label = n.op_type + "\n" + (n.name or str(idx))
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    label += "\n" + str(onnx.numpy_helper.to_array(attr.t).tolist())[:60]
        graph.node(name, label, fillcolor=color)
        for inp in n.input:
            if inp in producer:
                graph.edge(producer[inp], name, label=shapes.get(inp, "").replace("FLOAT16", "f16").replace("FLOAT", "f32"), fontsize="7")
        for out in n.output:
            producer[out] = name
    for out in m.graph.output:
        oid = "out_" + out.name
        graph.node(oid, out.name + "\n" + shapes.get(out.name, ""), fillcolor="#DDDDDD")
        graph.edge(producer[out.name], oid)
    graph.render(filename=str(fig / ("graph-" + stem)), cleanup=True)
# A compact original overview, accompanying the complete model dumps.
g = graphviz.Digraph("pipeline", format="svg")
g.attr(rankdir="LR")
g.attr("node", shape="box", style="rounded,filled", fillcolor="#E8EEF7", fontname="Helvetica")
for name, label in [
    ("sdpa", "PyTorch SDPA"),
    ("legacy", "Legacy exporter\nMatMul–Softmax–MatMul\nmasked safe-softmax guard"),
    ("modern", "Modern exporter\nAttention-23"),
    ("override", "Scoped override\nAttention-24 + valid lengths"),
    ("cpu", "ORT CPU\noptimized CPU graph / kernels"),
    ("cuda", "ORT CUDA\nCUDA attention dispatch"),
    ("amd", "ORT MIGraphX\nparse → fuse → compile"),
    ("proof", "Kernel traces + allocations\nprove implementation"),
]:
    g.node(name, label)
for x, y in [("sdpa", "legacy"), ("sdpa", "modern"), ("sdpa", "override")]:
    g.edge(x, y)
for x in ["legacy", "modern", "override"]:
    for y in ["cpu", "cuda", "amd"]:
        g.edge(x, y)
for x in ["cpu", "cuda", "amd"]:
    g.edge(x, "proof")
g.render(filename=str(fig / "export-runtime-pipeline"), cleanup=True)
# Machine-readable index includes failures; plots must not silently erase them.
index = {
    cfg: {"counts": dict(collections.Counter(r["status"] for r in rows)), "failed": [r for r in rows if r["status"] != "ok"]}
    for cfg, rows in {**data, **batch_data}.items()
}
(a.docs / "evidence-index.json").write_text(json.dumps(index, indent=2))
with (a.docs / "measurements.csv").open("w") as f:
    fields = [
        "configuration",
        "kind",
        "dtype",
        "mode",
        "batch",
        "seq",
        "dynamic",
        "status",
        "timing",
        "valid_tokens",
        "median_ms",
        "p95_ms",
        "session_ms",
        "first_call_ms",
        "device_io_median_ms",
        "mae",
        "max_abs",
        "atol",
        "rtol",
        "output_type",
        "error",
    ]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for cfg, rows in {**data, **batch_data}.items():
        for row in rows:
            w.writerow({"configuration": cfg, **row, "device_io_median_ms": row.get("device_io", {}).get("median_ms")})
print("Wrote figures to", fig)
