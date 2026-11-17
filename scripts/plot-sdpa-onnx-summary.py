#!/usr/bin/env python3
"""Plot attention scaling with separate panels for device, precision and workload."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# Color, marker and dash all encode implementation, consistently across figures.
STYLES = {
    "pt_math": ("PyTorch math", "#0072B2", "o", "-"),
    "pt_flash": ("PyTorch Flash", "#0072B2", "o", "-"),
    "pt_efficient": ("PyTorch efficient", "#CC79A7", "v", "-."),
    "ort_attention24": ("ORT Attention-24", "#009E73", "D", "--"),
    "ort_mha": ("ORT MHA", "#D55E00", "s", ":"),
    "onnx_sdpa_export": ("ORT decomposed", "#CC79A7", "^", "-."),
    "onnx_sdpa_static": ("ORT INT8", "#555555", "X", "-."),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-csv", type=Path, help="CPU results (legacy input option).")
    parser.add_argument("--gpu-csv", type=Path, help="GPU results (legacy input option).")
    parser.add_argument("--input", action="append", default=[], metavar="LABEL=CSV", help="Add labeled results, e.g. A100=path/summary.csv.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--sweep", choices=["sequence", "batch"], default="sequence")
    return parser.parse_args()


def read_rows(path, device, include_int8):
    if not path.exists():
        return []
    # Older CSVs omitted batch size and valid lengths; recover them from JSON.
    summary = path.with_suffix(".json")
    payload = json.loads(summary.read_text()) if summary.exists() else {}
    config = payload.get("config", {})
    raw_rows = payload.get("results")
    if raw_rows is None:
        with path.open(newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
    rows = []
    for row in raw_rows:
        variant = row["variant"]
        if row["status"] != "ok":
            print(f"Excluded {device} {variant}: {row.get('error', row['status'])}")
            continue
        if "int8" in variant and not include_int8:
            continue
        batch = int(row.get("batch_size", config.get("batch_size", 1)))
        length = int(row["sequence_length"])
        masked = variant.endswith("_masked")
        valid = row.get("valid_elements")
        if valid is None and "mask_fraction" in config:
            nmasked = int(length * config["mask_fraction"])
            offset = 0.5 if config.get("length_policy") == "balanced" else 1
            valid = sum(length - min(length - 1, int(nmasked * (i + offset) / batch)) for i in range(batch)) if masked else length * batch
        latency = float(row["p50_ms"])
        family = next((key for key in STYLES if variant.startswith(key)), variant)
        export_shape = row.get("export_shape", "native" if family.startswith("pt_") else "dynamic")
        rows.append(
            {
                "device": device,
                "variant": variant,
                "family": family,
                "export_shape": export_shape,
                "dtype": row.get("dtype", next((d for d in ["fp32", "fp16", "int8"] if d in variant), "unknown")),
                "masked": masked,
                "batch_size": batch,
                "sequence_length": length,
                "latency": latency,
                "throughput": 1000 * batch / latency,
                "tokens": 1000 * float(valid) / latency if valid is not None else math.nan,
                "memory": float(row.get("peak_memory_mib") or math.nan),
            }
        )
    return rows


def plot_panel(ax, rows, xkey, metric, title):
    for family in sorted({r["family"] for r in rows}, key=lambda f: list(STYLES).index(f) if f in STYLES else len(STYLES)):
        shapes = sorted({r["export_shape"] for r in rows if r["family"] == family})
        for shape in shapes:
            vals = sorted((r for r in rows if r["family"] == family and r["export_shape"] == shape), key=lambda r: r[xkey])
            xs = [r[xkey] for r in vals]
            if len(xs) != len(set(xs)):
                raise ValueError("Multiple workloads at the same x value; split by batch size / mask mode before plotting")
            label, color, marker, dash = STYLES.get(family, (family, "black", "x", "-"))
            if family == "pt_flash" and vals[0]["masked"]:
                label = "PyTorch varlen Flash"
            if shape == "static" or len(shapes) > 1:
                label += f" ({shape})"
            ax.plot(
                xs,
                [r[metric] for r in vals],
                label=label,
                color=color,
                marker=marker,
                linestyle="-" if shape == "static" else dash,
                linewidth=2.3,
                markersize=10 if marker == "D" else (8 if marker == "o" else 6),
                markerfacecolor=color if shape == "static" else "white",
                markeredgewidth=1.7,
            )
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ticks = sorted({r[xkey] for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Batch size" if xkey == "batch_size" else "Padded sequence length")
    ax.legend(fontsize=9, frameon=True, framealpha=0.95, handlelength=3.5)


def make_figure(rows, sweep, metric, outdir):
    xkey = "batch_size" if sweep == "batch" else "sequence_length"
    # Split precision as well as device to avoid many nearly identical curves.
    row_keys = sorted({(r["device"], r["dtype"]) for r in rows})
    if sweep == "batch":
        columns = sorted({(r["sequence_length"], r["masked"]) for r in rows})
    else:
        columns = sorted({(r["batch_size"], r["masked"]) for r in rows})
    fig, axes = plt.subplots(
        len(row_keys), len(columns), figsize=(6 * len(columns), 3.8 * len(row_keys)), squeeze=False, sharey="row", layout="constrained"
    )
    labels = {
        "latency": "Median latency [ms / batch]",
        "throughput": "Throughput [events / s]",
        "tokens": "Valid elements / s",
        "memory": "Peak process memory [MiB]",
    }
    for i, (device, dtype) in enumerate(row_keys):
        for j, (fixed, masked) in enumerate(columns):
            fixedkey = "sequence_length" if sweep == "batch" else "batch_size"
            selected = [r for r in rows if (r["device"], r["dtype"], r[fixedkey], r["masked"]) == (device, dtype, fixed, masked)]
            mode = "unequal valid lengths" if masked else "no padding"
            size = f"L={fixed}" if sweep == "batch" else f"B={fixed}"
            title = f"{device} · {dtype.upper()} · {size}\n{mode}"
            if selected:
                plot_panel(axes[i, j], selected, xkey, metric, title)
            else:
                axes[i, j].set_visible(False)
            axes[i, j].set_ylabel(labels[metric] if metric != "memory" else ("Process RSS [MiB]" if device == "CPU" else "Process VRAM [MiB]"))
    fig.suptitle(f"Attention {'batch-size' if sweep == 'batch' else 'sequence-length'} scaling — {labels[metric].split(' [')[0]}", fontsize=15)
    fig.supxlabel(
        (
            "ORT includes host↔device transfers on GPU; PyTorch inputs are device-resident. Memory is sampled process usage."
            if metric == "memory"
            else "Warm measurements · ORT includes GPU transfers · PyTorch includes packing for variable lengths"
        ),
        fontsize=9,
    )
    name = "sdpa_onnx_summary" if sweep == "sequence" and metric == "latency" else f"{sweep}_scaling_{metric}"
    for extension in ["png", "pdf"]:
        path = outdir / f"{name}.{extension}"
        fig.savefig(path, dpi=170)
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sources = []
    if args.cpu_csv is not None:
        sources.append(("CPU", args.cpu_csv))
    if args.gpu_csv is not None:
        sources.append(("GPU", args.gpu_csv))
    for item in args.input:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Invalid --input {item!r}; expected LABEL=CSV")
        sources.append((label, Path(path)))
    if not sources:
        raise ValueError("Provide --input LABEL=CSV or a legacy --cpu-csv/--gpu-csv input")
    if len({label for label, _ in sources}) != len(sources):
        raise ValueError("Input labels must be unique")
    for _, path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
    rows = [row for label, path in sources for row in read_rows(path, label, args.include_int8)]
    if not rows:
        raise ValueError("No successful benchmark rows")
    for metric in ["latency", "throughput", "tokens", "memory"]:
        make_figure(rows, args.sweep, metric, args.outdir)


if __name__ == "__main__":
    main()
