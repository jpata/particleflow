#!/usr/bin/env python3
"""Create summary plots for SDPA ONNX benchmark CSV outputs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREFERRED_ORDER = [
    "pt_math_fp32_unmasked",
    "pt_math_fp32_masked",
    "pt_math_fp16_unmasked",
    "pt_math_fp16_masked",
    "pt_flash_fp16_unmasked",
    "pt_efficient_fp32_unmasked",
    "pt_efficient_fp32_masked",
    "onnx_sdpa_export_fp32_unmasked",
    "onnx_sdpa_export_fp32_masked",
    "onnx_sdpa_export_fp16_unmasked",
    "onnx_sdpa_export_fp16_masked",
    "ort_mha_fp32_unmasked",
    "ort_mha_fp32_masked",
    "ort_mha_fp16_unmasked",
    "ort_mha_fp16_masked",
    "onnx_sdpa_static_int8_unmasked",
    "onnx_sdpa_static_int8_masked",
]

LABELS = {
    "pt_math_fp32_unmasked": "PyTorch math FP32",
    "pt_math_fp32_masked": "PyTorch math FP32 masked",
    "pt_math_fp16_unmasked": "PyTorch math FP16",
    "pt_math_fp16_masked": "PyTorch math FP16 masked",
    "pt_flash_fp16_unmasked": "PyTorch flash FP16",
    "pt_efficient_fp32_unmasked": "PyTorch efficient FP32",
    "pt_efficient_fp32_masked": "PyTorch efficient FP32 masked",
    "onnx_sdpa_export_fp32_unmasked": "ORT decomposed SDPA FP32",
    "onnx_sdpa_export_fp32_masked": "ORT decomposed SDPA FP32 masked",
    "onnx_sdpa_export_fp16_unmasked": "ORT decomposed SDPA FP16",
    "onnx_sdpa_export_fp16_masked": "ORT decomposed SDPA FP16 masked",
    "ort_mha_fp32_unmasked": "ORT fused MHA FP32",
    "ort_mha_fp32_masked": "ORT fused MHA FP32 masked",
    "ort_mha_fp16_unmasked": "ORT fused MHA FP16",
    "ort_mha_fp16_masked": "ORT fused MHA FP16 masked",
    "onnx_sdpa_static_int8_unmasked": "ORT quantized SDPA INT8",
    "onnx_sdpa_static_int8_masked": "ORT quantized SDPA INT8 masked",
}

VARIANT_COLORS = {
    "pt_math_fp32_unmasked": "#1f77b4",
    "pt_math_fp32_masked": "#aec7e8",
    "pt_math_fp16_unmasked": "#17becf",
    "pt_math_fp16_masked": "#9edae5",
    "pt_flash_fp16_unmasked": "#ff7f0e",
    "pt_efficient_fp32_unmasked": "#2ca02c",
    "pt_efficient_fp32_masked": "#98df8a",
    "onnx_sdpa_export_fp32_unmasked": "#9467bd",
    "onnx_sdpa_export_fp32_masked": "#c5b0d5",
    "onnx_sdpa_export_fp16_unmasked": "#8c564b",
    "onnx_sdpa_export_fp16_masked": "#c49c94",
    "ort_mha_fp32_unmasked": "#d62728",
    "ort_mha_fp32_masked": "#ff9896",
    "ort_mha_fp16_unmasked": "#7f7f7f",
    "ort_mha_fp16_masked": "#c7c7c7",
    "onnx_sdpa_static_int8_unmasked": "#bcbd22",
    "onnx_sdpa_static_int8_masked": "#dbdb8d",
    "other": "#111111",
}

VARIANT_MARKERS = {
    "pt_math": "o",
    "pt_flash": "P",
    "pt_efficient": "h",
    "onnx_sdpa_export": "D",
    "onnx_sdpa_static": "X",
    "ort_mha": "s",
    "other": "^",
}

MASK_LINESTYLES = {
    "unmasked": "-",
    "masked": (0, (4, 2)),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SDPA ONNX benchmark runtime and memory summaries.")
    parser.add_argument("--cpu-csv", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--include-int8", action="store_true", help="Include successful INT8 rows if any exist.")
    return parser.parse_args()


def read_rows(path: Path, device: str, include_int8: bool):
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            variant = row.get("variant", "")
            if not include_int8 and "int8" in variant:
                continue
            try:
                rows.append(
                    {
                        "device": device,
                        "variant": variant,
                        "sequence_length": int(row["sequence_length"]),
                        "mean_ms": float(row["mean_ms"]),
                        "peak_memory_mib": float(row["peak_memory_mib"]) if row.get("peak_memory_mib") else math.nan,
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def variant_order(variant):
    try:
        return PREFERRED_ORDER.index(variant)
    except ValueError:
        return len(PREFERRED_ORDER), variant


def variant_family(variant):
    if variant.startswith("pt_math"):
        return "pt_math"
    if variant.startswith("pt_flash"):
        return "pt_flash"
    if variant.startswith("pt_efficient"):
        return "pt_efficient"
    if variant.startswith("onnx_sdpa_export"):
        return "onnx_sdpa_export"
    if variant.startswith("onnx_sdpa_static"):
        return "onnx_sdpa_static"
    if variant.startswith("ort_mha"):
        return "ort_mha"
    return "other"


def variant_dtype(variant):
    for dtype in ["fp32", "fp16", "bf16", "int8"]:
        if dtype in variant:
            return dtype
    return "other"


def variant_mask_state(variant):
    return "masked" if variant.endswith("_masked") else "unmasked"


def variant_style(variant):
    masked = variant_mask_state(variant) == "masked"
    color = VARIANT_COLORS.get(variant, VARIANT_COLORS["other"])
    return {
        "color": color,
        "marker": VARIANT_MARKERS[variant_family(variant)],
        "linestyle": MASK_LINESTYLES[variant_mask_state(variant)],
        "markerfacecolor": "white" if masked else color,
    }


def plot_panel(ax, rows, metric, title, ylabel):
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    variants = sorted({row["variant"] for row in rows}, key=variant_order)
    for variant in variants:
        vals = sorted([row for row in rows if row["variant"] == variant], key=lambda row: row["sequence_length"])
        xs = [row["sequence_length"] for row in vals]
        ys = [row[metric] for row in vals]
        style = variant_style(variant)
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6,
            markerfacecolor=style["markerfacecolor"],
            markeredgewidth=1.0,
            markeredgecolor="black",
            label=LABELS.get(variant, variant),
        )

    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cpu_rows = read_rows(args.cpu_csv, "CPU", args.include_int8)
    gpu_rows = read_rows(args.gpu_csv, "GPU", args.include_int8)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    plot_panel(axes[0, 0], cpu_rows, "mean_ms", "CPU Runtime", "Mean runtime [ms]")
    plot_panel(axes[0, 1], cpu_rows, "peak_memory_mib", "CPU Memory", "Peak RSS [MiB]")
    plot_panel(axes[1, 0], gpu_rows, "mean_ms", "GPU Runtime", "Mean runtime [ms]")
    plot_panel(axes[1, 1], gpu_rows, "peak_memory_mib", "GPU Memory", "Peak VRAM [MiB]")

    handles, labels = [], []
    for ax in axes.flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend = fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, handlelength=3.2)
    for text in legend.get_texts():
        if text.get_text() == "ORT fused MHA FP16":
            text.set_fontweight("bold")
    fig.suptitle("SDPA ONNX Benchmark Summary", fontsize=16)

    png_path = args.outdir / "sdpa_onnx_summary.png"
    pdf_path = args.outdir / "sdpa_onnx_summary.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
