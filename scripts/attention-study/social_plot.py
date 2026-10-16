"""Plot measured attention speedups for a compact social-media figure."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
with (ROOT / "docs/attention-study/measurements.csv").open() as stream:
    rows = list(csv.DictReader(stream))


def latency(config, kind, field):
    matches = [
        r
        for r in rows
        if r["configuration"] == config
        and r["kind"] == kind
        and r["dtype"] == "fp16"
        and r["mode"] == "unmasked"
        and r["batch"] == "4"
        and r["seq"] == "2560"
        and r["dynamic"] == "False"
        and r["status"] == "ok"
    ]
    assert len(matches) == 1
    return float(matches[0][field])


pairs = [
    ("L40S · PyTorch", "cuda129", "pytorch_math", "cuda129", "pytorch_flash_attention", "median_ms"),
    ("L40S · ONNX Runtime", "cuda129", "sdpa_legacy", "cuda129", "attention24_dense", "device_io_median_ms"),
    ("W7900 · PyTorch", "rocm-default", "pytorch_math", "rocm-default", "pytorch_flash_attention", "median_ms"),
    ("W7900 · ONNX Runtime", "rocm-default", "sdpa_legacy", "rocm-attention", "sdpa_legacy", "device_io_median_ms"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(12, 7.2))
fig.subplots_adjust(left=0.29, right=0.94, top=0.77, bottom=0.36)
for i, (label, c0, k0, c1, k1, field) in enumerate(pairs):
    baseline, fused = latency(c0, k0, field), latency(c1, k1, field)
    gain = baseline / fused
    y = 3 - i
    ax.barh(y, gain, height=0.45, color="#0072B2" if i < 2 else "#009E73", zorder=3)
    ax.text(gain + 1, y, f"{gain:.1f}×", va="center", weight="bold", fontsize=16)
    ax.text(2.5, y - 0.36, f"{baseline:.2f} → {fused:.2f} ms", va="center", fontsize=10, color="#555555")
    print(f"{label}: {baseline:.6f} / {fused:.6f} = {gain:.4f}x")
ax.set_yticks([3, 2, 1, 0], [p[0] for p in pairs])
ax.set_xlim(0, 80)
ax.set_xticks([1, 10, 20, 40, 60])
ax.set_xlabel("Speedup over each runtime’s unfused baseline", labelpad=12)
ax.axvline(1, color="#555555", linewidth=1, linestyle="--")
ax.grid(axis="x", alpha=0.18, zorder=0)
ax.tick_params(axis="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(0.06, 0.92, "Fused attention: how much faster?", fontsize=26, weight="bold")
fig.text(0.06, 0.865, "L40S and W7900 · FP16 · sequence length 2,560 · batch 4 · 16 heads · head dim 16", fontsize=13)
fig.text(
    0.06,
    0.15,
    "PyTorch: MATH → FLASH_ATTENTION backend.\n"
    "ONNX Runtime / L40S: legacy SDPA decomposition → Attention-24 (FlashAttention).\n"
    "ONNX Runtime / W7900: same legacy graph, default → explicit MIGraphX attention fusion.",
    fontsize=11,
    linespacing=1.7,
)
fig.text(
    0.06,
    0.045,
    "Unmasked attention microbenchmark; median steady-state latency. ORT uses device I/O binding (no host transfers).\n"
    "PyTorch 2.9.1 · ORT 1.29 CUDA / 1.23.2 MIGraphX · ROCm 7.2.4. Gains are within each stack, not end-to-end MLPF speedups.",
    fontsize=9,
    color="#555555",
    linespacing=1.6,
)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(ROOT / f"docs/attention-study/figures/flash-attention-speedup.{ext}", dpi=180, facecolor="white")
