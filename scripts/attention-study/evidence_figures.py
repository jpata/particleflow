"""Kernel evidence plots from actual rocprof dispatches and Nsight summaries."""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--docs", type=Path, required=True)
a = p.parse_args()
fig = a.docs / "figures"
fig.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
root = a.root / "diagnostics"
evidence = []
for cfg in [
    "rocm-default-sdpa_legacy_fp16_unmasked",
    "rocm-attention-sdpa_legacy_fp16_unmasked",
    "rocm-default-explicit_safe_fp16_padding",
    "rocm-attention-explicit_safe_fp16_padding",
]:
    folder = root / cfg
    log = folder / "compile.log"
    if not log.exists():
        continue
    text = log.read_text()
    at = text.rfind('module: "main"')
    end = text.find("\x1b", at)
    program = text[at : end if end != -1 else None]
    (folder / "final-program.txt").write_text(program)
    n = program.count("gpu::code_object")
    traces = list(folder.rglob("*kernel_trace.csv"))
    if not traces:
        continue
    with traces[0].open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["Kernel_Name"] != "__amd_rocclr_copyBuffer"][-n:]
    evidence.append(
        {"case": cfg, "kernels": rows, "full_attention_intermediate": bool(re.search(r"\{4, 16, 512, 512\}", program)), "compiled_code_objects": n}
    )
if evidence:
    f, ax = plt.subplots(figsize=(12, 5))
    colors = {"Fused attention core": "#009E73", "Matmul": "#4477AA", "Softmax": "#CC79A7", "Mask preparation / output": "#E69F00"}
    for y, item in enumerate(evidence):
        start = min(int(r["Start_Timestamp"]) for r in item["kernels"])
        for r in item["kernels"]:
            name = r["Kernel_Name"]
            typ = (
                "Fused attention core"
                if name.count("dot") >= 2
                else "Matmul" if "dot" in name else "Softmax" if "reduce_max" in name else "Mask preparation / output"
            )
            left = (int(r["Start_Timestamp"]) - start) / 1000
            width = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000
            ax.barh(y, width, left=left, height=0.55, color=colors[typ], edgecolor="white")
        text = f"{item['compiled_code_objects']} kernel(s); full score tensor {'present' if item['full_attention_intermediate'] else 'absent'}"
        ax.text(0, y + 0.37, text, fontsize=8)
    ax.set_yticks(range(len(evidence)))
    ax.set_yticklabels(
        [
            e["case"]
            .replace("rocm-", "")
            .replace("-sdpa_legacy_fp16_unmasked", " · unmasked")
            .replace("-explicit_safe_fp16_padding", " · safe padded")
            for e in evidence
        ]
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="GPU time within the last profiled inference [µs]",
        title="W7900: actual kernel dispatches at B=4, H=16, S=512, D=16\nCompiler tuning and transfers excluded from this view",
    )
    ax.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], fontsize=8, loc="lower right")
    f.tight_layout()
    for ext in ["svg", "pdf", "png"]:
        f.savefig(fig / f"rocm-kernel-timeline.{ext}", bbox_inches="tight")
    plt.close(f)
(a.docs / "rocm-kernel-evidence.json").write_text(json.dumps(evidence, indent=2))
# Nsight sqlite data retains full demangled kernel names and exact dispatch counts.
cuda = []
for folder in root.glob("cuda-*"):
    sql = folder / "trace.sqlite"
    if not sql.exists():
        continue
    db = sqlite3.connect(sql)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        db.close()
        continue
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    namecol = "demangledName" if "demangledName" in cols else "shortName"
    rows = list(db.execute(f"SELECT s.value,k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.{namecol}=s.id ORDER BY k.start"))
    db.close()
    grouped = {}
    for name, start, end in rows:
        item = grouped.setdefault(name, {"name": name, "calls": 0, "total_us": 0})
        item["calls"] += 1
        item["total_us"] += (end - start) / 1000
    cuda.append(
        {
            "case": folder.name,
            "kernels": list(grouped.values()),
            "has_flash_named_kernel": any("flash" in name.lower() for name, _, _ in rows),
            "has_memory_efficient_kernel": any("fmha" in name.lower() or "attentionkernel" in name.lower() for name, _, _ in rows),
        }
    )
if cuda:
    f, axs = plt.subplots(len(cuda), 1, figsize=(12, 3 * len(cuda)), squeeze=False)
    for ax, item in zip(axs.flat, cuda):
        kernels = sorted(item["kernels"], key=lambda k: k["total_us"], reverse=True)[:7]

        def shorten(name):
            if "flash" in name.lower():
                return "FlashAttention forward"
            if "attention_kernel_batched" in name.lower() or "fmha" in name.lower():
                return "CUTLASS memory-efficient attention"
            if "softmax" in name.lower():
                return "Softmax (separate kernel)"
            if "gemm" in name.lower() or "cutlass::Kernel2" in name:
                return "GEMM (QK or probability × V)"
            if "ConvertBoolMask" in name:
                return "Mask → additive bias"
            if "ZeroFullyMaskedRows" in name:
                return "Zero fully masked rows"
            if "ConvertNonpad" in name:
                return "Convert valid lengths"
            if "reduce_sum" in name:
                return "Compute valid lengths"
            if "Transpose" in name:
                return "Layout transpose"
            if "Expand" in name:
                return "Expand dense mask"
            if "OP_Cast" in name:
                return "Cast"
            if "OP_Sqrt" in name:
                return "Scale: square root"
            if "OP_Div" in name:
                return "Scale: divide"
            if "OP_Mul" in name:
                return "Elementwise multiply"
            return name[:65]

        ax.barh(
            range(len(kernels)),
            [k["total_us"] / k["calls"] for k in kernels],
            color=["#009E73" if "flash" in k["name"].lower() else "#4477AA" for k in kernels],
        )
        ax.set_yticks(range(len(kernels)))
        ax.set_yticklabels([shorten(k["name"]) for k in kernels], fontsize=9)
        ax.invert_yaxis()
        ax.set(title=item["case"], xlabel="Mean kernel duration [µs] (profiled execution, not API timing)")
    f.tight_layout()
    for ext in ["svg", "pdf", "png"]:
        f.savefig(fig / f"cuda-kernel-summary.{ext}", bbox_inches="tight")
    plt.close(f)
(a.docs / "cuda-kernel-evidence.json").write_text(json.dumps(cuda, indent=2))
