"""Preserve representative ORT optimized graphs and compare serialized node counts."""

import argparse
import collections
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnx

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--docs", type=Path, required=True)
a = p.parse_args()
target = a.docs / "models" / "optimized"
target.mkdir(parents=True, exist_ok=True)
rows = []
for model in ["sdpa_legacy_fp16_unmasked_b4_s512", "sdpa_legacy_fp16_padding_b4_s512", "attention24_lengths_fp16_padding_b4_s512"]:
    for cfg in ["bundle", "cpu129", "cuda129"]:
        path = a.root / cfg / (model + (".onnx" if cfg == "bundle" else ".optimized.onnx"))
        if not path.exists():
            continue
        m = onnx.load(path)
        dest = target / (cfg + "-" + model + ".onnx")
        shutil.copy2(path, dest)
        dest.with_suffix(".txt").write_text(onnx.printer.to_text(m))
        rows.append(
            {
                "configuration": cfg,
                "model": model,
                "nodes": dict(collections.Counter((n.domain + "::" if n.domain else "") + n.op_type for n in m.graph.node)),
                "artifact": dest.name,
            }
        )
(a.docs / "runtime-graphs.json").write_text(json.dumps(rows, indent=2))
f, ax = plt.subplots(figsize=(11, 5))
families = {"Attention": "#009E73", "MatMul": "#4477AA", "Softmax": "#CC79A7", "IsNaN": "#D55E00", "Other": "#BBBBBB"}
for i, r in enumerate(rows):
    counts = {k: 0 for k in families}
    for name, count in r["nodes"].items():
        simple = name.split("::")[-1]
        key = simple if simple in families else "MatMul" if simple == "FusedMatMul" else "Other"
        counts[key] += count
    left = 0
    for kind, color in families.items():
        if counts[kind]:
            ax.barh(i, counts[kind], left=left, color=color, label=kind if i == 0 else None)
        left += counts[kind]
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["configuration"] + " · " + r["model"].replace("_fp16_", " ").replace("_b4_s512", "") for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="Serialized node count", title="Original versus ORT optimized graphs · node count alone does not prove kernel fusion")
ax.legend(handles=[matplotlib.patches.Patch(color=c, label=k) for k, c in families.items()], fontsize=8)
f.tight_layout()
for ext in ["svg", "pdf", "png"]:
    f.savefig(a.docs / "figures" / f"runtime-graph-counts.{ext}", bbox_inches="tight")
