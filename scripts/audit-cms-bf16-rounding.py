"""Audit the BF16 lattice of classification heads exported before FP32 I/O casts."""

import argparse
import json
from pathlib import Path
import numpy as np
import ml_dtypes


def audit(root, device="all"):
    cases = json.loads((root / "input-manifest.json").read_text())["cases"]
    rows = []
    for model in ["attention", "heptv2"]:
        paths = [("pytorch-cuda", root / "reference" / model), ("pytorch-amd", root / "reference-amd" / model)]
        paths += [
            (f"{provider}-{mode}", root / "results" / model / provider / profile / mode)
            for provider, profile in [("cuda", "attention24-bf16"), ("migraphx", "decomposed-bf16")]
            for mode in ["on", "off"]
        ]
        for label, folder in paths:
            if device != "all" and (("amd" if label == "pytorch-amd" or label.startswith("migraphx") else "cuda") != device):
                continue
            heads = {k: dict(elements=0, non_bf16_values=0, affected_events=0, max_distance_to_bf16=0.0) for k in ["bid", "id"]}
            for case in cases:
                with np.load(folder / case["file"]) as outputs:
                    for key, record in heads.items():
                        value = outputs[key]
                        assert np.isfinite(value).all()
                        rounded = value.astype(ml_dtypes.bfloat16).astype(np.float32)
                        count = int((value != rounded).sum())
                        record["elements"] += value.size
                        record["non_bf16_values"] += count
                        record["affected_events"] += int(count > 0)
                        record["max_distance_to_bf16"] = max(record["max_distance_to_bf16"], float(np.abs(value - rounded).max()))
            rows.append(dict(model=model, scenario=label, events=len(cases), heads=heads))
    result = dict(
        description="bid/id graph outputs are BF16 before the external FP32 cast; any finite off-lattice values indicate changed rounding semantics.",
        rows=rows,
    )
    filename = "bf16-rounding-audit.json" if device == "all" else f"bf16-rounding-audit-{device}.json"
    (root / filename).write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["model"], row["scenario"], row["heads"]["id"]["non_bf16_values"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", choices=["all", "cuda", "amd"], default="all")
    args = p.parse_args()
    audit(args.root, args.device)
