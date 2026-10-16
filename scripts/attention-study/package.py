"""Package the article, executable sources, and curated raw evidence without credentials."""

import argparse
import tarfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--docs", type=Path, required=True)
a = p.parse_args()
archive = a.docs / "attention-export-study.tar.gz"
raw_names = [
    "bundle",
    "batch-bundle",
    "cpu129",
    "cuda129",
    "rocm-default",
    "rocm-attention",
    "cpu123",
    "cuda123",
    "cpu-batch",
    "cuda-batch",
    "rocm-batch",
    "memory",
    "diagnostics",
    "override-demo",
    "override-demo-fp16",
    "causal-bundle",
    "causal-cpu",
    "causal-cuda",
    "causal-rocm-default",
    "causal-rocm-attention",
    "precision-cpu123",
    "precision-cpu129",
    "audit.json",
    "pods.json",
    "commit.txt",
    "cxx-runtime.json",
    "rocm-contract-tests.log",
    "cuda-contract-tests.log",
]


def keep(info):
    if "__pycache__" in Path(info.name).parts or "compiled-cache" in Path(info.name).parts:
        return None
    return info


with tarfile.open(archive, "w:gz") as out:
    for name in [
        "index.md",
        "reproduce.md",
        "figures",
        "models",
        "data",
        "measurements.csv",
        "evidence-index.json",
        "runtime-graphs.json",
        "rocm-kernel-evidence.json",
        "cuda-kernel-evidence.json",
    ]:
        path = a.docs / name
        if path.exists():
            out.add(path, arcname="docs/attention-study/" + name, filter=keep)
    out.add(Path(__file__).parent, arcname="scripts/attention-study", filter=keep)
    for name in ["README.md", "session-l40s.yaml", "session-w7900.yaml", "setup-w7900.sh", "kubectl.sh", "session.sh", ".env.example"]:
        out.add(Path("ngt") / name, arcname="ngt/" + name)
    for name in raw_names:
        path = a.root / name
        if path.exists():
            out.add(path, arcname="onnx_validation/attention-study/" + name, filter=keep)
print(archive, archive.stat().st_size, "bytes")
