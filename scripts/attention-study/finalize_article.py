"""Insert measured tables, version summaries, and portable evidence links into the draft."""

import argparse
import json
import re
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--docs", type=Path, required=True)
a = p.parse_args()
text = (a.docs / "index.md").read_text()
data = a.docs / "data"
data.mkdir(exist_ok=True)
configs = ["cpu129", "cuda129", "rocm-default", "rocm-attention"]
provs = {}
results = {}
for cfg in configs + [
    "cpu123",
    "cuda123",
    "cpu-batch",
    "cuda-batch",
    "rocm-batch",
    "causal-cpu",
    "causal-cuda",
    "causal-rocm-default",
    "causal-rocm-attention",
    "precision-cpu123",
    "precision-cpu129",
]:
    folder = a.root / cfg
    if not folder.exists():
        continue
    dest = data / cfg
    dest.mkdir(exist_ok=True)
    for name in [
        "provenance.json",
        "provenance-after.json",
        "runtime-provenance.json",
        "results.json",
        "validated-results.json",
        "edge-correctness.json",
        "results-tf32-default.json",
    ]:
        if (folder / name).exists():
            shutil.copy2(folder / name, dest / name)
    if (folder / "provenance.json").exists():
        provs[cfg] = json.loads(
            (folder / ("runtime-provenance.json" if (folder / "runtime-provenance.json").exists() else "provenance.json")).read_text()
        )
    rp = folder / ("validated-results.json" if (folder / "validated-results.json").exists() else "results.json")
    if rp.exists():
        results[cfg] = json.loads(rp.read_text())
for name in ["pods.json", "audit.json", "commit.txt", "cxx-runtime.json"]:
    if (a.root / name).exists():
        shutil.copy2(a.root / name, data / name)
if (a.root / "bundle/modern-export-provenance.json").exists():
    shutil.copy2(a.root / "bundle/modern-export-provenance.json", data / "modern-export-provenance.json")


def replace(marker, body):
    global text
    start = f"<!-- {marker} -->"
    end = f"<!-- /{marker} -->"
    if end in text:
        text = re.sub(re.escape(start) + ".*?" + re.escape(end), lambda m: start + "\n\n" + body + "\n\n" + end, text, flags=re.DOTALL)
    else:
        text = text.replace(start, start + "\n\n" + body + "\n\n" + end)


def pkg(prov, name):
    return next(
        (r["version"] for r in prov["python_packages"] if r["name"].lower().replace("_", "-") == name.lower().replace("_", "-")), "not installed"
    )


lines = ["| Recorded component | CPU / NVIDIA environment | AMD environment |", "|---|---|---|"]
n = provs.get("cuda129", {})
r = provs.get("rocm-attention", {})
if n and r:

    def model(d):
        return next((line.split(":", 1)[1].strip() for line in d["commands"]["cpu"]["stdout"].splitlines() if "Model name" in line), "unknown")

    driver = re.search(r"Driver Version\s*:\s*([^\n]+)", n["commands"]["nvidia"].get("stdout", ""))
    rows = [
        ("Hardware", model(n) + "; full NVIDIA L40S (sm_89)", model(r) + "; Radeon PRO W7900 (gfx1100)"),
        ("Python", n["python"].split()[0], r["python"].split()[0]),
        ("PyTorch", n["torch"]["version"], r["torch"]["version"]),
        ("ONNX Runtime", n["ort"]["version"] + "; controls: 1.23.2", r["ort"]["version"] + " MIGraphX build"),
        ("NVIDIA driver / AMD module", driver.group(1).strip() if driver else "see provenance", str(r.get("amdgpu_module", "unavailable")).strip()),
        ("Host kernel", n["platform"].split("-x86")[0], r["platform"].split("-x86")[0]),
        ("PyTorch CUDA / HIP build", str(n["torch"]["cuda_build"]), str(r["torch"]["hip_build"])),
        (
            "Runtime API version (raw integer)",
            str(n.get("cuda_driver_api", {}).get("version_integer", "unavailable")) + " (CUDA driver API)",
            str(r.get("hip_runtime_api", {}).get("version_integer", "unavailable")) + " (HIP runtime API)",
        ),
        (
            "CUDA runtime Python package / ROCm native",
            "nvidia-cuda-runtime-cu12 " + pkg(n, "nvidia-cuda-runtime-cu12"),
            "ROCm 7.2.4; MIGraphX 2.15.0",
        ),
        ("cuDNN package / MIOpen build", pkg(n, "nvidia-cudnn-cu12"), "MIOpen 3.5.1 (PyTorch build)"),
        (
            "CUDA actual runtime API",
            str(n.get("cuda_runtime_api", {}).get("version_integer", "see runtime-provenance.json")),
            "HIP API recorded in preceding row",
        ),
        (
            "Installed toolkit",
            n["commands"]["cuda_toolkit"].get("stdout", "unavailable").strip().split("\n")[-1],
            "hipconfig --full and ROCm RPM inventory in runtime provenance",
        ),
        ("cuBLAS package", pkg(n, "nvidia-cublas-cu12"), "Native rocBLAS/hipBLASLt package versions in provenance"),
        ("ONNX / NumPy", pkg(n, "onnx") + " / " + pkg(n, "numpy"), pkg(r, "onnx") + " / " + pkg(r, "numpy")),
        ("Exporter dependencies", "ONNX Script 0.5.3 / ONNX IR 0.1.12", "ONNX Script 0.5.3 / ONNX IR 0.1.12 (modern export)"),
        (
            "Auxiliary C++ runtime",
            "System runtime; exact loaded paths recorded",
            "Conda libstdcxx/libgcc 14.4.0; package builds and hashes in data/cxx-runtime.json",
        ),
        ("CPU compute threads", "1 PyTorch; ORT intra/inter-op 1", "same for reference calculations"),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += [
        "",
        "Loaded-library records: [CUDA](data/cuda129/runtime-provenance.json), [MIGraphX](data/rocm-attention/runtime-provenance.json). Full inventories: [NVIDIA](data/cuda129/provenance.json), [CPU](data/cpu129/provenance.json), [AMD default](data/rocm-default/provenance.json), [AMD fusion](data/rocm-attention/provenance.json), and [pod image digests](data/pods.json).",
    ]
replace("VERSIONS", "\n".join(lines))
lines = [
    "Representative **FP16-input, sequence-2560** medians, in milliseconds. Native PyTorch and ORT bound-I/O share device-resident boundaries; ORT host columns include transfers.",
    "",
    "| Backend / representation | Unmasked host/native | Unmasked bound GPU I/O | Padded host/native | Padded bound GPU I/O |",
    "|---|---:|---:|---:|---:|",
]
for cfg in configs:
    for kind in ["pytorch_math", "pytorch_flash_attention", "sdpa_legacy", "explicit_safe", "attention24_dense", "attention24_lengths"]:
        cells = []
        present = False
        for mode in ["unmasked", "padding"]:
            rr = [
                r
                for r in results.get(cfg, [])
                if not r["dynamic"] and r["kind"] == kind and r["dtype"] == "fp16" and r["mode"] == mode and r["seq"] == 2560
            ]
            if rr:
                present = True
                row = rr[0]
                good = row["status"] == "ok"
                cells.extend(
                    [
                        f"{row['median_ms']:.3f}" if good else row["status"],
                        f"{row['device_io']['median_ms']:.3f}" if good and "device_io" in row else "—",
                    ]
                )
            else:
                cells.extend(["—", "—"])
        if present:
            lines.append("| " + cfg + " / " + kind + " | " + " | ".join(cells) + " |")
lines += ["", "![FP16 padded valid-token throughput](figures/throughput.svg)"]
if (a.docs / "figures/batch-scaling.svg").exists():
    lines += ["", "![Balanced-padding batch scaling](figures/batch-scaling.svg)"]
replace("RESULTS", "\n".join(lines))
if (a.docs / "figures/memory.svg").exists():
    replace("MEMORY", "![Separate memory measurements](figures/memory.svg)")
if (a.docs / "figures/dynamic-compilation.svg").exists():
    body = "![Changing dynamic sizes](figures/dynamic-compilation.svg)\n\nIn this MIGraphX no-cache probe, returning to a previously used size still triggered compilation: the provider reused the current specialization, not an in-memory pool of all previously seen shapes. A separate disk-cache probe aborted with a GPU memory-access fault after changing sizes. That result is retained as a failed scenario; this study does not establish reliable disk-cache reuse for the tested wheel/library combination."
    replace("DYNAMIC", body)
links = [
    "Portable data include the [raw measurements](data/cuda129/results.json), [validated classifications](data/cpu129/validated-results.json), [artifact audit](data/audit.json), and full environment records linked above. Numerical reassessment is explicitly marked: initial FP16-input graphs with FP32 outputs were judged against an output-dtype-based tolerance; where the recorded maximum absolute error is at most 0.002, that bound proves the intended FP16-input absolute criterion. Raw measurements and original classifications are preserved."
]
for folder in (a.root / "diagnostics").glob("*"):
    if not folder.is_dir():
        continue
    target = data / "diagnostics" / folder.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ["probe.json", "final-program.txt", "kernels.csv"]:
        if (folder / name).exists():
            shutil.copy2(folder / name, target / name)
    if "dynamic-disk" in folder.name and (folder.with_suffix(".log")).exists():
        shutil.copy2(folder.with_suffix(".log"), target / "failure.log")
links += [
    "",
    "[ROCm kernel evidence](rocm-kernel-evidence.json) · [CUDA kernel evidence](cuda-kernel-evidence.json) · [dynamic calls](data/diagnostics/rocm-dynamic-none/probe.json) · [cache failure log](data/diagnostics/rocm-dynamic-disk/failure.log).",
]
if (a.root / "override-demo-fp16/validation.json").exists():
    shutil.copytree(a.root / "override-demo-fp16", data / "override-demo-fp16", dirs_exist_ok=True)
    links += [
        "",
        "[FP16 scoped wrapper validation](data/override-demo-fp16/validation.json) · [CUDA execution of the actual wrapper export](data/override-demo-fp16/cuda-proof/probe.json).",
    ]
if (a.root / "override-demo/validation.json").exists():
    target = data / "override-demo"
    shutil.copytree(a.root / "override-demo", target, dirs_exist_ok=True)
    links += [
        "",
        "[Actual scoped wrapper validation](data/override-demo/validation.json) · [placeholder ONNX](data/override-demo/override-placeholder.onnx) · [standard Attention-24 ONNX](data/override-demo/override-attention24.onnx).",
    ]
replace("EVIDENCE", "\n".join(links))
(a.docs / "index.md").write_text(text)
