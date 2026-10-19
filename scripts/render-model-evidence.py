"""Render the aggregate table and embedded scaling figures as a portable HTML report."""

import argparse
import html
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
d = json.loads((a.root / "summary.json").read_text())


def fmt(value):
    return "—" if value is None else f"{value:.3g}"


rows = "".join(
    "<tr>"
    + "".join(
        f"<td>{html.escape(str(v))}</td>"
        for v in [
            r["machine"],
            r["model"],
            r["batch"],
            r["slots_per_event"],
            r["valid_inputs"],
            fmt(r["p50_ms"]),
            fmt(r["p95_ms"]),
            fmt(r["max_abs"]),
            fmt(r["max_relative_l2"]),
            fmt(r["inference_vram_delta_mib"]),
            fmt(r["inference_rss_mib"]),
            fmt(r["session_seconds"]),
            fmt(r["first_run_seconds"]),
            "PASS" if r["passed"] else "FAIL",
        ]
    )
    + "</tr>"
    for r in d["rows"]
)
figures = ""
for sweep in ["sequence", "batch", "numerical"]:
    path = a.root / f"{sweep}-scaling.svg"
    if path.exists():
        svg = path.read_text()
        figures += svg[svg.index("<svg") :]
status = f"{d['completed_cells']}/{d['expected_cells']} complete; {len(d['failed'])} failed"
body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>MLPF full-model export evidence</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1600px;color:#18202b}}p{{max-width:1000px;line-height:1.5}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}svg{{max-width:100%;height:auto}}.status{{padding:1rem;background:#eef2f7}}a{{color:#175ca0}}</style>
<h1>MLPF full-model export evidence · 2026-10-19</h1><p class="status">{status}. Overall passed: {d['passed']}.</p>
<p>Seeded, untrained attention and deterministic-hashing HEPTv2 full models. This package establishes numerical export correctness and runtime scaling, not trained physics performance. The NVIDIA GPU is <strong>L40S</strong>; AMD is W7900. CPU fallback runs on the L40S host with one ORT thread.</p>
<p>Both models use three backbone layers and eight heads; they have different backbone implementations and are not matched in parameter count. Inputs are bit-identical CMS-derived features, deterministically repeated to the desired size, with 10% right padding. Each model has independent eager CPU PyTorch FP32 references. The portable baseline uses static FP32 decomposed exports (42 cases). A separate seven-case NVIDIA profile, labeled attention24-fp16, uses Attention-24 with FP16 attention and FP32 public inputs/outputs; CUDA sdpa_kernel=1 is requested, and an independent ORT-only Nsight trace confirms onnxruntime::flash::flash_fwd_kernel execution at N=4000. A second seven-case optimized profile, attention-decomposed-fp16 on W7900, exports decomposed FP16 attention with FP32 public inputs/outputs and requests MIGraphX MLIR attention fusion. An independent rocprofv3 trace confirms fused score–mask–softmax–value kernels at N=4000; per-shape final compiled programs verify all five attention operations are fused without quadratic attention-score allocations. Both optimized profiles use the same FP32 references and remain distinct from the baseline.</p>
<p>Latency includes host input/output transfers and output allocation: 10 warmups followed by 100 unprofiled synchronous session.run calls. GPU placement is checked in separate fresh processes with CPU fallback disabled. Reported VRAM is sampled whole-device used memory above pre-session baseline, including allocator reservations; 10 ms polling can miss transient peaks. RSS includes the Python process and loaded arrays. Model export, specialization and reference calculation are excluded from latency; compilation and first call are listed separately.</p>
{figures}
<h2>All measured shapes</h2><p>Errors cover valid positions across all four output heads; padded outputs must be exactly zero. Absolute/relative tolerances are 1e-3/3e-3. Relative L2 is the largest per-head value. RSS and VRAM values are MiB. Session and first-call durations are seconds.</p>
<table><thead><tr>{''.join('<th>'+v+'</th>' for v in ['Machine','Model','B','N slots','Valid inputs','p50 ms','p95 ms','Max abs','Rel L2','VRAM Δ MiB','RSS MiB','Session s','First s','Status'])}</tr></thead><tbody>{rows}</tbody></table>
<h2>Retained evidence</h2><p><a href="measurements.csv">CSV</a> · <a href="summary.json">Aggregate JSON</a> · <a href="provenance.json">Source provenance</a> · <a href="shared-inputs.json">Shared input hashes</a> · <a href="amd-fusion-verification.json">AMD fusion verification</a> · <a href="l40s-machine.txt">L40S machine</a> · <a href="w7900-machine.txt">W7900 machine</a>. Model/reference manifests and per-case raw latency samples, per-head errors, memory traces, placement profiles and logs are retained in the adjacent model and results directories. Software stacks differ: CUDA uses ORT 1.29; AMD uses native ORT 1.23.2 with MIGraphX 2.18 userspace libraries and recorded integer/compiler workarounds. These are complete-stack measurements.</p>
<p>Single runs characterize this setup; p99 from 100 samples is descriptive, not a precise tail estimate. No kernel-only latency or trained-model accuracy claim is made. CPU affinity and compiler flags are retained in each result. See the repository's docs/validation/export-evidence.md for the reproduction protocol.</p>
<p>Missing cells: {html.escape(', '.join(d['missing']) or 'none')}. Failed cells: {html.escape(', '.join(d['failed']) or 'none')}.</p></html>"""
(a.root / "report.html").write_text(body)
print(a.root / "report.html")
