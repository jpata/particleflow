"""Render retained full-model evidence and embedded scaling figures as HTML."""

import argparse
import html
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
d = json.loads((a.root / "summary.json").read_text())
attention = json.loads((a.root / "attention/manifest.json").read_text())
hept = json.loads((a.root / "heptv2/manifest.json").read_text())
trained = "checkpoint" in attention


def fmt(value):
    return "—" if value is None else f"{value:.3g}"


rows = "".join(
    "<tr>"
    + "".join(
        f"<td>{html.escape(str(v))}</td>"
        for v in [
            r["machine"],
            r["model"],
            r.get("case", ""),
            r["batch"],
            r["slots_per_event"],
            r["valid_inputs"],
            fmt(r["p50_ms"]),
            fmt(r["p95_ms"]),
            fmt(r["max_abs"]),
            fmt(r["max_relative_l2"]),
            r.get("violating_elements", "—"),
            r.get("bid_argmax_disagreements", "—"),
            r.get("id_argmax_disagreements", "—"),
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
if trained:
    scope = "Actual trained attention and deterministic-hashing HEPTv2 checkpoints, with strict state-dict loading. This package measures export equivalence on actual TFDS events."
    inputs = "Actual held-out test events from cms_pf_ttbar, cms_pf_qcd and cms_pf_ztt, configuration 1, version 3.0.0: two events per process plus a mixed three-event batch. Inputs are neither truncated nor repeated. Zero element-type rows are removed, feature width follows the saved config, and padding is rounded to the saved model requirements. The checkpoints were trained on 2.8.0. Features come directly from TFDS X: this export-equivalence study does not apply the training loader's CMS HO pT correction or target transformations. Different events have different content as well as different lengths, so these plots show observed scaling rather than a controlled input-length experiment."
    checkpoints = "<h2>Checkpoint provenance</h2><table><tr><th>Model</th><th>Step</th><th>Checkpoint SHA256</th></tr>"
    for label, manifest in [("Attention", attention), ("HEPTv2", hept)]:
        c = manifest["checkpoint"]
        checkpoints += f'<tr><td>{label}</td><td>{c["step"]}</td><td>{c["sha256"]}</td></tr>'
    checkpoints += (
        "</table><p>Model structures come from saved model_kwargs.pkl, including HEPTv2 width "
        + str(hept["config"]["model"]["heptv2"]["width"])
        + ". CPU PyTorch FP32 MATH attention supplies the independent references; only inference kernel selection is overridden. Training-only loss-weight buffers are omitted by explicit name, and all model weights are loaded strictly. The two checkpoints have different training durations.</p>"
    )
else:
    scope = "Seeded, untrained attention and deterministic-hashing HEPTv2 full models. This package establishes export correctness and implementation scaling."
    inputs = "Bit-identical CMS-derived features, repeated deterministically to controlled shapes with 10% right padding. Both models have three backbone layers and eight heads, but are not matched in parameter count. Each model has independent eager CPU PyTorch FP32 references."
    checkpoints = ""
precision = ""
if trained and (a.root / "precision-diagnostic/summary.json").exists():
    diagnostic = json.loads((a.root / "precision-diagnostic/summary.json").read_text())
    details = []
    for label, entry in diagnostic["profiles"].items():
        contract = entry["torch_fp16_contract_vs_fp32"]
        runtime = entry["onnx_vs_torch_fp16_contract"]
        details.append(
            f"{label}: PyTorch mixed-precision versus FP32 max absolute error {max(x['max_abs'] for x in contract.values()):.4g}; ONNX versus that PyTorch contract {max(x['max_abs'] for x in runtime.values()):.4g}."
        )
    precision = (
        "<h2>Precision diagnostic</h2><p>This diagnostic evaluates both export formats on L40S/CUDA. For "
        + html.escape(diagnostic["case"])
        + ", PyTorch evaluation of each mixed-precision export contract already exceeds the FP32 tolerance. ONNX adds further differences, so the failures cannot be attributed solely to either export conversion or precision casting. This diagnostic does not qualify the FP16 profiles. "
        + html.escape(" ".join(details))
        + ' <a href="precision-diagnostic/summary.json">Detailed comparisons</a>.</p>'
    )
status = f"{d['completed_cells']}/{d['expected_cells']} complete; {len(d['failed'])} failed"
groups = {}
for row in d["rows"]:
    groups.setdefault((row["machine"], row["model"]), []).append(row)
profile_rows = []
summary_batch = max((r["batch"] for r in d["rows"]), default=1)
for (machine, model), measured in groups.items():
    singles = [r["p50_ms"] for r in measured if r["batch"] == 1]
    mixed = [r["p50_ms"] for r in measured if r["batch"] == summary_batch]
    vram = [r["inference_vram_delta_mib"] for r in measured if r["inference_vram_delta_mib"] is not None]
    values = [
        machine,
        model,
        f"{sum(r['passed'] for r in measured)}/{len(measured)}",
        f"{fmt(min(singles))}–{fmt(max(singles))}" if singles else "—",
        fmt(mixed[0]) if mixed else "—",
        fmt(max(r["max_abs"] for r in measured)),
        fmt(max(r["max_relative_l2"] for r in measured)),
        fmt(max(vram)) if vram else "—",
        fmt(max(r["inference_rss_mib"] for r in measured)),
    ]
    profile_rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>")
profile_headers = [
    "Machine",
    "Profile",
    "Passed cases",
    "B=1 p50 range ms",
    f"B={summary_batch} p50 ms",
    "Max abs",
    "Max rel L2",
    "Max sampled VRAM Δ MiB",
    "Max RSS MiB",
]
profiles = (
    "<h2>Profile summary</h2><p>Latency ranges span the measured single events, not repeated measurements of one event. The largest-batch latency uses the largest measured batch. Memory maxima span all measured cases.</p><table><thead><tr>"
    + "".join("<th>" + h + "</th>" for h in profile_headers)
    + "</tr></thead><tbody>"
    + "".join(profile_rows)
    + "</tbody></table>"
)
headers = [
    "Machine",
    "Profile",
    "Case",
    "B",
    "N slots",
    "Valid inputs",
    "p50 ms",
    "p95 ms",
    "Max abs",
    "Rel L2",
    "Elements outside tolerance",
    "Binary flips",
    "PID flips",
    "VRAM Δ MiB",
    "RSS MiB",
    "Session s",
    "First s",
    "Status",
]
body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>MLPF full-model export evidence</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1800px;color:#18202b}}p{{max-width:1100px;line-height:1.5}}table{{border-collapse:collapse;font-size:12px}}th,td{{padding:6px;border-bottom:1px solid #ddd;text-align:right}}td:first-child{{text-align:left}}svg{{max-width:100%;height:auto}}.status{{padding:1rem;background:#eef2f7}}a{{color:#175ca0}}</style>
<h1>MLPF full-model export evidence · {html.escape(a.root.name)}</h1><p class="status">{status}. Overall passed: {d['passed']}.</p>
<p>{scope} NVIDIA is <strong>L40S</strong>; AMD is W7900. CPU fallback runs on the L40S host with one ORT thread.</p>
{checkpoints}
<p>{inputs}</p>
<p>The portable baseline uses decomposed FP32 exports. The NVIDIA attention24-fp16 profile uses Attention-24 with CUDA sdpa_kernel=1. The W7900 attention-decomposed-fp16 profile requests MIGraphX MLIR attention fusion. Both retain FP32 public I/O, FP16 attention internals, and the same FP32 reference arrays. Actual kernel traces and per-shape AMD compiler programs are retained; requests or provider names alone are not proof of fusion.</p>
<p>Latency includes host transfers and output allocation: 10 warmups followed by 100 unprofiled synchronous session.run calls. Placement is checked separately with GPU CPU-fallback disabled. Session creation and first-call durations are separate. Sampled whole-device VRAM above baseline includes allocator reservation; 10 ms polling can miss transient peaks. Process RSS includes the Python runtime and loaded arrays.</p>
{profiles}
{figures}
{precision}
<h2>All measured cases</h2><p>Finite valid outputs must satisfy atol=1e-3, rtol=3e-3, and padded outputs must be exactly zero. These thresholds are unchanged for the trained checkpoints. FP16 deviations are reported against FP32, even when the mixed-precision graph executes correctly. Binary/PID flips compare per-input argmax decisions with the FP32 reference. Relative L2 is the largest per-head value. Memory is MiB; session/first-call durations are seconds.</p>
<table><thead><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr></thead><tbody>{rows}</tbody></table>
<h2>Retained evidence</h2><p><a href="measurements.csv">CSV</a> · <a href="summary.json">Aggregate JSON</a> · <a href="provenance.json">Provenance</a> · <a href="shared-inputs.json">Input hashes</a> · <a href="amd-fusion-verification.json">AMD fusion checks</a> · <a href="l40s-machine.txt">L40S host</a> · <a href="w7900-machine.txt">W7900 host</a>. Manifests, model graphs, references, raw timing/memory samples, placement profiles, traces and logs remain adjacent. CUDA uses ORT 1.29; AMD uses native ORT 1.23.2 with MIGraphX 2.18 and recorded ROCm userspace libraries. Environments are managed with uv.</p>
<p>Single-run p99 estimates from 100 samples are descriptive. These are complete-stack host-to-host measurements, not kernel-only timings or a physics-quality assessment. See docs/validation/export-evidence.md and ngt/README.md for the protocol and environment setup.</p>
<p>Missing: {html.escape(', '.join(d['missing']) or 'none')}. Failed: {html.escape(', '.join(d['failed']) or 'none')}.</p></html>"""
(a.root / "report.html").write_text(body)
print(a.root / "report.html")
