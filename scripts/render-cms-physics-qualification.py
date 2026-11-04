"""Render the complete CMS numerical/physics qualification matrix and strict gate."""

import argparse
import html
import json
from pathlib import Path

from mlpf.standalone_eval.cms_evidence import input_manifest_path

PROFILES = [
    ("attention-cuda-fp32", "Attention", "NVIDIA", "FP32"),
    ("heptv2-cuda-fp32", "HEPTv2", "NVIDIA", "FP32"),
    ("attention-amd-fp32", "Attention", "AMD", "FP32"),
    ("heptv2-amd-fp32", "HEPTv2", "AMD", "FP32, original export"),
    ("heptv2-amd-stable-logsumexp", "HEPTv2", "AMD", "FP32, stable log-sum-exp export"),
    ("attention-cpu-fp32", "Attention", "CPU", "FP32, default optimization"),
    ("heptv2-cpu-basic", "HEPTv2", "CPU", "FP32, basic optimization / no memory pattern"),
    ("attention-cuda-fp16-on", "Attention", "NVIDIA", "FP16 fused"),
    ("attention-cuda-fp16-off", "Attention", "NVIDIA", "FP16 unfused"),
    ("attention-amd-fp16-on", "Attention", "AMD", "FP16 fused"),
    ("attention-amd-fp16-off", "Attention", "AMD", "FP16 unfused"),
    ("attention-cuda-fused-vs-unfused", "Attention", "NVIDIA", "FP16 fused vs unfused"),
    ("attention-amd-fused-vs-unfused", "Attention", "AMD", "FP16 fused vs unfused"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true", help="Exit nonzero for missing analyses or numerical failures")
    args = p.parse_args()
    source = input_manifest_path(args.root)
    manifest = json.loads(source.read_text())
    cases = manifest["cases"]
    assert len(cases) >= 1000
    assert len({c["event_sha256"] for c in cases}) == len(cases)
    rows, missing = [], []
    table = ""
    for name, model, device, profile in PROFILES:
        path = args.root / "analysis" / name / "summary.json"
        if not path.exists():
            missing.append(name)
            continue
        data = json.loads(path.read_text())
        assert data["events"] == len(cases) and data["full_sample"]
        numeric = data["numerical"]
        row = {
            "name": name,
            "model": model,
            "device": device,
            "profile": profile,
            "events": data["events"],
            "numerical_passed": data["numerical_passed"],
            "max_abs": max(v["max_abs"] for v in numeric.values()),
            "violating_elements": sum(v["violations"] for v in numeric.values()),
            "binary_argmax_changes": numeric["bid"]["argmax_changes"],
            "pid_argmax_changes": numeric["id"]["argmax_changes"],
            "paired_event_shifts": data["paired_event_shifts"],
            "jet30_matching": data["jet30_matching"],
        }
        rows.append(row)
        shifts = data["paired_event_shifts"]
        met = shifts["met"]
        met_text = f'{met["mean_candidate_minus_reference"]:.4g} [{met["mean_ci95"][0]:.4g}, {met["mean_ci95"][1]:.4g}]'
        cells = [
            model,
            device,
            profile,
            data["events"],
            "PASS" if data["numerical_passed"] else "FAIL",
            f'{row["max_abs"]:.5g}',
            row["violating_elements"],
            row["binary_argmax_changes"],
            row["pid_argmax_changes"],
            int(shifts["particle_count"]["sum"]),
            int(shifts["jet_count"]["sum"]),
            int(shifts["jet30_count"]["sum"]),
            met_text,
        ]
        table += "<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells)
        table += f'<td><a href="analysis/{name}/report.html">Plots</a> · <a href="analysis/{name}/summary.json">JSON</a></td></tr>'
    complete = not missing
    passed = complete and all(row["numerical_passed"] for row in rows)
    result = {
        "complete": complete,
        "numerical_passed": passed,
        "events": len(cases),
        "completed_analyses": len(rows),
        "expected_analyses": len(PROFILES),
        "missing": missing,
        "rows": rows,
        "physics_acceptance": "Report-only; no physics pass/fail assigned",
    }
    (args.root / "qualification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    status = "All analyses collected" if complete else f"{len(rows)}/{len(PROFILES)} analyses collected; missing: " + ", ".join(missing)
    headers = [
        "Model",
        "Device",
        "Profile / comparison",
        "Events",
        "Numerical",
        "Max abs",
        "Violations",
        "Binary flips",
        "PID flips",
        "Δ particles",
        "Δ jets >3",
        "Δ jets ≥30",
        "Mean ΔMET [95% CI], GeV",
        "Evidence",
    ]
    body = """<!doctype html><html lang="en"><meta charset="utf-8"><title>CMS v3.2.0 ONNX numerical and physics closure</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1900px;color:#18202b}p{max-width:1150px;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}.status{padding:1rem;background:#eef2f7}</style>
<h1>CMS v3.2.0 ONNX numerical and physics closure · 2026-10-30</h1>"""
    body += '<p class="status">' + html.escape(status) + ". Overall numerical gate: " + str(passed) + ". Physics is report-only.</p>"
    body += """<p>Both 100000-step CMS v3.2.0 checkpoints are evaluated on the same 1024 unique cms_pf_ttbar/1:2.8.0 test events. All 64 shards are represented. Production preprocessing includes the CMS HO correction and target mapping. Valid lengths are 2018–5516; static lengths are 3000, 4000, 5000, 6000. No events are repeated or truncated. Reference evaluation is independent PyTorch CPU FP32 MATH; each model has its own reference. Models have different architectures and parameter counts.</p>
<p>Raw-head numerical acceptance is unchanged: atol=1e-3, rtol=3e-3. The particle decoder is the production predict_particles implementation. Clustered jets use FastJet anti-kT R=0.4, pT&gt;3 GeV; the ≥30 GeV view separates harder jets. Both inclusive and ispu&lt;0.8 collections are retained. The existing maximum-cardinality, minimum-angle one-to-one matcher uses ΔR&lt;0.1. Jet clustering and selection are discontinuous: small tensor differences can change particles near decision thresholds or alter nearly degenerate jet clustering. Review matched tails, unmatched jets and per-event outliers as well as inclusive histograms.</p>
<p>Δ values are candidate minus reference. Count differences are totals across the same events. MET intervals and plot bands use 400 paired event-bootstrap resamples. Histogram bands are pointwise, not a simultaneous statistical test; empty reference bins are omitted from ratios, and underflow/overflow remain in histograms.npz. Targets and CMS PF candidates appear as context; gen jets and MET are retained in paired-events.parquet. No formal physics acceptance is assigned, as requested. Numerical failures are not overridden by visually similar distributions.</p>"""
    body += "<table><thead><tr>" + "".join("<th>" + h + "</th>" for h in headers) + "</tr></thead><tbody>" + table + "</tbody></table>"
    body += """<p>The CPU-only ORT 1.29 wheel is recorded separately from the CUDA 12 ORT 1.29 wheel. Default CPU execution produced SIGBUS during extended tests; HEPTv2's basic-optimization/no-memory-pattern configuration is separately labeled and uses eight threads for lengths 3000/4000 and one thread for lengths 5000/6000. This mixed configuration is a correctness control; no CPU performance comparison is claimed. AMD HEPTv2's original zero-output failure is retained. The separate stable log-sum-exp export row tests the repair: MIGraphX 2.18's naive exp/sum/log reduction overflows in the second attention layer. Max-shifted lowering prevents this overflow without changing checkpoint weights or the eager reference. <a href="heptv2-amd-debug/README.txt">Debug evidence</a> includes an operator reproducer and intermediate layer comparisons. Earlier attempts are under revisions/.</p>
<p><a href="qualification-summary.json">Aggregate JSON</a> · <a href="release.json">Release/checkpoint hashes</a> · <a href="release-revision.json">Pinned Hugging Face revision</a> · <a href="input-manifest.json">Unique test sample</a> · <a href="fusion-verification.json">Fusion evidence</a>. Source snapshots, runtime/library hashes, static graphs, raw heads, placement traces, compiler programs, kernel traces, failed attempts and checksums are retained in this package.</p></html>"""
    body = body.replace('href="input-manifest.json"', f'href="{input_manifest_path(args.root).relative_to(args.root).as_posix()}"')
    (args.root / "report.html").write_text(body)
    print(json.dumps({"complete": complete, "numerical_passed": passed, "completed_analyses": len(rows), "missing": missing}), flush=True)
    if args.require_complete and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
