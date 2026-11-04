"""Render paired CMS BF16 runtime, numerical and particle/jet physics results."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import statistics


def timing(records):
    values = sorted(c["inference_ms"] for c in records)

    def quantile(q):
        at = (len(values) - 1) * q
        low = int(at)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)

    return dict(events=len(values), mean_ms=statistics.mean(values), p50_ms=quantile(0.5), p95_ms=quantile(0.95))


def performance(root, model, device, fusion):
    cases = json.loads((root / "input-manifest.json").read_text())["cases"]
    buckets = sorted({case["slots"] for case in cases})
    if fusion == "pytorch":
        path = root / ("reference" if device == "cuda" else "reference-amd") / model
        manifest = json.loads((path / "manifest.json").read_text())
        assert len(manifest["cases"]) == len(cases)
        assert {c["case"] for c in manifest["cases"]} == {c["file"] for c in cases}
        return dict(
            **timing(manifest["cases"]),
            warmup=manifest["warmup"],
            scope=manifest["timing_scope"],
            environment={k: manifest[k] for k in ["torch", "cuda", "hip", "gpu", "python"]},
            per_bucket={str(b): timing([c for c in manifest["cases"] if c["slots"] == b]) for b in buckets},
        )
    provider, profile = ("cuda", "attention24-bf16") if device == "cuda" else ("migraphx", "decomposed-bf16")
    path = root / "results" / model / provider / profile / fusion
    manifests = {str(b): json.loads((path / f"bucket_{b}.json").read_text()) for b in buckets}
    records = [c for m in manifests.values() for c in m["cases"]]
    assert len(records) == len(cases) and {c["case"] for c in records} == {c["file"] for c in cases}
    return dict(
        **timing(records),
        per_bucket={b: dict(**timing(m["cases"]), cold=m["timing"]) for b, m in manifests.items()},
        scope=next(iter(manifests.values()))["timing"]["scope"],
        ort=next(iter(manifests.values()))["ort"],
    )


def physics_report(root, result, physics):
    """Keep physics distributions and latency in the main view."""
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>CMS physics and latency</title>'
    body += "<style>body{font:16px system-ui;margin:2rem;max-width:1450px;color:#18202b}p{max-width:1150px;line-height:1.5}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;max-width:1400px}a{color:#165b9e}summary{cursor:pointer;margin:1rem 0}</style>"
    body += f'<h1>CMS physics distributions and latency</h1><p>{result["events"]:,} unique ttbar test events, two trained models, NVIDIA L40S and AMD W7900. Physics spectra use native NVIDIA PyTorch BF16 as the common reference. Particle and jet spectra retain absolute yields per event; event sum-pT is the scalar sum over all predicted particles.</p>'
    body += "<h2>Physics distributions</h2><p>Assess changes in particle pT, jet pT and event sum-pT against the statistical fluctuations of this validation sample. Grey bands show ±2 reference sampling standard errors, treating whole events as independent. Coloured whiskers show the narrower paired-difference 95% bootstrap intervals.</p>"
    fused = [r for r in physics["rows"] if r["scenario"] in ["cuda-on", "amd-on"]]
    departures = [(r, key, v) for r in fused for key, v in r["observables"].items() if v["bins_outside_2se"]]
    if not departures:
        body += "<p>All assessed fused spectra stay within the reference ±2σ sampling band in populated bins.</p>"
    else:
        findings = []
        for row, key, values in departures:
            title = {"particle_pt": "particle pT", "jet_pt": "jet pT", "sumpt": "event sum-pT"}[key]
            model = "Attention" if row["model"] == "attention" else "HEPTv2"
            device = "NVIDIA" if row["scenario"].startswith("cuda") else "AMD"
            worst = values["worst_bin"]
            findings.append(
                f'{model} on {device}: {values["bins_outside_2se"]} {title} bins exceed the band, with a largest departure of {values["max_abs_shift_in_reference_se"]:.1f}σ in {worst["low"]:.2g}–{worst["high"]:.2g} GeV'
            )
            if key == "particle_pt":
                findings[-1] += f' (departing bins extend up to {max(v["high"] for v in values["departures"]):.2g} GeV)'
        body += (
            "<p><strong>" + html.escape("; ".join(findings)) + ".</strong> Other assessed fused spectra stay within the reference sampling band.</p>"
        )
    body += "<table><tr><th>Fused ONNX</th><th>Particle pT</th><th>Jet pT</th><th>Event sum-pT</th><th>Mean event sum-pT shift [95% paired CI], GeV</th></tr>"
    for row in fused:
        model = "Attention" if row["model"] == "attention" else "HEPTv2"
        device = "NVIDIA" if row["scenario"].startswith("cuda") else "AMD"
        shift = row["mean_sumpt_shift"]
        values = [f"{model} / {device}"]
        values += [f'{row["observables"][key]["max_abs_shift_in_reference_se"]:.2f}σ' for key in ["particle_pt", "jet_pt", "sumpt"]]
        values.append(f'{shift["mean_candidate_minus_reference"]:+.2f} [{shift["mean_ci95"][0]:+.2f}, {shift["mean_ci95"][1]:+.2f}]')
        body += "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>"
    body += "</table><p>σ columns give the largest absolute bin shift divided by the reference sampling standard error. Assessed bins require at least 20 contributing reference events. This is a pointwise, sample-size-dependent comparison, not a global significance test or an equivalence bound.</p>"
    for model in ["attention", "heptv2"]:
        body += f'<p><a href="physics-summary/{model}-fused.pdf">PDF</a></p><img src="physics-summary/{model}-fused.png" alt="{model} particle pT, jet pT and event sum-pT">'
    body += "<details><summary>Unfused ONNX and native AMD controls</summary>"
    for model in ["attention", "heptv2"]:
        body += f'<p><a href="physics-summary/{model}-controls.pdf">PDF</a></p><img src="physics-summary/{model}-controls.png" alt="{model} unfused and native controls">'
    body += "</details><details><summary>Physics methods and detailed evidence</summary><p>Production particle decoding; anti-kT R=0.4 jets with pT &gt;3 GeV. Spectra are not independently normalized to unit area. Reference bin uncertainty is sqrt(N × sample variance of per-event bin counts), preserving correlations among objects in an event. Paired intervals use the retained 400 event-bootstrap resamples. Small systematic shifts may be resolved by pairing even when smaller than the reference sample fluctuations. Sparse bins remain in the spectra; ratio panels and the σ summary assess populated bins only. Underflow, overflow and sparse-bin information are retained in the JSON.</p>"
    body += '<p><a href="physics-summary/summary.json">Spectral statistics and departing bins</a> · <a href="physics-summary/provenance.json">Source hashes</a>. '
    body += " · ".join(f'<a href="analysis/{row["name"]}/report.html">{html.escape(row["name"])}</a>' for row in result["rows"])
    body += "</p></details><h2>Latency</h2><p>Synchronous batch-one CPU inputs to CPU FP32 outputs, including transfers and conversion. Timings exclude warmup, compilation, file I/O and physics. The table gives event-weighted mean latency; plots retain every event and tail.</p>"
    body += "<table><tr><th>Model / GPU</th><th>Native ms</th><th>ONNX fused ms</th><th>ONNX unfused ms</th><th>Fused speedup</th></tr>"
    for model in ["attention", "heptv2"]:
        for device in ["cuda", "amd"]:
            records = [r for r in result["runtimes"] if r["model"] == model and r["device"] == device]
            native, on, off = records
            label = ("Attention" if model == "attention" else "HEPTv2") + " / " + ("L40S" if device == "cuda" else "W7900")
            values = [label, f'{native["mean_ms"]:.2f}', f'{on["mean_ms"]:.2f}', f'{off["mean_ms"]:.2f}', f'{on["speedup_vs_native_pytorch"]:.2f}×']
            body += "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>"
    body += "</table>"
    if (root / "runtime-plots" / "summary.json").exists():
        body += '<p>Event-size plots count valid input elements; dotted lines mark static ONNX buckets. Dots show events, lines show medians, and bands show the 10th–90th percentile spread in 250-element bins (at least 20 events). <a href="runtime-plots/report.html">Timing details, PDFs and per-event CSV</a>.</p>'
        for name in ["latency-distributions", "latency-vs-event-size"]:
            body += f'<img src="runtime-plots/{name}.png" alt="{name}">'
    body += '<details><summary>Technical diagnostics and reproducibility</summary><p><a href="qualification-summary.json">Full numerical diagnostics and timings</a> · <a href="fusion-verification.json">Fusion and GPU-placement checks</a> · <a href="bf16-rounding-audit.json">BF16 rounding audit</a> · <a href="release.json">Checkpoints</a> · <a href="input-manifest.json">Events</a> · <a href="README.txt">Archive layout</a> · <a href="SHA256SUMS">Checksums</a>.</p></details></html>'
    (root / "report.html").write_text(body)


def main(root):
    cases = json.loads((root / "input-manifest.json").read_text())["cases"]
    event_count = len(cases)
    bucket_counts = Counter(case["slots"] for case in cases)
    expected_shapes = 8 * len(bucket_counts)
    rows, runtimes = [], []
    for model in ["attention", "heptv2"]:
        for device in ["cuda", "amd"]:
            native = performance(root, model, device, "pytorch")
            runtimes.append(dict(model=model, device=device, engine="PyTorch BF16 native fused", **native))
            for fusion in ["on", "off"]:
                perf = performance(root, model, device, fusion)
                runtimes.append(
                    dict(
                        model=model,
                        device=device,
                        engine=f"ONNX BF16 fusion {fusion}",
                        speedup_vs_native_pytorch=native["mean_ms"] / perf["mean_ms"],
                        **perf,
                    )
                )
            names = [f"{model}-{device}-{f}" for f in ["on", "off", "fused-vs-unfused"]]
            if device == "amd":
                names.append(f"{model}-amd-pytorch-vs-nvidia")
            for name in names:
                data = json.loads((root / "analysis" / name / "summary.json").read_text())
                assert data["events"] == event_count and data["full_sample"]
                numeric = data["numerical"]
                migration = data["particle_pid_migration"]
                reference_counts = dict(
                    particle_count=sum(sum(row) for row in migration[1:]),
                    jet_count=data["jet_matching"]["reference_count"],
                    jet30_count=data["jet30_matching"]["reference_count"],
                )
                relative_counts = {
                    key: dict(
                        reference_total=total,
                        delta_total=data["paired_event_shifts"][key]["sum"],
                        delta_percent=100 * data["paired_event_shifts"][key]["sum"] / total if total else None,
                    )
                    for key, total in reference_counts.items()
                }
                rows.append(
                    dict(
                        name=name,
                        model=model,
                        device=device,
                        events=data["events"],
                        numerical_passed=data["numerical_passed"],
                        max_abs=max(v["max_abs"] for v in numeric.values()),
                        numerical=numeric,
                        presence_flips=numeric["bid"]["argmax_changes"],
                        pid_flips=numeric["id"]["argmax_changes"],
                        paired_event_shifts=data["paired_event_shifts"],
                        relative_counts=relative_counts,
                        jet_matching=data["jet_matching"],
                        jet30_matching=data["jet30_matching"],
                    )
                )
    fusion = json.loads((root / "fusion-verification.json").read_text())
    assert fusion["verified_shapes"] == expected_shapes
    result = dict(
        complete=True,
        numerical_passed=all(row["numerical_passed"] for row in rows),
        events=event_count,
        completed_analyses=len(rows),
        expected_analyses=14,
        rows=rows,
        runtimes=runtimes,
        numerical_tolerance=dict(atol=1e-3, rtol=3e-3),
        physics_acceptance="Report-only; no formal physics pass/fail assigned",
        fusion_verified_shapes=expected_shapes,
        baseline="NVIDIA PyTorch BF16 SDPA Flash for Attention; compiled FlexAttention with FP32 LSE for HEPTv2",
    )
    (root / "qualification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    physics_path = root / "physics-summary" / "summary.json"
    if physics_path.exists():
        physics = json.loads(physics_path.read_text())
        assert physics["events"] == event_count
        physics_report(root, result, physics)
        print(json.dumps(dict(complete=True, physics_report=True, analyses=len(rows), fusion_verified_shapes=expected_shapes)))
        return
    body = """<!doctype html><html lang="en"><meta charset="utf-8"><title>CMS BF16 export qualification</title>
<style>body{font:15px system-ui;margin:2rem;color:#18202b;max-width:1800px}p{max-width:1200px;line-height:1.5}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}a{color:#165b9e}</style>
<h1>CMS v3.2.0 BF16 ONNX qualification · PACKAGE_NAME</h1>
<p>Both step-100000 checkpoints, the same EVENT_COUNT unique CMS 2.8.0 ttbar test events. SAMPLE_DESCRIPTION. No truncation or repeats. The common reference is NVIDIA L40S PyTorch BF16: forced SDPA Flash with production jagged packing for Attention; compiled FlexAttention with FP32 LSE for HEPTv2. AMD W7900 native PyTorch BF16 is also evaluated independently.</p>
<p>Full-model GPU BF16 autocast; FP32 LayerNorm, original FP32 input geometry and exact int64 HEPT hashing, FP32 external I/O. This differs from the earlier attention-only FP16 cast profiles. TF32 is disabled. ONNX BF16 predicates are widened exactly to FP32; ELU uses FP32 opmath and one BF16 rounding; BF16 constants use raw bits for MIGraphX parser compatibility. Stable max-shifted FP32 log-sum-exp prevents the previously diagnosed AMD overflow. Checkpoint weights are unchanged.</p>
<p>These are BF16-typed exports, not a guarantee of identical compiler rounding. The <a href="bf16-rounding-audit.json">classification-head audit</a> finds exact BF16 values in both native PyTorch baselines and NVIDIA ONNX, but off-lattice values in AMD ONNX. A <a href="bf16-rounding-probe.log">small AMD operator test</a> reproduces missing BF16 rounding without attention: BF16 addition followed by FP32 output returns 1.003006 where BF16 rounding gives 1.0, including with datatype elimination disabled. The <a href="probe-bf16-rounding.py">reproducer</a> and final compiler programs retain this backend discrepancy; numerical tolerances are unchanged.</p>
<h2>Runtime</h2><p>Batch-one synchronous CPU inputs to CPU FP32 outputs, including transfers and output conversion, excluding file I/O, physics, warmup and compilation. All EVENT_COUNT events are timed once after six warmups per bucket, without profiling. Means are weighted by the actual event sample; p50 and p95 describe this mixed-shape sample. Runtime is not peak throughput. Native PyTorch uses a persistent autocast weight cache. Cold session creation and first run, compilation warmup and per-bucket timing are retained in the aggregate JSON. No concurrent GPU workloads were used during timing.</p>"""
    body += (
        "<table><tr><th>Model</th><th>Device</th><th>Engine</th><th>Mean ms</th><th>p50 ms</th><th>p95 ms</th><th>Speedup vs native PyTorch</th></tr>"
    )
    for row in runtimes:
        vals = [
            row["model"],
            row["device"],
            row["engine"],
            f"{row['mean_ms']:.3f}",
            f"{row['p50_ms']:.3f}",
            f"{row['p95_ms']:.3f}",
            f"{row.get('speedup_vs_native_pytorch',1):.2f}×",
        ]
        body += "<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in vals) + "</tr>"
    body += "</table>"
    if (root / "runtime-plots" / "summary.json").exists():
        body += "<h2>Latency distributions and event-size dependence</h2><p>All measured event timings, after warmup. Histograms use shared logarithmic latency bins and include every recorded tail. The event-size plot uses valid input element counts; dotted vertical lines mark static ONNX bucket boundaries. Dots show individual events; lines show medians and bands show the 10th–90th percentile spread in 250-element bins with at least 20 events. Sparse bins retain their individual dots. Logarithmic latency axes keep slow-event tails visible. Bands describe event spread, not uncertainty on the median.</p>"
        body += '<p><a href="runtime-plots/report.html">Runtime plot details</a> · <a href="runtime-plots/per-event-latency.csv">Per-event latency CSV</a></p>'
        for name, title in [("latency-distributions", "Latency distributions"), ("latency-vs-event-size", "Latency versus event size")]:
            body += f'<p><a href="runtime-plots/{name}.pdf">{title} · PDF</a></p><img style="width:100%;max-width:1400px" src="runtime-plots/{name}.png" alt="{title}">'
    body += """<h2>Numerical and physics closure</h2><p>Count percentages are 100 × (total candidate − total reference) / total reference; absolute totals are retained in the aggregate JSON. <a href="relative-counts/report.html">Relative count distributions and particle gains/losses</a>. ONNX on/off and AMD PyTorch rows use the common NVIDIA PyTorch reference. Fused-vs-unfused rows use the same-device unfused ONNX reference, isolating fusion at BF16 precision. Numerical tolerance is unchanged: atol=1e-3, rtol=3e-3. Physics is report-only, with 400 paired event-bootstrap resamples and pointwise 95% histogram intervals. No physics acceptance is inferred from these results.</p><p>Production particle decoding; FastJet anti-kT R=0.4 inclusive jets pT&gt;3 GeV and hard jets pT≥30 GeV, inclusive and ispu&lt;0.8 collections. Maximum-cardinality, minimum-angle one-to-one matching uses ΔR&lt;0.1. Tiny particle differences can change threshold decisions and jet clustering; inspect unmatched counts, matched response tails and event outliers alongside inclusive distributions. Targets and CMS PF candidates provide context. Histograms retain underflow/overflow; ratios omit empty reference bins.</p>"""
    body += (
        "<table><tr>"
        + "".join(
            "<th>" + x + "</th>"
            for x in [
                "Comparison",
                "Numerical",
                "Max abs",
                "Binary argmax flips",
                "PID argmax flips",
                "Δ particles [%]",
                "Δ jets >3 [%]",
                "Δ jets ≥30 [%]",
                "Mean ΔMET [95% CI], GeV",
                "Max |ΔMET|, GeV",
                "Evidence",
            ]
        )
        + "</tr>"
    )
    for row in rows:
        s = row["paired_event_shifts"]
        met = s["met"]
        vals = [
            row["name"],
            "PASS" if row["numerical_passed"] else "FAIL",
            f"{row['max_abs']:.5g}",
            row["presence_flips"],
            row["pid_flips"],
            *[
                f"{row['relative_counts'][key]['delta_percent']:+.5f}" if row["relative_counts"][key]["delta_percent"] is not None else "undefined"
                for key in ["particle_count", "jet_count", "jet30_count"]
            ],
            f"{met['mean_candidate_minus_reference']:.4g} [{met['mean_ci95'][0]:.4g}, {met['mean_ci95'][1]:.4g}]",
            f"{met['max_abs_event_difference']:.4g}",
        ]
        body += (
            "<tr>"
            + "".join("<td>" + html.escape(str(v)) + "</td>" for v in vals)
            + f'<td><a href="analysis/{row["name"]}/report.html">Plots</a> · <a href="analysis/{row["name"]}/summary.json">JSON</a> · <a href="analysis/{row["name"]}/met-outliers.json">MET tails</a></td></tr>'
        )
    body += """</table><p>Fusion evidence covers all FUSION_SHAPES model/device/mode/shape combinations. NVIDIA selects BF16 FLASH_ATTENTION or MATH explicitly, with debug dispatch evidence from the exact same static graph hashes. AMD final compiler programs contain three BF16 fused attention code objects when enabled and none when disabled; Attention global quadratic buffers disappear when fused. HEPTv2 fusion is partial: a separate FP32 QK/log-sum-exp branch is still materialized for bucket merging because standard ONNX Attention does not expose LSE. This limits its performance relative to native compiled FlexAttention.</p>
<p><a href="qualification-summary.json">Aggregate results and timings</a> · <a href="fusion-verification.json">Fusion evidence</a> · <a href="release.json">Checkpoint hashes</a> · <a href="release-revision.json">Pinned release revision</a> · <a href="input-manifest.json">Unique test events</a>. Raw heads, graphs, placement profiles, compiler programs, PyTorch kernel traces, source snapshots, failed attempts and checksums are retained alongside the report.</p></html>"""
    sample_description = (
        f"Valid lengths {min(c['length'] for c in cases)}–{max(c['length'] for c in cases)}; static bucket event counts "
        + ", ".join(f"{b}: {n}" for b, n in sorted(bucket_counts.items()))
    )
    for key, value in {
        "PACKAGE_NAME": root.name,
        "EVENT_COUNT": event_count,
        "SAMPLE_DESCRIPTION": sample_description,
        "FUSION_SHAPES": expected_shapes,
    }.items():
        body = body.replace(key, html.escape(str(value)))
    (root / "report.html").write_text(body)
    print(json.dumps(dict(complete=True, analyses=len(rows), fusion_verified_shapes=expected_shapes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
