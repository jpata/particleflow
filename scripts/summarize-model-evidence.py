"""Aggregate retained model evidence, reject missing cells, and plot scaling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

OPTIMIZED_PROFILES = {"attention24-fp16": "l40s", "attention-decomposed-fp16": "w7900"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--require-complete", action="store_true", help="Fail unless every expected cell passes")
    a = p.parse_args()
    rows, missing, failed = [], [], []
    expected_cells = 0
    matrix = [(machine, model) for machine in ["l40s", "w7900", "cpu-l40s-host"] for model in ["attention", "heptv2"]]
    for profile, machine in OPTIMIZED_PROFILES.items():
        if (a.root / profile / "manifest.json").exists():
            matrix.append((machine, profile))
    for machine, model in matrix:
        manifest = json.loads((a.root / model / "manifest.json").read_text())
        for case in manifest["cases"]:
            expected_cells += 1
            base = (a.root / "supplementary" / model if model in OPTIMIZED_PROFILES else a.root / "results" / machine / model) / Path(
                case["file"]
            ).stem
            paths = [base / mode / "result.json" for mode in ["timing", "placement"]]
            if not all(path.exists() for path in paths):
                missing.append(str(base.relative_to(a.root)))
                continue
            timing, placement = [json.loads(path.read_text()) for path in paths]
            expected_provider = {"l40s": "cuda", "w7900": "migraphx", "cpu-l40s-host": "cpu"}[machine]
            ok = (
                timing["passed"]
                and placement["passed"]
                and not timing["profiling_enabled"]
                and placement["profiling_enabled"]
                and timing["provider"] == placement["provider"] == expected_provider
                and timing["case"] == placement["case"] == case
                and len(timing["latency_ms"]["samples"]) == 100
                and timing["warmup"] == 10
                and timing["model_sha256"] == placement["model_sha256"] == manifest["model_sha256"]
                and placement["placement"]["full_provider_placement"]
            )
            if model == "attention24-fp16":
                trace = a.root / "kernel-evidence/attention24-kernel-summary-bound.csv"
                ok &= timing.get("flash_requested", False) and placement.get("flash_requested", False)
                ok &= trace.exists() and "onnxruntime::flash::flash_fwd_kernel" in trace.read_text()
            if model == "attention-decomposed-fp16":
                proof_path = base / "placement/fusion-verification.json"
                program = base / "placement/compiled-program.txt"
                traces = list((a.root / "kernel-evidence/w7900-fused").rglob("*kernel_trace.csv"))
                ok &= any("reduce_max" in path.read_text() and "reduce_sum" in path.read_text() for path in traces)
                ok &= timing["compiler_environment"].get("MIGRAPHX_MLIR_USE_SPECIFIC_OPS") == "attention"
                ok &= placement["compiler_environment"].get("MIGRAPHX_MLIR_USE_SPECIFIC_OPS") == "attention"
                ok &= manifest["export_profile"] == {"attention_format": "decomposed", "attention_dtype": "float16"}
                if proof_path.exists() and program.exists():
                    proof = json.loads(proof_path.read_text())
                    ok &= proof["verified"] and proof["model_sha256"] == manifest["model_sha256"]
                    ok &= proof["case"] == case["file"]
                    ok &= proof["fused_attention_code_objects"] == proof["expected_attention_operations"] > 0
                    ok &= not proof["quadratic_attention_allocations"]
                    ok &= hashlib.sha256(program.read_bytes()).hexdigest() == proof["program_sha256"]
                    static = a.root / "static-models" / (proof["static_model_sha256"] + ".onnx")
                    ok &= static.exists() and hashlib.sha256(static.read_bytes()).hexdigest() == proof["static_model_sha256"]
                else:
                    ok = False
            if not ok:
                failed.append(str(base.relative_to(a.root)))
            mem = timing["memory"]
            gpu = mem["inference_peak_vram_bytes"]
            gpu_base = mem["baseline"]["vram_bytes"]
            rows.append(
                {
                    "machine": machine,
                    "model": model,
                    "batch": case["shape"][0],
                    "slots_per_event": case["shape"][1],
                    "valid_inputs": sum(case["lengths"]),
                    "passed": ok,
                    "p50_ms": timing["latency_ms"]["p50"],
                    "p95_ms": timing["latency_ms"]["p95"],
                    "session_seconds": timing["session_seconds"],
                    "first_run_seconds": timing["first_run_seconds"],
                    "max_abs": max(e["max_abs"] for e in timing["outputs"].values()),
                    "max_relative_l2": max(e["relative_l2"] for e in timing["outputs"].values()),
                    "inference_vram_mib": gpu / 2**20 if gpu is not None else None,
                    "inference_vram_delta_mib": (gpu - gpu_base) / 2**20 if gpu is not None else None,
                    "inference_rss_mib": mem["inference_peak_rss_bytes"] / 2**20,
                    "inference_rss_delta_mib": (mem["inference_peak_rss_bytes"] - mem["baseline"]["rss_bytes"]) / 2**20,
                    "session_peak_rss_mib": mem["session_peak_rss_bytes"] / 2**20 if mem["session_peak_rss_bytes"] else None,
                }
            )
    (a.root / "summary.json").write_text(
        json.dumps(
            {
                "complete": not missing,
                "passed": not missing and not failed,
                "expected_cells": expected_cells,
                "completed_cells": len(rows),
                "missing": missing,
                "failed": failed,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if rows:
        with (a.root / "measurements.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if a.plots and rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for sweep in ["sequence", "batch"]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for machine in ["l40s", "w7900", "cpu-l40s-host"]:
                for model, style in [("attention", "-"), ("heptv2", "--"), ("attention24-fp16", "-."), ("attention-decomposed-fp16", ":")]:
                    selected = [
                        r
                        for r in rows
                        if r["machine"] == machine
                        and r["model"] == model
                        and (r["batch"] == 1 if sweep == "sequence" else r["slots_per_event"] == 1000)
                    ]
                    selected.sort(key=lambda r: r["slots_per_event"] if sweep == "sequence" else r["batch"])
                    x = [r["slots_per_event"] if sweep == "sequence" else r["batch"] for r in selected]
                    for ax, key in zip(axes, ["p50_ms", "inference_vram_delta_mib", "inference_rss_mib"]):
                        points = [(v, r[key]) for v, r in zip(x, selected) if r[key] is not None]
                        if points:
                            ax.plot(*zip(*points), style, marker="o", label=f"{model} / {machine}")
            for ax, title in zip(axes, ["Median session.run latency (ms)", "Sampled VRAM above baseline (MiB)", "Sampled process RSS (MiB)"]):
                ax.set_title(title)
                ax.set_xlabel("Input slots per event (B=1)" if sweep == "sequence" else "Batch size (N=1000)")
                ax.grid(alpha=0.3)
            axes[0].set_yscale("log")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
            fig.suptitle("Seeded full models · FP32 baseline + labeled FP16 profiles · host transfers included")
            fig.tight_layout(rect=[0, 0.15, 1, 0.94])
            for suffix in ["png", "pdf", "svg"]:
                fig.savefig(a.root / f"{sweep}-scaling.{suffix}", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for machine in ["l40s", "w7900", "cpu-l40s-host"]:
            for model, style in [("attention", "-"), ("heptv2", "--"), ("attention24-fp16", "-."), ("attention-decomposed-fp16", ":")]:
                selected = sorted(
                    [r for r in rows if r["machine"] == machine and r["model"] == model and r["batch"] == 1], key=lambda r: r["slots_per_event"]
                )
                if not selected:
                    continue
                for ax, key in zip(axes, ["max_abs", "max_relative_l2"]):
                    ax.plot([r["slots_per_event"] for r in selected], [r[key] for r in selected], style, marker="o", label=f"{model} / {machine}")
        for ax, title in zip(axes, ["Maximum absolute error across heads", "Maximum relative L2 error across heads"]):
            ax.set_title(title)
            ax.set_xlabel("Input slots per event (B=1)")
            ax.grid(alpha=0.3)
            ax.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle("Export outputs versus eager CPU PyTorch FP32 · valid inputs")
        fig.tight_layout(rect=[0, 0.15, 1, 0.94])
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(a.root / f"numerical-scaling.{suffix}", dpi=160)
        plt.close(fig)
    print(json.dumps({"completed": len(rows), "missing": len(missing), "failed": failed}))
    if a.require_complete and (missing or failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
