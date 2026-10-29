"""Run isolated training cases for a hardware batch-size sweep.

Use a GPU allocation and the platform's Python environment. OOM is an observed
capacity limit; other failures make the overall command fail after saving results.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from mlpf.benchmark_report import main as report_main
from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)


def case_scenario(base, batch_size, steps, warmup):
    case = base.model_copy(deep=True)
    case.name = f"{base.name}_batch{batch_size}"
    case.training.global_batch_size = batch_size
    case.training.parameters.update(
        {
            "num_steps": steps,
            "patience": steps,
            "checkpoint_freq": steps,
            "val_freq": steps,
            "nvalid": 32,
            "ntest": 8,
            "tensorboard_step_freq": 20,
            "task_loss_weights.calibration_steps": max(1, warmup),
            "benchmark.enabled": True,
            "benchmark.warmup_steps": warmup,
            "benchmark.gpu_timing_interval": 10,
        }
    )
    return case


def classify(returncode, log, timed_out=False):
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "completed"
    if "out of memory" in log.lower():
        return "oom"
    return "error"


def execute(command, log_path, timeout):
    timed_out = False
    with log_path.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return returncode, timed_out


def save_state(output, state):
    temporary = output / "sweep-state.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(output / "sweep-state.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--case-timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= args.warmup or args.warmup < 0 or args.case_timeout <= 0 or any(batch <= 0 for batch in args.batch_sizes):
        parser.error("Require positive batches/timeout and steps > warmup >= 0")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("Batch sizes must be unique")
    base = load_training_scenario(args.scenario)
    platform = load_platform_profile(args.platform)
    cases = []
    for batch in args.batch_sizes:
        case = case_scenario(base, batch, args.steps, args.warmup)
        jobs = resolve_scenario_jobs(case, platform, spec_file=case.spec_file)
        for job in jobs:
            cases.append((case, job))
    print(f"Resolved {len(cases)} cases: batches={args.batch_sizes}, steps={args.steps}, warmup={args.warmup}", flush=True)
    if args.dry_run:
        for case, job in cases:
            print(case.name, job.variant_name, job.seed, "per-GPU batch", job.per_gpu_batch_size)
        return
    output = args.output.resolve()
    if (output / "sweep-state.json").exists():
        parser.error("Output already contains a sweep; use a new output directory")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    state = {
        "schema_version": 1,
        "scenario": str(args.scenario.resolve()),
        "platform": str(args.platform.resolve()),
        "steps_per_case": args.steps,
        "warmup_steps": args.warmup,
        "purpose": "throughput/memory sweep; event exposure differs across batch sizes",
        "cases": [],
    }
    save_state(output, state)
    unexpected_failure = False
    for case, job in cases:
        case_path = output / "cases" / f"{case.name}.yaml"
        case_path.write_text(yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False))
        # Freeze the platform too, including resolved paths and benchmark settings.
        profile_path = output / "cases" / "platform.yaml"
        profile_path.write_text(yaml.safe_dump(platform.model_dump(mode="json"), sort_keys=False))
        case_id = f"{job.variant_name}_batch{job.global_batch_size}_seed{job.seed}"
        log_path = output / "logs" / f"{case_id}.log"
        command = [
            sys.executable,
            "scripts/training/run_scenario.py",
            "--scenario",
            str(case_path),
            "--platform",
            str(profile_path),
            "--variant",
            job.variant_name,
            "--seed",
            str(job.seed),
            "--experiments-dir",
            str(output / "runs"),
        ]
        record = {
            "case": case_id,
            "variant": job.variant_name,
            "global_batch_size": job.global_batch_size,
            "seed": job.seed,
            "status": "running",
            "command": command,
            "log": str(log_path),
            "started_at": time.time(),
        }
        state["cases"].append(record)
        save_state(output, state)
        print("Starting", case_id, flush=True)
        returncode, timed_out = execute(command, log_path, args.case_timeout)
        record.update(status=classify(returncode, log_path.read_text(errors="replace"), timed_out), returncode=returncode, finished_at=time.time())
        unexpected_failure |= record["status"] not in ("completed", "oom")
        save_state(output, state)
        print("Finished", case_id, record["status"], flush=True)
        if list((output / "runs").glob("**/benchmark/*/summary.json")):
            report_main([str(output / "runs")])
    state["status"] = "failed" if unexpected_failure else "completed"
    save_state(output, state)
    if unexpected_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
