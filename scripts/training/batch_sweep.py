"""Run isolated training cases for a hardware batch-size sweep.

Use a GPU allocation and the platform's Python environment. OOM is an observed
capacity limit; other failures make the overall command fail after saving results.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from mlpf.benchmark_report import main as report_main
from scripts.training.check_batch_data import check_data
from scripts.training.monitor_batch_memory import monitor_memory
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
            "benchmark.sample_interval_seconds": 1.0,
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = ROOT / "configs/training/scenarios/cld_backbone_benchmark.yaml"


def validate_protocol_overrides(case, platform):
    """A platform may choose implementation details, but cannot silently change timing/precision."""
    protected = {key: value for key, value in case.training.parameters.items() if key.startswith("benchmark.") or key in {"dtype", "compile"}}
    for source, overrides in [
        (platform.name, platform.runtime_overrides),
        *((name, variant.overrides) for name, variant in case.variants.items()),
        ("common_overrides", case.common_overrides),
    ]:
        for key in protected.keys() & overrides.keys():
            if overrides[key] != protected[key]:
                raise ValueError(f"{source} overrides shared benchmark setting {key}; change the scientific recipe explicitly")


def git_state(root=ROOT):
    """Record provenance even for development runs; strict comparisons can reject dirt."""
    try:

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)

        return {
            "revision": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(git("status", "--porcelain")),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"unavailable": True}


def verify_data(cases, manifest_path):
    if manifest_path is None:
        return {"status": "not_checked", "scope": "No data manifest supplied"}
    manifest = json.loads(manifest_path.read_text())
    roots = set()
    for _, job in cases:
        config = job.resolved_config.model_dump(mode="json")
        selected = {
            f"{name}/{split}/{sample['version']}"
            for groups in config["train_dataset"].values()
            for group in groups.values()
            for name, sample in group["samples"].items()
            for split in sample["splits"]
        }
        if selected != {manifest["relative_directory"]}:
            raise ValueError("Data manifest must match every selected training dataset/config/version")
        roots.add(job.data_dir)
    for root in roots:
        errors = check_data(root, manifest)
        if errors:
            raise ValueError("Dataset preflight failed: " + "; ".join(errors[:10]))
    return {
        "status": "verified",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset": manifest["relative_directory"],
        "scope": "Filenames, sizes and metadata hashes only; shard payload checksums not checked",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=os.environ.get("MLPF_BATCH_SWEEP_SCENARIO", DEFAULT_SCENARIO))
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=os.environ.get("MLPF_BATCH_SWEEP_OUTPUT"))
    parser.add_argument("--data-dir", help="Override the profile TFDS root for every variant")
    parser.add_argument("--data-manifest", type=Path, help="Require matching dataset inventory before any training")
    parser.add_argument("--require-clean-git", action="store_true", help="Reject missing or dirty source provenance")
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
    if args.data_dir:
        platform.data_dir = args.data_dir
    if platform.gpus != 1:
        parser.error("Batch sweeps require a single-device platform profile")
    spec_file = Path(base.spec_file)
    if not spec_file.is_absolute():
        spec_file = ROOT / spec_file
    base.spec_file = str(spec_file.resolve())
    cases = []
    for batch in args.batch_sizes:
        case = case_scenario(base, batch, args.steps, args.warmup)
        validate_protocol_overrides(case, platform)
        jobs = resolve_scenario_jobs(case, platform, spec_file=case.spec_file)
        for job in jobs:
            if job.resolved_config.gpus != 1:
                parser.error("Every batch-sweep variant must use one visible device")
            cases.append((case, job))
    print(f"Resolved {len(cases)} cases: batches={args.batch_sizes}, steps={args.steps}, warmup={args.warmup}", flush=True)
    if args.dry_run:
        for case, job in cases:
            print(case.name, job.variant_name, job.seed, "per-GPU batch", job.per_gpu_batch_size)
        return
    source = git_state()
    if args.require_clean_git and (source.get("unavailable") or source.get("dirty")):
        parser.error("A clean Git checkout is required for this sweep")
    verification = verify_data(cases, args.data_manifest)
    default_name = f"{platform.name}_batch_sweep_{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d_%H%M%S'))}"
    output = (args.output or Path(platform.experiments_dir) / default_name).resolve()
    if (output / "sweep-state.json").exists():
        parser.error("Output already contains a sweep; use a new output directory")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    # A frozen spec prevents later source edits from changing later cases in this sweep.
    spec_bytes = spec_file.read_bytes()
    (output / "particleflow_spec.yaml").write_bytes(spec_bytes)
    for case, _ in cases:
        case.spec_file = str(output / "particleflow_spec.yaml")
    if args.data_manifest:
        (output / "data-manifest.json").write_bytes(args.data_manifest.read_bytes())
    state = {
        "schema_version": 2,
        "git": source,
        "data_verification": verification,
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "scenario": str(args.scenario.resolve()),
        "platform": str(args.platform.resolve()),
        "steps_per_case": args.steps,
        "warmup_steps": args.warmup,
        "purpose": "throughput/memory sweep; event exposure differs across batch sizes",
        "cases": [],
    }
    print(f"Batch sweep output: {output}", flush=True)
    save_state(output, state)
    with monitor_memory(output / "memory-diagnostics.jsonl"):
        run_cases(output, state, cases, platform, args.case_timeout)


def run_cases(output, state, cases, platform, case_timeout):
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
            str(ROOT / "scripts/training/run_scenario.py"),
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
        returncode, timed_out = execute(command, log_path, case_timeout)
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
