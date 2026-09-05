"""Build and submit site-specific Slurm jobs for reusable training scenarios."""

import argparse
import shlex
import subprocess
from pathlib import Path

from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)


def resolve_scenario_path(reference, repo_root):
    path = Path(reference).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    candidate = repo_root / "configs/training/scenarios" / path.name
    if not candidate.is_file():
        raise ValueError(f"Unknown training scenario {reference!r}")
    return candidate.resolve()


def resolve_platform_profile_path(reference, repo_root, site):
    path = Path(reference).expanduser()
    if path.is_file():
        return path.resolve()
    name = path.stem
    prefix = f"{site}_"
    if not name.startswith(prefix):
        name = f"{prefix}{name}"
    candidate = repo_root / "configs/training/platforms" / f"{name}.yaml"
    if not candidate.is_file():
        raise ValueError(f"Unknown {site} accelerator/profile {reference!r}")
    return candidate.resolve()


def resolve_flatiron_profile_path(reference, repo_root):
    """Backward-compatible Flatiron profile resolver."""
    return resolve_platform_profile_path(reference, repo_root, "flatiron")


def available_choices(repo_root, site="flatiron"):
    scenarios = sorted(path.stem for path in (repo_root / "configs/training/scenarios").glob("*.yaml"))
    accelerators = sorted(path.stem.removeprefix(f"{site}_") for path in (repo_root / "configs/training/platforms").glob(f"{site}_*.yaml"))
    return scenarios, accelerators


def _worker_for_site(repo_root, site):
    relative_paths = {
        "flatiron": "scripts/flatiron/run_uv_scenario.sh",
        "tallinn": "scripts/tallinn/run_scenario.sh",
        "lumi": "scripts/lumi/run_scenario.sh",
    }
    try:
        return repo_root / relative_paths[site]
    except KeyError as exc:
        raise ValueError(f"No scenario worker is configured for site {site!r}") from exc


def _gpu_request_args(slurm, gpus):
    if slurm.gpu_request == "gpus-per-node":
        return ["--gpus-per-node", str(gpus)]
    if slurm.gpu_request == "gpus-per-task":
        return ["--gpus-per-task", str(gpus)]
    resource = f"gpu:{gpus}"
    if slurm.gpu_type:
        resource = f"gpu:{slurm.gpu_type}:{gpus}"
    return ["--gres", resource]


def build_slurm_submission(
    scenario_path,
    profile_path,
    repo_root,
    *,
    seed=None,
    worker=None,
):
    scenario = load_training_scenario(scenario_path)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        scenario.seeds = [seed]
    profile = load_platform_profile(profile_path)
    if profile.slurm is None:
        raise ValueError(f"Platform profile {profile.name!r} has no Slurm configuration")

    spec_file = Path(scenario.spec_file)
    if not spec_file.is_absolute():
        spec_file = repo_root / spec_file
    jobs = resolve_scenario_jobs(scenario, profile, spec_file=spec_file)
    if not jobs:
        raise ValueError("Scenario did not resolve to any jobs")

    slurm = profile.slurm
    logs_dir = repo_root / "logs_slurm"
    worker = Path(worker) if worker is not None else _worker_for_site(repo_root, "flatiron")
    command = [
        "sbatch",
        "--time",
        slurm.time,
        "--nodes",
        str(slurm.nodes),
        "--ntasks-per-node",
        str(slurm.tasks_per_node),
        "--partition",
        slurm.partition,
        *_gpu_request_args(slurm, profile.gpus),
        "--cpus-per-task",
        str(slurm.cpus_per_task),
    ]
    if slurm.constraint:
        command.extend(["--constraint", slurm.constraint])
    if slurm.account:
        command.extend(["--account", slurm.account])
    if slurm.memory:
        command.extend(["--mem", slurm.memory])
    if slurm.memory_per_gpu:
        command.extend(["--mem-per-gpu", slurm.memory_per_gpu])
    if slurm.no_requeue:
        command.append("--no-requeue")
    command.extend(
        [
            "--array",
            f"0-{len(jobs) - 1}",
            "--job-name",
            scenario.name,
            "--output",
            str(logs_dir / "log_%x_%A_%a.out"),
            "--error",
            str(logs_dir / "log_%x_%A_%a.err"),
            "--chdir",
            str(repo_root),
            str(worker),
            str(Path(scenario_path).resolve()),
            str(Path(profile_path).resolve()),
            "--repo-root",
            str(repo_root),
        ]
    )
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command, jobs


def main(argv=None, *, site="flatiron"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="Scenario name or YAML path")
    parser.add_argument(
        "accelerator",
        nargs="?",
        help="Accelerator name or platform profile path",
    )
    parser.add_argument("--seed", type=int, help="Replace the scenario seed list")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without submitting",
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and accelerators")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    scenarios, accelerators = available_choices(repo_root, site)
    if args.list or args.scenario is None or args.accelerator is None:
        print("Scenarios: " + ", ".join(scenarios))
        print("Accelerators: " + ", ".join(accelerators))
        if args.list:
            return
        parser.error("scenario and accelerator are required")

    scenario_path = resolve_scenario_path(args.scenario, repo_root)
    profile_path = resolve_platform_profile_path(args.accelerator, repo_root, site)
    command, jobs = build_slurm_submission(
        scenario_path,
        profile_path,
        repo_root,
        seed=args.seed,
        worker=_worker_for_site(repo_root, site),
    )
    print(
        f"Submitting {len(jobs)} jobs for {args.scenario} on {args.accelerator}:\n" + shlex.join(command),
        flush=True,
    )
    if args.dry_run:
        return

    (repo_root / "logs_slurm").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
