"""Build and submit Slurm jobs for reusable training scenarios."""

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


def resolve_flatiron_profile_path(reference, repo_root):
    path = Path(reference).expanduser()
    if path.is_file():
        return path.resolve()
    name = path.stem
    if not name.startswith("flatiron_"):
        name = f"flatiron_{name}"
    candidate = repo_root / "configs/training/platforms" / f"{name}.yaml"
    if not candidate.is_file():
        raise ValueError(f"Unknown Flatiron accelerator/profile {reference!r}")
    return candidate.resolve()


def available_choices(repo_root):
    scenarios = sorted(
        path.stem for path in (repo_root / "configs/training/scenarios").glob("*.yaml")
    )
    accelerators = sorted(
        path.stem.removeprefix("flatiron_")
        for path in (repo_root / "configs/training/platforms").glob("flatiron_*.yaml")
    )
    return scenarios, accelerators


def build_slurm_submission(
    scenario_path,
    profile_path,
    repo_root,
    *,
    seed=None,
):
    scenario = load_training_scenario(scenario_path)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        scenario.seeds = [seed]
    profile = load_platform_profile(profile_path)
    if profile.slurm is None:
        raise ValueError(
            f"Platform profile {profile.name!r} has no Slurm configuration"
        )

    spec_file = Path(scenario.spec_file)
    if not spec_file.is_absolute():
        spec_file = repo_root / spec_file
    jobs = resolve_scenario_jobs(scenario, profile, spec_file=spec_file)
    if not jobs:
        raise ValueError("Scenario did not resolve to any jobs")

    slurm = profile.slurm
    logs_dir = repo_root / "logs_slurm"
    worker = repo_root / "scripts/flatiron/run_uv_scenario.sh"
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
        "--gpus-per-node",
        str(profile.gpus),
        "--cpus-per-task",
        str(slurm.cpus_per_task),
        "--constraint",
        slurm.constraint,
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
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command, jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="Scenario name or YAML path")
    parser.add_argument(
        "accelerator",
        nargs="?",
        help="Accelerator name (for example h100) or profile path",
    )
    parser.add_argument("--seed", type=int, help="Replace the scenario seed list")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without submitting",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenarios and accelerators"
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    scenarios, accelerators = available_choices(repo_root)
    if args.list or args.scenario is None or args.accelerator is None:
        print("Scenarios: " + ", ".join(scenarios))
        print("Accelerators: " + ", ".join(accelerators))
        if args.list:
            return
        parser.error("scenario and accelerator are required")

    scenario_path = resolve_scenario_path(args.scenario, repo_root)
    profile_path = resolve_flatiron_profile_path(args.accelerator, repo_root)
    command, jobs = build_slurm_submission(
        scenario_path,
        profile_path,
        repo_root,
        seed=args.seed,
    )
    print(
        f"Submitting {len(jobs)} jobs for {args.scenario} on {args.accelerator}:\n"
        + shlex.join(command),
        flush=True,
    )
    if args.dry_run:
        return

    (repo_root / "logs_slurm").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
