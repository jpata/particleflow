"""Submit one serial benchmark sweep using resources from any platform profile."""

import argparse
from pathlib import Path
import shlex
import subprocess

from mlpf.training_scenarios import load_platform_profile
from mlpf.training_submission import slurm_resource_args

ROOT = Path(__file__).resolve().parents[2]


def build_submission(profile_path, python_wrapper=None, sweep_args=(), repo_root=ROOT):
    profile_path = Path(profile_path).resolve()
    profile = load_platform_profile(profile_path)
    slurm = profile.slurm
    if slurm is None:
        raise ValueError("A Slurm profile is required; use batch_sweep.py directly for local allocations")
    if profile.gpus != 1 or slurm.nodes != 1 or slurm.tasks_per_node != 1:
        raise ValueError("Batch sweeps require one node, one task and one visible device")
    wrapper = str(Path(python_wrapper).resolve()) if python_wrapper else "-"
    if python_wrapper and not Path(wrapper).is_file():
        raise ValueError(f"Python environment wrapper does not exist: {wrapper}")
    return [
        "sbatch",
        *slurm_resource_args(slurm, profile.gpus),
        "--job-name",
        f"{profile.name}_batch_sweep",
        "--output",
        str(repo_root / "logs_slurm/benchmark_%x_%j.log"),
        "--chdir",
        str(repo_root),
        str(repo_root / "scripts/training/run_batch_sweep.sh"),
        wrapper,
        "--platform",
        str(profile_path),
        *sweep_args,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--python-wrapper", type=Path, help="Optional shell script that prepares an environment and execs Python")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command without submitting")
    args, sweep_args = parser.parse_known_args(argv)
    if sweep_args[:1] == ["--"]:
        sweep_args = sweep_args[1:]
    command = build_submission(args.platform, args.python_wrapper, sweep_args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        (ROOT / "logs_slurm").mkdir(exist_ok=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
