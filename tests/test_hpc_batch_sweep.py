import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)
from scripts.training.batch_sweep import case_scenario
from scripts.training.check_batch_data import check_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workers,python_command", [(0, ["python3"]), (3, ["uv", "run", "python3"])])
def test_batch_cases_use_platform_runtime_settings(training_files, workers, python_command):
    from mlpf.training_scenarios import _pipeline_command

    scenario_path, profile_path = training_files
    scenario = load_training_scenario(scenario_path)
    profile = load_platform_profile(profile_path)
    profile.runtime_overrides["num_workers"] = workers
    profile.python_command = python_command
    case = case_scenario(scenario, 32, 11, 2)
    jobs = resolve_scenario_jobs(case, profile, spec_file=case.spec_file)
    for job in jobs:
        assert job.resolved_config.gpus == profile.gpus
        assert job.per_gpu_batch_size * profile.gpus == 32
        assert job.resolved_config.num_workers == workers
        assert job.resolved_config.dtype == scenario.training.parameters["dtype"]
        command = _pipeline_command(job, case, profile, Path(case.spec_file), Path("experiment"))
        assert command[: len(python_command) + 1] == [*python_command, "-u"]


def shell_environment(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("BASH_FUNC_")}
    env.update(
        PATH=f"{stub}:/usr/bin:/bin",
        BASH_ENV="/dev/null",
        USER="testuser",
        MLPF_REPO_ROOT=str(ROOT),
        SLURM_JOB_ID="123",
        MIOPEN_USER_DB_PATH=str(tmp_path / "miopen"),
    )
    return stub, env


def executable(path, body):
    path.write_text("#!/bin/bash\nset -e\n" + body + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize("site", ["lumi", "flatiron", "empireai"])
def test_site_worker_binds_one_gpu_and_forwards_sweep_args(tmp_path, site):
    stub, env = shell_environment(tmp_path)
    executable(stub / "srun", 'printf "%s\\n" "$@"')
    env["MLPF_BATCH_SWEEP_OUTPUT"] = str(tmp_path / "output")
    result = subprocess.run(
        ["bash", str(ROOT / f"scripts/{site}/run_batch_sweep.sh"), "--batch-sizes", "8", "16"], env=env, check=True, capture_output=True, text=True
    )
    assert "--ntasks=1" in result.stdout
    assert "--cpu-bind=cores" in result.stdout
    args = result.stdout.splitlines()
    profile_path = ROOT / args[args.index("--platform") + 1]
    assert load_platform_profile(profile_path).gpus == 1
    assert "scripts.training.batch_sweep" in result.stdout
    assert result.stdout.splitlines()[-3:] == ["--batch-sizes", "8", "16"]


def test_lumi_python_helper_preserves_single_gcd_visibility_and_image(tmp_path):
    stub, env = shell_environment(tmp_path)
    executable(stub / "module", "exit 0")
    executable(stub / "singularity", 'printf "VISIBLE=%s\\n" "$ROCR_VISIBLE_DEVICES"; printf "%s\\n" "$@"')
    image = tmp_path / "image with spaces.sif"
    image.touch()
    overlay = tmp_path / "overlay with spaces"
    (overlay / "bin").mkdir(parents=True)
    (overlay / "bin/activate").touch()
    env.update(IMG=str(image), LUMI_VENV=str(overlay), ROCR_VISIBLE_DEVICES="3")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/lumi/batch_python.sh"), "payload.py", "--dry-run"], env=env, check=True, capture_output=True, text=True
    )
    assert "VISIBLE=3" in result.stdout
    assert str(image) in result.stdout.splitlines()
    assert result.stdout.splitlines()[-2:] == ["payload.py", "--dry-run"]


def test_flatiron_python_helper_uses_prepared_uv_environment(tmp_path):
    stub, env = shell_environment(tmp_path)
    executable(stub / "module", "exit 0")
    executable(stub / "uv", 'printf "%s\\n" "$@"')
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/flatiron/batch_python.sh"), "payload.py", "--dry-run"], env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.splitlines() == ["run", "--no-sync", "python3", "payload.py", "--dry-run"]


def test_data_inventory_detects_missing_sizes_and_metadata_changes(tmp_path):
    directory = tmp_path / "dataset/1/3.2.2"
    directory.mkdir(parents=True)
    (directory / "shard").write_bytes(b"data")
    (directory / "metadata.json").write_bytes(b"{}")
    manifest = {
        "relative_directory": "dataset/1/3.2.2",
        "files": {"shard": 4, "metadata.json": 2},
        "metadata_sha256": {"metadata.json": hashlib.sha256(b"{}").hexdigest()},
    }
    assert not check_data(tmp_path, manifest)
    (directory / "shard").write_bytes(b"x")
    (directory / "metadata.json").write_bytes(b"[]")
    errors = check_data(tmp_path, manifest)
    assert any("size mismatch" in error for error in errors)
    assert any("checksum mismatch" in error for error in errors)
    (directory / "shard").unlink()
    assert any("missing" in error for error in check_data(tmp_path, manifest))


def test_generic_submission_uses_profile_resources_and_preserves_arguments(training_files):
    from scripts.training.submit_batch_sweep import build_submission

    _, path = training_files
    command = build_submission(path, sweep_args=["--output", "/output with spaces", "--batch-sizes", "16"])
    resolved = load_platform_profile(path)
    assert command[command.index("--partition") + 1] == resolved.slurm.partition
    assert command[command.index("--cpus-per-task") + 1] == str(resolved.slurm.cpus_per_task)
    assert command[-4:] == ["--output", "/output with spaces", "--batch-sizes", "16"]
    assert ("--no-requeue" in command) == resolved.slurm.no_requeue


def test_new_platform_needs_no_site_dispatch(tmp_path):
    import yaml
    from scripts.training.submit_batch_sweep import build_submission

    profile = tmp_path / "future.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "name": "future_device",
                "gpus": 1,
                "data_dir": "/data",
                "experiments_dir": "/runs",
                "slurm": {"partition": "new_partition", "gpu_request": "gres", "gpu_type": "new_gpu", "cpus_per_task": 3},
            }
        )
    )
    command = build_submission(profile)
    assert command[command.index("--gres") + 1] == "gpu:new_gpu:1"
    assert command[command.index("--partition") + 1] == "new_partition"
    assert command[-3:] == ["-", "--platform", str(profile)]


def test_memory_monitor_stops_when_case_raises(tmp_path):
    import threading
    from scripts.training.monitor_batch_memory import monitor_memory

    with pytest.raises(RuntimeError):
        with monitor_memory(tmp_path / "memory.jsonl", interval=0.01):
            raise RuntimeError("failed case")
    assert not any(t.name == "mlpf-sweep-memory" for t in threading.enumerate())
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) >= 2
