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


@pytest.mark.parametrize("profile_name", ["empireai_gh200", "lumi_mi250x_single", "flatiron_h100_single"])
def test_cross_platform_profiles_resolve_single_device_cases(profile_name, monkeypatch):
    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    profile = load_platform_profile(ROOT / f"configs/training/platforms/{profile_name}.yaml")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_gh200_backbone_benchmark.yaml")
    for batch in (8, 16, 32, 64, 128):
        jobs = resolve_scenario_jobs(case_scenario(scenario, batch, 120, 20), profile, spec_file=ROOT / scenario.spec_file)
        assert len(jobs) == 2
        assert all(job.resolved_config.gpus == 1 and job.per_gpu_batch_size == batch for job in jobs)
        assert all(job.resolved_config.dtype == "bfloat16" and job.resolved_config.num_workers == 4 for job in jobs)
        assert jobs[0].resolved_config.model.set_decoder == jobs[1].resolved_config.model.set_decoder
        assert profile.python_command == ["python3"]


def shell_environment(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("BASH_FUNC_")}
    env.update(
        PATH=f"{stub}:/usr/bin:/bin", BASH_ENV="/dev/null", MLPF_REPO_ROOT=str(ROOT), SLURM_JOB_ID="123", MIOPEN_USER_DB_PATH=str(tmp_path / "miopen")
    )
    return stub, env


def executable(path, body):
    path.write_text("#!/bin/bash\nset -e\n" + body + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize("site,profile", [("lumi", "lumi_mi250x_single"), ("flatiron", "flatiron_h100_single")])
def test_site_worker_binds_one_gpu_and_forwards_sweep_args(tmp_path, site, profile):
    stub, env = shell_environment(tmp_path)
    executable(stub / "srun", 'printf "%s\\n" "$@"')
    env["MLPF_BATCH_SWEEP_OUTPUT"] = str(tmp_path / "output")
    result = subprocess.run(
        ["bash", str(ROOT / f"scripts/{site}/run_batch_sweep.sh"), "--batch-sizes", "8", "16"], env=env, check=True, capture_output=True, text=True
    )
    assert "--gpus-per-task=1" in result.stdout
    assert "--cpu-bind=cores" in result.stdout
    assert f"configs/training/platforms/{profile}.yaml" in result.stdout
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
