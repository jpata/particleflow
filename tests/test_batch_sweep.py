import sys
from pathlib import Path

from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)
from scripts.training.batch_sweep import case_scenario, classify, execute, main

ROOT = Path(__file__).resolve().parents[1]


def test_batch_case_preserves_matched_models_and_resolves_measurement_settings(monkeypatch):
    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    base = load_training_scenario(ROOT / "configs/training/scenarios/cld_gh200_backbone_benchmark.yaml")
    profile = load_platform_profile(ROOT / "configs/training/platforms/empireai_gh200.yaml")
    case = case_scenario(base, 32, 120, 20)
    jobs = resolve_scenario_jobs(case, profile, spec_file=ROOT / case.spec_file)
    assert len(jobs) == 2
    assert base.training.global_batch_size == 8
    assert base.training.parameters["num_steps"] == 1000
    assert all(job.global_batch_size == job.per_gpu_batch_size == 32 for job in jobs)
    assert jobs[0].resolved_config.model.set_decoder == jobs[1].resolved_config.model.set_decoder
    for job in jobs:
        config = job.resolved_config
        assert config.num_steps == config.checkpoint_freq == config.val_freq == 120
        assert config.benchmark.enabled and config.benchmark.warmup_steps == 20
        assert config.task_loss_weights.calibration_steps == 20


def test_failure_classification():
    assert classify(0, "") == "completed"
    assert classify(1, "torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert classify(1, "ImportError") == "error"
    assert classify(-15, "", timed_out=True) == "timeout"


def test_case_timeout_terminates_isolated_process(tmp_path):
    code, timed_out = execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "case.log", 0.1)
    assert timed_out and code != 0


def test_dry_run_resolves_all_cases_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    monkeypatch.chdir(ROOT)
    output = tmp_path / "sweep"
    main(
        [
            "--scenario",
            "configs/training/scenarios/cld_gh200_backbone_benchmark.yaml",
            "--platform",
            "configs/training/platforms/empireai_gh200.yaml",
            "--output",
            str(output),
            "--batch-sizes",
            "8",
            "16",
            "--dry-run",
        ]
    )
    assert not output.exists()


def test_shared_defaults_and_frozen_spec(tmp_path, monkeypatch):
    import json
    import yaml
    from scripts.training import batch_sweep as sweep

    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    monkeypatch.setenv("MLPF_BATCH_SWEEP_OUTPUT", str(tmp_path / "sweep"))
    monkeypatch.setenv("MLPF_BATCH_SWEEP_SCENARIO", str(ROOT / "configs/training/scenarios/cld_backbone_benchmark.yaml"))
    monkeypatch.setattr(sweep, "git_state", lambda: {"revision": "abc", "dirty": False})
    invoked = []

    def execute_stub(command, log, timeout):
        invoked.append(command)
        log.write_text("CUDA out of memory" if len(invoked) == 1 else "finished")
        return (1 if len(invoked) == 1 else 0), False

    monkeypatch.setattr(sweep, "execute", execute_stub)
    sweep.main(["--platform", str(ROOT / "configs/training/platforms/empireai_gh200.yaml"), "--batch-sizes", "8", "--require-clean-git"])
    output = tmp_path / "sweep"
    state = json.loads((output / "sweep-state.json").read_text())
    assert state["status"] == "completed"
    assert [case["status"] for case in state["cases"]] == ["oom", "completed"]
    assert state["data_verification"]["status"] == "not_checked"
    assert state["git"]["revision"] == "abc"
    assert (output / "particleflow_spec.yaml").read_bytes() == (ROOT / "particleflow_spec.yaml").read_bytes()
    frozen = yaml.safe_load(next((output / "cases").glob("cld*.yaml")).read_text())
    assert frozen["spec_file"] == str(output / "particleflow_spec.yaml")
    assert len((output / "memory-diagnostics.jsonl").read_text().splitlines()) >= 2
    assert len(invoked) == 2


def test_data_preflight_matches_config_and_fails_before_training(tmp_path):
    import json
    from types import SimpleNamespace
    import pytest
    from scripts.training.batch_sweep import verify_data

    config = {"train_dataset": {"cld_hits": {"physical": {"samples": {"cld_edm_ttbar_hits": {"version": "3.2.1", "splits": ["1"]}}}}}}
    job = SimpleNamespace(data_dir=str(tmp_path), resolved_config=SimpleNamespace(model_dump=lambda **_: config))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"relative_directory": "cld_edm_ttbar_hits/1/3.2.2", "files": {}, "metadata_sha256": {}}))
    with pytest.raises(ValueError, match="match every"):
        verify_data([(None, job)], manifest)
    config["train_dataset"]["cld_hits"]["physical"]["samples"]["cld_edm_ttbar_hits"]["version"] = "3.2.2"
    manifest.write_text(json.dumps({"relative_directory": "cld_edm_ttbar_hits/1/3.2.2", "files": {"shard": 4}, "metadata_sha256": {}}))
    with pytest.raises(ValueError, match="missing"):
        verify_data([(None, job)], manifest)
    (tmp_path / "cld_edm_ttbar_hits/1/3.2.2").mkdir(parents=True)
    (tmp_path / "cld_edm_ttbar_hits/1/3.2.2/shard").write_bytes(b"data")
    assert verify_data([(None, job)], manifest)["status"] == "verified"


def test_strict_source_check_rejects_dirty_tree_before_creating_output(tmp_path, monkeypatch):
    import pytest
    from scripts.training import batch_sweep as sweep

    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    monkeypatch.setattr(sweep, "git_state", lambda: {"dirty": True})
    with pytest.raises(SystemExit):
        sweep.main(
            [
                "--platform",
                str(ROOT / "configs/training/platforms/empireai_gh200.yaml"),
                "--output",
                str(tmp_path / "sweep"),
                "--batch-sizes",
                "8",
                "--require-clean-git",
            ]
        )
    assert not (tmp_path / "sweep").exists()


def test_platform_cannot_silently_change_shared_precision_or_warmup(monkeypatch):
    import pytest
    from scripts.training.batch_sweep import validate_protocol_overrides

    monkeypatch.setenv("MLPF_CLD_DATA_DIR", "/test/tfds")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_backbone_benchmark.yaml")
    case = case_scenario(scenario, 16, 120, 20)
    profile = load_platform_profile(ROOT / "configs/training/platforms/empireai_gh200.yaml")
    validate_protocol_overrides(case, profile)
    for key, value in [("dtype", "float32"), ("benchmark.warmup_steps", 2), ("benchmark.enabled", False)]:
        changed = profile.model_copy(deep=True)
        changed.runtime_overrides[key] = value
        with pytest.raises(ValueError, match="shared benchmark setting"):
            validate_protocol_overrides(case, changed)
