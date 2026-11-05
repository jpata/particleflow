import sys

import pytest
from pathlib import Path

from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)
from scripts.training.batch_sweep import case_scenario, classify, execute, main


@pytest.mark.parametrize("batch,steps,warmup", [(16, 11, 0), (32, 23, 3)])
def test_batch_case_preserves_recipe_and_resolves_measurement_settings(training_files, batch, steps, warmup):
    scenario_path, profile_path = training_files
    base = load_training_scenario(scenario_path)
    original = base.model_dump()
    case = case_scenario(base, batch, steps, warmup)
    assert base.model_dump() == original
    assert case.variants == base.variants
    assert case.common_overrides == base.common_overrides
    assert case.seeds == base.seeds
    jobs = resolve_scenario_jobs(case, load_platform_profile(profile_path), spec_file=case.spec_file)
    assert [job.variant_name for job in jobs] == list(base.variants)
    for job in jobs:
        config = job.resolved_config
        assert job.global_batch_size == job.per_gpu_batch_size == batch
        assert config.num_steps == config.checkpoint_freq == config.val_freq == steps
        assert config.benchmark.enabled and config.benchmark.warmup_steps == warmup
        assert config.task_loss_weights.calibration_steps == max(1, warmup)


def test_failure_classification():
    assert classify(0, "") == "completed"
    assert classify(1, "torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert classify(1, "ImportError") == "error"
    assert classify(-15, "", timed_out=True) == "timeout"


def test_case_timeout_terminates_isolated_process(tmp_path):
    code, timed_out = execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "case.log", 0.1)
    assert timed_out and code != 0


def test_dry_run_resolves_all_cases_without_creating_output(tmp_path, training_files, capsys):
    scenario_path, profile_path = training_files
    output = tmp_path / "sweep"
    main(
        [
            "--scenario",
            str(scenario_path),
            "--platform",
            str(profile_path),
            "--output",
            str(output),
            "--batch-sizes",
            "8",
            "16",
            "--dry-run",
        ]
    )
    assert not output.exists()
    assert "Resolved 4 cases" in capsys.readouterr().out


def test_environment_overrides_and_frozen_spec(tmp_path, monkeypatch, training_files):
    scenario_path, profile_path = training_files
    base = load_training_scenario(scenario_path)
    import json
    import yaml
    from scripts.training import batch_sweep as sweep

    monkeypatch.setenv("MLPF_BATCH_SWEEP_OUTPUT", str(tmp_path / "sweep"))
    monkeypatch.setenv("MLPF_BATCH_SWEEP_SCENARIO", str(scenario_path))
    monkeypatch.setattr(sweep, "git_state", lambda: {"revision": "abc", "dirty": False})
    invoked = []

    def execute_stub(command, log, timeout):
        invoked.append(command)
        log.write_text("CUDA out of memory" if len(invoked) == 1 else "finished")
        return (1 if len(invoked) == 1 else 0), False

    monkeypatch.setattr(sweep, "execute", execute_stub)
    sweep.main(["--platform", str(profile_path), "--batch-sizes", "8", "--require-clean-git"])
    output = tmp_path / "sweep"
    state = json.loads((output / "sweep-state.json").read_text())
    assert state["status"] == "completed"
    assert [case["status"] for case in state["cases"]] == ["oom", "completed"]
    assert state["data_verification"]["status"] == "not_checked"
    assert state["git"]["revision"] == "abc"
    assert (output / "particleflow_spec.yaml").read_bytes() == Path(base.spec_file).read_bytes()
    frozen = yaml.safe_load(next((output / "cases").glob(f"{base.name}_batch*.yaml")).read_text())
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


def test_strict_source_check_rejects_dirty_tree_before_creating_output(tmp_path, monkeypatch, training_files):
    scenario_path, profile_path = training_files
    monkeypatch.setenv("MLPF_BATCH_SWEEP_SCENARIO", str(scenario_path))
    import pytest
    from scripts.training import batch_sweep as sweep

    monkeypatch.setattr(sweep, "git_state", lambda: {"dirty": True})
    with pytest.raises(SystemExit):
        sweep.main(
            [
                "--platform",
                str(profile_path),
                "--output",
                str(tmp_path / "sweep"),
                "--batch-sizes",
                "8",
                "--require-clean-git",
            ]
        )
    assert not (tmp_path / "sweep").exists()


def test_platform_cannot_silently_change_shared_precision_or_warmup(training_files):
    scenario_path, profile_path = training_files
    import pytest
    from scripts.training.batch_sweep import validate_protocol_overrides

    scenario = load_training_scenario(scenario_path)
    case = case_scenario(scenario, 16, 120, 20)
    profile = load_platform_profile(profile_path)
    validate_protocol_overrides(case, profile)
    for key, value in [("dtype", "bfloat16"), ("benchmark.warmup_steps", 2), ("benchmark.enabled", False)]:
        changed = profile.model_copy(deep=True)
        changed.runtime_overrides[key] = value
        with pytest.raises(ValueError, match="shared benchmark setting"):
            validate_protocol_overrides(case, changed)
