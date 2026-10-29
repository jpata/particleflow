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
