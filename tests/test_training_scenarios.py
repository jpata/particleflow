from copy import deepcopy
from pathlib import Path

import pytest

from mlpf.training_scenarios import (
    PlatformProfile,
    ScenarioVariant,
    ScenarioTraining,
    _experiment_path,
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
    validate_variant_invariants,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/training/scenarios/cld_hits_output_comparison.yaml"
PLATFORMS = ROOT / "configs/training/platforms"


def test_comparison_scenario_resolves_both_output_modes_with_same_seed():
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == ["elementwise", "set"]
    assert {job.seed for job in jobs} == {12345}
    assert {job.gpu_batch_multiplier for job in jobs} == {8}
    assert {job.resolved_config.model.output_mode.value for job in jobs} == {
        "elementwise",
        "set",
    }
    assert all(job.resolved_config.seed == 12345 for job in jobs)


@pytest.mark.parametrize(
    ("profile_name", "expected_multiplier"),
    [
        ("flatiron_h100.yaml", 64),
        ("flatiron_a100.yaml", 128),
        ("flatiron_b200.yaml", 64),
        ("tallinn_l40.yaml", 256),
        ("lumi_mi250x.yaml", 64),
    ],
)
def test_platform_profiles_preserve_global_batch(profile_name, expected_multiplier):
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / profile_name)

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )

    assert {job.global_batch_size for job in jobs} == {512}
    assert {job.gpu_batch_multiplier for job in jobs} == {expected_multiplier}
    assert {job.per_gpu_batch_size for job in jobs} == {512 // platform.gpus}


def test_variant_invariant_check_rejects_unapproved_difference():
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )
    bad_jobs = deepcopy(jobs)
    bad_jobs[1].resolved_config.lr *= 2

    with pytest.raises(ValueError, match="variants differ.*lr"):
        validate_variant_invariants(bad_jobs, scenario.allowed_variant_differences)


def test_global_batch_must_be_divisible_by_hardware_layout():
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "flatiron_a100.yaml")

    with pytest.raises(ValueError, match="global_batch_size=130 is not divisible"):
        resolve_scenario_jobs(
            scenario,
            platform,
            spec_file=ROOT / "particleflow_spec.yaml",
            global_batch_size=130,
        )


def test_scenario_and_platform_reject_misplaced_settings():
    with pytest.raises(ValueError, match="derived keys"):
        ScenarioTraining(
            global_batch_size=8,
            parameters={"gpu_batch_multiplier": 8},
        )

    with pytest.raises(ValueError, match="runtime-specific"):
        PlatformProfile(
            name="bad",
            gpus=1,
            data_dir="/tmp/data",
            experiments_dir="/tmp/experiments",
            runtime_overrides={"lr": 0.1},
        )

    with pytest.raises(ValueError, match="derived keys"):
        ScenarioVariant(
            model_name="pyg-cld-hits-v1",
            overrides={"seed": 17},
        )


def test_cli_seed_replaces_scenario_seed_for_task_selection(capsys):
    from mlpf.training_scenarios import main

    main(
        [
            "--scenario",
            str(SCENARIO),
            "--platform",
            str(PLATFORMS / "local.yaml"),
            "--spec-file",
            str(ROOT / "particleflow_spec.yaml"),
            "--global-batch-size",
            "8",
            "--seed",
            "17",
            "--task-index",
            "1",
            "--dry-run",
        ]
    )

    command = capsys.readouterr().out
    assert "--seed 17" in command
    assert "--model-name pyg-cld-hits-set-v1" in command


def test_experiments_are_grouped_under_the_scenario_directory():
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    job = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )[0]

    path = _experiment_path(platform, job, timestamp="TIMESTAMP")

    assert path == Path(
        "experiments/cld_hits_output_comparison/elementwise_seed12345_TIMESTAMP"
    )
