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
BACKBONE_SCENARIO = ROOT / "configs/training/scenarios/cld_hits_backbone_comparison.yaml"
PF_HITS_SCENARIO = ROOT / "configs/training/scenarios/cld_pf_hits_comparison.yaml"
CLIC_CLD_SCENARIO = ROOT / "configs/training/scenarios/clic_cld_pf_set_hits_comparison.yaml"
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


def test_backbone_comparison_scenario_keeps_elementwise_output_and_depth_fixed():
    scenario = load_training_scenario(BACKBONE_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == ["attention", "heptv2"]
    assert [job.resolved_config.model.type.value for job in jobs] == ["attention", "heptv2"]
    assert {job.resolved_config.model.output_mode.value for job in jobs} == {"elementwise"}
    assert {job.resolved_config.model.backbone.num_convs for job in jobs} == {6}
    assert jobs[1].resolved_config.model.heptv2.block_size == 128


def test_pf_hits_comparison_scenario_resolves_three_40k_variants():
    scenario = load_training_scenario(PF_HITS_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == ["pf", "elementwise_hits", "set_hits"]
    assert [job.model_name for job in jobs] == ["pyg-cld-v1", "pyg-cld-hits-v1", "pyg-cld-hits-set-v1"]
    assert [job.resolved_config.dataset.value for job in jobs] == ["cld", "cld_hits", "cld_hits"]
    assert [job.resolved_config.model.output_mode.value for job in jobs] == ["elementwise", "elementwise", "set"]
    assert [job.resolved_config.model.binary_classification_focal_gamma for job in jobs] == [None, 2.0, 2.0]
    assert {job.resolved_config.model.backbone.num_convs for job in jobs} == {6}
    assert [
        (
            job.resolved_config.model.backbone.num_tracker_layers,
            job.resolved_config.model.backbone.num_calo_layers,
            job.resolved_config.model.backbone.num_common_layers,
        )
        for job in jobs
    ] == [(None, None, None), (2, 2, 2), (2, 2, 2)]
    assert {job.resolved_config.num_steps for job in jobs} == {40000}
    assert {job.resolved_config.val_freq for job in jobs} == {5000}
    assert {job.resolved_config.checkpoint_freq for job in jobs} == {5000}
    assert {job.resolved_config.lr for job in jobs} == {0.001}
    assert {job.global_batch_size for job in jobs} == {8}
    assert {job.seed for job in jobs} == {12345}


def test_clic_cld_scenario_resolves_pf_and_set_hits_per_detector():
    scenario = load_training_scenario(CLIC_CLD_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == ["cld_pf", "cld_set_hits", "clic_pf", "clic_set_hits"]
    assert [job.model_name for job in jobs] == ["pyg-cld-v1", "pyg-cld-hits-set-v1", "pyg-clic-v1", "pyg-clic-hits-set-v1"]
    assert [job.production_name for job in jobs] == ["cld", "cld", "clic", "clic"]
    assert [job.data_dir for job in jobs] == [
        platform.data_dir["cld"],
        platform.data_dir["cld"],
        platform.data_dir["clic"],
        platform.data_dir["clic"],
    ]
    assert [job.resolved_config.data_dir for job in jobs] == [job.data_dir for job in jobs]
    assert [job.resolved_config.dataset.value for job in jobs] == ["cld", "cld_hits", "clic", "clic_hits"]
    assert [job.resolved_config.model.output_mode.value for job in jobs] == ["elementwise", "set", "elementwise", "set"]
    # The set-based hit models run twice the backbone depth of the PF models.
    assert [job.resolved_config.model.backbone.num_convs for job in jobs] == [6, 12, 6, 12]
    assert [
        (
            job.resolved_config.model.backbone.num_tracker_layers,
            job.resolved_config.model.backbone.num_calo_layers,
            job.resolved_config.model.backbone.num_common_layers,
        )
        for job in jobs
    ] == [(None, None, None), (4, 4, 4), (None, None, None), (4, 4, 4)]
    assert {job.resolved_config.model.set_decoder.num_layers for job in jobs if job.resolved_config.model.set_decoder} == {8}
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.resolved_config.val_freq for job in jobs} == {5000}
    assert {job.resolved_config.lr for job in jobs} == {0.001}
    assert {job.seed for job in jobs} == {12345}


def test_platform_data_dir_mapping_requires_the_variant_production():
    scenario = load_training_scenario(CLIC_CLD_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    platform.data_dir = {"cld": platform.data_dir["cld"]}

    with pytest.raises(ValueError, match="no data_dir for production 'clic'"):
        resolve_scenario_jobs(
            scenario,
            platform,
            spec_file=ROOT / "particleflow_spec.yaml",
            global_batch_size=8,
        )

    platform.data_dir = "/tmp/shared_tfds"
    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )
    assert {job.data_dir for job in jobs} == {"/tmp/shared_tfds"}


def test_cli_dry_run_uses_per_production_data_dir(capsys):
    from mlpf.training_scenarios import main

    main(
        [
            "--scenario",
            str(CLIC_CLD_SCENARIO),
            "--platform",
            str(PLATFORMS / "local.yaml"),
            "--spec-file",
            str(ROOT / "particleflow_spec.yaml"),
            "--global-batch-size",
            "8",
            "--variant",
            "clic_set_hits",
            "--dry-run",
        ]
    )

    command = capsys.readouterr().out
    assert "--production-name clic" in command
    assert "--data-dir /mnt/work/mlpf/clic/v1.2.5_key4hep_2025-05-29/tfds" in command
    assert "--model.backbone.num_convs 12" in command


@pytest.mark.parametrize(
    ("profile_name", "expected_multiplier"),
    [
        ("flatiron_h100.yaml", 64),
        ("flatiron_a100.yaml", 128),
        ("flatiron_h200.yaml", 64),
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

    assert path == Path("experiments/cld_hits_output_comparison/elementwise_seed12345_TIMESTAMP")
