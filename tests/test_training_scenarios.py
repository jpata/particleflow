import json
import shlex
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mlpf.training_scenarios import (
    PlatformProfile,
    ScenarioVariant,
    ScenarioTraining,
    _experiment_path,
    find_scenario_continuation,
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
    validate_variant_invariants,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform_name", ["local", "lumi_mi250x", "flatiron_h100"])
def test_clic_cld_benchmark_pairs_and_recommended_hit_recipe(platform_name):
    platform = load_platform_profile(ROOT / f"configs/training/platforms/{platform_name}.yaml")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/clic_cld_backbone_benchmark.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    reference = resolve_scenario_jobs(
        load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_default_50k.yaml"),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )[0].resolved_config
    by_name = {job.variant_name: job for job in jobs}
    assert len(jobs) == 8
    for job in jobs:
        config = job.resolved_config
        assert job.global_batch_size == 128
        assert config.num_steps == 50000 and config.seed == 12345
        assert config.val_freq == config.checkpoint_freq == 1000
        assert config.lr == 0.001 and config.load is None
        assert config.optimizer == reference.optimizer and config.lr_schedule == reference.lr_schedule
        assert config.model.backbone.num_convs == 6
        expected_gpus = 1 if "_pf_" in job.variant_name else platform.gpus
        assert config.gpus == expected_gpus
        assert job.per_gpu_batch_size == 128 // expected_gpus
    for detector in ("cld", "clic"):
        for representation in ("pf", "hits"):
            attention_job = by_name[f"{detector}_{representation}_attention"]
            heptv2_job = by_name[f"{detector}_{representation}_heptv2"]
            attention = attention_job.resolved_config
            heptv2 = heptv2_job.resolved_config
            assert attention.dataset.value == (detector if representation == "pf" else f"{detector}_hits")
            assert attention.model.type.value == "attention"
            assert heptv2.model.type.value == "heptv2"
            assert attention.model.attention.num_heads * attention.model.attention.head_dim == heptv2.model.heptv2.embedding_dim == 256
            # The broad campaign allowlist must not hide differences within a
            # backbone pair: data, objective and training budget stay identical.
            validate_variant_invariants(
                [attention_job, heptv2_job], ["conv_type", "model.type", "model.attention", "model.heptv2"]
            )
            if representation == "hits":
                expected_decoder = reference.model.set_decoder.model_copy(update={"tracker_momentum_readout": True})
                assert heptv2.model.set_decoder == expected_decoder
                assert heptv2.model.hit_feature_engineering == reference.model.hit_feature_engineering
                backbone = heptv2.model.heptv2.model_copy(update={"activation_checkpointing": False})
                assert backbone == reference.model.heptv2
            else:
                assert heptv2.model.output_mode.value == "elementwise"


def test_heptv2_model_scaling_preserves_baseline_except_size_and_checkpointing():
    platform = load_platform_profile(ROOT / "configs/training/platforms/lumi_mi250x.yaml")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_model_scaling_50k.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    default = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_default_50k.yaml")
    reference = resolve_scenario_jobs(default, platform, spec_file=ROOT / "particleflow_spec.yaml")[0].resolved_config
    assert len(jobs) == 3
    for job, (width, backbone_layers, decoder_layers) in zip(jobs, ((256, 6, 4), (384, 9, 6), (512, 12, 8))):
        config = job.resolved_config
        assert job.global_batch_size == 128 and job.per_gpu_batch_size == 16
        assert config.val_freq == config.checkpoint_freq == 1000
        assert config.num_steps == reference.num_steps == 50000
        assert config.lr == reference.lr and config.seed == reference.seed
        assert config.optimizer == reference.optimizer and config.lr_schedule == reference.lr_schedule
        assert config.model.heptv2.embedding_dim == config.model.heptv2.width == width
        assert config.model.backbone.num_convs == config.model.heptv2.num_convs == backbone_layers
        assert config.model.set_decoder.num_layers == decoder_layers
        assert width % config.model.heptv2.num_heads == 0
        assert width % config.model.set_decoder.num_heads == 0
        assert config.model.heptv2.activation_checkpointing
        model = config.model.model_dump()
        for field in ("embedding_dim", "width", "num_convs", "activation_checkpointing"):
            model["heptv2"][field] = getattr(reference.model.heptv2, field)
        model["backbone"]["num_convs"] = reference.model.backbone.num_convs
        model["set_decoder"]["num_layers"] = reference.model.set_decoder.num_layers
        assert model == reference.model.model_dump()


def test_tracker_momentum_ablation_changes_only_input_features():
    platform = load_platform_profile(ROOT / "configs/training/platforms/lumi_mi250x.yaml")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_tracker_momentum_50k.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    assert len(jobs) == 3
    flags = [job.resolved_config.model.hit_feature_engineering for job in jobs]
    assert [(f.tracker_momentum, f.tracker_doublets) for f in flags] == [(False, False), (True, False), (True, True)]
    assert all(job.global_batch_size == 384 and job.per_gpu_batch_size == 48 for job in jobs)
    assert all(job.resolved_config.model.heptv2.activation_checkpointing for job in jobs)


@pytest.mark.parametrize("platform_name", ["flatiron_h100", "lumi_mi250x"])
def test_heptv2_batch512_uses_reduced_batch_for_memory_headroom(platform_name):
    platform = load_platform_profile(ROOT / f"configs/training/platforms/{platform_name}.yaml")
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_batch512_50k.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    pf, hits = (job.resolved_config for job in jobs)
    assert all(job.global_batch_size == 384 and job.gpu_batch_multiplier == 48 for job in jobs)
    for config in (pf, hits):
        assert config.num_steps == 50000
        assert config.lr == 0.001
        assert config.optimizer.value == "lamb"
        assert config.lr_schedule.value == "cosinedecay"
        assert config.seed == 12345
        assert config.val_freq == config.checkpoint_freq == 5000
        assert config.model.backbone.num_convs == 6
    assert hits.model.heptv2.activation_checkpointing
    assert hits.model.hit_feature_engineering.tracker_neighborhood
    assert hits.model.set_decoder.hit_grouping
    assert not hits.model.set_decoder.tracker_group_readout
    assert not hits.model.set_decoder.tracker_fit_correction
    default_scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_default_50k.yaml")
    default = resolve_scenario_jobs(default_scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")[0].resolved_config
    assert not default.model.heptv2.activation_checkpointing
    default.model.heptv2.activation_checkpointing = True
    default.model.backbone.num_convs = 6
    assert hits.model.model_dump() == default.model.model_dump()


def test_cld_default_matches_winning_tracker_feature_configuration():
    platform = load_platform_profile(ROOT / "configs/training/platforms/local.yaml")
    configs = []
    for name in ("cld_pf_heptv2_default_50k", "cld_pf_heptv2_grouped_tracking_50k"):
        scenario = load_training_scenario(ROOT / f"configs/training/scenarios/{name}.yaml")
        jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml", global_batch_size=8)
        if name == "cld_pf_heptv2_default_50k":
            assert len(jobs) == 1
        configs.append(next(job.resolved_config for job in jobs if job.variant_name == "hit_heptv2_grouped_tracker_features"))
    default, winner = configs
    assert default.model.model_dump() == winner.model.model_dump()
    for field in ("lr", "num_steps", "val_freq", "checkpoint_freq", "nvalid", "ntest", "sampler_mode", "seed"):
        assert getattr(default, field) == getattr(winner, field)
    assert default.model.hit_feature_engineering.tracker_neighborhood
    assert default.model.set_decoder.hit_grouping
    assert not default.model.set_decoder.tracker_group_readout
    assert not default.model.set_decoder.tracker_fit_correction


def test_training_scenario_rejects_empty_variants():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_default_50k.yaml")
    payload = scenario.model_dump()
    payload["variants"] = {}
    with pytest.raises(ValueError, match="at least one variant"):
        type(scenario).model_validate(payload)


def test_fit_correction_scenario_resolves_three_controlled_variants():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_fit_correction_50k.yaml")
    platform = load_platform_profile(ROOT / "configs/training/platforms/local.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml", global_batch_size=8)
    assert len(jobs) == 3
    configs = [job.resolved_config for job in jobs]
    assert {c.num_steps for c in configs} == {50000}
    assert [c.model.set_decoder.tracker_group_readout for c in configs] == [False, True, True]
    assert [c.model.set_decoder.tracker_fit_correction for c in configs] == [False, False, True]
    assert configs[-1].model.set_decoder.tracker_fit_max_log_correction == pytest.approx(0.4054651081081644)


SCENARIO = ROOT / "configs/training/scenarios/cld_hits_output_comparison.yaml"
BACKBONE_SCENARIO = ROOT / "configs/training/scenarios/cld_hits_backbone_comparison.yaml"
PF_HITS_SCENARIO = ROOT / "configs/training/scenarios/cld_pf_hits_comparison.yaml"
CLIC_CLD_SCENARIO = ROOT / "configs/training/scenarios/clic_cld_pf_set_hits_comparison.yaml"
SET_IMPROVEMENT_SCENARIO = ROOT / "configs/training/scenarios/cld_set_hits_improvement_comparison.yaml"
SET_QUERY_MATCHING_SCENARIO = ROOT / "configs/training/scenarios/cld_set_hits_query_matching_comparison.yaml"
OBJECT_FORMATION_SCENARIO = ROOT / "configs/training/scenarios/cld_set_hits_object_formation_comparison.yaml"
HEPTV2_SECTOR_SCENARIO = ROOT / "configs/training/scenarios/cld_set_hits_heptv2_sector_comparison.yaml"
HEPTV2_MOMENTUM_SCENARIO = ROOT / "configs/training/scenarios/cld_pf_heptv2_momentum_50k_comparison.yaml"
HEPTV2_GROUPING_SCENARIO = ROOT / "configs/training/scenarios/cld_pf_heptv2_hit_grouping_50k.yaml"
PLATFORMS = ROOT / "configs/training/platforms"


def test_heptv2_momentum_scenario_isolates_50k_loss_and_matcher_changes():
    scenario = load_training_scenario(HEPTV2_MOMENTUM_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == [
        "track_cluster_mlpf",
        "hit_heptv2_baseline",
        "hit_heptv2_energy_match",
        "hit_heptv2_huber_momentum",
    ]
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.resolved_config.lr for job in jobs} == {0.001}
    assert {job.seed for job in jobs} == {12345}
    assert [job.resolved_config.model.set_decoder.matcher.energy for job in jobs[1:]] == [0.0, 0.5, 0.0]
    assert [job.resolved_config.model.set_decoder.momentum_loss for job in jobs[1:]] == ["mse", "mse", "huber"]


def test_heptv2_grouping_scenario_changes_only_hit_grouping():
    scenario = load_training_scenario(HEPTV2_GROUPING_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )
    assert [job.variant_name for job in jobs] == ["hit_heptv2_baseline", "hit_heptv2_grouped"]
    baseline, grouped = (job.resolved_config for job in jobs)
    assert not baseline.model.set_decoder.hit_grouping
    assert grouped.model.set_decoder.hit_grouping
    assert grouped.num_steps == baseline.num_steps == 50000


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
    assert [job.resolved_config.model.type.value for job in jobs] == [
        "attention",
        "heptv2",
    ]
    assert {job.resolved_config.model.output_mode.value for job in jobs} == {"elementwise"}
    assert {job.resolved_config.model.backbone.num_convs for job in jobs} == {6}
    assert jobs[1].resolved_config.model.heptv2.block_size == 128


def test_cms_backbone_scenario_pins_all_datasets_and_resolves_global_batch():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cms_backbone_comparison.yaml")
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    platform.data_dir = "/tmp/cms-tfds"
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")

    assert [job.resolved_config.model.type.value for job in jobs] == ["attention", "heptv2"]
    for job in jobs:
        config = job.resolved_config
        assert config.dataset.value == "cms"
        assert config.model.output_mode.value == "elementwise"
        assert config.model.backbone.num_convs == 3
        assert config.pad_to_multiple_elements == 100
        assert config.seed == 12345
        assert job.per_gpu_batch_size * platform.gpus == 64
        for datasets in (config.train_dataset["cms"], config.valid_dataset["cms"]):
            assert {dataset.batch_size for dataset in datasets.values()} == {1}
            assert sum(len(dataset.samples) for dataset in datasets.values()) == 6
            assert {sample.version for dataset in datasets.values() for sample in dataset.samples.values()} == {"2.8.0"}
            assert all(sample.splits == [str(i) for i in range(1, 11)] for dataset in datasets.values() for sample in dataset.samples.values())
        assert len(config.test_dataset) == 3
        assert {sample.version for sample in config.test_dataset.values()} == {"2.8.0"}
    from mlpf.conf import MLPFConfig

    default = MLPFConfig.from_spec(str(ROOT / "particleflow_spec.yaml"), "pyg-cms-v1", "cms_run3")
    attention = jobs[0].resolved_config.model.attention
    heptv2 = jobs[1].resolved_config.model.heptv2
    assert default.model.attention.num_heads == attention.num_heads == heptv2.num_heads == 16
    assert default.model.attention.head_dim == attention.head_dim == 32
    assert attention.num_heads * attention.head_dim == heptv2.embedding_dim == heptv2.width == 512
    assert attention.num_convs == heptv2.num_convs == 3
    assert attention.dropout_ff == attention.dropout_conv_id_ff == attention.dropout_conv_id_mha == heptv2.dropout_ff == 0.0
    assert jobs[1].resolved_config.model.heptv2.block_size == 100
    assert jobs[1].resolved_config.model.heptv2.deterministic_hashing


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
    assert [job.model_name for job in jobs] == [
        "pyg-cld-v1",
        "pyg-cld-hits-v1",
        "pyg-cld-hits-set-v1",
    ]
    assert [job.resolved_config.dataset.value for job in jobs] == [
        "cld",
        "cld_hits",
        "cld_hits",
    ]
    assert [job.resolved_config.model.output_mode.value for job in jobs] == [
        "elementwise",
        "elementwise",
        "set",
    ]
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

    assert [job.variant_name for job in jobs] == [
        "cld_pf",
        "cld_set_hits",
        "clic_pf",
        "clic_set_hits",
    ]
    assert [job.model_name for job in jobs] == [
        "pyg-cld-v1",
        "pyg-cld-hits-set-v1",
        "pyg-clic-v1",
        "pyg-clic-hits-set-v1",
    ]
    assert [job.production_name for job in jobs] == ["cld", "cld", "clic", "clic"]
    assert [job.data_dir for job in jobs] == [
        platform.data_dir["cld"],
        platform.data_dir["cld"],
        platform.data_dir["clic"],
        platform.data_dir["clic"],
    ]
    assert [job.resolved_config.data_dir for job in jobs] == [job.data_dir for job in jobs]
    assert [job.resolved_config.dataset.value for job in jobs] == [
        "cld",
        "cld_hits",
        "clic",
        "clic_hits",
    ]
    assert [job.resolved_config.model.output_mode.value for job in jobs] == [
        "elementwise",
        "set",
        "elementwise",
        "set",
    ]
    # The hit models split the same six-layer budget across detector branches.
    assert [job.resolved_config.model.backbone.num_convs for job in jobs] == [
        6,
        6,
        6,
        6,
    ]
    assert [
        (
            job.resolved_config.model.backbone.num_tracker_layers,
            job.resolved_config.model.backbone.num_calo_layers,
            job.resolved_config.model.backbone.num_common_layers,
        )
        for job in jobs
    ] == [(None, None, None), (2, 2, 2), (None, None, None), (2, 2, 2)]
    assert {job.resolved_config.model.set_decoder.num_layers for job in jobs if job.resolved_config.model.set_decoder} == {8}
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.resolved_config.val_freq for job in jobs} == {5000}
    assert {job.resolved_config.lr for job in jobs} == {0.001}
    assert {job.seed for job in jobs} == {12345}


def test_set_improvement_scenario_resolves_four_isolated_cld_variants():
    scenario = load_training_scenario(SET_IMPROVEMENT_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == [
        "control_8layer_local",
        "decoder4_local",
        "decoder8_global",
        "decoder8_kinematic_match",
    ]
    assert {job.model_name for job in jobs} == {"pyg-cld-hits-set-v1"}
    assert {job.resolved_config.dataset.value for job in jobs} == {"cld_hits"}
    assert {job.resolved_config.model.output_mode.value for job in jobs} == {"set"}
    assert {
        (
            job.resolved_config.model.backbone.num_tracker_layers,
            job.resolved_config.model.backbone.num_calo_layers,
            job.resolved_config.model.backbone.num_common_layers,
        )
        for job in jobs
    } == {(2, 2, 2)}
    assert [job.resolved_config.model.set_decoder.num_layers for job in jobs] == [8, 4, 8, 8]
    assert [job.resolved_config.model.set_decoder.local_attention_radius for job in jobs] == [0.4, 0.4, None, 0.4]
    assert [job.resolved_config.model.set_decoder.matcher.energy for job in jobs] == [0.0, 0.0, 0.0, 1.0]
    assert [job.resolved_config.model.set_decoder.matcher.presence for job in jobs] == [1.0, 1.0, 1.0, 0.0]
    assert [job.resolved_config.model.set_decoder.matcher.pid for job in jobs] == [1.0, 1.0, 1.0, 0.0]
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.seed for job in jobs} == {12345}


def test_set_query_matching_scenario_resolves_four_factorial_variants():
    scenario = load_training_scenario(SET_QUERY_MATCHING_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )

    assert [job.variant_name for job in jobs] == [
        "baseline_4layer_local",
        "calo_balanced_queries",
        "pid_free_match",
        "calo_balanced_pid_free_match",
    ]
    assert {job.model_name for job in jobs} == {"pyg-cld-hits-set-v1"}
    assert {job.resolved_config.dataset.value for job in jobs} == {"cld_hits"}
    assert {job.resolved_config.model.output_mode.value for job in jobs} == {"set"}
    assert {job.resolved_config.model.set_decoder.num_layers for job in jobs} == {4}
    assert {job.resolved_config.model.set_decoder.local_attention_radius for job in jobs} == {0.4}
    assert [job.resolved_config.model.set_decoder.tracker_query_fraction for job in jobs] == [0.6, 0.5, 0.6, 0.5]
    assert [job.resolved_config.model.set_decoder.matcher.presence for job in jobs] == [1.0, 1.0, 1.0, 1.0]
    assert [job.resolved_config.model.set_decoder.matcher.pid for job in jobs] == [1.0, 1.0, 0.0, 0.0]
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.resolved_config.val_freq for job in jobs} == {5000}
    assert {job.resolved_config.checkpoint_freq for job in jobs} == {1000}
    assert {job.seed for job in jobs} == {12345}


def test_object_formation_scenario_resolves_scalable_two_by_two_campaign():
    scenario = load_training_scenario(OBJECT_FORMATION_SCENARIO)
    platform = load_platform_profile(PLATFORMS / "lumi_mi250x.yaml")

    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )

    assert [job.variant_name for job in jobs] == [
        "baseline_topk_absolute",
        "grid_diverse_absolute",
        "topk_aggregate_anchors",
        "grid_diverse_aggregate_anchors",
    ]
    assert [
        (
            job.resolved_config.model.set_decoder.proposal_mode,
            job.resolved_config.model.set_decoder.use_aggregate_anchors,
        )
        for job in jobs
    ] == [
        ("topk", False),
        ("grid-diverse", False),
        ("topk", True),
        ("grid-diverse", True),
    ]
    assert {job.resolved_config.model.backbone.num_convs for job in jobs} == {6}
    assert {job.resolved_config.model.set_decoder.num_layers for job in jobs} == {4}
    assert {job.resolved_config.num_steps for job in jobs} == {50000}
    assert {job.resolved_config.checkpoint_freq for job in jobs} == {1000}
    assert {job.resolved_config.compile for job in jobs} == {False}
    assert {job.global_batch_size for job in jobs} == {512}
    assert {job.gpu_batch_multiplier for job in jobs} == {64}


def test_heptv2_sector_scenario_matches_grid_aggregate_baseline():
    platform = load_platform_profile(PLATFORMS / "lumi_mi250x.yaml")
    scenario = load_training_scenario(HEPTV2_SECTOR_SCENARIO)
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    baseline_scenario = load_training_scenario(OBJECT_FORMATION_SCENARIO)
    baseline_jobs = resolve_scenario_jobs(baseline_scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    previous_baseline = next(job for job in baseline_jobs if job.variant_name == "grid_diverse_aggregate_anchors")

    assert [job.variant_name for job in jobs] == [
        "attention_grid_diverse_aggregate",
        "heptv2_sector_grid_diverse_aggregate",
        "heptv2_global_grid_diverse_aggregate",
    ]
    # Keep the paired comparison aligned while lowering its memory footprint.
    previous_baseline.resolved_config.gpu_batch_multiplier = 8
    assert jobs[0].resolved_config.model_dump(mode="json") == previous_baseline.resolved_config.model_dump(mode="json")
    hept_global = jobs[2].resolved_config.model_dump(mode="json")
    hept_sector = jobs[1].resolved_config.model_dump(mode="json")
    hept_sector["model"]["set_decoder"]["attention_mode"] = "global-flash"
    assert hept_global == hept_sector
    assert [job.resolved_config.model.type.value for job in jobs] == ["attention", "heptv2", "heptv2"]
    assert [job.resolved_config.model.set_decoder.attention_mode for job in jobs] == ["global-flash", "sectorized", "global-flash"]
    assert {job.resolved_config.model.set_decoder.proposal_mode for job in jobs} == {"grid-diverse"}
    assert {job.resolved_config.model.set_decoder.use_aggregate_anchors for job in jobs} == {True}
    assert {job.resolved_config.model.backbone.num_convs for job in jobs} == {6}
    assert {job.resolved_config.model.set_decoder.num_layers for job in jobs} == {4}
    assert {job.resolved_config.pad_to_multiple_elements for job in jobs} == {128}
    assert {job.global_batch_size for job in jobs} == {64}
    assert {job.gpu_batch_multiplier for job in jobs} == {8}


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


def test_cli_dry_run_routes_selected_variant_to_its_production_data(tmp_path, capsys):
    from mlpf.training_scenarios import main

    first_data = tmp_path / "first-production"
    selected_data = tmp_path / "selected-production"
    scenario_path = tmp_path / "scenario.yaml"
    platform_path = tmp_path / "platform.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "production-routing",
                "production_name": "cld",
                "variants": {
                    "first": {"model_name": "pyg-cld-v1"},
                    "selected": {
                        "model_name": "pyg-clic-v1",
                        "production_name": "clic",
                    },
                },
                "seeds": [1],
                "training": {"global_batch_size": 8},
                # Detector productions legitimately differ throughout their
                # resolved data and model configuration. This test is about
                # CLI selection and routing, not cross-detector invariants.
                "allowed_variant_differences": [
                    "data_dir",
                    "dataset",
                    "enabled_test_datasets",
                    "input_dim",
                    "train_dataset",
                    "valid_dataset",
                    "test_dataset",
                    "model",
                ],
            }
        )
    )
    platform_path.write_text(
        yaml.safe_dump(
            {
                "name": "test-local",
                "gpus": 1,
                "data_dir": {
                    "cld": str(first_data),
                    "clic": str(selected_data),
                },
                "experiments_dir": str(tmp_path / "experiments"),
            }
        )
    )

    main(
        [
            "--scenario",
            str(scenario_path),
            "--platform",
            str(platform_path),
            "--spec-file",
            str(ROOT / "particleflow_spec.yaml"),
            "--global-batch-size",
            "8",
            "--variant",
            "selected",
            "--dry-run",
        ]
    )

    command = shlex.split(capsys.readouterr().out)
    assert command[command.index("--production-name") + 1] == "clic"
    assert command[command.index("--data-dir") + 1] == str(selected_data)
    assert str(first_data) not in command


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


def test_continuation_chooses_most_advanced_compatible_checkpoint(tmp_path):
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    platform.experiments_dir = str(tmp_path)
    job = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=8,
    )[0]

    for suffix, step in [("older", 10000), ("newer", 5000), ("best", 15000)]:
        run_dir = tmp_path / scenario.name / f"{job.variant_name}_seed{job.seed}_{suffix}"
        (run_dir / "checkpoints").mkdir(parents=True)
        manifest = {
            "job": job.model_dump(mode="json", exclude={"resolved_config"}),
            "resolved_config": job.resolved_config.model_dump(mode="json"),
        }
        (run_dir / "scenario-manifest.json").write_text(json.dumps(manifest))
        (run_dir / "checkpoints" / f"checkpoint-{step}.pth").touch()

    continuation = find_scenario_continuation(job, platform)

    assert continuation.step == 15000
    assert continuation.checkpoint.name == "checkpoint-15000.pth"
    assert continuation.experiment_dir.name.endswith("_best")


@pytest.mark.parametrize("profile, expected", [("lumi_mi250x", ["python3", "-u"]), ("local", ["uv", "run", "python3", "-u"])])
def test_pipeline_command_uses_platform_python_environment(profile, expected):
    from mlpf.training_scenarios import _pipeline_command

    scenario = load_training_scenario(SET_IMPROVEMENT_SCENARIO)
    platform = load_platform_profile(PLATFORMS / f"{profile}.yaml")
    job = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")[0]
    command = _pipeline_command(job, scenario, platform, ROOT / "particleflow_spec.yaml", Path("experiment"))
    assert command[: len(expected)] == expected


def test_grouped_tracking_scenario_isolates_each_mechanism():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_grouped_tracking_50k.yaml")
    platform = load_platform_profile(PLATFORMS / "lumi_mi250x.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")
    assert len(jobs) == 5
    assert all(job.seed == 12345 and job.resolved_config.num_steps == 50000 for job in jobs)
    assert all(job.resolved_config.validation_diagnostics_batches == 4 for job in jobs)
    baseline = jobs[0].resolved_config.model.model_dump()
    old = resolve_scenario_jobs(load_training_scenario(HEPTV2_GROUPING_SCENARIO), platform, spec_file=ROOT / "particleflow_spec.yaml")[1]
    assert baseline == old.resolved_config.model.model_dump()
    expected = [
        None,
        ("hit_feature_engineering", "tracker_neighborhood"),
        ("set_decoder", "hit_grouping_detector_balanced"),
        ("set_decoder", "tracker_group_readout"),
        ("set_decoder", "event_consistency_loss_weight"),
    ]
    for job, change in zip(jobs[1:], expected[1:]):
        variant = job.resolved_config.model.model_dump()
        assert variant[change[0]][change[1]] != baseline[change[0]][change[1]]
        variant[change[0]][change[1]] = baseline[change[0]][change[1]]
        assert variant == baseline


def test_tracker_fit_loss_ablation_changes_only_auxiliary_weights():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_fit_loss_50k.yaml")
    platform = load_platform_profile(ROOT / "configs/training/platforms/local.yaml")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml", global_batch_size=8)
    assert len(jobs) == 4
    reference = None
    weights = []
    for job in jobs:
        config = job.resolved_config
        assert config.num_steps == 50000
        assert config.model.hit_feature_engineering.tracker_neighborhood
        decoder = config.model.set_decoder
        assert decoder.hit_grouping and not decoder.tracker_group_readout and not decoder.tracker_fit_correction
        model = config.model.model_dump()
        weights.append((model["set_decoder"].pop("tracker_geometry_loss_weight"), model["set_decoder"].pop("tracker_curvature_loss_weight")))
        if reference is None:
            reference = model
        assert model == reference
    assert weights == [(0.0, 0.0), (0.01, 0.0), (0.0, 0.1), (0.01, 0.1)]


def test_tracker_readout_scenario_preserves_small_baseline():
    platform = load_platform_profile(ROOT / "configs/training/platforms/lumi_mi250x.yaml")
    reference = resolve_scenario_jobs(
        load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_model_scaling_50k.yaml"),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )[0].resolved_config
    jobs = resolve_scenario_jobs(
        load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_tracker_readout_50k.yaml"),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )
    assert len(jobs) == 4
    assert [
        (j.resolved_config.model.set_decoder.tracker_momentum_readout, j.resolved_config.model.set_decoder.hit_grouping_sparse) for j in jobs
    ] == [(False, False), (True, False), (False, True), (True, True)]
    for job in jobs:
        config = job.resolved_config
        assert job.global_batch_size == 128 and job.per_gpu_batch_size == 16
        assert config.num_steps == 50000 and config.val_freq == config.checkpoint_freq == 1000
        model = config.model.model_dump()
        for key in ["tracker_momentum_readout", "hit_grouping_sparse"]:
            model["set_decoder"][key] = False
        assert model == reference.model.model_dump()
        assert (config.lr, config.lr_schedule, config.optimizer) == (reference.lr, reference.lr_schedule, reference.optimizer)


def test_tracker_grouping_losses_scenario_is_from_scratch_small_factorial():
    platform = load_platform_profile(ROOT / "configs/training/platforms/lumi_mi250x.yaml")
    reference = resolve_scenario_jobs(
        load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_model_scaling_50k.yaml"),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )[0].resolved_config
    jobs = resolve_scenario_jobs(
        load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_tracker_grouping_losses_50k.yaml"),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
    )
    assert len(jobs) == 4
    assert [
        (j.resolved_config.model.set_decoder.hit_grouping_tracker_particle_balanced, j.resolved_config.model.set_decoder.tracker_overlap_loss_weight)
        for j in jobs
    ] == [(False, 0), (True, 0), (False, 0.05), (True, 0.05)]
    for job in jobs:
        c = job.resolved_config
        assert c.load is None and c.model.trainable == "all"
        assert job.global_batch_size == 128 and job.per_gpu_batch_size == 16
        assert c.num_steps == 50000 and c.val_freq == c.checkpoint_freq == 1000
        model = c.model.model_dump()
        model["set_decoder"]["hit_grouping_tracker_particle_balanced"] = False
        model["set_decoder"]["tracker_overlap_loss_weight"] = 0.0
        assert model == reference.model.model_dump()
        assert (c.lr, c.lr_schedule, c.optimizer) == (reference.lr, reference.lr_schedule, reference.optimizer)


def test_single_gpu_benchmark_worker_checks_selected_job_allocation(monkeypatch):
    import mlpf.training_scenarios as runner

    executed = []
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "1")
    monkeypatch.setattr(runner, "run_scenario_job", lambda job, *args, **kwargs: executed.append(job))
    args = [
        "--scenario",
        str(ROOT / "configs/training/scenarios/clic_cld_backbone_benchmark.yaml"),
        "--platform",
        str(PLATFORMS / "flatiron_h100.yaml"),
        "--spec-file",
        str(ROOT / "particleflow_spec.yaml"),
        "--task-index",
        "0",
    ]
    runner.main(args)
    assert len(executed) == 1
    assert executed[0].resolved_config.gpus == 1
    assert executed[0].per_gpu_batch_size == 128
    args[-1] = "2"
    with pytest.raises(ValueError, match="Slurm allocated 1"):
        runner.main(args)
    assert len(executed) == 1


@pytest.mark.parametrize(
    "environment,requested,valid",
    [
        ({"SLURM_GPUS_PER_TASK": "1", "SLURM_GPUS_ON_NODE": "8", "SLURM_GPUS_PER_NODE": "8"}, 1, True),
        ({"SLURM_GPUS_PER_TASK": "1", "SLURM_GPUS_ON_NODE": "8"}, 8, False),
        ({"SLURM_GPUS_ON_NODE": "8"}, 1, True),
        ({"SLURM_GPUS_PER_NODE": "8"}, 1, True),
        ({"SLURM_GPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "8"}, 8, True),
        ({"SLURM_GPUS_ON_NODE": "1"}, 8, False),
        ({"SLURM_GPUS_PER_TASK": "mi250:1", "SLURM_GPUS_ON_NODE": "8"}, 8, False),
        ({"SLURM_GPUS_PER_NODE": "a100:1,h100:1"}, 2, True),
        ({}, 1, True),
    ],
)
def test_slurm_allocation_validates_task_capacity(monkeypatch, environment, requested, valid):
    from mlpf.training_scenarios import _validate_slurm_allocation

    for variable in ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    profile = load_platform_profile(PLATFORMS / "lumi_mi250x.yaml")
    profile.gpus = requested
    if valid:
        _validate_slurm_allocation(profile)
    else:
        with pytest.raises(ValueError, match="Slurm allocated"):
            _validate_slurm_allocation(profile)
