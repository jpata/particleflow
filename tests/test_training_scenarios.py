import argparse
import json
import shlex
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mlpf.conf import parse_data_dir
from mlpf.training_scenarios import (
    PlatformProfile,
    ScenarioVariant,
    ScenarioTraining,
    TrainingScenario,
    _experiment_path,
    _global_batch_size_arg,
    _pipeline_command,
    find_scenario_continuation,
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
    validate_variant_invariants,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "root,subdir",
    [(None, "relative"), ("scan", "/absolute"), ("scan", "../escape"), ("scan", "valid/../../escape")],
)
def test_variant_data_subdirectory_requires_a_root_and_stays_within_it(root, subdir):
    with pytest.raises(ValueError, match="data_subdir requires data_root.*within that root"):
        ScenarioVariant(model_name="fixture_model", data_root=root, data_subdir=subdir)


@pytest.mark.parametrize("subdir", [None, "geometry/tfds"])
@pytest.mark.parametrize("explicit_dir", [False, True])
def test_variant_data_directory_precedence_and_cli_roundtrip(training_files, tmp_path, subdir, explicit_dir):
    from mlpf.conf import MLPFConfig
    from mlpf.pipeline import get_parser
    from mlpf.training_scenarios import resolve_scenario_job

    scenario_path, platform_path = training_files
    scenario = load_training_scenario(scenario_path)
    platform = load_platform_profile(platform_path)
    platform.data_dir = {"fixture": str(tmp_path / "default"), "scan": str(tmp_path / "scan root")}
    scenario.variants["first"].data_root = "scan"
    scenario.variants["first"].data_subdir = subdir
    original = deepcopy(platform.model_dump())
    override = str(tmp_path / "explicit data with spaces")
    expected = override if explicit_dir else str(Path(platform.data_dir["scan"]) / (subdir or ""))

    job = resolve_scenario_job(scenario, platform, "first", scenario.seeds[0], extra_overrides={"data_dir": override} if explicit_dir else {})
    command = _pipeline_command(job, scenario, platform, scenario.spec_file, tmp_path / "experiment")
    command = shlex.split(shlex.join(command))
    args, extra_args = get_parser().parse_known_args(command[command.index("mlpf/pipeline.py") + 1 :])
    reparsed = MLPFConfig.from_spec(scenario.spec_file, job.model_name, job.production_name, args=args, extra_args=extra_args)

    assert job.data_dir == job.resolved_config.data_dir == reparsed.data_dir == expected
    assert command.count("--data-dir") == 1
    assert "--data_dir" not in command
    assert "data_dir" not in job.settings
    assert platform.model_dump() == original


def test_variant_data_root_must_exist_on_platform(training_files):
    from mlpf.training_scenarios import resolve_scenario_job

    scenario_path, platform_path = training_files
    scenario = load_training_scenario(scenario_path)
    platform = load_platform_profile(platform_path)
    platform.data_dir = {"fixture": "/data/default"}
    scenario.variants["first"].data_root = "missing"
    with pytest.raises(ValueError, match="no data_dir for production 'missing'"):
        resolve_scenario_job(scenario, platform, "first", scenario.seeds[0])


@pytest.mark.parametrize("replace_samples", [False, True], ids=["leaf-overrides", "sample-map-replacement"])
def test_dataset_overrides_survive_resolution_and_command_roundtrip(training_files, tmp_path, replace_samples):
    from mlpf.conf import MLPFConfig
    from mlpf.pipeline import get_parser
    from mlpf.training_scenarios import resolve_scenario_job

    scenario_path, platform_path = training_files
    scenario = load_training_scenario(scenario_path)
    platform = load_platform_profile(platform_path)
    sample = "cld_custom_pf" if replace_samples else "fixture_sample"
    selection = {sample: {"version": "2.0.0", "splits": ["split with spaces", "another"]}}
    overrides = {}
    for section in ("train_dataset.cld.physical.samples", "valid_dataset.cld.physical.samples", "test_dataset"):
        if replace_samples:
            overrides[section] = selection
        else:
            overrides[f"{section}.{sample}.version"] = selection[sample]["version"]
            overrides[f"{section}.{sample}.splits"] = selection[sample]["splits"]
    scenario.variants["first"].overrides.update(overrides)
    # A new model-spec default must not replace an explicit scenario selection.
    spec_path = Path(scenario.spec_file)
    spec = yaml.safe_load(spec_path.read_text())
    model = spec["models"]["fixture_model"]
    for section in ("train_datasets", "validation_datasets"):
        model[section]["physical"]["samples"][0]["version"] = "3.0.0"
    model["test_datasets"][0]["version"] = "3.0.0"
    spec_path.write_text(yaml.safe_dump(spec))

    job = resolve_scenario_job(scenario, platform, "first", scenario.seeds[0])
    command = _pipeline_command(job, scenario, platform, spec_path, tmp_path / "experiment")
    command = shlex.split(shlex.join(command))
    args, extra_args = get_parser().parse_known_args(command[command.index("mlpf/pipeline.py") + 1 :])
    args.train = args.test = True
    reparsed = MLPFConfig.from_spec(str(spec_path), job.model_name, job.production_name, args=args, extra_args=extra_args)

    for config in (job.resolved_config, reparsed):
        for groups in (config.train_dataset, config.valid_dataset):
            assert {name: entry.model_dump(include={"version", "splits"}) for name, entry in groups["cld"]["physical"].samples.items()} == selection
        assert {name: entry.model_dump(include={"version", "splits"}) for name, entry in config.test_dataset.items()} == selection


@pytest.mark.parametrize(
    "weights_only,reset_weights,has_checkpoint",
    [(True, True, True), (True, False, True), (False, False, True), (True, True, False)],
)
def test_continuation_resumes_training_state_without_restarting_finetuning(
    training_files, tmp_path, monkeypatch, capsys, weights_only, reset_weights, has_checkpoint
):
    import mlpf.training_scenarios as runner
    from unittest.mock import Mock

    scenario_path, platform_path = training_files
    scenario = load_training_scenario(scenario_path)
    platform = load_platform_profile(platform_path)
    source = str(tmp_path / "source checkpoint.pth")
    scenario.training.parameters.update(load=source, load_weights_only=weights_only, reset_task_loss_weights=reset_weights)
    job = runner.resolve_scenario_job(scenario, platform, "first", scenario.seeds[0])
    experiment = tmp_path / scenario.name / "prior-run"
    platform.experiments_dir = str(tmp_path)
    (experiment / "checkpoints").mkdir(parents=True)
    checkpoint = experiment / "checkpoints/checkpoint-5.pth"
    if has_checkpoint:
        checkpoint.touch()
    manifest_path = experiment / "scenario-manifest.json"
    manifest_path.write_text(
        json.dumps({"job": job.model_dump(mode="json", exclude={"resolved_config"}), "resolved_config": job.resolved_config.model_dump(mode="json")})
    )
    run = Mock()
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "_git_revision", lambda: "fixture-revision")

    runner.run_scenario_job(job, scenario, platform, scenario.spec_file, continue_run=True)

    command = shlex.split(capsys.readouterr().out.strip())
    assert command[command.index("--load") + 1] == (str(checkpoint) if has_checkpoint else source)
    for flag, initial in (("load_weights_only", weights_only), ("reset_task_loss_weights", reset_weights)):
        assert command[command.index("--" + flag) + 1] == str(False if has_checkpoint else initial).lower()
    assert command[command.index("--experiment-dir") + 1] == str(experiment)
    run.assert_called_once()
    assert run.call_args.args[0] == command
    assert run.call_args.kwargs["check"] is True
    manifest = json.loads(manifest_path.read_text())
    assert manifest["continuations"][-1]["command"] == command
    assert manifest["resolved_config"]["load"] == source


def test_training_scenario_rejects_empty_variants():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_pf_heptv2_default_50k.yaml")
    payload = scenario.model_dump()
    payload["variants"] = {}
    with pytest.raises(ValueError, match="at least one variant"):
        type(scenario).model_validate(payload)


SCENARIO = ROOT / "configs/training/scenarios/cld_hits_output_comparison.yaml"
CLIC_CLD_SCENARIO = ROOT / "configs/training/scenarios/clic_cld_pf_set_hits_comparison.yaml"
PLATFORMS = ROOT / "configs/training/platforms"


def _pileup_weighting_scenario():
    """pu0 and pu200 trained together, one group each; the variants differ only in the groups' step weights."""
    model = "pyg-colliderml-pu0pu200-groups-v1"
    weights = ["train_dataset.colliderml.pu0.weight", "train_dataset.colliderml.pu200.weight"]
    return TrainingScenario.model_validate(
        {
            "name": "colliderml_pileup_weighting",
            "production_name": "colliderml",
            "variants": {
                "unweighted": {"model_name": model},
                "w11": {"model_name": model, "overrides": dict(zip(weights, [1, 1]))},
                "w13": {"model_name": model, "overrides": dict(zip(weights, [1, 3]))},
            },
            "seeds": [12345],
            "training": {"global_batch_size": {"pu0": 768, "pu200": 64}, "parameters": {"num_steps": 20000}},
            "common_overrides": {
                "train_dataset.colliderml.pu0.batch_size": 12,
                "train_dataset.colliderml.pu200.batch_size": 1,
            },
            "allowed_variant_differences": weights,
        }
    )


def _train_groups(job):
    config = job.resolved_config
    return {
        name: (group.batch_size * config.gpu_batch_multiplier, group.weight) for name, group in config.train_dataset[config.dataset.value].items()
    }


def test_pileup_weighting_scenario_differs_only_in_step_weights():
    scenario = _pileup_weighting_scenario()
    jobs = resolve_scenario_jobs(
        scenario,
        PlatformProfile(name="test", gpus=8, data_dir="/tmp/data", experiments_dir="/tmp/experiments"),
        spec_file=ROOT / "particleflow_spec.yaml",
    )

    assert [job.variant_name for job in jobs] == ["unweighted", "w11", "w13"]
    assert {job.gpu_batch_multiplier for job in jobs} == {8}
    assert all(job.global_batch_size == {"pu0": 768, "pu200": 64} for job in jobs)
    assert all(job.per_gpu_batch_size == {"pu0": 96, "pu200": 8} for job in jobs)
    assert [_train_groups(job) for job in jobs] == [
        {"pu0": (96, None), "pu200": (8, None)},
        {"pu0": (96, 1.0), "pu200": (8, 1.0)},
        {"pu0": (96, 1.0), "pu200": (8, 3.0)},
    ]
    assert all(list(job.resolved_config.valid_dataset["colliderml"]) == ["pu0", "pu200"] for job in jobs)


@pytest.mark.parametrize(
    "global_batch_size, error",
    [
        (32, "different dataset batch sizes"),
        ({"pu0": 768}, "names groups"),
        ({"pu0": 384, "pu200": 64}, "one multiplier serves all groups"),
        ({"pu0": 768, "pu200": 68}, r"global_batch_size\[pu200\]=68 is not divisible"),
    ],
)
def test_per_group_global_batch_must_match_the_groups(global_batch_size, error):
    scenario = _pileup_weighting_scenario()
    platform = PlatformProfile(name="test", gpus=8, data_dir="/tmp/data", experiments_dir="/tmp/experiments")
    scenario.training.global_batch_size = global_batch_size
    with pytest.raises(ValueError, match=error):
        resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")


def test_global_batch_size_cli_takes_one_value_or_one_per_group():
    assert _global_batch_size_arg("1536") == 1536
    assert _global_batch_size_arg("pu0=768, pu200=64") == {"pu0": 768, "pu200": 64}
    for bad in ("pu0=768,pu200", "big", "pu0=x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _global_batch_size_arg(bad)
    platform = PlatformProfile(name="test", gpus=8, data_dir="/tmp/data", experiments_dir="/tmp/experiments")
    jobs = resolve_scenario_jobs(
        _pileup_weighting_scenario(),
        platform,
        spec_file=ROOT / "particleflow_spec.yaml",
        global_batch_size=_global_batch_size_arg("pu0=1536,pu200=128"),
    )
    assert {job.gpu_batch_multiplier for job in jobs} == {16}


@pytest.mark.parametrize("platform_name", ["local", "lumi_mi250x"])
def test_cms_backbone_scenario_pins_all_datasets_and_resolves_global_batch(platform_name):
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cms_backbone_comparison.yaml")
    platform = load_platform_profile(PLATFORMS / f"{platform_name}.yaml")
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
        assert job.global_batch_size == {"physical_pu": 96, "physical_nopu": 768}
        assert {name: batch * platform.gpus for name, batch in job.per_gpu_batch_size.items()} == job.global_batch_size
        assert config.num_steps == 320000
        assert config.lr == 0.0004 and config.lr_schedule.value == "cosinedecay"
        assert config.nvalid == 5000 and config.ntest == 512
        for datasets in (config.train_dataset["cms"], config.valid_dataset["cms"]):
            assert {name: dataset.batch_size for name, dataset in datasets.items()} == {"physical_pu": 1, "physical_nopu": 8}
            assert all(dataset.weight is None for dataset in datasets.values())
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
    platform = PlatformProfile(name="test", gpus=4, data_dir="/tmp/data", experiments_dir="/tmp/experiments")

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
    scenario = load_training_scenario(SCENARIO)
    selected = list(scenario.variants.values())[1]
    assert f"--model-name {selected.model_name}" in command


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

    assert path == Path(platform.experiments_dir) / scenario.name / f"{job.variant_name}_seed{job.seed}_TIMESTAMP"


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

    for suffix, step in [("older", 10000), ("newer", 5000), ("best", 15000), ("different_limit", 20000)]:
        run_dir = tmp_path / scenario.name / f"{job.variant_name}_seed{job.seed}_{suffix}"
        (run_dir / "checkpoints").mkdir(parents=True)
        manifest = {
            "job": job.model_dump(mode="json", exclude={"resolved_config"}),
            "resolved_config": job.resolved_config.model_dump(mode="json"),
        }
        if suffix == "best":
            del manifest["resolved_config"]["validation_particle_metrics_max_events"]
        if suffix == "different_limit":
            manifest["resolved_config"]["validation_particle_metrics_max_events"] = 0
        (run_dir / "scenario-manifest.json").write_text(json.dumps(manifest))
        (run_dir / "checkpoints" / f"checkpoint-{step}.pth").touch()

    continuation = find_scenario_continuation(job, platform)

    assert continuation.step == 15000
    assert continuation.checkpoint.name == "checkpoint-15000.pth"
    assert continuation.experiment_dir.name.endswith("_best")


@pytest.mark.parametrize("python_command", [["python3"], ["uv", "run", "python3"]])
def test_pipeline_command_uses_platform_python_environment(python_command):
    from mlpf.training_scenarios import _pipeline_command

    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORMS / "local.yaml")
    platform.python_command = python_command
    job = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / scenario.spec_file, global_batch_size=8)[0]
    command = _pipeline_command(job, scenario, platform, ROOT / scenario.spec_file, Path("experiment"))
    assert command[: len(python_command) + 1] == [*python_command, "-u"]


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


@pytest.mark.parametrize("scenario_path", sorted((ROOT / "configs/training/scenarios").glob("*.yaml")), ids=lambda path: path.stem)
def test_shipped_scenarios_resolve(scenario_path):
    """Recipes may change, but must still produce valid jobs and satisfy their declared invariants."""
    scenario = load_training_scenario(scenario_path)
    platform = PlatformProfile(name="test", gpus=1, data_dir="/tmp/data", experiments_dir="/tmp/experiments")
    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / scenario.spec_file)
    assert [(job.seed, job.variant_name) for job in jobs] == [(seed, variant) for seed in scenario.seeds for variant in scenario.variants]


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
@pytest.mark.parametrize("global_batch_size", [32, 96])
def test_global_batch_is_distributed_across_gpus(gpus, global_batch_size):
    scenario = TrainingScenario(
        name="batch-test",
        production_name="cld",
        variants={"baseline": ScenarioVariant(model_name="pyg-cld-v1")},
        seeds=[17],
        training=ScenarioTraining(global_batch_size=global_batch_size),
    )
    platform = PlatformProfile(name="test", gpus=gpus, data_dir="/tmp/data", experiments_dir="/tmp/experiments")
    job = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / scenario.spec_file)[0]
    config = job.resolved_config
    assert job.global_batch_size == global_batch_size
    assert job.per_gpu_batch_size * gpus == global_batch_size
    assert config.gpus == gpus
    assert config.gpu_batch_multiplier == job.gpu_batch_multiplier
    for group in config.train_dataset[config.dataset.value].values():
        assert group.batch_size * job.gpu_batch_multiplier == job.per_gpu_batch_size


def test_override_precedence():
    scenario = TrainingScenario(
        name="override-test",
        production_name="cld",
        variants={"baseline": ScenarioVariant(model_name="pyg-cld-v1", overrides={"lr": 0.003})},
        seeds=[17],
        training=ScenarioTraining(global_batch_size=32, parameters={"lr": 0.001}),
        common_overrides={"lr": 0.002},
    )
    platform = PlatformProfile(name="test", gpus=2, data_dir="/tmp/data", experiments_dir="/tmp/experiments")

    def resolve(**kwargs):
        return resolve_scenario_jobs(scenario, platform, spec_file=ROOT / scenario.spec_file, **kwargs)[0]

    assert resolve().resolved_config.lr == 0.003
    assert resolve(extra_overrides={"lr": 0.004}).resolved_config.lr == 0.004
    scenario.variants["baseline"].overrides.clear()
    assert resolve().resolved_config.lr == 0.002
    scenario.common_overrides.clear()
    assert resolve().resolved_config.lr == 0.001


@pytest.mark.parametrize("profile_path", sorted(PLATFORMS.glob("*.yaml")), ids=lambda path: path.stem)
def test_shipped_platform_profiles_load(profile_path):
    load_platform_profile(profile_path)


def test_worker_validates_selected_variant_gpu_allocation(monkeypatch):
    import mlpf.training_scenarios as runner

    scenario = TrainingScenario(
        name="allocation-test",
        production_name="cld",
        variants={"single": ScenarioVariant(model_name="pyg-cld-v1", gpus=1), "multi": ScenarioVariant(model_name="pyg-cld-v1")},
        seeds=[17],
        training=ScenarioTraining(global_batch_size=32),
    )
    platform = PlatformProfile(name="test", gpus=4, data_dir="/tmp/data", experiments_dir="/tmp/experiments")
    executed = []
    for variable in ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "1")
    monkeypatch.setattr(runner, "load_training_scenario", lambda path: scenario)
    monkeypatch.setattr(runner, "load_platform_profile", lambda path: platform)
    monkeypatch.setattr(runner, "run_scenario_job", lambda job, *args, **kwargs: executed.append(job))
    args = ["--scenario", "scenario.yaml", "--platform", "profile.yaml", "--spec-file", str(ROOT / scenario.spec_file), "--task-index", "0"]
    runner.main(args)
    assert len(executed) == 1
    assert executed[0].resolved_config.gpus == 1
    args[-1] = "1"
    with pytest.raises(ValueError, match="Slurm allocated 1"):
        runner.main(args)
    assert len(executed) == 1


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("cld_edm_ttbar_pf", "cld"),
        ("clic_edm_qq_pf", "clic"),
        ("clic_edm_qq_hits", "clic_hits"),
        ("colliderml_ttbar_nopu_pf", "colliderml"),
        ("cms_pf_ttbar", "cms"),
        ("unrecognized_sample", "cld"),
    ],
)
def test_dataset_for_sample_follows_the_sample_detector(sample, expected):
    from mlpf.conf import Dataset, dataset_for_sample

    assert dataset_for_sample(sample, Dataset.CLD).value == expected


def test_feature_bag_scenario_differs_only_in_the_feature_bag():
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_clic_colliderml_feature_bag.yaml")
    platform = load_platform_profile(PLATFORMS / "flatiron_h100.yaml")

    jobs = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")

    assert [job.variant_name for job in jobs] == ["feature_bag", "no_feature_bag"]
    assert [job.resolved_config.model.feature_bag for job in jobs] == [True, False]
    for job in jobs:
        config = job.resolved_config
        assert config.model.input_normalization
        # Each detector's samples come from that detector's platform data_dir.
        assert config.data_dir == {name: platform.data_dir[name] for name in ("cld", "clic", "colliderml")}
        assert config.data_dir_for("colliderml_ttbar_nopu_pf") == platform.data_dir["colliderml"]
        assert config.dataset_versions() == {"cld": "3.3.0", "clic": "3.3.0", "colliderml": "1.5.0"}
        # Each detector gets the same share of the training steps.
        assert {name: group.weight for name, group in config.train_dataset["cld"].items()} == {"cld": 1.0, "clic": 1.0, "colliderml": 1.0}
    reference, candidate = (job.resolved_config.model_dump(mode="json") for job in jobs)
    candidate["model"]["feature_bag"] = True
    assert candidate == reference


def test_joint_production_passes_one_data_dir_per_detector_to_the_pipeline(tmp_path):
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_clic_colliderml_feature_bag.yaml")
    platform = PlatformProfile(
        name="test", gpus=1, data_dir={"cld": "/d/cld", "clic": "/d/clic", "colliderml": "/d/cml"}, experiments_dir=str(tmp_path)
    )
    job = resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")[0]

    command = _pipeline_command(job, scenario, platform, ROOT / "particleflow_spec.yaml", tmp_path)
    data_dir_args = [command[i + 1] for i, arg in enumerate(command) if arg == "--data-dir"]
    assert data_dir_args == ["cld=/d/cld", "clic=/d/clic", "colliderml=/d/cml"]
    # The pipeline reads them back into the same mapping.
    assert parse_data_dir(data_dir_args) == job.resolved_config.data_dir == {"cld": "/d/cld", "clic": "/d/clic", "colliderml": "/d/cml"}


def test_joint_production_needs_every_detector_on_the_platform(tmp_path):
    scenario = load_training_scenario(ROOT / "configs/training/scenarios/cld_clic_colliderml_feature_bag.yaml")
    platform = PlatformProfile(name="test", gpus=1, data_dir={"cld": "/d/cld", "clic": "/d/clic"}, experiments_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no data_dir for production 'colliderml'"):
        resolve_scenario_jobs(scenario, platform, spec_file=ROOT / "particleflow_spec.yaml")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (None, None),
        (["/data/tfds"], "/data/tfds"),
        (["cld=/a", "colliderml=/b"], {"cld": "/a", "colliderml": "/b"}),
    ],
)
def test_parse_data_dir(values, expected):
    assert parse_data_dir(values) == expected


def test_parse_data_dir_rejects_a_path_mixed_with_per_detector_entries():
    with pytest.raises(ValueError, match="SOURCE=PATH"):
        parse_data_dir(["/data/tfds", "cld=/a"])


def test_config_rejects_a_data_dir_map_without_a_sample_detector():
    from mlpf.conf import MLPFConfig

    config = {
        "dataset": "cld",
        "data_dir": {"cld": "/a"},
        "model": {"type": "attention"},
        "conv_type": "attention",
        "test_dataset": {"cld_edm_ttbar_pf": {"version": "3.3.0"}, "colliderml_ttbar_nopu_pf": {"version": "1.5.0"}},
    }
    with pytest.raises(ValueError, match="not for the samples \\['colliderml_ttbar_nopu_pf'\\]"):
        MLPFConfig.model_validate(config)
    config["data_dir"]["colliderml"] = "/b"
    assert MLPFConfig.model_validate(config).data_dir_for("colliderml_ttbar_nopu_pf") == "/b"
