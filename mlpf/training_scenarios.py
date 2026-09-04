"""Resolve reusable scientific training scenarios against hardware profiles."""

import argparse
import contextlib
import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlpf.conf import MLPFConfig


PLATFORM_OVERRIDE_KEYS = {
    "compile",
    "dtype",
    "max_open_readers",
    "model.attention.use_flash_attn_varlen",
    "num_workers",
    "prefetch_factor",
}
DERIVED_KEYS = {"gpu_batch_multiplier", "gpus", "seed"}
KNOWN_BOOLEAN_FLAGS = {"comet", "comet_offline", "compile", "make_plots"}
KNOWN_VALUE_FLAGS = {"dtype", "load", "test_datasets"}


class ScenarioVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_derived_overrides(self):
        invalid = DERIVED_KEYS.intersection(self.overrides)
        if invalid:
            raise ValueError(f"Variant overrides must not set derived keys: {sorted(invalid)}")
        return self


class ScenarioTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_batch_size: int = Field(gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_derived_parameters(self):
        invalid = DERIVED_KEYS.intersection(self.parameters)
        if invalid:
            raise ValueError(f"Scenario parameters must not set derived keys: {sorted(invalid)}")
        return self


class TrainingScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    spec_file: str = "particleflow_spec.yaml"
    production_name: str
    variants: dict[str, ScenarioVariant]
    seeds: list[int] = Field(min_length=1)
    training: ScenarioTraining
    common_overrides: dict[str, Any] = Field(default_factory=dict)
    allowed_variant_differences: list[str] = Field(default_factory=lambda: ["model.output_mode", "model.set_decoder"])

    @model_validator(mode="after")
    def validate_scenario(self):
        if len(self.variants) < 2:
            raise ValueError("A comparison scenario requires at least two variants")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Scenario seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Scenario seeds must be non-negative")
        invalid = DERIVED_KEYS.intersection(self.common_overrides)
        if invalid:
            raise ValueError(f"Common overrides must not set derived keys: {sorted(invalid)}")
        return self


class SlurmProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partition: str = "gpu"
    constraint: str | None = None
    account: str | None = None
    time: str = "12:00:00"
    nodes: int = Field(default=1, gt=0)
    tasks_per_node: int = Field(default=1, gt=0)
    cpus_per_task: int = Field(default=64, gt=0)
    gpu_request: Literal["gpus-per-node", "gpus-per-task", "gres"] = "gpus-per-node"
    gpu_type: str | None = None
    memory: str | None = None
    memory_per_gpu: str | None = None
    no_requeue: bool = False

    @model_validator(mode="after")
    def validate_resources(self):
        if self.gpu_type is not None and self.gpu_request != "gres":
            raise ValueError("gpu_type is only valid with gpu_request='gres'")
        if self.memory is not None and self.memory_per_gpu is not None:
            raise ValueError("Set only one of memory and memory_per_gpu")
        return self


class PlatformProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gpus: int = Field(gt=0)
    data_dir: str
    experiments_dir: str
    environment: dict[str, str] = Field(default_factory=dict)
    runtime_overrides: dict[str, Any] = Field(default_factory=dict)
    slurm: SlurmProfile | None = None

    @model_validator(mode="after")
    def validate_runtime_overrides(self):
        invalid = set(self.runtime_overrides).difference(PLATFORM_OVERRIDE_KEYS)
        if invalid:
            raise ValueError("Platform profiles may only set runtime-specific overrides; " f"invalid keys: {sorted(invalid)}")
        return self


class ResolvedScenarioJob(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    platform_name: str
    variant_name: str
    model_name: str
    seed: int
    global_batch_size: int
    per_gpu_batch_size: int
    gpu_batch_multiplier: int
    settings: dict[str, Any]
    resolved_config: MLPFConfig


def _read_yaml(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def load_training_scenario(path):
    return TrainingScenario.model_validate(_read_yaml(path))


def load_platform_profile(path):
    profile = PlatformProfile.model_validate(_read_yaml(path))
    profile.data_dir = os.path.expandvars(os.path.expanduser(profile.data_dir))
    profile.experiments_dir = os.path.expandvars(os.path.expanduser(profile.experiments_dir))
    profile.environment = {key: os.path.expandvars(os.path.expanduser(value)) for key, value in profile.environment.items()}
    return profile


@contextlib.contextmanager
def _temporary_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _serialize_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _settings_as_extra_args(settings):
    args = []
    for key, value in settings.items():
        if key in KNOWN_BOOLEAN_FLAGS or key in KNOWN_VALUE_FLAGS:
            continue
        args.extend([f"--{key}", _serialize_cli_value(value)])
    return args


def _config_args(profile, settings):
    return SimpleNamespace(
        train=True,
        test=True,
        pipeline=False,
        data_dir=profile.data_dir,
        gpus=profile.gpus,
        compile=settings.get("compile"),
        comet=settings.get("comet"),
        comet_offline=settings.get("comet_offline"),
        dtype=settings.get("dtype"),
        load=settings.get("load"),
        make_plots=settings.get("make_plots"),
        test_datasets=settings.get("test_datasets", []),
    )


def _training_batch_size(config):
    physical_datasets = config.train_dataset[config.dataset.value].values()
    batch_sizes = {dataset.batch_size for dataset in physical_datasets}
    if len(batch_sizes) != 1:
        raise ValueError(
            "Automatic global-batch resolution requires every physical training dataset " f"to use the same batch size, got {sorted(batch_sizes)}"
        )
    return next(iter(batch_sizes))


def _merge_settings(scenario, platform, variant, extra_overrides):
    return {
        **scenario.training.parameters,
        **scenario.common_overrides,
        **platform.runtime_overrides,
        **variant.overrides,
        **extra_overrides,
    }


def resolve_scenario_job(
    scenario,
    platform,
    variant_name,
    seed,
    *,
    spec_file=None,
    global_batch_size=None,
    extra_overrides=None,
):
    if variant_name not in scenario.variants:
        raise ValueError(f"Unknown variant {variant_name!r}; choose from {sorted(scenario.variants)}")
    variant = scenario.variants[variant_name]
    extra_overrides = extra_overrides or {}
    invalid = DERIVED_KEYS.intersection(extra_overrides)
    if invalid:
        raise ValueError(f"Use the dedicated runner options for derived settings, not --set: {sorted(invalid)}")
    settings = _merge_settings(scenario, platform, variant, extra_overrides)
    settings["seed"] = seed
    selected_spec = str(spec_file or scenario.spec_file)

    with _temporary_environment(platform.environment):
        config = MLPFConfig.from_spec(
            selected_spec,
            variant.model_name,
            scenario.production_name,
            args=_config_args(platform, settings),
            extra_args=_settings_as_extra_args(settings),
        )

    target_global_batch = global_batch_size if global_batch_size is not None else scenario.training.global_batch_size
    if target_global_batch <= 0:
        raise ValueError("global_batch_size must be positive")
    dataset_batch_size = _training_batch_size(config)
    divisor = platform.gpus * dataset_batch_size
    if target_global_batch % divisor:
        raise ValueError(
            f"global_batch_size={target_global_batch} is not divisible by " f"gpus={platform.gpus} * dataset_batch_size={dataset_batch_size}"
        )
    multiplier = target_global_batch // divisor
    settings["gpu_batch_multiplier"] = multiplier

    with _temporary_environment(platform.environment):
        config = MLPFConfig.from_spec(
            selected_spec,
            variant.model_name,
            scenario.production_name,
            args=_config_args(platform, settings),
            extra_args=_settings_as_extra_args(settings),
        )

    return ResolvedScenarioJob(
        scenario_name=scenario.name,
        platform_name=platform.name,
        variant_name=variant_name,
        model_name=variant.model_name,
        seed=seed,
        global_batch_size=target_global_batch,
        per_gpu_batch_size=dataset_batch_size * multiplier,
        gpu_batch_multiplier=multiplier,
        settings=settings,
        resolved_config=config,
    )


def _flatten(value, prefix=""):
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(child, child_prefix))
    else:
        flattened[prefix] = value
    return flattened


def _difference_allowed(path, allowed_paths):
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)


def validate_variant_invariants(jobs, allowed_paths):
    if len(jobs) < 2:
        return
    reference = _flatten(jobs[0].resolved_config.model_dump(mode="json"))
    for job in jobs[1:]:
        candidate = _flatten(job.resolved_config.model_dump(mode="json"))
        differences = {
            path: (reference.get(path), candidate.get(path))
            for path in sorted(set(reference) | set(candidate))
            if reference.get(path) != candidate.get(path) and not _difference_allowed(path, allowed_paths)
        }
        if differences:
            details = ", ".join(f"{path}: {values[0]!r} != {values[1]!r}" for path, values in differences.items())
            raise ValueError(f"Scenario variants differ outside allowed fields: {details}")


def resolve_scenario_jobs(
    scenario,
    platform,
    *,
    spec_file=None,
    global_batch_size=None,
    extra_overrides=None,
):
    jobs = [
        resolve_scenario_job(
            scenario,
            platform,
            variant_name,
            seed,
            spec_file=spec_file,
            global_batch_size=global_batch_size,
            extra_overrides=extra_overrides,
        )
        for seed in scenario.seeds
        for variant_name in scenario.variants
    ]
    for seed in scenario.seeds:
        seed_jobs = [job for job in jobs if job.seed == seed]
        validate_variant_invariants(seed_jobs, scenario.allowed_variant_differences)
    return jobs


def _pipeline_command(job, scenario, platform, spec_file, experiment_dir):
    settings = dict(job.settings)
    command = [
        "uv",
        "run",
        "python3",
        "-u",
        "mlpf/pipeline.py",
        "--spec-file",
        str(spec_file),
        "--model-name",
        job.model_name,
        "--production-name",
        scenario.production_name,
        "--data-dir",
        platform.data_dir,
        "--experiment-dir",
        str(experiment_dir),
        "train",
        "--gpus",
        str(platform.gpus),
    ]
    for name in sorted(KNOWN_VALUE_FLAGS):
        value = settings.pop(name, None)
        if value is None:
            continue
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, list):
            command.extend([flag, *(str(item) for item in value)])
        else:
            command.extend([flag, str(value)])
    for name in sorted(KNOWN_BOOLEAN_FLAGS):
        value = settings.pop(name, None)
        if value:
            command.append(f"--{name.replace('_', '-')}")
    for key, value in settings.items():
        command.extend([f"--{key}", _serialize_cli_value(value)])
    return command


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _experiment_path(platform, job, timestamp=None):
    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"{job.variant_name}_seed{job.seed}_{timestamp}"
    return Path(platform.experiments_dir) / job.scenario_name / experiment_name


def run_scenario_job(job, scenario, platform, spec_file, *, dry_run=False):
    experiment_dir = _experiment_path(platform, job, timestamp="TIMESTAMP" if dry_run else None)
    command = _pipeline_command(job, scenario, platform, spec_file, experiment_dir)
    print(shlex.join(command), flush=True)
    if dry_run:
        return

    experiment_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scenario": scenario.model_dump(mode="json"),
        "platform": platform.model_dump(mode="json"),
        "job": job.model_dump(mode="json", exclude={"resolved_config"}),
        "resolved_config": job.resolved_config.model_dump(mode="json"),
        "command": command,
        "git_revision": _git_revision(),
    }
    with (experiment_dir / "scenario-manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    environment = os.environ.copy()
    environment.update(platform.environment)
    subprocess.run(command, check=True, env=environment)


def _parse_set_overrides(values):
    overrides = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def _validate_slurm_allocation(platform):
    allocated = os.environ.get("SLURM_GPUS_PER_NODE")
    if allocated and allocated.isdigit() and int(allocated) != platform.gpus:
        raise ValueError(f"Platform profile requests {platform.gpus} GPUs but Slurm allocated {allocated}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--spec-file")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variant")
    selection.add_argument("--task-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--experiments-dir")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    scenario = load_training_scenario(args.scenario)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("seed must be non-negative")
        # Apply the same seed override to direct and Slurm-array execution.
        scenario.seeds = [args.seed]
    platform = load_platform_profile(args.platform)
    if args.data_dir:
        platform.data_dir = args.data_dir
    if args.experiments_dir:
        platform.experiments_dir = args.experiments_dir
    _validate_slurm_allocation(platform)

    spec_file = args.spec_file or scenario.spec_file
    jobs = resolve_scenario_jobs(
        scenario,
        platform,
        spec_file=spec_file,
        global_batch_size=args.global_batch_size,
        extra_overrides=_parse_set_overrides(args.set),
    )
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(jobs):
            raise ValueError(f"task-index must be between 0 and {len(jobs) - 1}")
        jobs = [jobs[args.task_index]]
    else:
        if args.variant:
            jobs = [job for job in jobs if job.variant_name == args.variant]
        if not jobs:
            raise ValueError("No jobs matched the requested variant")

    for job in jobs:
        run_scenario_job(job, scenario, platform, spec_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
