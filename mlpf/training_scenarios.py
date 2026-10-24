"""Resolve reusable scientific training scenarios against hardware profiles."""

import argparse
import contextlib
import datetime
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

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
    # Defaults to the scenario-level production; set it to compare detectors.
    production_name: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_derived_overrides(self):
        invalid = DERIVED_KEYS.intersection(self.overrides)
        if invalid:
            raise ValueError(f"Variant overrides must not set derived keys: {sorted(invalid)}")
        return self


class ScenarioTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Events per optimizer step, summed over all GPUs. Each step draws one batch from one train
    # group, so with groups of different dataset batch sizes give one value per group.
    global_batch_size: PositiveInt | dict[str, PositiveInt]
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
        if not self.variants:
            raise ValueError("A training scenario requires at least one variant")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Scenario seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Scenario seeds must be non-negative")
        invalid = DERIVED_KEYS.intersection(self.common_overrides)
        if invalid:
            raise ValueError(f"Common overrides must not set derived keys: {sorted(invalid)}")
        return self

    def variant_production(self, variant_name):
        return self.variants[variant_name].production_name or self.production_name


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
    python_command: list[str] = Field(default_factory=lambda: ["uv", "run", "python3"], min_length=1)
    # Either one TFDS directory shared by every production, or a mapping from
    # production name to that production's TFDS directory.
    data_dir: str | dict[str, str]
    experiments_dir: str
    environment: dict[str, str] = Field(default_factory=dict)
    runtime_overrides: dict[str, Any] = Field(default_factory=dict)
    slurm: SlurmProfile | None = None

    @model_validator(mode="after")
    def validate_runtime_overrides(self):
        invalid = set(self.runtime_overrides).difference(PLATFORM_OVERRIDE_KEYS)
        if invalid:
            raise ValueError("Platform profiles may only set runtime-specific overrides; " f"invalid keys: {sorted(invalid)}")
        if isinstance(self.data_dir, dict) and not self.data_dir:
            raise ValueError("Platform data_dir mapping must name at least one production")
        return self

    def data_dir_for(self, production_name):
        if isinstance(self.data_dir, str):
            return self.data_dir
        if production_name not in self.data_dir:
            raise ValueError(
                f"Platform profile {self.name!r} has no data_dir for production {production_name!r}; " f"available: {sorted(self.data_dir)}"
            )
        return self.data_dir[production_name]


class ResolvedScenarioJob(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    platform_name: str
    variant_name: str
    model_name: str
    production_name: str
    data_dir: str
    seed: int
    # one value, or one per train group when the scenario gives global_batch_size per group
    global_batch_size: int | dict[str, int]
    per_gpu_batch_size: int | dict[str, int]
    gpu_batch_multiplier: int
    settings: dict[str, Any]
    resolved_config: MLPFConfig


class ScenarioContinuation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_dir: Path
    checkpoint: Path | None
    step: int = 0


def _read_yaml(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def load_training_scenario(path):
    return TrainingScenario.model_validate(_read_yaml(path))


def load_platform_profile(path):
    profile = PlatformProfile.model_validate(_read_yaml(path))
    if isinstance(profile.data_dir, str):
        profile.data_dir = os.path.expandvars(os.path.expanduser(profile.data_dir))
    else:
        profile.data_dir = {key: os.path.expandvars(os.path.expanduser(value)) for key, value in profile.data_dir.items()}
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
    if value is None:
        return "null"
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


def _config_args(profile, settings, data_dir):
    return SimpleNamespace(
        train=True,
        test=True,
        pipeline=False,
        data_dir=data_dir,
        gpus=profile.gpus,
        compile=settings.get("compile"),
        comet=settings.get("comet"),
        comet_offline=settings.get("comet_offline"),
        dtype=settings.get("dtype"),
        load=settings.get("load"),
        make_plots=settings.get("make_plots"),
        test_datasets=settings.get("test_datasets", []),
    )


def _train_group_batch_sizes(config):
    return {name: group.batch_size for name, group in config.train_dataset[config.dataset.value].items()}


def _gpu_batch_multiplier(target, group_batch_sizes, gpus):
    """gpu_batch_multiplier for the target global batch size (events per step, summed over all GPUs).

    A train group's global batch size is gpus * its dataset batch_size * multiplier. target is either
    one value for all groups (only if they share a dataset batch_size) or a {group: value} dict.
    Example: 8 GPUs, batch sizes {pu0: 12, pu200: 1} and target {pu0: 768, pu200: 64} give multiplier 8.
    There is only one multiplier, so every group must give the same one, i.e. the target values must
    have the same ratios as the dataset batch sizes.
    """
    per_group = isinstance(target, dict)
    if not per_group:
        if len(set(group_batch_sizes.values())) != 1:
            raise ValueError(
                f"train groups have different dataset batch sizes {group_batch_sizes}; give global_batch_size per group, "
                f"e.g. {{{', '.join(f'{name}: {gpus * size}' for name, size in group_batch_sizes.items())}}} for multiplier 1"
            )
        target = dict.fromkeys(group_batch_sizes, target)

    if set(target) != set(group_batch_sizes):
        raise ValueError(f"global_batch_size names groups {sorted(target)}, but the train groups are {sorted(group_batch_sizes)}")
    multipliers = {}
    for name, events in target.items():
        label = f"global_batch_size[{name}]" if per_group else "global_batch_size"
        if events <= 0:
            raise ValueError(f"{label} must be positive")
        if events % (gpus * group_batch_sizes[name]):
            raise ValueError(f"{label}={events} is not divisible by gpus={gpus} * dataset_batch_size={group_batch_sizes[name]}")
        multipliers[name] = events // (gpus * group_batch_sizes[name])
    if len(set(multipliers.values())) != 1:
        raise ValueError(
            f"global_batch_size {target} implies gpu_batch_multiplier {multipliers}, but one multiplier serves all groups: "
            f"keep the dataset batch_size ratios {group_batch_sizes} or override train_dataset.<detector>.<group>.batch_size"
        )
    return next(iter(multipliers.values()))


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
    production_name = scenario.variant_production(variant_name)
    data_dir = platform.data_dir_for(production_name)

    with _temporary_environment(platform.environment):
        config = MLPFConfig.from_spec(
            selected_spec,
            variant.model_name,
            production_name,
            args=_config_args(platform, settings, data_dir),
            extra_args=_settings_as_extra_args(settings),
        )

    target_global_batch = global_batch_size if global_batch_size is not None else scenario.training.global_batch_size
    group_batch_sizes = _train_group_batch_sizes(config)
    multiplier = _gpu_batch_multiplier(target_global_batch, group_batch_sizes, platform.gpus)
    settings["gpu_batch_multiplier"] = multiplier
    per_gpu_batch_size = {name: size * multiplier for name, size in group_batch_sizes.items()}

    with _temporary_environment(platform.environment):
        config = MLPFConfig.from_spec(
            selected_spec,
            variant.model_name,
            production_name,
            args=_config_args(platform, settings, data_dir),
            extra_args=_settings_as_extra_args(settings),
        )

    return ResolvedScenarioJob(
        scenario_name=scenario.name,
        platform_name=platform.name,
        variant_name=variant_name,
        model_name=variant.model_name,
        production_name=production_name,
        data_dir=data_dir,
        seed=seed,
        global_batch_size=target_global_batch,
        per_gpu_batch_size=per_gpu_batch_size if isinstance(target_global_batch, dict) else next(iter(per_gpu_batch_size.values())),
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
        *platform.python_command,
        "-u",
        "mlpf/pipeline.py",
        "--spec-file",
        str(spec_file),
        "--model-name",
        job.model_name,
        "--production-name",
        job.production_name,
        "--data-dir",
        job.data_dir,
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


def _checkpoint_step(path):
    match = re.fullmatch(r"checkpoint-(\d+)(?:-.*)?\.pth", Path(path).name)
    return int(match.group(1)) if match else None


def find_scenario_continuation(job, platform):
    """Find the most advanced compatible run for one scenario job."""
    scenario_dir = Path(platform.experiments_dir) / job.scenario_name
    desired_config = job.resolved_config.model_dump(mode="json")
    candidates = []
    for manifest_path in scenario_dir.glob("*/scenario-manifest.json"):
        try:
            with manifest_path.open() as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        saved_job = manifest.get("job", {})
        if (
            saved_job.get("scenario_name") != job.scenario_name
            or saved_job.get("platform_name") != job.platform_name
            or saved_job.get("variant_name") != job.variant_name
            or saved_job.get("seed") != job.seed
            or manifest.get("resolved_config") != desired_config
        ):
            continue

        experiment_dir = manifest_path.parent
        checkpoints = []
        for checkpoint_path in (experiment_dir / "checkpoints").glob("checkpoint-*.pth"):
            step = _checkpoint_step(checkpoint_path)
            if step is not None:
                checkpoints.append((step, checkpoint_path))
        step, checkpoint = max(checkpoints, default=(0, None), key=lambda item: item[0])
        candidates.append((step, manifest_path.stat().st_mtime, experiment_dir, checkpoint))

    if not candidates:
        return None
    step, _, experiment_dir, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    return ScenarioContinuation(experiment_dir=experiment_dir, checkpoint=checkpoint, step=step)


def run_scenario_job(job, scenario, platform, spec_file, *, dry_run=False, continue_run=False):
    continuation = find_scenario_continuation(job, platform) if continue_run else None
    if continue_run and continuation is None:
        raise ValueError(f"No compatible prior run found for {job.variant_name} seed {job.seed}")
    experiment_dir = (
        continuation.experiment_dir if continuation is not None else _experiment_path(platform, job, timestamp="TIMESTAMP" if dry_run else None)
    )
    if continuation is not None and continuation.checkpoint is not None:
        job.settings["load"] = str(continuation.checkpoint)
    command = _pipeline_command(job, scenario, platform, spec_file, experiment_dir)
    print(shlex.join(command), flush=True)
    if dry_run:
        return

    manifest_path = experiment_dir / "scenario-manifest.json"
    if continuation is None:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "scenario": scenario.model_dump(mode="json"),
            "platform": platform.model_dump(mode="json"),
            "job": job.model_dump(mode="json", exclude={"resolved_config"}),
            "resolved_config": job.resolved_config.model_dump(mode="json"),
            "command": command,
            "git_revision": _git_revision(),
        }
    else:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest.setdefault("continuations", []).append(
            {
                "checkpoint": str(continuation.checkpoint) if continuation.checkpoint else None,
                "step": continuation.step,
                "command": command,
                "git_revision": _git_revision(),
                "submitted_at": datetime.datetime.now().isoformat(),
            }
        )
    with manifest_path.open("w") as handle:
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


def _global_batch_size_arg(value):
    """--global-batch-size: N for every train group, or one per group as GROUP=N,GROUP=N."""
    try:
        if "=" not in value:
            return int(value)
        return {name.strip(): int(size) for name, size in (item.split("=", 1) for item in value.split(","))}
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected N or GROUP=N,GROUP=N, got {value!r}") from None


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
    parser.add_argument("--global-batch-size", type=_global_batch_size_arg, metavar="N|GROUP=N,...")
    parser.add_argument("--data-dir")
    parser.add_argument("--experiments-dir")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Resume this unfinished scenario job",
    )
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
        if args.continue_run:
            continuation = find_scenario_continuation(job, platform)
            if continuation is None:
                raise ValueError(f"No compatible prior run found for {job.variant_name} seed {job.seed}")
            target_step = job.resolved_config.num_steps
            if continuation.step >= target_step:
                print(f"Skipping completed job {job.variant_name} seed {job.seed}: step {continuation.step}/{target_step}")
                continue
        run_scenario_job(
            job,
            scenario,
            platform,
            spec_file,
            dry_run=args.dry_run,
            continue_run=args.continue_run,
        )


if __name__ == "__main__":
    main()
