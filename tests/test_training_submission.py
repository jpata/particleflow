import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from mlpf.training_scenarios import SlurmProfile, load_platform_profile, load_training_scenario, resolve_scenario_jobs
from mlpf.training_submission import (
    available_choices,
    build_slurm_submission,
    build_slurm_submissions,
    resolve_platform_profile_path,
    resolve_scenario_path,
    slurm_resource_args,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def submission_files(tmp_path):
    """Controlled inputs exercise submission behavior independently of site and experiment recipes."""
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "submission-test",
                "production_name": "cld",
                "variants": {"first": {"model_name": "pyg-cld-v1", "gpus": 1}, "second": {"model_name": "pyg-cld-v1"}},
                "seeds": [11, 29],
                "training": {"global_batch_size": 96, "parameters": {"num_steps": 100}},
            },
            sort_keys=False,
        )
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "gpus": 4,
                "data_dir": str(tmp_path / "data"),
                "experiments_dir": str(tmp_path / "experiments"),
                "slurm": {"partition": "test", "gpu_request": "gpus-per-node"},
            }
        )
    )
    return scenario_path, profile_path


@pytest.mark.parametrize(
    "gpu_request,gpu_type,expected",
    [
        ("gpus-per-node", None, ["--gpus-per-node", "3"]),
        ("gpus-per-task", None, ["--gpus-per-task", "3"]),
        ("gres", None, ["--gres", "gpu:3"]),
        ("gres", "testgpu", ["--gres", "gpu:testgpu:3"]),
    ],
)
def test_slurm_gpu_request_format(gpu_request, gpu_type, expected):
    args = slurm_resource_args(SlurmProfile(gpu_request=gpu_request, gpu_type=gpu_type), 3)
    index = args.index(expected[0])
    assert args[index : index + 2] == expected
    assert sum(flag in args for flag in ("--gpus-per-node", "--gpus-per-task", "--gres")) == 1


@pytest.mark.parametrize("memory_field,memory_flag", [("memory", "--mem"), ("memory_per_gpu", "--mem-per-gpu")])
def test_slurm_optional_resources_are_forwarded(memory_field, memory_flag):
    options = {"constraint": "test-constraint", "account": "test-account", "qos": "test-qos", "reservation": "test-reservation"}
    profile = SlurmProfile(**options, **{memory_field: "24G"}, no_requeue=True)
    args = slurm_resource_args(profile, 2)
    for name, value in options.items():
        assert args[args.index("--" + name) + 1] == value
    assert args[args.index(memory_flag) + 1] == "24G"
    assert "--no-requeue" in args


def test_slurm_optional_resources_are_omitted_when_unset():
    args = slurm_resource_args(SlurmProfile(), 2)
    assert not set(args).intersection({"--constraint", "--account", "--qos", "--reservation", "--mem", "--mem-per-gpu", "--no-requeue"})


@pytest.mark.parametrize(
    "cpus_per_gpu,cpus_per_task,gpus,expected",
    [(None, 7, 3, 7), (5, 7, 3, 15), (5, 7, 1, 5)],
)
def test_slurm_cpu_request_uses_per_gpu_setting_when_present(cpus_per_gpu, cpus_per_task, gpus, expected):
    args = slurm_resource_args(SlurmProfile(cpus_per_gpu=cpus_per_gpu, cpus_per_task=cpus_per_task), gpus)
    assert args.count("--cpus-per-task") == 1
    assert args[args.index("--cpus-per-task") + 1] == str(expected)


@pytest.mark.parametrize("cpus_per_gpu", [0, -1])
def test_slurm_rejects_nonpositive_cpus_per_gpu(cpus_per_gpu):
    with pytest.raises(ValueError, match="cpus_per_gpu"):
        SlurmProfile(cpus_per_gpu=cpus_per_gpu)


def test_submission_scales_cpu_request_with_effective_gpu_count(submission_files):
    scenario_path, profile_path = submission_files
    profile_data = yaml.safe_load(profile_path.read_text())
    profile_data["slurm"].update(cpus_per_gpu=5, cpus_per_task=7)
    profile_path.write_text(yaml.safe_dump(profile_data))
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT)
    assert len(submissions) == 2
    for (command, jobs), gpus, cpus in zip(submissions, (1, 4), (5, 20)):
        assert command[command.index("--cpus-per-task") + 1] == str(cpus)
        assert all(job.resolved_config.gpus == gpus for job in jobs)


def test_partial_node_settings_apply_only_below_the_platform_gpu_count(submission_files):
    scenario_path, profile_path = submission_files
    profile_data = yaml.safe_load(profile_path.read_text())
    profile_data["slurm"].update(
        partition="full",
        memory="480G",
        cpus_per_task=32,
        partial_node={"partition": "shared", "cpus_per_gpu": 7, "memory": None, "memory_per_gpu": "60G"},
    )
    profile_path.write_text(yaml.safe_dump(profile_data))
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT)

    def option(command, flag):
        return command[command.index(flag) + 1] if flag in command else None

    (single, single_jobs), (full, full_jobs) = submissions
    assert {job.resolved_config.gpus for job in single_jobs} == {1} and {job.resolved_config.gpus for job in full_jobs} == {4}
    assert [option(single, flag) for flag in ("--partition", "--cpus-per-task", "--mem", "--mem-per-gpu")] == ["shared", "7", None, "60G"]
    assert [option(full, flag) for flag in ("--partition", "--cpus-per-task", "--mem", "--mem-per-gpu")] == ["full", "32", "480G", None]


@pytest.mark.parametrize(
    "partial_node,error",
    [
        ({"memory_per_gpu": "60G"}, "only one of memory and memory_per_gpu"),
        ({"partial_node": {}}, "cannot be nested"),
        ({"no_such_field": 1}, "no_such_field"),
    ],
)
def test_partial_node_settings_are_validated_with_the_profile(partial_node, error):
    with pytest.raises(ValueError, match=error):
        SlurmProfile(memory="480G", partial_node=partial_node)


def test_lumi_release_checkpoints_share_nodes_only_for_single_gpu_jobs():
    submissions = build_slurm_submissions(
        ROOT / "configs/training/scenarios/release_checkpoints.yaml", ROOT / "configs/training/platforms/lumi_mi250x.yaml", ROOT
    )
    partitions = {}
    for command, jobs in submissions:
        for job in jobs:
            partitions[job.variant_name] = (command[command.index("--partition") + 1], command[command.index("--gpus-per-task") + 1])
    assert {name for name, request in partitions.items() if request == ("small-g", "1")} == {
        f"{detector}_pf_{backbone}" for detector in ("cld", "clic", "colliderml") for backbone in ("attention", "heptv2")
    }
    assert {request for name, request in partitions.items() if "_pf_" not in name} == {("standard-g", "8")}


def test_picker_discovers_files_for_selected_site(tmp_path):
    scenarios = tmp_path / "configs/training/scenarios"
    profiles = tmp_path / "configs/training/platforms"
    scenarios.mkdir(parents=True)
    profiles.mkdir(parents=True)
    for name in ("second.yaml", "first.yaml", "ignore.txt"):
        (scenarios / name).touch()
    for name in ("test_fast.yaml", "test_small.yaml", "other_fast.yaml"):
        (profiles / name).touch()
    assert available_choices(tmp_path, "test") == (["first", "second"], ["fast", "small"])
    assert resolve_scenario_path("first", tmp_path) == scenarios / "first.yaml"
    assert resolve_platform_profile_path("fast", tmp_path, "test") == profiles / "test_fast.yaml"


@pytest.mark.parametrize("gpu_request", ["gpus-per-node", "gpus-per-task", "gres"])
def test_submission_groups_tasks_by_gpu_count(submission_files, gpu_request):
    from mlpf.training_scenarios import _pipeline_command

    scenario_path, profile_path = submission_files
    profile_data = yaml.safe_load(profile_path.read_text())
    profile_data["slurm"]["gpu_request"] = gpu_request
    if gpu_request == "gres":
        profile_data["slurm"]["gpu_type"] = "testgpu"
    profile_path.write_text(yaml.safe_dump(profile_data))
    worker = ROOT / "scripts/tallinn/run_scenario.sh"
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT, worker=worker)
    scenario = load_training_scenario(scenario_path)
    profile = load_platform_profile(profile_path)
    assert len(submissions) == 2
    for (command, jobs), count, indices in zip(submissions, (1, 4), ("0,2", "1,3")):
        request = f"gpu:testgpu:{count}" if gpu_request == "gres" else str(count)
        assert command[command.index("--" + gpu_request) + 1] == request
        assert command[command.index("--array") + 1] == indices
        assert [job.seed for job in jobs] == [11, 29]
        assert str(worker) in command
        for job in jobs:
            assert job.resolved_config.gpus == count
            assert job.per_gpu_batch_size * count == 96
            pipeline = _pipeline_command(job, scenario, profile, ROOT / scenario.spec_file, Path("experiment"))
            assert pipeline[pipeline.index("--gpus") + 1] == str(count)
    with pytest.raises(ValueError, match="multiple GPU resource groups"):
        build_slurm_submission(scenario_path, profile_path, ROOT)


def test_variant_selection_keeps_original_array_indices(submission_files):
    scenario_path, profile_path = submission_files
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT, variants=["second"])
    assert len(submissions) == 1
    command, jobs = submissions[0]
    assert [job.variant_name for job in jobs] == ["second", "second"]
    assert command[command.index("--array") + 1] == "1,3"
    with pytest.raises(ValueError, match="Unknown variants"):
        build_slurm_submissions(scenario_path, profile_path, ROOT, variants=["no_such_variant"])


def test_seed_override_replaces_all_scenario_seeds(submission_files):
    scenario_path, profile_path = submission_files
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT, seed=2468)
    assert sum(len(jobs) for _, jobs in submissions) == 2
    for command, jobs in submissions:
        assert {job.seed for job in jobs} == {2468}
        assert command[-2:] == ["--seed", "2468"]
        assert command[command.index("--repo-root") + 1] == str(ROOT)


def test_single_resource_array_includes_all_seeds(submission_files):
    scenario_path, profile_path = submission_files
    payload = yaml.safe_load(scenario_path.read_text())
    del payload["variants"]["first"]["gpus"]
    scenario_path.write_text(yaml.safe_dump(payload))
    command, jobs = build_slurm_submission(scenario_path, profile_path, ROOT)
    assert len(jobs) == 4
    assert command[command.index("--array") + 1] == "0-3"


def test_continue_preserves_original_indices_across_resource_groups(submission_files):
    scenario_path, profile_path = submission_files
    scenario = load_training_scenario(scenario_path)
    profile = load_platform_profile(profile_path)
    jobs = resolve_scenario_jobs(scenario, profile, spec_file=ROOT / scenario.spec_file)
    for index, job in enumerate(jobs):
        run_dir = Path(profile.experiments_dir) / scenario.name / f"{job.variant_name}_seed{job.seed}_test"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        manifest = {"job": job.model_dump(mode="json", exclude={"resolved_config"}), "resolved_config": job.resolved_config.model_dump(mode="json")}
        (run_dir / "scenario-manifest.json").write_text(json.dumps(manifest))
        step = 40 if index in {1, 2} else 100
        (checkpoint_dir / f"checkpoint-{step}.pth").touch()
    submissions = build_slurm_submissions(scenario_path, profile_path, ROOT, continue_run=True)
    assert [command[command.index("--array") + 1] for command, _ in submissions] == ["1", "2"]
    assert [command[command.index("--gpus-per-node") + 1] for command, _ in submissions] == ["4", "1"]
    assert all(command[-1] == "--continue" for command, _ in submissions)
    assert [(job.variant_name, job.seed) for _, jobs in submissions for job in jobs] == [("second", 11), ("first", 29)]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_container_submission_script_preserves_arguments_and_dry_run(tmp_path, monkeypatch, dry_run, num_groups):
    import mlpf.training_submission as submission

    scenario_path = tmp_path / "scenario.yaml"
    profile_path = tmp_path / "profile.yaml"
    script_path = tmp_path / "submit.sh"
    payload = "argument with spaces; $(false) 'quoted'"
    command = ["printf", "%s", payload]
    monkeypatch.setattr(submission, "available_choices", lambda *args: ([], []))
    monkeypatch.setattr(submission, "resolve_scenario_path", lambda *args: scenario_path)
    monkeypatch.setattr(submission, "resolve_platform_profile_path", lambda *args: profile_path)
    monkeypatch.setattr(submission, "build_slurm_submissions", lambda *args, **kwargs: [(command, [object()])] * num_groups)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)

    # Preparing a container submission must never invoke sbatch itself.
    with monkeypatch.context() as patch:
        patch.setattr(submission.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected submission"))
        args = ["scenario", "mi250x", "--submission-script", str(script_path)]
        if dry_run:
            args.append("--dry-run")
        submission.main(args, site="lumi")

    if dry_run:
        assert not script_path.exists()
    else:
        result = subprocess.run(["bash", str(script_path)], check=True, capture_output=True, text=True)
        assert result.stdout == payload * num_groups


@pytest.mark.parametrize("task_gpus,expected_visible", [("1", "3"), ("8", "unset")])
def test_lumi_worker_preserves_single_gpu_binding(tmp_path, monkeypatch, task_gpus, expected_visible):
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, body in {
        "module": "exit 0",
        "rocm-smi": "exit 0",
        "singularity": 'printf "TEST_VISIBLE=%s\\n" "${ROCR_VISIBLE_DEVICES-unset}"',
    }.items():
        executable = stub_bin / name
        executable.write_text("#!/bin/bash\n" + body + "\n")
        executable.chmod(0o755)
    container = tmp_path / "container.sif"
    container.touch()
    overlay = tmp_path / "overlay"
    (overlay / "bin").mkdir(parents=True)
    (overlay / "bin/activate").touch()
    monkeypatch.setenv("PATH", str(stub_bin) + ":/usr/bin:/bin")
    monkeypatch.setenv("IMG", str(container))
    monkeypatch.setenv("LUMI_VENV", str(overlay))
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", str(tmp_path / "miopen"))
    monkeypatch.setenv("MLPF_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_GPUS_PER_TASK", task_gpus)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/lumi/run_scenario.sh"), "scenario.yaml", "profile.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"TEST_VISIBLE={expected_visible}" in result.stdout


def test_lumi_component_scripts_pin_runs_and_resources(tmp_path, monkeypatch):
    from mlpf.training_submission import write_component_scripts

    # The CI runtime has Bash but no Git.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "bash").symlink_to(shutil.which("bash"))
    monkeypatch.setenv("PATH", str(bin_dir))

    scenario_path = resolve_scenario_path("cms_backbone_comparison", ROOT)
    profile_data = yaml.safe_load(resolve_platform_profile_path("mi250x", ROOT, "lumi").read_text())
    profile_data["experiments_dir"] = str(tmp_path / "experiments")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_data))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "particleflow_spec.yaml").write_text((ROOT / "particleflow_spec.yaml").read_text())
    submissions = build_slurm_submissions(scenario_path, profile_path, repo, worker=repo / "scripts/lumi/run_scenario.sh", seed=2468)
    generated = write_component_scripts(submissions, scenario_path, profile_path, repo)
    assert len(generated) == 2
    manifests = []
    for command, jobs in generated:
        script = Path(command[1])
        content = script.read_text()
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert "#SBATCH --array" not in content
        assert "#SBATCH --gpus-per-task=8" in content
        assert "--run-manifest" in content
        launch = shlex.split(content.splitlines()[-1])
        manifest_path = Path(launch[launch.index("--run-manifest") + 1])
        manifest = json.loads(manifest_path.read_text())
        assert manifest["git_revision"] is None
        assert manifest["job"]["seed"] == 2468
        assert manifest["command"][manifest["command"].index("--experiment-dir") + 1] == str(manifest_path.parent)
        assert Path(manifest["command"][manifest["command"].index("--spec-file") + 1]).is_file()
        manifests.append(manifest_path)
    assert len(set(manifests)) == 2


def test_lumi_component_continue_keeps_saved_manifest(tmp_path):
    from mlpf.training_submission import write_component_scripts

    scenario_path = resolve_scenario_path("cms_backbone_comparison", ROOT)
    profile_data = yaml.safe_load(resolve_platform_profile_path("mi250x", ROOT, "lumi").read_text())
    profile_data["experiments_dir"] = str(tmp_path / "experiments")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_data))
    scenario = load_training_scenario(scenario_path)
    profile = load_platform_profile(profile_path)
    jobs = resolve_scenario_jobs(scenario, profile, spec_file=ROOT / scenario.spec_file)
    for job in jobs:
        run = Path(profile.experiments_dir) / scenario.name / f"{job.variant_name}_seed{job.seed}_test"
        (run / "checkpoints").mkdir(parents=True)
        (run / "scenario-manifest.json").write_text(
            json.dumps(
                {
                    "job": job.model_dump(mode="json", exclude={"resolved_config"}),
                    "resolved_config": job.resolved_config.model_dump(mode="json"),
                }
            )
        )
        step = 10000 if job.variant_name == "heptv2" else job.resolved_config.num_steps
        (run / "checkpoints" / f"checkpoint-{step}.pth").touch()
    run_manifest = next(Path(profile.experiments_dir).glob("*/heptv2*/scenario-manifest.json"))
    original = run_manifest.read_text()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "particleflow_spec.yaml").write_text((ROOT / "particleflow_spec.yaml").read_text())
    submissions = build_slurm_submissions(scenario_path, profile_path, repo, worker=repo / "scripts/lumi/run_scenario.sh", continue_run=True)
    generated = write_component_scripts(submissions, scenario_path, profile_path, repo, continue_run=True)
    assert len(generated) == 1
    assert str(run_manifest.resolve()) in Path(generated[0][0][1]).read_text()
    assert run_manifest.read_text() == original
