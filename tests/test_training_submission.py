import json
from pathlib import Path

import yaml

from mlpf.training_submission import (
    available_choices,
    build_slurm_submission,
    resolve_flatiron_profile_path,
    resolve_platform_profile_path,
    resolve_scenario_path,
)
from mlpf.training_scenarios import (
    load_platform_profile,
    load_training_scenario,
    resolve_scenario_jobs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_picker_discovers_scenarios_and_accelerators():
    scenarios, accelerators = available_choices(ROOT)

    assert "cld_hits_output_comparison" in scenarios
    assert "cld_hits_backbone_comparison" in scenarios
    assert "cld_pf_hits_comparison" in scenarios
    assert "clic_cld_pf_set_hits_comparison" in scenarios
    assert {"a100", "h100", "h200"}.issubset(accelerators)


def test_multi_production_scenario_submits_one_array_task_per_variant():
    scenario = resolve_scenario_path("clic_cld_pf_set_hits_comparison", ROOT)
    profile = resolve_flatiron_profile_path("h100", ROOT)

    command, jobs = build_slurm_submission(scenario, profile, ROOT)

    assert [job.production_name for job in jobs] == ["cld", "cld", "clic", "clic"]
    assert command[command.index("--array") + 1] == "0-3"


def test_h100_submission_is_derived_from_scenario_and_profile():
    scenario = resolve_scenario_path("cld_hits_output_comparison", ROOT)
    profile = resolve_flatiron_profile_path("h100", ROOT)

    command, jobs = build_slurm_submission(
        scenario,
        profile,
        ROOT,
        seed=2468,
    )

    assert [job.variant_name for job in jobs] == ["elementwise", "set"]
    assert {job.seed for job in jobs} == {2468}
    assert command[command.index("--gpus-per-node") + 1] == "8"
    assert command[command.index("--constraint") + 1] == "h100"
    assert command[command.index("--array") + 1] == "0-1"
    assert command[command.index("--repo-root") + 1] == str(ROOT)
    assert command[-2:] == ["--seed", "2468"]


def test_array_size_includes_all_scenario_seeds():
    scenario = resolve_scenario_path("cld_hits_output_comparison", ROOT)
    profile = resolve_flatiron_profile_path("a100", ROOT)

    command, jobs = build_slurm_submission(scenario, profile, ROOT)

    assert len(jobs) == 2
    assert command[command.index("--array") + 1] == "0-1"
    assert command[command.index("--gpus-per-node") + 1] == "4"


def test_tallinn_submission_uses_typed_gres_and_site_worker():
    scenario = resolve_scenario_path("cld_hits_output_comparison", ROOT)
    profile = resolve_platform_profile_path("l40", ROOT, "tallinn")
    worker = ROOT / "scripts/tallinn/run_scenario.sh"

    command, jobs = build_slurm_submission(
        scenario,
        profile,
        ROOT,
        seed=2468,
        worker=worker,
    )

    assert len(jobs) == 2
    assert command[command.index("--gres") + 1] == "gpu:l40:2"
    assert command[command.index("--mem-per-gpu") + 1] == "80G"
    assert str(worker) in command
    assert "--constraint" not in command


def test_lumi_submission_uses_task_gpus_account_and_container_worker():
    scenario = resolve_scenario_path("cld_hits_output_comparison", ROOT)
    profile = resolve_platform_profile_path("mi250x", ROOT, "lumi")
    worker = ROOT / "scripts/lumi/run_scenario.sh"

    command, jobs = build_slurm_submission(
        scenario,
        profile,
        ROOT,
        worker=worker,
    )

    assert len(jobs) == 2
    assert command[command.index("--gpus-per-task") + 1] == "8"
    assert command[command.index("--account") + 1] == "project_465001293"
    assert command[command.index("--mem") + 1] == "450G"
    assert "--no-requeue" in command
    assert str(worker) in command


def test_picker_discovers_site_specific_accelerators():
    _, tallinn_accelerators = available_choices(ROOT, "tallinn")
    _, lumi_accelerators = available_choices(ROOT, "lumi")

    assert tallinn_accelerators == ["l40"]
    assert lumi_accelerators == ["mi250x"]


def _write_scenario_run(experiments_dir, scenario, profile, job, step):
    run_dir = experiments_dir / scenario.name / f"{job.variant_name}_seed{job.seed}_test"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    manifest = {
        "job": job.model_dump(mode="json", exclude={"resolved_config"}),
        "resolved_config": job.resolved_config.model_dump(mode="json"),
    }
    (run_dir / "scenario-manifest.json").write_text(json.dumps(manifest))
    (checkpoint_dir / f"checkpoint-{step}.pth").touch()


def test_continue_submission_selects_only_unfinished_original_array_indices(tmp_path):
    scenario_path = resolve_scenario_path("clic_cld_pf_set_hits_comparison", ROOT)
    original_profile_path = resolve_flatiron_profile_path("h100", ROOT)
    profile_data = yaml.safe_load(original_profile_path.read_text())
    profile_data["experiments_dir"] = str(tmp_path / "experiments")
    profile_path = tmp_path / "flatiron_h100.yaml"
    profile_path.write_text(yaml.safe_dump(profile_data))

    scenario = load_training_scenario(scenario_path)
    profile = load_platform_profile(profile_path)
    jobs = resolve_scenario_jobs(scenario, profile, spec_file=ROOT / scenario.spec_file)
    for index, job in enumerate(jobs):
        _write_scenario_run(
            Path(profile.experiments_dir),
            scenario,
            profile,
            job,
            40000 if index in {1, 3} else 50000,
        )

    command, selected_jobs = build_slurm_submission(
        scenario_path,
        profile_path,
        ROOT,
        worker=ROOT / "scripts/flatiron/run_uv_scenario.sh",
        continue_run=True,
    )

    assert [job.variant_name for job in selected_jobs] == [
        "cld_set_hits",
        "clic_set_hits",
    ]
    assert command[command.index("--array") + 1] == "1,3"
    assert command[-1] == "--continue"
