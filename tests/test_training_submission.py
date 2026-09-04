from pathlib import Path

from mlpf.training_submission import (
    available_choices,
    build_slurm_submission,
    resolve_flatiron_profile_path,
    resolve_platform_profile_path,
    resolve_scenario_path,
)


ROOT = Path(__file__).resolve().parents[1]


def test_picker_discovers_scenarios_and_accelerators():
    scenarios, accelerators = available_choices(ROOT)

    assert "cld_hits_output_comparison" in scenarios
    assert {"a100", "b200", "h100"}.issubset(accelerators)


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
