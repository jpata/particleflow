from pathlib import Path

from mlpf.training_submission import (
    available_choices,
    build_slurm_submission,
    resolve_flatiron_profile_path,
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
    assert command[-2:] == ["--seed", "2468"]


def test_array_size_includes_all_scenario_seeds():
    scenario = resolve_scenario_path("cld_hits_output_comparison", ROOT)
    profile = resolve_flatiron_profile_path("a100", ROOT)

    command, jobs = build_slurm_submission(scenario, profile, ROOT)

    assert len(jobs) == 2
    assert command[command.index("--array") + 1] == "0-1"
    assert command[command.index("--gpus-per-node") + 1] == "4"
