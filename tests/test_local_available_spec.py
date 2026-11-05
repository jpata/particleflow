import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/local/make_local_available_spec.py"
TRAIN_SCENARIO_SCRIPT = ROOT / "scripts/local/train_scenario.sh"


@pytest.mark.parametrize(
    "kind,dataset,models",
    [
        ("pf", "cld", ["pyg-cld-v1"]),
        ("pf", "clic", ["pyg-clic-v1"]),
        ("hit", "cld_hits", ["pyg-cld-hits-v1", "pyg-cld-hits-set-v1", "pyg-cld-hits-heptv2-sector-set-v1", "pyg-cld-hits-heptv2-set-v1"]),
        ("hit", "clic_hits", ["pyg-clic-hits-v1", "pyg-clic-hits-set-v1"]),
        ("idea", "idea", ["pyg-idea-pipeline-v1"]),
    ],
)
def test_local_spec_applies_dataset_overrides_and_preserves_other_settings(tmp_path, kind, dataset, models):
    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    original = {"models": {name: {"hyperparameters": {"lr": 0.123}} for name in models}}
    original["models"]["unrelated-model"] = {"dataset": "unrelated"}
    input_path.write_text(yaml.safe_dump(original))
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            input_path,
            output_path,
            f"--{kind}-version",
            "9.8.7",
            f"--{kind}-splits",
            "2",
            "4",
        ],
        check=True,
    )
    output = yaml.safe_load(output_path.read_text())
    assert output["models"]["unrelated-model"] == original["models"]["unrelated-model"]
    for name in models:
        model = output["models"][name]
        assert model["dataset"] == dataset
        assert model["hyperparameters"] == original["models"][name]["hyperparameters"]
        train = model["train_datasets"]["physical"]["samples"]
        assert len(train) == 1
        assert train[0]["version"] == "9.8.7"
        assert train[0]["splits"] == ["2", "4"]
        assert model["validation_datasets"]["physical"]["samples"] == model["test_datasets"] == train


@pytest.mark.parametrize("data_config", ["", "selected-split"])
def test_local_launcher_forwards_overrides_as_arguments(tmp_path, data_config):
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    log = tmp_path / "commands.jsonl"
    uv = stub_bin / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_UV_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    uv.chmod(0o755)
    scenario = tmp_path / "scenario with spaces.yaml"
    scenario.write_text("name: test\n")
    spec = str(tmp_path / "spec with spaces.yaml")
    local_spec = str(tmp_path / "local spec.yaml")
    subprocess.run(
        ["bash", TRAIN_SCENARIO_SCRIPT, scenario, "--dry-run"],
        check=True,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "TEST_UV_LOG": str(log),
            "SPEC_FILE": spec,
            "LOCAL_SPEC_FILE": local_spec,
            "USE_LOCAL_AVAILABLE_SPEC": "true",
            "PF_VERSION": "9.8.7",
            "PF_SPLITS": "2 4",
            "HIT_VERSION": "8.7.6",
            "HIT_SPLITS": "3 5",
            "DATA_CONFIG": data_config,
        },
    )
    prepare, run = [json.loads(line) for line in log.read_text().splitlines()]
    assert prepare[3:5] == [spec, local_spec]
    for flag, values in {"--pf-version": ["9.8.7"], "--pf-splits": ["2", "4"], "--hit-version": ["8.7.6"], "--hit-splits": ["3", "5"]}.items():
        index = prepare.index(flag) + 1
        assert prepare[index : index + len(values)] == values
    assert run[run.index("--scenario") + 1] == str(scenario)
    assert run[run.index("--spec-file") + 1] == local_spec
    assert run[-1] == "--dry-run"
    settings = [run[index + 1] for index, arg in enumerate(run) if arg == "--set"]
    assert [setting for setting in settings if setting.startswith("data_config=")] == ([f"data_config={data_config}"] if data_config else [])
