import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/local/make_local_available_spec.py"
TRAIN_SCENARIO_SCRIPT = ROOT / "scripts/local/train_scenario.sh"


def _model_dataset(output_spec, model_name):
    sample = output_spec["models"][model_name]["train_datasets"]["physical"]["samples"][0]
    return sample["version"], sample["splits"]


def test_local_spec_uses_current_cld_and_clic_dataset_versions(tmp_path):
    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "pyg-cld-v1": {},
                    "pyg-clic-v1": {},
                    "pyg-cld-hits-v1": {},
                    "pyg-clic-hits-v1": {},
                }
            }
        )
    )

    subprocess.run([sys.executable, SCRIPT, input_path, output_path], check=True)

    output_spec = yaml.safe_load(output_path.read_text())
    assert _model_dataset(output_spec, "pyg-cld-v1") == ("3.2.1", [str(index) for index in range(1, 11)])
    assert _model_dataset(output_spec, "pyg-clic-v1") == ("3.2.1", [str(index) for index in range(1, 11)])
    assert _model_dataset(output_spec, "pyg-cld-hits-v1") == ("3.2.1", ["1"])
    assert _model_dataset(output_spec, "pyg-clic-hits-v1") == ("3.2.1", ["1"])


def test_local_spec_accepts_pf_version_and_split_overrides(tmp_path):
    input_path = tmp_path / "input.yaml"
    output_path = tmp_path / "output.yaml"
    input_path.write_text(yaml.safe_dump({"models": {"pyg-cld-v1": {}}}))

    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            input_path,
            output_path,
            "--pf-version",
            "9.8.7",
            "--pf-splits",
            "2",
            "4",
        ],
        check=True,
    )

    output_spec = yaml.safe_load(output_path.read_text())
    assert _model_dataset(output_spec, "pyg-cld-v1") == ("9.8.7", ["2", "4"])


def test_local_training_launcher_forwards_pf_overrides_without_global_split_override():
    launcher = TRAIN_SCENARIO_SCRIPT.read_text()

    assert "PF_VERSION=${PF_VERSION:-3.2.1}" in launcher
    assert "PF_SPLITS=${PF_SPLITS:-1}" in launcher
    assert '--pf-version "$PF_VERSION"' in launcher
    assert '--pf-splits "${PF_SPLIT_LIST[@]}"' in launcher
    assert "DATA_CONFIG=${DATA_CONFIG:-}" in launcher
    assert 'if [[ -n "$DATA_CONFIG" ]]' in launcher
