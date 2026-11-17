"""FCC-ee generator physics contracts introduced by production PR #527."""

import ast
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "mlpf/data/key4hep/gen"


@pytest.mark.parametrize("path", ["cld/CLDConfig/pythia.py", "clic/pythia.py", "idea/pythia.py", "pythia/generate.py"])
def test_pythia_wrappers_do_not_apply_additional_vertex_smearing(path):
    tree = ast.parse((GENERATOR / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == "GaussSmearVertex")
            assert all(keyword.arg != "VertexSmearingTool" for keyword in node.keywords)
        if isinstance(node, ast.Attribute):
            assert node.attr != "VertexSmearingTool"


def test_fcc_cards_use_energy_dependent_2025_vertex_widths():
    expected = {
        91: (6.36e-3, 28.3e-6, 0.424),
        240: (9.19e-3, 22.6e-6, 0.606),
        365: (26.17e-3, 31.1e-6, 1.198),
    }
    cards = sorted((GENERATOR / "cld/CLDConfig/pythia").glob("p8_ee_*.cmd"))
    assert len(cards) >= 10
    for card in cards:
        settings = {}
        for line in card.read_text().splitlines():
            line = line.split("!", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
        energy = int(settings["Beams:eCM"])
        widths = tuple(float(settings[f"Beams:sigmaVertex{axis}"]) for axis in "XYZ")
        assert settings["Beams:allowVertexSpread"] == "on", card
        assert widths == pytest.approx(expected[energy]), card


def test_cld_guns_and_collisions_use_the_updated_release():
    script = (GENERATOR / "cld/run_sim.sh").read_text()
    assert 'source "$GENERATOR_ROOT/setup.sh"' in script
    assert "/generators/generate_events.sh" in script
    assert "--enableGun" in script
    assert "setup.sh -r 2025-05-29" not in script


def test_all_production_detectors_share_stable_runtime():
    import yaml

    spec = yaml.safe_load((REPOSITORY / "particleflow_spec.yaml").read_text())
    productions = spec["productions"]
    for detector in ("cld", "clic", "ild", "idea"):
        production = productions[detector]
        assert production["gen_environment"]["KEY4HEP_RELEASE"] == "2026-04-08"
        assert production["gen_container"].endswith("/gen/idea/idea-dev.sif")


@pytest.mark.parametrize("case", ["valid", "legacy", "conflicting", "stale", "rolling", "mixed"])
def test_shared_setup_checks_build_identity_and_removes_stock_fork_paths(tmp_path, case):
    import os
    import subprocess

    base = tmp_path / "releases/2026-04-08/setup.sh"
    base.parent.mkdir(parents=True)
    base.write_text("# mock stable stack\n")
    env = {
        "PATH": os.environ["PATH"],
        "KEY4HEP_STACK": str(base),
        "KEY4HEP_LOCAL_ROOT": str(tmp_path / "local"),
        "MLPF_KEY4HEP_REQUEST": "stable:2026-04-08",
        "KEY4HEP_BOOTSTRAP": "1",
    }
    script = f"""
set -eo pipefail
k4_local_repo() {{ :; }}
_replace_marlin_dll() {{ :; }}
source '{GENERATOR / "setup.sh"}'
for component in k4FWCore MarlinReco idea/k4RecTracker-local; do
    directory="$MLPF_KEY4HEP_ROOT/$component"
    mkdir -p "$(key4hep_install_dir "$directory")"
    key4hep_record_build "$directory"
done
unset KEY4HEP_BOOTSTRAP
"""
    if case == "stale":
        script += 'echo stale > "$KEY4HEP_LOCAL_ROOT/install/k4FWCore/.key4hep-build-id"\n'
    if case == "rolling":
        script += "export KEY4HEP_RELEASE=Mon\n"
    if case == "legacy":
        script += "export KEY4HEP_VERSION=2026-04-08\n"
    if case == "conflicting":
        script += "export KEY4HEP_VERSION=2025-05-29\n"
    if case == "mixed":
        script += "export LCG_VIEW_DIR=/mock/nightly\n"
    script += f"source '{GENERATOR / 'setup.sh'}'\n"
    script += """
export PYTHONPATH=/mock/k4rectracker/old/python:/keep/python
key4hep_use_local "$MLPF_KEY4HEP_ROOT/idea/k4RecTracker-local"
test "$PYTHONPATH" = /keep/python
"""
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    if case in ("valid", "legacy"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert {"stale": "Missing or stale", "rolling": "dated stable", "mixed": "clean shell", "conflicting": "disagree"}[case] in result.stderr
