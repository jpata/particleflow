"""FCC-ee generator physics contracts introduced by production PR #527."""

import ast
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY / "mlpf/data/key4hep/gen"


@pytest.mark.parametrize("path", ["cld/CLDConfig/pythia.py", "clic/pythia.py", "idea/pythia.py"])
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
    assert len(cards) == 10
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
    assert script.count("source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08") == 2
    assert "--enableGun" in script
    assert "setup.sh -r 2025-05-29" not in script
