"""Configuration checks for CLD validation and calibration particle guns."""

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GUN_SAMPLES = {"gun_e_10gev", "gun_mu_10gev", "gun_pi_10gev"}


def load_productions():
    with (REPOSITORY_ROOT / "particleflow_spec.yaml").open() as handle:
        return yaml.safe_load(handle)["productions"]


def test_cld_and_idea_campaigns_contain_only_requested_collisions():
    productions = load_productions()
    expected = {"ttbar", "qq", "ww_fullhad", "z_qq", "z_tautau"}
    assert set(productions["cld"]["samples"]) == set(productions["idea"]["samples"]) == expected


def test_cld_guns_remain_available_manually_and_outside_dataset_campaign():
    cld = load_productions()["cld"]
    script = (REPOSITORY_ROOT / "mlpf/data/key4hep/gen/cld/run_sim.sh").read_text()
    for particle in ("e", "mu", "pi"):
        assert f"gun_{particle}_*" in script
    assert GUN_SAMPLES.isdisjoint(cld["samples"])
    assert GUN_SAMPLES.isdisjoint(cld["tfds_mapping"])
    assert GUN_SAMPLES.isdisjoint(cld["tfds_hit_mapping"])
