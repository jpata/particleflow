"""Configuration checks for CLD validation and calibration particle guns."""

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GUN_SAMPLES = {"gun_e_10gev", "gun_mu_10gev", "gun_pi_10gev"}


def load_productions():
    with (REPOSITORY_ROOT / "particleflow_spec.yaml").open() as handle:
        return yaml.safe_load(handle)["productions"]


def test_cld_guns_match_idea_campaign():
    productions = load_productions()
    cld_samples = productions["cld"]["samples"]
    idea_samples = productions["idea"]["samples"]

    for sample_name in GUN_SAMPLES:
        assert cld_samples[sample_name]["process_name"] == idea_samples[sample_name]["process_name"]
        assert cld_samples[sample_name]["seed_range"] == idea_samples[sample_name]["seed_range"]
        assert cld_samples[sample_name]["events_per_job"] == idea_samples[sample_name]["events_per_job"]
        assert cld_samples[sample_name]["gen_script"] == "mlpf/data/key4hep/gen/cld/run_sim.sh"


def test_cld_guns_are_excluded_from_training_datasets():
    cld = load_productions()["cld"]

    assert GUN_SAMPLES.isdisjoint(cld["tfds_mapping"])
    assert GUN_SAMPLES.isdisjoint(cld["tfds_hit_mapping"])
