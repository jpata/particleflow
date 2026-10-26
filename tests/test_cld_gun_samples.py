"""Configuration checks for CLD validation and calibration particle guns."""

from pathlib import Path
from xml.etree import ElementTree

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


def test_paired_photon_readout_productions():
    productions = load_productions()
    baseline = productions["cld_photon_5p1"]
    fine = productions["cld_photon_2mm"]

    assert baseline["generation_only"] and fine["generation_only"]
    assert baseline["save_sim"] and fine["save_sim"]
    assert baseline["samples"] == fine["samples"]
    assert baseline["workspace_dir"] != fine["workspace_dir"]
    assert "tfds_mapping" not in baseline and "tfds_mapping" not in fine
    assert "tfds_hit_mapping" not in baseline and "tfds_hit_mapping" not in fine

    sample = baseline["samples"]["gun_gamma_10gev"]
    assert sample["gen_script"] == "scripts/cld_photon_readout/run_sim.sh"
    assert sample["seed_range"] == [810000, 810010]

    geometry = REPOSITORY_ROOT / fine["compact_file"].replace("${project.paths.project_root}/", "")
    assert geometry.exists()
    barrel = ElementTree.parse(geometry.parent / "ECalBarrel_o2_v01_03.xml")
    endcap = ElementTree.parse(geometry.parent / "ECalEndcap_o2_v01_03.xml")
    assert barrel.find(".//constant[@name='ECal_cell_size']").attrib["value"] == "2.0*mm"
    assert barrel.find(".//readout[@name='ECalBarrelCollection']/segmentation").attrib["grid_size_x"] == "ECal_cell_size"
    assert endcap.find(".//readout[@name='ECalEndcapCollection']/segmentation").attrib["grid_size_x"] == "ECal_cell_size"


def test_ttbar_2mm_validation_matches_existing_cld_sample():
    productions = load_productions()
    baseline = productions["cld"]["samples"]["ttbar"]
    fine = productions["cld_ecal_2mm"]
    sample = fine["samples"]["ttbar"]

    assert fine["generation_only"]
    assert "tfds_mapping" not in fine and "tfds_hit_mapping" not in fine
    assert sample["process_name"] == baseline["process_name"]
    assert sample["events_per_job"] == baseline["events_per_job"]
    assert sample["seed_range"] == [baseline["seed_range"][0], baseline["seed_range"][0] + 10]
    assert sample["gen_script"] == "scripts/cld_photon_readout/run_sim.sh"
    assert fine["compact_file"] == productions["cld_photon_2mm"]["compact_file"]
