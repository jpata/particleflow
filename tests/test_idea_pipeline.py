"""Tests for the temporary IDEA MLPF pipeline-validation path."""

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import yaml

from mlpf.conf import Dataset, MLPFConfig
from mlpf.data.key4hep.prepare_idea_pipeline import prepare_pipeline_file
from mlpf.heptfds.edm4hep_utils.utils_idea import (
    CANDIDATE_SOURCE,
    TRACK_SOURCE,
    find_idea_parquets,
    make_oracle_candidates,
    prepare_data_idea,
    repair_jet_indices,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IDEA_100_PARQUET = REPOSITORY_ROOT / "idea-100/postprocessing/reco_p8_ee_qq_ecm365_424242.parquet"


def test_oracle_candidate_is_an_independent_copy():
    target = np.arange(28, dtype=np.float32).reshape(2, 14)
    candidate = make_oracle_candidates(target)
    candidate[0, 0] = -1
    assert target[0, 0] == 0


def test_invalid_representative_jet_indices_are_marked_unclustered():
    target = np.zeros((3, 14), dtype=np.float32)
    target[:, 0] = [211, 22, 0]
    target[:, 12] = [0, 3, 8]

    repaired, count = repair_jet_indices(target, num_target_jets=1)

    assert count == 1
    np.testing.assert_array_equal(repaired[:, 12], [0, -1, 8])


def test_prepare_pipeline_file_adds_provenance_and_oracle_candidates(tmp_path):
    target_track = np.zeros((1, 14), dtype=np.float32)
    target_track[0, [0, 2, 6, 12, 13]] = [211, 5, 5.2, 0, 1]
    source = ak.Record(
        {
            "X_track": ak.Array([np.ones((1, 16), dtype=np.float32)]),
            "X_cluster": ak.Array([np.ones((1, 17), dtype=np.float32)]),
            "ytarget_track": ak.Array([target_track]),
            "ytarget_cluster": ak.Array([np.zeros((1, 14), dtype=np.float32)]),
            "targetjet": ak.Array([np.zeros((0, 4), dtype=np.float32)]),
        }
    )
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "prepared.parquet"
    ak.to_parquet(source, input_path)

    summary = prepare_pipeline_file(input_path, output_path)
    prepared = ak.from_parquet(output_path)

    assert summary == {"events": 1, "repaired_jet_indices": 1}
    assert prepared["metadata_track_source"][0] == TRACK_SOURCE
    assert prepared["metadata_candidate_source"][0] == CANDIDATE_SOURCE
    assert not prepared["metadata_suitable_for_physics"][0]
    np.testing.assert_array_equal(
        ak.to_numpy(prepared["ycand_track"][0]),
        ak.to_numpy(prepared["ytarget_track"][0]),
    )
    assert prepared["ytarget_track"][0, 0, 12] == -1


@pytest.mark.skipif(
    not IDEA_100_PARQUET.exists(),
    reason="workspace IDEA 100-event parquet is not available",
)
def test_current_idea_100_parquet_loads_into_mlpf_contract():
    examples = prepare_data_idea(IDEA_100_PARQUET, event_indices=[0, 31, 99])

    assert len(examples) == 3
    for example in examples:
        assert example["X"].ndim == 2 and example["X"].shape[1] == 17
        assert example["ytarget"].shape == example["ycand"].shape
        assert example["ytarget"].shape == (len(example["X"]), 14)
        assert np.all(np.isfinite(example["X"]))
        assert np.all(np.isfinite(example["ytarget"]))
        np.testing.assert_array_equal(example["ycand"], example["ytarget"])
        assert set(np.unique(example["ytarget"][:, 0])).issubset(set(range(6)))
        num_jets = len(example["targetjets"])
        representatives = example["ytarget"][:, 0] > 0
        jet_indices = example["ytarget"][representatives, 12]
        assert np.all((jet_indices == -1) | ((jet_indices >= 0) & (jet_indices < num_jets)))


def test_real_spec_builds_idea_pipeline_config():
    config = MLPFConfig.from_spec(REPOSITORY_ROOT / "particleflow_spec.yaml", "pyg-idea-pipeline-v1", "idea")

    assert config.dataset is Dataset.IDEA
    assert config.input_dim == 17
    assert config.num_classes == 6
    assert config.ntrain == 10
    assert "idea_edm_qq_pf" in config.test_dataset


def test_idea_tfds_mapping_excludes_validation_gun():
    with (REPOSITORY_ROOT / "particleflow_spec.yaml").open() as handle:
        mapping = yaml.safe_load(handle)["productions"]["idea"]["tfds_mapping"]

    assert set(mapping) == {"ttbar", "ww_fullhad", "qq"}


def test_idea_parquet_discovery_selects_requested_process(tmp_path):
    qq_dir = tmp_path / "p8_ee_qq_ecm365"
    ttbar_dir = tmp_path / "p8_ee_ttbar_ecm365"
    qq_dir.mkdir()
    ttbar_dir.mkdir()
    qq_file = qq_dir / "qq.parquet"
    ttbar_file = ttbar_dir / "ttbar.parquet"
    qq_file.touch()
    ttbar_file.touch()

    assert find_idea_parquets(tmp_path, "p8_ee_ttbar_ecm365") == [ttbar_file]


def test_idea_test_campaign_has_exactly_100_jobs_per_sample():
    with (REPOSITORY_ROOT / "particleflow_spec.yaml").open() as handle:
        samples = yaml.safe_load(handle)["productions"]["idea"]["samples"]

    expected_samples = {"gun_e_10gev", "gun_mu_10gev", "gun_pi_10gev", "ttbar", "ww_fullhad", "qq"}
    assert set(samples) == expected_samples
    for sample in samples.values():
        start, stop = sample["seed_range"]
        assert stop - start == 100

    for sample_name in expected_samples:
        assert samples[sample_name]["events_per_job"] == 100
