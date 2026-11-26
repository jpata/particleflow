"""Tests for the temporary IDEA MLPF pipeline-validation path."""

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import yaml

from mlpf.conf import Dataset, MLPFConfig
from mlpf.data.key4hep.prepare_idea_pipeline import prepare_pipeline_file
from mlpf.heptfds.edm4hep_utils.utils_pf import N_X_FEATURES
from mlpf.heptfds.edm4hep_utils.utils_idea import (
    CANDIDATE_SOURCE,
    TRACK_SOURCE,
    PROXY_TRACK_SOURCE,
    TRACK_COLLECTION,
    TRACK_QUALITY_POLICY,
    TRACK_TRUTH_POLICY,
    TRACK_TRUTH_MIN_PURITY,
    validate_track_provenance,
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
            "metadata_track_source": ak.Array([TRACK_SOURCE]),
            "metadata_track_collection": ak.Array([TRACK_COLLECTION]),
            "metadata_track_quality_policy": ak.Array([TRACK_QUALITY_POLICY]),
            "metadata_track_truth_policy": ak.Array([TRACK_TRUTH_POLICY]),
            "metadata_track_truth_min_purity": ak.Array([TRACK_TRUTH_MIN_PURITY]),
            "X_track": ak.Array([np.ones((1, 16), dtype=np.float32)]),
            "X_cluster": ak.Array([np.ones((1, 19), dtype=np.float32)]),
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
def test_legacy_idea_100_parquet_loads_only_in_explicit_proxy_mode():
    examples = prepare_data_idea(IDEA_100_PARQUET, event_indices=[0, 31, 99], expected_track_source=PROXY_TRACK_SOURCE)

    assert len(examples) == 3
    for example in examples:
        assert example["X"].ndim == 2 and example["X"].shape[1] == 19
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
    assert config.input_dim == N_X_FEATURES
    assert config.num_classes == 6


def test_idea_tfds_mapping_excludes_validation_gun():
    with (REPOSITORY_ROOT / "particleflow_spec.yaml").open() as handle:
        production = yaml.safe_load(handle)["productions"]["idea"]

    guns = {name for name in production["samples"] if name.startswith("gun_")}
    assert not guns
    assert all(not name.startswith("gun_") for name in production["tfds_mapping"])


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


def _point_on_helix(d0, phi0, z0, omega, tan_lambda, arc):
    """Position and direction after a transverse arc from the origin-referenced perigee (EDM4hep/LCIO convention)."""
    x = -d0 * np.sin(phi0) + (np.sin(phi0) - np.sin(phi0 - omega * arc)) / omega
    y = d0 * np.cos(phi0) + (np.cos(phi0 - omega * arc) - np.cos(phi0)) / omega
    return x, y, z0 + arc * tan_lambda, phi0 - omega * arc


def test_perigee_at_origin_recovers_the_helix_parameters():
    from mlpf.data.key4hep.postprocessing import perigee_at_origin

    rng = np.random.default_rng(0)
    n = 1000
    d0 = rng.uniform(-500, 500, n)
    phi0 = rng.uniform(-np.pi, np.pi, n)
    z0 = rng.uniform(-1000, 1000, n)
    omega = rng.choice([-1, 1], n) * rng.uniform(1e-4, 1e-2, n)
    tan_lambda = rng.uniform(-3, 3, n)
    # a perigee is the circle's closest point to the beam axis only if omega * d0 < 1
    d0 = np.where(omega * d0 < 1, d0, -d0)
    assert np.all(omega * d0 < 1)
    # anywhere within half a turn of the perigee, before or after it
    arc = rng.uniform(-0.99, 0.99, n) * np.pi / np.abs(omega)
    x, y, z, phi = _point_on_helix(d0, phi0, z0, omega, tan_lambda, arc)

    d0_out, phi0_out, z0_out = perigee_at_origin(x, y, z, phi, omega, tan_lambda)

    np.testing.assert_allclose(d0_out, d0, atol=1e-6)
    np.testing.assert_allclose(np.angle(np.exp(1j * (phi0_out - phi0))), 0, atol=1e-9)
    np.testing.assert_allclose(z0_out, z0, atol=1e-6)


def test_perigee_at_origin_keeps_tracks_from_the_origin_and_straight_tracks():
    from mlpf.data.key4hep.postprocessing import perigee_at_origin

    phi, omega, tan_lambda = np.array([0.3, -2.0]), np.array([1e-3, -5e-4]), np.array([0.5, -1.0])
    d0, phi0, z0 = perigee_at_origin(np.zeros(2), np.zeros(2), np.array([5.0, -3.0]), phi, omega, tan_lambda)
    np.testing.assert_allclose(d0, 0, atol=1e-12)
    np.testing.assert_allclose(phi0, phi)
    np.testing.assert_allclose(z0, [5.0, -3.0])

    # straight (omega = 0): the perigee is the foot of the perpendicular from the origin
    d0, phi0, z0 = perigee_at_origin(np.array([100.0]), np.array([50.0]), np.array([20.0]), np.array([0.0]), np.array([0.0]), np.array([0.5]))
    np.testing.assert_allclose([d0[0], phi0[0], z0[0]], [50.0, 0.0, 20.0 - 100.0 * 0.5])


def test_proxy_features_keep_upstream_perigee_and_first_hit_fixes():
    from mlpf.data.key4hep.postprocessing import idea_track_to_features

    coll = "TracksFromGenParticles"
    states = f"_{coll}_trackStates"
    x, y, z, phi = _point_on_helix(5.0, 0.3, 20.0, 0.001, 0.5, 100.0)
    track_fields = {"type": [0], "chi2": [0.0], "ndf": [0], "trackStates_begin": [0], "trackStates_end": [2]}
    state_fields = {
        "location": [1, 2],
        "phi": [phi, phi],
        "omega": [0.001, 0.001],
        "tanLambda": [0.5, 0.5],
        "D0": [999.0, 0.0],
        "Z0": [999.0, 0.0],
        "time": [0.0, 0.0],
        "referencePoint.x": [x, 30.0],
        "referencePoint.y": [y, 40.0],
        "referencePoint.z": [z, 0.0],
    }
    prop = {
        coll: [ak.Array({f"{coll}.{key}": value for key, value in track_fields.items()})],
        states: [ak.Array({f"{states}.{key}": value for key, value in state_fields.items()})],
    }
    features = idea_track_to_features(prop, 0, 2.0, coll=coll)
    np.testing.assert_allclose([features[name][0] for name in ("D0", "phi", "Z0")], [5.0, 0.3, 20.0])
    np.testing.assert_allclose(features["radiusOfInnermostHit"], [50.0])


@pytest.mark.parametrize("source", [None, PROXY_TRACK_SOURCE, "mixed"])
def test_current_loader_rejects_missing_proxy_or_mixed_provenance(source):
    fields = {"X_track": ak.Array([[], []])}
    if source is not None:
        fields["metadata_track_source"] = ak.Array([TRACK_SOURCE, PROXY_TRACK_SOURCE] if source == "mixed" else [source, source])
    with pytest.raises(ValueError, match="provenance"):
        validate_track_provenance(ak.Record(fields))


def test_pipeline_preparation_does_not_relabel_unmarked_legacy_data(tmp_path):
    data = ak.Record(
        {
            "X_track": ak.Array([[]]),
            "X_cluster": ak.Array([[]]),
            "ytarget_track": ak.Array([[]]),
            "ytarget_cluster": ak.Array([[]]),
            "targetjet": ak.Array([[]]),
        }
    )
    source, target = tmp_path / "source.parquet", tmp_path / "target.parquet"
    ak.to_parquet(data, source)
    with pytest.raises(ValueError, match="provenance"):
        prepare_pipeline_file(source, target)
    prepare_pipeline_file(source, target, legacy_proxy=True)
    assert ak.from_parquet(target)["metadata_track_source"].to_list() == [PROXY_TRACK_SOURCE]


def test_fitted_provenance_requires_quality_and_truth_policy():
    data = ak.Record({"X_track": ak.Array([[]]), "metadata_track_source": ak.Array([TRACK_SOURCE])})
    with pytest.raises(ValueError, match="metadata_track_collection"):
        validate_track_provenance(data)
