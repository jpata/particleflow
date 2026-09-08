import awkward as ak
import numpy as np
import pytest

from mlpf.jet_utils import jet_matching_metrics, match_jets
from mlpf.plotting.plot_utils import jet_response_metrics


def make_jets(eta_events, phi_events):
    return ak.zip({"eta": eta_events, "phi": phi_events})


def test_match_jets_is_one_to_one_and_maximizes_angular_matches():
    reference = make_jets([[0.0, 0.08]], [[0.0, 0.0]])
    candidate = make_jets([[0.04, 0.12]], [[0.0, 0.0]])

    reference_indices, candidate_indices = match_jets(reference, candidate, 0.1)

    assert reference_indices == [[0, 1]]
    assert candidate_indices == [[0, 1]]


def test_match_jets_does_not_reuse_a_candidate():
    reference = make_jets([[0.0, 0.05]], [[0.0, 0.0]])
    candidate = make_jets([[0.025]], [[0.0]])

    reference_indices, candidate_indices = match_jets(reference, candidate, 0.1)

    assert len(reference_indices[0]) == 1
    assert candidate_indices == [[0]]


def test_match_jets_wraps_phi_and_handles_empty_events():
    reference = make_jets([[0.0], []], [[np.pi - 0.01], []])
    candidate = make_jets([[0.0], []], [[-np.pi + 0.01], []])

    reference_indices, candidate_indices = match_jets(reference, candidate, 0.1)

    assert reference_indices == [[0], []]
    assert candidate_indices == [[0], []]


def test_jet_matching_metrics_separate_angular_and_response_quality():
    metrics = jet_matching_metrics(
        response_ratios=[1.0, 1.4, 1.6],
        num_reference_jets=4,
        num_candidate_jets=5,
        response_rel_pt_cut=0.5,
    )

    assert metrics["num_reference_jets"] == 4
    assert metrics["num_candidate_jets"] == 5
    assert metrics["num_angular_matches"] == 3
    assert metrics["num_response_qualified_matches"] == 2
    assert metrics["response_rel_pt_cut"] == 0.5
    assert metrics["angular_recall"] == pytest.approx(3 / 4)
    assert metrics["angular_precision"] == pytest.approx(3 / 5)
    assert metrics["angular_f1"] == pytest.approx(6 / 9)
    assert metrics["angular_fake_rate"] == pytest.approx(2 / 5)
    assert metrics["response_qualified_recall"] == pytest.approx(2 / 4)
    assert metrics["response_qualified_precision"] == pytest.approx(2 / 5)
    assert metrics["response_qualified_f1"] == pytest.approx(4 / 9)
    assert metrics["match_frac"] == metrics["angular_recall"]


def test_jet_matching_metrics_reject_invalid_counts_and_cuts():
    with pytest.raises(ValueError, match="cannot exceed"):
        jet_matching_metrics([1.0, 1.0], 1, 2)
    with pytest.raises(ValueError, match="must be positive"):
        jet_matching_metrics([], 0, 0, response_rel_pt_cut=0)


def test_jet_response_metrics_uses_both_collection_denominators():
    yvals = {
        "jet_ratio_target_to_pred_pt": np.asarray([1.0, 1.6]),
        "jets_target_pt": ak.Array([[10.0, 20.0, 30.0]]),
        "jets_pred_pt": ak.Array([[10.0, 32.0, 8.0, 6.0]]),
    }

    metrics = jet_response_metrics(yvals, "target", "pred", response_rel_pt_cut=0.5)

    assert metrics["med"] == pytest.approx(1.3)
    assert metrics["iqr"] == pytest.approx(0.3)
    assert metrics["angular_recall"] == pytest.approx(2 / 3)
    assert metrics["angular_fake_rate"] == pytest.approx(2 / 4)
    assert metrics["response_qualified_recall"] == pytest.approx(1 / 3)
    assert metrics["response_qualified_precision"] == pytest.approx(1 / 4)
