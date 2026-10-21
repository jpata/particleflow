from types import SimpleNamespace

import pytest
import torch

from mlpf.conf import Y_FEATURES
from mlpf.model.reconstruction_diagnostics import compute_hit_grouping_metrics
from mlpf.model.training import _finalize_diagnostics
from mlpf.model.validation_metrics import compute_validation_particle_metrics


def finish(values):
    return _finalize_diagnostics({k: [torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)] for k, (a, b) in values.items()}, 1)


def grouping_case():
    # Truth A: tracker 4 correct + 2 split; truth B: tracker 4 correct.
    # One background hit contaminates A; another is correctly rejected.
    truth = [20] * 6 + [10] * 4 + [0, 0] + [20, 20, 10, 0]
    winners = [1] * 4 + [0] * 6 + [1, 2] + [1, 0, 0, 1]
    count = len(truth)
    x = torch.zeros(1, count + 1, 15)
    x[0, :12, 0] = 1
    x[0, 12:, 0] = 2
    x[0, 12:16, 5] = torch.tensor([8.0, 2.0, 1.0, 3.0])
    x[0, :, 14] = torch.arange(count + 1)
    targets = torch.zeros(1, 2, len(Y_FEATURES))
    targets[0, :, Y_FEATURES.index("particle_number")] = torch.tensor([20, 10])
    targets[0, :, Y_FEATURES.index("pt")] = torch.tensor([2.0, 20.0]).log()
    targets[0, :, Y_FEATURES.index("eta")] = torch.tensor([0.1, 2.1])
    targets[0, :, Y_FEATURES.index("cos_phi")] = 1
    hit_target = torch.zeros(1, count + 1, len(Y_FEATURES))
    hit_target[0, :count, Y_FEATURES.index("particle_number")] = torch.tensor(truth)
    mask = torch.arange(count + 1)[None] < count
    keys = torch.zeros(1, count + 1, 2)
    bg = torch.full((1, count + 1, 1), -10.0)
    for i, slot in enumerate(winners):
        if slot < 2:
            keys[0, i, slot] = 1
        else:
            bg[0, i] = 20
    batch = SimpleNamespace(X=x, mask=mask, ytarget=hit_target, ytarget_set=targets, target_mask=torch.ones(1, 2, dtype=torch.bool))
    matches = [(torch.tensor([1, 0]), torch.tensor([0, 1]))]
    return (keys, torch.eye(2)[None], bg), batch, matches


def test_grouping_overlap_split_merge_and_energy_weighting():
    grouping, batch, matches = grouping_case()
    raw = compute_hit_grouping_metrics(grouping, batch, matches, chunk_size=1)
    values = finish(raw)
    assert values["tracker/hit_samples"] == 12  # padded calo hit excluded
    assert values["tracker/purity"] == pytest.approx(8 / 11)
    assert values["tracker/completeness"] == pytest.approx(8 / 10)
    assert values["tracker/background_recall"] == 0.5
    assert values["tracker/all/object_purity"] == pytest.approx((4 / 5 + 4 / 6) / 2)
    assert values["tracker/all/object_completeness"] == pytest.approx((4 / 6 + 1) / 2)
    assert values["tracker/all/split_fraction"] == 0.5
    assert values["tracker/merge_fraction"] == 0.5
    assert values["tracker/all/merged_slot_fraction"] == 0.5
    assert values["tracker/pt_1to10/object_completeness"] == pytest.approx(4 / 6)
    assert values["tracker/abs_eta_ge2/object_completeness"] == 1
    assert values["tracker/surfaces_4to7/object_samples"] == 2
    assert values["tracker/neighbors_0/object_samples"] == 2
    assert values["calorimeter/purity"] == 0.5
    assert values["calorimeter/completeness"] == pytest.approx(2 / 3)
    assert values["calorimeter/energy_purity"] == pytest.approx(9 / 14)
    assert values["calorimeter/energy_completeness"] == pytest.approx(9 / 11)
    assert raw == compute_hit_grouping_metrics(grouping, batch, matches, chunk_size=100)


def test_grouping_slot_permutation_and_missing_matched_slot():
    grouping, batch, matches = grouping_case()
    original = compute_hit_grouping_metrics(grouping, batch, matches)
    permuted = (grouping[0], grouping[1].flip(1), grouping[2])
    reordered_matches = [(1 - matches[0][0], matches[0][1])]
    assert original == compute_hit_grouping_metrics(permuted, batch, reordered_matches)
    values = finish(compute_hit_grouping_metrics(grouping, batch, [(torch.tensor([1]), torch.tensor([0]))]))
    assert values["tracker/completeness"] == pytest.approx(4 / 10)
    assert values["tracker/pt_ge10/object_completeness"] == 0
    assert values["tracker/pt_ge10/best_slot_completeness"] == 1


def test_grouping_empty_events_keep_keys_and_zero_support():
    grouping, batch, matches = grouping_case()
    original = compute_hit_grouping_metrics(grouping, batch, matches)
    batch.target_mask[:] = False
    batch.mask[:] = False
    raw = compute_hit_grouping_metrics(grouping, batch, [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))])
    assert list(raw) == list(original)
    values = finish(raw)
    assert values["tracker/hit_samples"] == 0
    assert "tracker/purity" not in values
    with pytest.raises(ValueError):
        compute_hit_grouping_metrics(grouping, batch, matches, chunk_size=0)


def collection(pt, eta, cls):
    return {
        key: torch.tensor([value])
        for key, value in {
            "pt": pt,
            "eta": eta,
            "cls_id": cls,
            "energy": pt,
            "sin_phi": [0.0] * len(pt),
            "cos_phi": [1.0] * len(pt),
        }.items()
    }


def test_momentum_keeps_large_failures_and_decomposes_event_vectors():
    truth = collection([10.0, 10.0], [0.0, 2.5], [1, 5])
    pred = collection([11.0, 20.0], [0.0, 2.5], [2, 5])
    mask = torch.ones(1, 2, dtype=torch.bool)
    raw = compute_validation_particle_metrics(truth, mask, pred, mask, 6, detailed=True)
    values = finish(raw)
    assert values["momentum/all/assigned/samples"] == 2
    assert values["momentum/all/accepted/samples"] == 1
    assert values["momentum/all/assigned/pt_relative_abs_mean"] == pytest.approx(0.55)
    assert values["momentum/all/accepted/pt_relative_abs_mean"] == pytest.approx(0.1)
    assert values["momentum/all/assigned/pt_relative_gt0p5_fraction"] == 0.5
    assert values["momentum/abs_eta_ge2/assigned/pz_abs_gt10_fraction"] == 1
    assert values["momentum/class_1/accepted/pt_relative_abs_mean"] == pytest.approx(0.1)  # truth PID defines matched slices
    assert values["momentum/class_1/event/px_bias"] == -10  # predicted PID defines event slices
    assert values["momentum/class_2/event/px_bias"] == 11
    assert values["momentum/all/event/pz_bias"] == pytest.approx(10 * torch.sinh(torch.tensor(2.5)).item())
    assert values["momentum/all/assigned/pt_relative_residual_rms"] == pytest.approx((0.505) ** 0.5)
    assert values["momentum/all/event/px_bias"] == sum(values[f"momentum/class_{i}/event/px_bias"] for i in range(1, 6))
    assert values["momentum/all/event/pz_bias"] == sum(values[f"momentum/{b}/event/pz_bias"] for b in ("abs_eta_lt1", "abs_eta_1to2", "abs_eta_ge2"))
    ordinary = compute_validation_particle_metrics(truth, mask, pred, mask, 6)
    assert ordinary == {key: raw[key] for key in ordinary}


def test_momentum_empty_predictions_and_weighted_finalization(monkeypatch):
    truth = collection([10.0], [2.0], [1])
    pred = collection([5.0], [0.0], [1])
    mask = torch.ones(1, 1, dtype=torch.bool)
    raw = compute_validation_particle_metrics(truth, mask, pred, ~mask, 6, detailed=True)
    values = finish(raw)
    assert values["momentum/all/target_samples"] == 1
    assert values["momentum/all/assigned/samples"] == 0
    assert values["momentum/all/event/px_bias"] == -10
    assert values["momentum/abs_eta_ge2/efficiency"] == 0
    # Simulate a second rank with an unequal number of residuals; one packed
    # reduction must compute global weighted means, support totals, and RMS.
    accum = {"x_residual_rms": [torch.tensor(4.0), torch.tensor(1.0)], "samples": [torch.tensor(1.0), torch.tensor(1.0)]}

    def reduce(packed):
        assert packed.shape == (2, 2)
        packed += torch.tensor([[3.0, 1.0], [27.0, 3.0]])  # sorted keys: samples, x_residual_rms

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    result = _finalize_diagnostics(accum, 2)
    assert result["samples"] == 4
    assert result["x_residual_rms"] == pytest.approx((31 / 4) ** 0.5)
