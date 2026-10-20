import math

import pytest
import torch
from mlpf.conf import Y_FEATURES
from mlpf.model.PFDataset import PFBatch
from mlpf.model.tracker_momentum_readout import TrackerMomentumReadout
from scripts.studies.frozen_tracker_pooling import pool_hit_associations
from scripts.studies.train_frozen_tracker_readout import oracle_associations, correction_loss


def test_oracle_uses_fixed_slot_matching_and_rejects_background():
    x = torch.zeros(1, 4, 15)
    x[:, :, 0] = 1
    y = torch.zeros(1, 4, len(Y_FEATURES))
    t = torch.zeros(1, 2, len(Y_FEATURES))
    pn = Y_FEATURES.index("particle_number")
    y[0, :, pn] = torch.tensor([17, 42, 0, 999])
    t[0, :, pn] = torch.tensor([42, 17])
    t[:, :, 0] = 1
    batch = PFBatch(X=x, ytarget=y, ytarget_set=t)
    dest, weights = oracle_associations(batch, [(torch.tensor([3, 0]), torch.tensor([0, 1]))], 4)
    assert dest[0, :2, 0].tolist() == [0, 3]
    assert weights.flatten().tolist() == [1, 1, 0, 0]


def test_override_uses_same_summary_operator_and_no_gradient_to_grouping():
    m = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]], requires_grad=True)
    x = torch.zeros(1, 3, 15)
    x[:, :, 0] = 1
    x[0, :, 6] = torch.tensor([50.0, 200.0, 900.0])
    keys = torch.randn(1, 3, 2, requires_grad=True)
    slots = torch.randn(1, 2, 2, requires_grad=True)
    bg = torch.zeros(1, 3, 1, requires_grad=True)
    dest = torch.zeros(1, 3, 1, dtype=torch.long)
    weights = torch.tensor([[[1.0], [1.0], [0.0]]])
    out, summary = pool_hit_associations(
        m, torch.ones(1, 3, dtype=torch.bool), x, (keys, slots, bg), 2, 2, 1, 1.0, False, True, fixed_associations=(dest, weights)
    )
    torch.testing.assert_close(summary[0, 0, :2], torch.tensor([4.0, 6.0]))
    assert summary[0, 1].count_nonzero() == 0
    bins = summary[0, 0, 2:].reshape(4, 11)
    torch.testing.assert_close(bins[:, -1].expm1(), torch.tensor([1.0, 1.0, 0.0, 0.0]))
    summary.sum().backward()
    assert m.grad[0, 2].count_nonzero() == 0
    assert keys.grad is None or keys.grad.count_nonzero() == 0


def test_frozen_loss_only_moves_charged_predictions_with_fixed_matches():
    base = torch.zeros(1, 3, 5)
    base[0, :, 0] = torch.tensor([1, 2, 1])
    truth = torch.zeros(1, 2, 5)
    truth[0, :, 1] = math.log(2.0)
    mapping = torch.tensor([[1, 0, -1]])
    correction = torch.zeros(1, 3, requires_grad=True)
    loss = correction_loss(correction, base, truth, mapping)
    assert loss.item() == pytest.approx(math.sqrt(2) * math.log(2) ** 2)
    loss.backward()
    assert correction.grad[0, 0] < 0
    assert correction.grad[0, 1] == correction.grad[0, 2] == 0
    assert base.grad is None and truth.grad is None


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_study_pooling_matches_production_reductions(sparse, chunk_size):
    from mlpf.model.tracker_momentum_readout import (
        association_weights,
        pool_hit_associations as model_pool,
    )

    torch.manual_seed(123)
    memory = torch.randn(2, 7, 3, requires_grad=True)
    features = torch.randn(2, 7, 15)
    features[..., 0] = torch.tensor([1, 2, 1, 2, 1, 1, 2])
    features[..., 6:9] *= 500
    mask = torch.tensor([[True] * 6 + [False], [False] * 7])
    grouping = (torch.randn(2, 7, 3), torch.randn(2, 4, 3), torch.randn(2, 7, 1))
    pairs = [association_weights(grouping[0][e], grouping[1][e], grouping[2][e], 2, 1.0, sparse) for e in range(2)]
    fixed = tuple(torch.stack([pair[i] for pair in pairs]) for i in range(2))
    args = (memory, mask, features, grouping, 4, chunk_size, 2, 1.0, sparse, True)
    expected = model_pool(*args)
    actual = pool_hit_associations(*args, fixed_associations=fixed)
    delegated = pool_hit_associations(*args)
    for reference, diagnostic, unchanged in zip(expected, actual, delegated):
        torch.testing.assert_close(diagnostic, reference)
        torch.testing.assert_close(unchanged, reference)
    (expected_grad,) = torch.autograd.grad(sum(x.sum() for x in expected), memory)
    (actual_grad,) = torch.autograd.grad(sum(x.sum() for x in actual), memory)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_zero_initialized_readout_has_no_initial_intervention():
    head = TrackerMomentumReadout(256, math.log(1.5))
    x = torch.randn(2, 5, 300)
    x[..., 256:] = 0
    x[..., 266] = math.log(3)
    assert head(x).count_nonzero() == 0
    head.output.bias.data.fill_(1)
    assert head(x).max() <= math.log(1.5)
    x[..., 266] = 0
    assert head(x).count_nonzero() == 0
