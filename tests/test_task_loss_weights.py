"""Tests for one-time task-loss weight calibration."""

import torch

from mlpf.conf import TaskLossWeights
from mlpf.model.losses import LOSS_TASKS, make_task_loss_weighter


def _losses():
    return {task: torch.tensor(0.1 * (i + 1), requires_grad=True) for i, task in enumerate(LOSS_TASKS)}


def _weighter(**kwargs):
    weights = TaskLossWeights(**kwargs)
    return make_task_loss_weighter(weights.model_dump())


def test_starts_with_unit_weights():
    w = _weighter(calibration_steps=2)
    assert torch.equal(w.weights, torch.ones(len(LOSS_TASKS)))
    assert not w.calibrated
    assert w.calibration_count.item() == 0


def test_calibrates_after_configured_number_of_steps_and_then_freezes():
    w = _weighter(calibration_steps=2)
    w.train()

    _, first_diag = w(_losses())
    assert not w.calibrated
    assert all(weight.item() == 1.0 for weight in first_diag["weight"].values())

    _, last_calibration_diag = w(_losses())
    assert w.calibrated
    # The final calibration batch still uses unit weights.
    assert all(weight.item() == 1.0 for weight in last_calibration_diag["weight"].values())

    expected = torch.tensor([1.0 / (i + 1) for i in range(len(LOSS_TASKS))])
    torch.testing.assert_close(w.weights, expected)

    _, calibrated_diag = w(_losses())
    for task, expected_weight in zip(LOSS_TASKS, expected):
        torch.testing.assert_close(calibrated_diag["weight"][task], expected_weight)

    frozen_weights = w.weights.clone()
    very_different_losses = {task: torch.tensor(1000.0) for task in LOSS_TASKS}
    w(very_different_losses)
    torch.testing.assert_close(w.weights, frozen_weights)
    assert w.calibration_count.item() == 2


def test_eval_does_not_advance_calibration():
    w = _weighter(calibration_steps=2)
    w.eval()
    w(_losses())
    assert w.calibration_count.item() == 0
    assert not w.calibrated


def test_calibration_does_not_break_loss_gradients():
    w = _weighter(calibration_steps=2)
    losses = _losses()
    total, _ = w(losses)
    total.backward()
    for loss in losses.values():
        torch.testing.assert_close(loss.grad, torch.tensor(1.0))


def test_calibration_state_is_checkpointed_but_not_trainable():
    w = _weighter(calibration_steps=2)
    w.train()
    w(_losses())

    assert list(w.parameters()) == []
    restored = _weighter(calibration_steps=2)
    restored.load_state_dict(w.state_dict())
    torch.testing.assert_close(restored.loss_sums, w.loss_sums)
    assert restored.calibration_count.item() == 1
    assert restored.calibration_observations.item() == 1
    assert not restored.calibrated
