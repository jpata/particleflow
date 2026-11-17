"""
Spec: per-dataset-group losses in 'mlpf.model.training'. '_GroupLossMeans' averages loss dicts per group (validation:
per validation group after summing over the ranks with 'reduce'; training: per train group since the last logging step);
'_log_group_losses_to_tensorboard' and '_comet_group_metrics' name them 'group/<group>/loss_<loss>' (TensorBoard) and
'<group>/<split>_loss_<loss>' (Comet). Key assertions: each group's means are over its own batches/steps, groups without
any are left out, reset starts a new window, 'reduce' turns the sums into floats, and the "group/<group>/<loss>"
validation keys are split off for logging.
"""

import unittest

import torch

from mlpf.model.losses import LOSS_TASKS
from mlpf.model.training import (
    UNWEIGHTED_TOTAL,
    _comet_group_metrics,
    _GroupLossMeans,
    _log_group_losses_to_tensorboard,
    _split_group_losses,
    _validation_group_losses,
)


class _Recorder:
    """Stands in for a TensorBoard writer."""

    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)


def _losses(value, as_tensor=True):
    losses = {task: value for task in LOSS_TASKS}
    losses["Total"] = 10 * value
    return {name: torch.tensor(v) for name, v in losses.items()} if as_tensor else losses


class TestGroupLossMeans(unittest.TestCase):
    def test_means_per_group_and_reset(self):
        means = _GroupLossMeans(["pu0", "pu200", "rare"])
        for group, value in [("pu0", 1.0), ("pu200", 5.0), ("pu0", 3.0), ("pu200", 7.0)]:
            means.add(group, _losses(value, as_tensor=False))
        out = means.means()
        self.assertEqual(list(out), ["pu0", "pu200"])  # a group without steps is left out
        self.assertAlmostEqual(out["pu0"]["Total"], 20.0)
        self.assertAlmostEqual(out["pu200"][LOSS_TASKS[0]], 6.0)

        means.reset()
        means.add("rare", _losses(9.0, as_tensor=False))
        self.assertEqual(means.means(), {"rare": _losses(9.0, as_tensor=False)})

    def test_reduce_gives_float_batch_averages(self):
        means = _GroupLossMeans(["pu0", "pu200", "empty"])
        for group, value in [("pu0", 1.0), ("pu200", 4.0), ("pu0", 3.0), ("pu200", 6.0), ("pu200", 8.0)]:
            means.add(group, _losses(value))
        means.reduce(sorted(_losses(0.0)), "cpu", 1)
        out = means.means()
        self.assertEqual(list(out), ["pu0", "pu200"])
        self.assertIsInstance(out["pu0"]["Total"], float)
        self.assertAlmostEqual(out["pu0"]["Total"], 20.0)
        self.assertAlmostEqual(out["pu200"][LOSS_TASKS[0]], 6.0)


class TestValidationGroupLosses(unittest.TestCase):
    def test_flat_keys_with_unweighted_total_and_nothing_for_a_single_group(self):
        means = _GroupLossMeans(["pu0", "pu200"])
        means.add("pu0", _losses(2.0))
        out = _validation_group_losses(means, sorted(_losses(0.0)), "cpu", 1)
        self.assertAlmostEqual(out["group/pu0/Total"], 20.0)
        self.assertAlmostEqual(out[f"group/pu0/{UNWEIGHTED_TOTAL}"], 2.0 * len(LOSS_TASKS))
        self.assertFalse(any(key.startswith("group/pu200/") for key in out))

        single = _GroupLossMeans(["physical"])
        single.add("physical", _losses(1.0))
        self.assertEqual(_validation_group_losses(single, sorted(_losses(0.0)), "cpu", 1), {})


class TestGroupLossLogging(unittest.TestCase):
    def test_split_and_log_names(self):
        combined, groups = _split_group_losses({"Total": 3.0, "metrics/particle/x": 0.5, "group/pu0/Total": 2.0, "group/pu200/Regression_pt": 4.0})
        self.assertEqual(combined, {"Total": 3.0, "metrics/particle/x": 0.5})
        self.assertEqual(groups, {"pu0": {"Total": 2.0}, "pu200": {"Regression_pt": 4.0}})

        recorder = _Recorder()
        _log_group_losses_to_tensorboard(recorder, groups, 7)
        self.assertEqual(recorder.scalars, {"group/pu0/loss_Total": (2.0, 7), "group/pu200/loss_Regression_pt": (4.0, 7)})
        self.assertEqual(_comet_group_metrics(groups, "valid"), {"pu0/valid_loss_Total": 2.0, "pu200/valid_loss_Regression_pt": 4.0})


if __name__ == "__main__":
    unittest.main()
