"""
Spec: Ensures 'restore_checkpoint' / 'restore_loader_states' (shared by training.py and distributed_ray.py) resume a run fully
by default, and with weights_only load only the model weights. Key assertions: a full resume restores the step, optimizer, RNG
and LR schedule; weights_only starts from step 1 with a fresh optimizer, LR schedule and RNG stream.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch.optim import AdamW

from mlpf.conf import LRSchedule
from mlpf.model.training import restore_checkpoint, restore_loader_states
from mlpf.model.utils import get_lr_schedule, save_checkpoint

TOTAL_STEPS = 100
SAVED_STEP = 7


def _config(**kwargs):
    defaults = dict(lr_schedule=LRSchedule.COSINEDECAY, lr_schedule_config={}, relaxed_load=False, sampler_from_scratch=False)
    ns = SimpleNamespace(**{**defaults, **kwargs})
    if not hasattr(ns, "lr"):
        ns.lr = 5e-4
    return ns


def _setup(lr):
    model = torch.nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_schedule = get_lr_schedule(_config(lr=lr), optimizer, TOTAL_STEPS)
    return model, optimizer, lr_schedule


class _Loader:
    def __init__(self):
        self.state = None

    def load_state_dict(self, state):
        self.state = state


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model, optimizer, lr_schedule = _setup(lr=1e-3)
    for _ in range(SAVED_STEP):
        model(torch.randn(3, 4)).sum().backward()
        optimizer.step()
        lr_schedule.step()
    extra_state = {
        "step": SAVED_STEP,
        "lr_schedule_state_dict": lr_schedule.state_dict(),
        "train_loader_state_dict": {"split": "train"},
        "valid_loader_state_dict": {"split": "valid"},
    }
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, model, optimizer, extra_state)
    return torch.load(path, weights_only=False), model


def test_full_resume(checkpoint):
    checkpoint, saved_model = checkpoint
    model, optimizer, lr_schedule = _setup(lr=1e-3)
    torch.manual_seed(123)

    start_step = restore_checkpoint(checkpoint, model, optimizer, lr_schedule, _config())

    assert start_step == SAVED_STEP + 1
    torch.testing.assert_close(model.state_dict(), saved_model.state_dict())
    assert len(optimizer.state) == len(list(model.parameters()))
    assert lr_schedule.last_epoch == SAVED_STEP
    assert optimizer.param_groups[0]["lr"] == checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
    torch.testing.assert_close(torch.get_rng_state(), checkpoint["extra_state"]["rng_state"])

    loaders = {"train": _Loader(), "valid": _Loader()}
    restore_loader_states(loaders, checkpoint, _config())
    assert loaders["train"].state == {"split": "train"}
    assert loaders["valid"].state == {"split": "valid"}

    loaders = {"train": _Loader(), "valid": _Loader()}
    restore_loader_states(loaders, checkpoint, _config(sampler_from_scratch=True))
    assert loaders["train"].state is None and loaders["valid"].state is None


def test_full_resume_keeps_checkpoint_lr(checkpoint, caplog):
    """A resume continues the checkpoint's LR schedule and warns that a different config lr is ignored."""
    checkpoint, _ = checkpoint
    model, optimizer, lr_schedule = _setup(lr=5e-4)

    with caplog.at_level(logging.WARNING, logger="mlpf"):
        restore_checkpoint(checkpoint, model, optimizer, lr_schedule, _config())

    assert lr_schedule.base_lrs == [1e-3]
    assert any("ignoring config lr" in r.getMessage() for r in caplog.records)


def test_weights_only(checkpoint):
    checkpoint, saved_model = checkpoint
    model, optimizer, lr_schedule = _setup(lr=5e-4)
    torch.manual_seed(123)
    rng_state = torch.get_rng_state()

    start_step = restore_checkpoint(checkpoint, model, optimizer, lr_schedule, _config(), weights_only=True)

    assert start_step == 1
    torch.testing.assert_close(model.state_dict(), saved_model.state_dict())
    assert len(optimizer.state) == 0
    assert lr_schedule.last_epoch == 0
    assert lr_schedule.base_lrs == [5e-4]
    assert optimizer.param_groups[0]["lr"] == 5e-4
    torch.testing.assert_close(torch.get_rng_state(), rng_state)


def test_weights_only_checks_keys(checkpoint):
    checkpoint, _ = checkpoint
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    lr_schedule = get_lr_schedule(_config(lr=1e-3), optimizer, TOTAL_STEPS)
    with pytest.raises(KeyError):
        restore_checkpoint(checkpoint, model, optimizer, lr_schedule, _config(), weights_only=True)
