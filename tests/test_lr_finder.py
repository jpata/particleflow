"""
Spec: the LR range test (mlpf/model/lr_finder.py, `pipeline.py lr-find`). Key assertions: the lr schedule holds
start_lr through the calibration steps and then ramps exponentially to exactly end_lr, also when driven through
torch's LambdaLR on an optimizer that already carries another scheduler; the tracker stops on divergence or a
non-finite loss and suggests learning rates around the minimum of a synthetic loss curve; results are written as
json + png; config validation rejects an empty lr range.
"""

import json
import math

import numpy as np
import pytest
import torch

from mlpf.conf import LRFinderConfig, LRSchedule, MLPFConfig
from mlpf.model.lr_finder import LRRangeTest, lr_at_step, suggest_lrs
from mlpf.model.utils import get_lr_schedule


def test_lr_schedule_holds_during_warmup_then_ramps_to_end_lr():
    cfg = LRFinderConfig(start_lr=1e-6, end_lr=1e-1, num_steps=101)
    lrs = [lr_at_step(i, cfg, warmup_steps=5) for i in range(5 + 101)]

    assert lrs[:6] == [1e-6] * 6  # warmup, then the first ramp step
    assert lrs[-1] == pytest.approx(1e-1)
    ratios = np.diff(np.log10(lrs[5:]))
    np.testing.assert_allclose(ratios, 5 / 100)  # 5 decades in 100 equal log steps
    assert lr_at_step(10_000, cfg, warmup_steps=5) == pytest.approx(1e-1)  # clamped past the end


def test_lambda_lr_reproduces_the_ramp_on_an_already_scheduled_optimizer():
    cfg = LRFinderConfig(start_lr=1e-7, end_lr=1e-2, num_steps=20)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    class Config:
        lr_schedule = LRSchedule.COSINEDECAY
        lr = 1e-4
        lr_schedule_config = {}

    get_lr_schedule(Config, optimizer, 1000)  # what run() creates before the range test takes over
    base_lr = optimizer.param_groups[0].setdefault("initial_lr", optimizer.param_groups[0]["lr"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: lr_at_step(index, cfg, 3) / base_lr)

    seen = []
    for _ in range(3 + cfg.num_steps):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()
    np.testing.assert_allclose(seen, [lr_at_step(i, cfg, 3) for i in range(3 + cfg.num_steps)], rtol=1e-12)


def synthetic_loss(lr):
    # flat at low lr, fastest descent around 1e-4, minimum at 1e-3, blows up above
    x = math.log10(lr)
    return 2.0 - 1.0 / (1 + math.exp(-3 * (x + 4))) + 50.0 * max(0.0, x + 3) ** 2


def run_tracker(cfg, loss_fn, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    tracker = LRRangeTest(cfg)
    for i in range(cfg.num_steps):
        lr = lr_at_step(i, cfg)
        loss = loss_fn(lr) * (1 + noise * rng.standard_normal())
        if tracker.record(lr, {"Total": loss, "Classification": 0.5 * loss, "Regression_pt": 0.5 * loss}):
            break
    return tracker


def test_tracker_stops_on_divergence_and_suggests_lrs_below_the_blowup():
    cfg = LRFinderConfig(start_lr=1e-7, end_lr=1.0, num_steps=400, smoothing=0.9)
    tracker = run_tracker(cfg, synthetic_loss, noise=0.02)

    assert tracker.stop_reason is not None and tracker.stop_reason.startswith("diverged")
    assert len(tracker.lrs) < cfg.num_steps
    assert tracker.lrs[-1] < 1e-1  # stopped well before the end of the ramp

    suggestions = tracker.suggestions()
    # the EMA lags, so the smoothed minimum sits somewhat above the true one
    assert 5e-4 < suggestions["minimum"] < 1e-2
    assert suggestions["minimum_over_10"] == pytest.approx(suggestions["minimum"] / 10)
    assert 1e-5 < suggestions["steepest"] < suggestions["minimum"]


def test_tracker_stops_on_non_finite_loss():
    cfg = LRFinderConfig(start_lr=1e-6, end_lr=1e-1, num_steps=100)
    tracker = run_tracker(cfg, lambda lr: float("inf") if lr > 1e-3 else 1.0)

    assert tracker.stop_reason.startswith("non-finite")
    assert math.isnan(tracker.smoothed[-1])
    assert tracker.suggestions()["minimum"] <= 1e-3


def test_tracker_without_divergence_runs_the_full_ramp():
    cfg = LRFinderConfig(start_lr=1e-6, end_lr=1e-3, num_steps=50)
    tracker = run_tracker(cfg, lambda lr: 1.0 - 0.1 * math.log10(lr / 1e-6))

    assert tracker.stop_reason is None
    assert len(tracker.lrs) == cfg.num_steps


def test_suggestions_handle_short_and_empty_curves():
    assert suggest_lrs([], []) == {}
    short = suggest_lrs([1e-5, 1e-4], [1.0, 0.9])
    assert short["minimum"] == 1e-4 and "steepest" not in short


def test_save_writes_json_and_plot(tmp_path):
    cfg = LRFinderConfig(start_lr=1e-7, end_lr=1.0, num_steps=200, smoothing=0.9)
    tracker = run_tracker(cfg, synthetic_loss)
    tracker.save(tmp_path / "lr_find")

    result = json.loads((tmp_path / "lr_find" / "lr_find.json").read_text())
    assert result["steps_recorded"] == len(result["lr"]) == len(result["loss"]) == len(result["smoothed_loss"])
    assert set(result["loss_components"][0]) == {"Classification", "Regression_pt"}
    assert result["suggestions"] == tracker.suggestions()
    assert (tmp_path / "lr_find" / "lr_find.png").stat().st_size > 0


def test_config_validation():
    with pytest.raises(ValueError, match="start_lr must be below"):
        LRFinderConfig(start_lr=1e-2, end_lr=1e-3)

    config = MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "model": {"type": "attention", "attention": {}},
            "conv_type": "attention",
            "lr_finder": {"num_steps": 50, "end_lr": 1e-2},
        }
    )
    assert config.lr_finder.num_steps == 50 and config.lr_finder.start_lr == 1e-7
    assert config.lr_find is False
    with pytest.raises(ValueError):
        MLPFConfig.model_validate(
            {
                "dataset": "cld_hits",
                "data_dir": "/tmp",
                "model": {"type": "attention", "attention": {}},
                "conv_type": "attention",
                "lr_finder": {"num_step": 50},  # typo: extra="forbid"
            }
        )
