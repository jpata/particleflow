"""LR range test ("LR finder"): https://arxiv.org/abs/1506.01186, Sec. 3.3, and fastai's lr_find.

Trains for `lr_finder.num_steps` steps while the learning rate grows exponentially from
`lr_finder.start_lr` to `lr_finder.end_lr`, recording the training loss at every step. The loss
curve against the learning rate shows where training starts to make progress, where it is
fastest and where it diverges; the run stops early once the smoothed loss exceeds
`lr_finder.divergence_factor` times its minimum.

The first `task_loss_weights.calibration_steps` steps run at `start_lr` and are not recorded:
the task-loss weights are calibrated on them (as at the start of a normal training) and the
optimized loss changes scale once the calibrated weights apply.

Run via `mlpf/pipeline.py ... lr-find`, which reuses the model, optimizer, data and DDP setup of
`train`. Results go to `<experiment dir>/lr_find/` (`lr_find.json`, `lr_find.png`).
"""

import json
import math
from pathlib import Path

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from mlpf.conf import LRFinderConfig, MLPFConfig
from mlpf.logger import _logger
from mlpf.model.training import UNIT_REGRESSION_WEIGHTS, UNWEIGHTED_TOTAL, train_step

MIN_STEPS_BEFORE_STOP = 10  # the bias-corrected EMA is still noisy over the first few steps
SAVE_FREQ = 50  # steps between intermediate result dumps (kept if a later step crashes)


def lr_at_step(index, cfg: LRFinderConfig, warmup_steps=0):
    """Learning rate of the 0-based step `index`: start_lr during warmup, then an exponential ramp."""
    if index < warmup_steps:
        return cfg.start_lr
    t = min((index - warmup_steps) / (cfg.num_steps - 1), 1.0)
    return cfg.start_lr * (cfg.end_lr / cfg.start_lr) ** t


def suggest_lrs(lrs, smoothed_losses, skip_start=10):
    """Heuristic learning-rate suggestions from a recorded range test.

    minimum: lr at the minimum of the smoothed loss (already in the unstable regime).
    minimum_over_10: fastai's rule of thumb, minimum / 10.
    steepest: lr of the steepest smoothed-loss descent per decade of lr, before the minimum.
    """
    lrs = np.asarray(lrs, dtype=float)
    smoothed = np.asarray(smoothed_losses, dtype=float)
    if len(smoothed) == 0 or not np.isfinite(smoothed).any():
        return {}
    i_min = int(np.nanargmin(np.where(np.isfinite(smoothed), smoothed, np.nan)))
    suggestions = {"minimum": float(lrs[i_min]), "minimum_over_10": float(lrs[i_min] / 10)}
    start = min(skip_start, i_min)
    if i_min - start >= 2:
        gradient = np.gradient(smoothed[start : i_min + 1], np.log10(lrs[start : i_min + 1]))
        suggestions["steepest"] = float(lrs[start + int(np.argmin(gradient))])
    return suggestions


class LRRangeTest:
    """Records (lr, loss) pairs, smooths the loss and decides when the range test has diverged."""

    def __init__(self, cfg: LRFinderConfig):
        self.cfg = cfg
        self.lrs = []
        self.losses = []
        self.smoothed = []
        self.components = []
        self.best = math.inf
        self.stop_reason = None
        self._ema = 0.0

    def record(self, lr, losses):
        """Add one step's losses (dict with a "Total" entry); returns True if the test should stop."""
        loss = float(losses["Total"])
        self.lrs.append(float(lr))
        self.losses.append(loss)
        self.components.append({name: float(value) for name, value in losses.items() if name not in ("Total", UNWEIGHTED_TOTAL)})
        if not math.isfinite(loss):
            self.smoothed.append(math.nan)
            self.stop_reason = f"non-finite loss {loss} at lr={lr:.3e}"
            return True

        beta = self.cfg.smoothing
        self._ema = beta * self._ema + (1 - beta) * loss
        smoothed = self._ema / (1 - beta ** len(self.losses))
        self.smoothed.append(smoothed)
        self.best = min(self.best, smoothed)
        if len(self.losses) > MIN_STEPS_BEFORE_STOP and smoothed > self.cfg.divergence_factor * self.best:
            self.stop_reason = f"diverged: smoothed loss {smoothed:.4g} > {self.cfg.divergence_factor} x minimum {self.best:.4g} at lr={lr:.3e}"
            return True
        return False

    def suggestions(self):
        return suggest_lrs(self.lrs, self.smoothed)

    def to_dict(self):
        return {
            "config": self.cfg.model_dump(),
            "steps_recorded": len(self.lrs),
            "stop_reason": self.stop_reason,
            "suggestions": self.suggestions(),
            "lr": self.lrs,
            "loss": self.losses,
            "smoothed_loss": self.smoothed,
            "loss_components": self.components,
        }

    def save(self, outdir, plot=True):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "lr_find.json", "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        if plot and len(self.lrs) > 1:
            self.plot(outdir / "lr_find.png")

    def plot(self, path):
        lrs = np.asarray(self.lrs)
        smoothed = np.asarray(self.smoothed)
        finite = np.isfinite(smoothed)
        fig, (ax_total, ax_components) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

        ax_total.plot(lrs, self.losses, color="0.75", lw=0.8, label="loss")
        ax_total.plot(lrs, smoothed, color="C0", lw=2, label=f"EMA (beta={self.cfg.smoothing})")
        if finite.any():
            head = smoothed[finite][: max(1, finite.sum() // 10)]
            ax_total.set_ylim(0.9 * np.nanmin(smoothed), 1.2 * np.nanmax(head))
        for (name, lr), color in zip(self.suggestions().items(), ["C3", "C1", "C2"]):
            ax_total.axvline(lr, color=color, ls="--", label=f"{name}: {lr:.2e}")
        ax_total.set_ylabel("training loss (Total)")
        ax_total.legend(loc="upper left", fontsize=9)

        # per task, smoothed and normalized to its first value: shows which task diverges first
        beta = self.cfg.smoothing
        for name in self.components[0] if self.components else []:
            values = np.array([c[name] for c in self.components])
            ema = np.zeros_like(values)
            avg = 0.0
            for i, value in enumerate(values):
                avg = beta * avg + (1 - beta) * value
                ema[i] = avg / (1 - beta ** (i + 1))
            if ema[0] > 0:
                ax_components.plot(lrs, ema / ema[0], lw=1.5, label=name)
        ax_components.set_yscale("log")
        ax_components.set_ylabel("smoothed task loss / initial")
        ax_components.legend(loc="upper left", fontsize=8, ncol=2)

        ax_components.set_xscale("log")
        ax_components.set_xlabel("learning rate")
        title = f"LR range test, {len(lrs)} steps"
        if self.stop_reason:
            title += "\n" + self.stop_reason
        ax_total.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def lr_range_test(
    rank,
    world_size,
    model,
    optimizer,
    train_loader,
    config: MLPFConfig,
    outdir,
    dtype=torch.float32,
    comet_experiment=None,
):
    """Run the range test in place of train_all_steps (same per-step train_step, DDP-aware)."""
    cfg = config.lr_finder
    warmup_steps = config.task_loss_weights.calibration_steps
    total_steps = warmup_steps + cfg.num_steps
    is_main = (rank == 0) or (rank == "cpu")
    device_type = "cuda" if isinstance(rank, int) else "cpu"
    resultdir = Path(outdir) / "lr_find"

    # LambdaLR scales the group's initial_lr, which an earlier scheduler may already have set
    base_lr = optimizer.param_groups[0].setdefault("initial_lr", optimizer.param_groups[0]["lr"])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: lr_at_step(index, cfg, warmup_steps) / base_lr)
    scaler = torch.amp.GradScaler()
    tracker = LRRangeTest(cfg)
    tensorboard_writer = SummaryWriter(f"{outdir}/runs/lr_find") if is_main else None

    _logger.info(
        f"LR range test: {warmup_steps} calibration steps at lr={cfg.start_lr:.1e}, then {cfg.num_steps} steps "
        f"from {cfg.start_lr:.1e} to {cfg.end_lr:.1e} on rank {rank} of {world_size}"
    )
    train_iterator = iter(train_loader)
    for step in range(1, total_steps + 1):
        lr = optimizer.param_groups[0]["lr"]  # the lr this step trains with (the schedule steps after it)
        batch = next(train_iterator)
        losses, _, _ = train_step(
            rank=rank,
            world_size=world_size,
            model=model,
            optimizer=optimizer,
            batch=batch,
            lr_schedule=lr_schedule,
            step=step,
            regression_weights=UNIT_REGRESSION_WEIGHTS,
            tensorboard_writer=tensorboard_writer,
            tensorboard_step_freq=config.tensorboard_step_freq,
            device_type=device_type,
            dtype=dtype,
            scaler=scaler,
            loader_state_dict=train_loader.state_dict()["loader_state_dict"],
        )
        if step <= warmup_steps:
            continue

        # losses are already averaged over ranks, so every rank takes the same decision; the
        # all_reduce only guards against any rank disagreeing
        stop = tracker.record(lr, losses)
        if world_size > 1:
            flag = torch.tensor(float(stop), device=rank)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())

        if is_main:
            metrics = {"lr": lr, "loss": tracker.losses[-1], "smoothed_loss": tracker.smoothed[-1]}
            for name, value in metrics.items():
                tensorboard_writer.add_scalar(f"lr_find/{name}", value, step)
            if comet_experiment is not None:
                comet_experiment.log_metrics(metrics, prefix="lr_find", step=step)
            if step % config.tensorboard_step_freq == 0:
                _logger.info(
                    f"LR range test step {step}/{total_steps} | lr={lr:.3e} | loss={metrics['loss']:.4f} | smoothed={metrics['smoothed_loss']:.4f}"
                )
            if len(tracker.lrs) % SAVE_FREQ == 0:
                tracker.save(resultdir, plot=False)

        if stop:
            _logger.info(f"LR range test stopped at step {step}: {tracker.stop_reason}")
            break

    if is_main:
        tracker.save(resultdir)
        tensorboard_writer.close()
        suggestions = tracker.suggestions()
        _logger.info(f"LR range test done ({len(tracker.lrs)} steps recorded); suggestions: {suggestions}", color="bold")
        _logger.info(f"LR range test results in {resultdir}")
        if comet_experiment is not None:
            comet_experiment.log_others({f"lr_find_{name}": lr for name, lr in suggestions.items()})
            comet_experiment.log_image(str(resultdir / "lr_find.png"), name="lr_find")
    if world_size > 1:
        dist.barrier()
    return tracker
