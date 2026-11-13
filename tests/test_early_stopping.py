from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlpf.model import training


@pytest.mark.parametrize("rank", ["cpu", 0, 1])
@pytest.mark.parametrize(
    "validation_losses,patience,expected_step,best_step",
    [
        ([5, 5, 6, 7, 1], 3, 8, 2),  # Equal loss is stale; stop on the third failure.
        ([5, 6, 4, 4, 5, 6, 1], 3, 12, 6),  # Improvement resets patience.
        ([5, 4, 3, 2, 1, 0], 3, 12, 12),  # Keep improving up to the step budget.
        ([5, 5, 1], 0, 4, 2),  # Zero patience still allows improvements.
    ],
)
def test_training_stops_on_validation_patience(tmp_path, monkeypatch, rank, validation_losses, patience, expected_step, best_step):
    """Exercise the loop and real validation state updates on main and worker ranks."""
    losses = iter(validation_losses)
    evaluate = Mock(side_effect=lambda **kwargs: {"Total": next(losses), "Total_unweighted": 1.0})
    monkeypatch.setattr(training, "evaluate", evaluate)
    train_step = Mock(return_value=({"Total": 1.0, "Total_unweighted": 1.0}, {}, {}))
    monkeypatch.setattr(training, "train_step", train_step)
    save = Mock()
    monkeypatch.setattr(training, "save_checkpoint", save)
    monkeypatch.setattr(training, "SummaryWriter", Mock())
    monkeypatch.setattr(training, "get_train_epoch", lambda loader: 0.5)
    monkeypatch.setattr(training, "active_benchmark", lambda: None)
    monkeypatch.setattr(training, "_current_step_memory_summary", lambda _: {})
    monkeypatch.setattr(training.torch.amp, "GradScaler", Mock())
    monkeypatch.setattr(training.dist, "barrier", Mock())
    monkeypatch.setattr(training.torch.cuda, "empty_cache", Mock())
    for name in (
        "log_memory",
        "_log_and_checkpoint_step",
        "_log_validation_results_to_tensorboard",
        "_log_group_losses_to_tensorboard",
        "_log_validation_probe_results",
    ):
        monkeypatch.setattr(training, name, Mock())

    class Loader:
        names = []

        def __len__(self):
            return 12

        def __iter__(self):
            return iter([SimpleNamespace(X=SimpleNamespace(shape=(1,)))] * 12)

        def state_dict(self):
            return {}

        def reset(self):
            pass

    training.train_all_steps(
        rank=rank,
        world_size=1 if rank == "cpu" else 2,
        model=None,
        optimizer=None,
        train_loader=Loader(),
        valid_loader=Loader(),
        num_steps=12,
        patience=patience,
        outdir=str(tmp_path),
        config=SimpleNamespace(tensorboard_step_freq=100, patience=patience, enabled_test_datasets=[], benchmark=SimpleNamespace(enabled=False)),
        val_freq=2,
        checkpoint_dir=str(tmp_path),
    )

    assert train_step.call_count == expected_step
    assert evaluate.call_count == expected_step // 2
    if rank in ("cpu", 0):
        assert save.call_args.args[0] == tmp_path / "best.pth"
        assert save.call_args.args[3]["step"] == best_step
    else:
        save.assert_not_called()
