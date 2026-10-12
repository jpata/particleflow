import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mlpf.model.instance_probe import compute_instance_ari, empty_ari_metrics
from mlpf.model.training import (
    _finalize_diagnostics,
    _log_validation_probe_results,
    _log_validation_results_to_tensorboard,
)


def events():
    ret = []
    for i in range(8):
        labels = np.repeat([1, 2], 4)
        x = np.zeros((8, 15), dtype=np.float32)
        x[:, 0] = 2
        x[:, 6] = np.arange(8)
        x[:, 2] = x[:, 6] / 10
        ret.append(dict(raw=x, embedding=np.eye(2)[labels - 1], labels=labels))
    return ret


def test_ari_uses_calibration_not_evaluation_and_has_static_keys():
    batch = events()
    result = compute_instance_ari(batch, "calorimeter")
    prefix = "metrics/linear_probe/calorimeter/"
    assert result[prefix + "ari_fit_event_samples"][0] == 4
    assert result[prefix + "ari_calibration_event_samples"][0] == 2
    assert result[prefix + "ari_eval_event_samples"][0] == 2
    assert result[prefix + "embedding/ari"] == [2, 2]
    for i in (3, 7):
        batch[i]["embedding"] = np.zeros((8, 2))
    changed = compute_instance_ari(batch, "calorimeter")
    assert (
        changed[prefix + "embedding/ari_threshold"]
        == result[prefix + "embedding/ari_threshold"]
    )
    assert changed[prefix + "embedding/ari"][0] == 0
    assert (
        set(result)
        == set(compute_instance_ari([], "calorimeter"))
        == set(empty_ari_metrics("calorimeter"))
    )


def test_ari_written_to_real_tensorboard_and_log(tmp_path, caplog):
    metrics = compute_instance_ari(events(), "calorimeter")
    values = _finalize_diagnostics(
        {k: [torch.tensor(a), torch.tensor(b)] for k, (a, b) in metrics.items()}, 1
    )
    with SummaryWriter(str(tmp_path)) as writer:
        _log_validation_results_to_tensorboard(writer, values, 5000)
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    for rep in ("raw", "embedding"):
        tag = f"validation/linear_probe/calorimeter/{rep}/ari"
        assert accumulator.Scalars(tag)[0].step == 5000
        assert (
            accumulator.Scalars(tag)[0].value
            == values[f"metrics/linear_probe/calorimeter/{rep}/ari"]
        )
    with caplog.at_level("INFO"):
        _log_validation_probe_results(values, 5000)
    assert (
        "VALIDATION_PROBE | Step=5000 | linear_probe/calorimeter/embedding/ari=1"
        in caplog.text
    )
    assert "linear_probe/calorimeter/raw/ari=" in caplog.text
