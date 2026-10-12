from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlpf.conf import Y_FEATURES
from mlpf.model.validation_probes import PREFIX, ValidationCaloProbe, empty_metrics
from mlpf.model.training import (
    _finalize_diagnostics,
    _log_validation_results_to_tensorboard,
)


def example():
    # Two interleaved particles in two layers, with competing hits at similar distances.
    x = np.zeros((8, 15), dtype=np.float32)
    x[:, 0] = 2
    x[:, 4] = 1
    x[:, 12] = 10
    x[:, 14] = [0] * 4 + [1] * 4
    x[:, 6] = [0, 0.1, 0.2, 0.3, 10, 10.1, 10.2, 10.3]
    x[:, 2] = x[:, 6] / 100
    labels = np.array([1, 2, 1, 2] * 2)
    return dict(raw=x, labels=labels, embedding=np.eye(2, dtype=np.float32)[labels - 1])


def test_probe_refits_and_evaluates_disjoint_events():
    p = ValidationCaloProbe(8)
    p.events = [example() for _ in range(8)]
    result = p.finalize()
    assert set(result) == set(empty_metrics())
    assert result[PREFIX + "fit_event_samples"][0] == 6
    assert result[PREFIX + "eval_event_samples"][0] == 2
    assert result[PREFIX + "fit_success_fraction"] == [1, 1]
    assert result[PREFIX + "embedding/triplet_accuracy"] == [2, 2]
    assert result[PREFIX + "coverage/distance_matched_triplets"] == [8, 16]
    # Strongly change held-out embeddings: fit is unchanged but test accuracy falls.
    for index in (3, 7):
        p.events[index]["embedding"][4:] = p.events[index]["embedding"][4:, ::-1]
    result = p.finalize()
    assert result[PREFIX + "embedding/triplet_accuracy"] == [0, 2]


def test_insufficient_events_keep_static_keys_without_fake_accuracy():
    p = ValidationCaloProbe()
    p.events = [example()]
    result = p.finalize()
    assert set(result) == set(empty_metrics())
    assert result[PREFIX + "fit_success_fraction"] == [0, 1]
    assert result[PREFIX + "embedding/triplet_accuracy"] == [0, 0]
    finalized = _finalize_diagnostics(
        {k: [torch.tensor(a), torch.tensor(b)] for k, (a, b) in result.items()}, 1
    )
    assert PREFIX + "embedding/triplet_accuracy" not in finalized


def test_capture_detaches_and_removes_hook_on_success_and_exception():
    event = example()
    x = torch.tensor(event["raw"])[None]
    targets = torch.zeros(1, 8, len(Y_FEATURES))
    targets[0, :, Y_FEATURES.index("particle_number")] = torch.tensor(event["labels"])
    batch = SimpleNamespace(
        X=x, ytarget=targets, mask=torch.ones(1, 8, dtype=torch.bool)
    )
    model = SimpleNamespace(set_decoder=torch.nn.Identity())
    p = ValidationCaloProbe()
    embedding = torch.tensor(event["embedding"], requires_grad=True)[None]
    with p.capture(model, batch):
        out = model.set_decoder(embedding)
    assert out.requires_grad
    assert len(p.events) == 1
    p.add_batch(batch, embedding)
    assert len(p.events) == 1  # repeated validation record cannot enter the other split
    assert not model.set_decoder._forward_pre_hooks
    with pytest.raises(RuntimeError), p.capture(model, batch):
        raise RuntimeError("forward failed")
    assert not model.set_decoder._forward_pre_hooks
    assert len(p.events) == 1


def test_tensorboard_probe_tag():
    calls = []
    writer = SimpleNamespace(add_scalar=lambda *args: calls.append(args))
    _log_validation_results_to_tensorboard(
        writer, {PREFIX + "raw/triplet_accuracy": 0.6}, 5000
    )
    assert calls == [
        ("validation/linear_probe/calorimeter/raw/triplet_accuracy", 0.6, 5000)
    ]
