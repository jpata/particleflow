import copy

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset

from mlpf.model.ssl_snapshots import SnapshotConfig, save_snapshot
from mlpf.pretrain import PretrainConfig
from tests.test_ssl import setup_model


def test_fixed_cohort_and_isolation(tmp_path):
    model, X, _ = setup_model()
    X[0, 1] = 0
    data = ConcatDataset([[{"X": x.numpy(), "source_id": 1} for x in X]] * 2)
    config = PretrainConfig(data_dir="/tmp", datasets=["a", "b"], snapshots=SnapshotConfig(enabled=True, events_per_dataset=1, elements_per_event=4))
    model.train()
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    a = save_snapshot(model, data, config, tmp_path, 0, torch.device("cpu"), torch.float32)
    b = save_snapshot(model, data, config, tmp_path, 10, torch.device("cpu"), torch.float32)
    assert a["cohort_sha256"] == b["cohort_sha256"]
    assert model.training and not model.teacher.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with np.load(tmp_path / a["path"], allow_pickle=False) as first, np.load(tmp_path / b["path"], allow_pickle=False) as second:
        assert first["student"].shape == (8, 8)
        for key in ("X", "student", "teacher", "dataset_index", "event_index", "element_index"):
            np.testing.assert_array_equal(first[key], second[key])
        assert set(first["dataset_index"]) == {0, 1}
        np.testing.assert_array_equal(first["X"][:4], X[0, first["element_index"][:4]].numpy())
    # Refuse a different input cohort in the same history.
    data.datasets[0][0]["X"][0, 1] += 5
    config.snapshots.elements_per_event = 8
    with pytest.raises(ValueError, match="cohort changed"):
        save_snapshot(model, data, config, tmp_path, 20, torch.device("cpu"), torch.float32)


def test_rotation_invariant_embedding_comparison():
    from scripts.local.plot_ssl_evolution import compare_embeddings

    rng = np.random.default_rng(7)
    ref = rng.normal(size=(32, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    aligned, metrics = compare_embeddings(ref, ref @ q + 3)
    assert metrics["linear_cka_to_first"] == pytest.approx(1)
    np.testing.assert_allclose(aligned, ref - ref.mean(0), atol=1e-12)
