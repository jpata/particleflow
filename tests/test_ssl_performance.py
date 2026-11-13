"""Numerical and data-order contracts for vectorized/prefetched SSL."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow_datasets as tfds
import torch
from torch.utils.data import ConcatDataset, Subset

from mlpf.model.PFDataset import TFDSDataSource
from mlpf.model.ssl import balanced_mean, subsample_view
from mlpf.model.ssl_io import StepBatches, fetch_records, training_loader
from mlpf.model.ssl_timing import StepTimings


@pytest.mark.parametrize("width", [None, 7])
def test_vectorized_loss_matches_reference_and_gradients(width):
    types = torch.tensor([[1, 1, 2, 0, 0], [2, 2, 2, 11, 1], [11, 0, 0, 0, 0]])
    mask = types != 0
    shape = (*types.shape, width) if width else types.shape
    values = torch.randn(shape, requires_grad=True)
    reference = torch.stack(
        [torch.stack([value[valid & (kind == t)].mean(0) for t in kind[valid].unique()]).mean(0) for value, kind, valid in zip(values, types, mask)]
    ).mean(0)
    actual = balanced_mean(values, types, mask)
    torch.testing.assert_close(actual, reference)
    expected_grad = torch.autograd.grad(reference.sum(), values, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_vectorized_sampling_has_correct_quotas_and_uniform_inclusion():
    X = torch.zeros(4000, 10, 3)
    X[:, :, 0] = torch.tensor([1, 1, 1, 1, 11, 11, 11, 11, 11, 0])
    X[:, :, 1] = torch.arange(10)
    view, valid, indices = subsample_view(X, X[..., 0] != 0, 0.6, torch.Generator().manual_seed(7))
    assert torch.all((view[..., 0] == 1).sum(1) == 2)
    assert torch.all((view[..., 0] == 11).sum(1) == 3)
    assert torch.all(valid.sum(1) == 5)
    assert torch.all(indices[:, 1:] > indices[:, :-1])
    freq = torch.bincount(indices.flatten(), minlength=10).float() / len(X)
    torch.testing.assert_close(freq[:4], torch.full((4,), 0.5), rtol=0, atol=0.04)
    torch.testing.assert_close(freq[4:9], torch.full((5,), 0.6), rtol=0, atol=0.04)
    assert freq[9] == 0


def test_bulk_projected_tfds_decode_preserves_order_and_duplicates():
    features = tfds.features.FeaturesDict(
        {"X": tfds.features.Tensor(shape=(None, 3), dtype=np.float32), "ytarget": tfds.features.Tensor(shape=(None, 3), dtype=np.float32)}
    )
    records = [features.serialize_example({"X": np.asarray([[1, i, 2]], np.float32), "ytarget": np.zeros((2, 3), np.float32)}) for i in range(3)]
    calls = []

    def read(indices):
        calls.append(indices)
        return [records[i] for i in indices]

    ds = SimpleNamespace(
        dataset_info=SimpleNamespace(name="cld_edm_ttbar_pf", config_name="1", features=features),
        data_source=SimpleNamespace(__getitems__=read),
        decoders=None,
    )
    source = TFDSDataSource(ds, sort=False, input_only=True)
    batch = source.__getitems__([2, 0, 2])
    assert calls == [[2, 0, 2]]
    assert list(source.ds.dataset_info.features) == ["X"]
    assert [r["X"][0, 1] for r in batch] == [2, 0, 2]
    assert all("ytarget" not in r for r in batch)


class BulkFixture:
    def __init__(self, offset=0):
        self.offset = offset
        self.calls = []

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"X": np.asarray([[1, index + self.offset, 2]], dtype=np.float32), "source_id": 1}

    def __getitems__(self, indices):
        self.calls.append(indices)
        return [self[i] for i in indices]


def test_bulk_subsets_restore_interleaved_order():
    a, b = BulkFixture(), BulkFixture(10)
    data = ConcatDataset([Subset(a, [4, 2, 1]), b])
    out = fetch_records(data, [4, 0, 1, 4, 2])
    assert [r["X"][0, 1] for r in out] == [11, 4, 2, 11, 1]
    assert a.calls == [[4, 2, 1]] and b.calls == [[1, 1]]


def test_prefetch_and_resume_preserve_batches_and_model_rng():
    data = ConcatDataset([BulkFixture(), BulkFixture(10)])
    config = SimpleNamespace(seed=7, batch_size=8, steps=4, num_workers=2, prefetch_factor=2)
    expected = StepBatches(data, config.seed, config.batch_size, 1, 4)
    before = torch.get_rng_state().clone()
    loader = training_loader(data, config, 3, torch.device("cpu"))
    actual = list(loader)
    torch.testing.assert_close(torch.get_rng_state(), before)
    for i, batch in enumerate(actual):
        torch.testing.assert_close(batch["X"], expected[i + 2]["X"])


def test_cpu_phase_timing():
    timer = StepTimings(torch.device("cpu"))
    with timer.phase("work"):
        sum(range(10))
    assert timer.results()["cpu_wall_seconds"]["work"] >= 0
