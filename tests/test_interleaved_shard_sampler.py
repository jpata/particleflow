"""
Spec: Validates 'InterleavedShardSampler' for mixed-source training.
Key tests: Ensures early batches cover multiple ConcatDataset shards, verifies
deterministic shuffled ordering, and checks distributed ranks cover the same
global interleaved sequence without overlap when no padding is required.
'_build_interleaved_shard_indices' (the interleaved event order) mixes the shards in
proportion to their sizes over the whole epoch: the order is a permutation with each
shard's own order shuffled, every prefix holds each shard's proportional share to
within less than one plus the number of shards (no tail of only the largest shards),
equal-size shards give one event per shard per round, the order is deterministic per
seed, empty shards are skipped, and each rank of the distributed sampler gets a
contiguous, hence proportional, block of the order.
"""

import pytest
import numpy as np
from torch.utils.data import ConcatDataset

from mlpf.conf import INPUT_TYPE_IDS, SOURCE_IDS, dataset_input_type_id, dataset_source_id
from mlpf.model.PFDataset import (
    Collater,
    DistributedInterleavedShardSampler,
    InterleavedShardSampler,
    ShardConsecutiveSampler,
    TFDSDataSource,
    _build_interleaved_shard_indices,
)
from tests.mock_data import MockDataset


class DummyFeatures:
    def __init__(self, example):
        self.example = example

    def deserialize_example_np(self, record, decoders=None):
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.example.items()}


class DummyBackingSource:
    def __getitems__(self, items):
        return items


class DummyTFDS:
    def __init__(self, name, example):
        self.dataset_info = type("DatasetInfo", (), {"name": name, "config_name": "test", "features": DummyFeatures(example)})()
        self.data_source = DummyBackingSource()
        self.decoders = {}


def shard_id(index, shard_size):
    return index // shard_size


def _concat(sizes):
    return ConcatDataset([MockDataset(n) for n in sizes])


def _shard_of(dataset, indices):
    return np.searchsorted(dataset.cumulative_sizes, indices, side="right")


def test_interleaved_sampler_cycles_across_shards_without_shuffle():
    shard_size = 4
    datasets = [MockDataset(shard_size) for _ in range(4)]
    concat_ds = ConcatDataset(datasets)

    sampler = InterleavedShardSampler(concat_ds, shuffle=False)
    indices = list(iter(sampler))

    assert indices[:8] == [0, 4, 8, 12, 1, 5, 9, 13]
    assert [shard_id(idx, shard_size) for idx in indices[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert sorted(indices) == list(range(16))


def test_interleaved_sampler_covers_all_shards_early_with_shuffle():
    shard_size = 50
    num_shards = 4
    datasets = [MockDataset(shard_size) for _ in range(num_shards)]
    concat_ds = ConcatDataset(datasets)

    interleaved = InterleavedShardSampler(concat_ds, shuffle=True, seed=123)
    consecutive = ShardConsecutiveSampler(concat_ds, shuffle=True, seed=123)

    interleaved_first_cycle = list(iter(interleaved))[:num_shards]
    consecutive_first_cycle = list(iter(consecutive))[:num_shards]

    assert {shard_id(idx, shard_size) for idx in interleaved_first_cycle} == set(range(num_shards))
    assert len({shard_id(idx, shard_size) for idx in consecutive_first_cycle}) == 1


def test_interleaved_sampler_shuffle_is_deterministic_and_epoch_dependent():
    datasets = [MockDataset(10) for _ in range(3)]
    concat_ds = ConcatDataset(datasets)

    sampler_a = InterleavedShardSampler(concat_ds, shuffle=True, seed=42)
    sampler_b = InterleavedShardSampler(concat_ds, shuffle=True, seed=42)
    sampler_c = InterleavedShardSampler(concat_ds, shuffle=True, seed=42)
    sampler_c.set_epoch(1)

    indices_a = list(iter(sampler_a))
    indices_b = list(iter(sampler_b))
    indices_c = list(iter(sampler_c))

    assert indices_a == indices_b
    assert indices_a != indices_c
    assert sorted(indices_a) == list(range(30))
    assert sorted(indices_c) == list(range(30))


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_distributed_interleaved_sampler_sync_without_padding(world_size):
    shard_size = 8
    datasets = [MockDataset(shard_size) for _ in range(4)]
    concat_ds = ConcatDataset(datasets)

    rank_indices = [
        list(
            iter(
                DistributedInterleavedShardSampler(
                    concat_ds,
                    world_size=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            )
        )
        for rank in range(world_size)
    ]

    assert len({len(indices) for indices in rank_indices}) == 1
    combined = [idx for indices in rank_indices for idx in indices]
    assert len(combined) == len(set(combined))
    assert set(combined) == set(range(len(concat_ds)))


def test_distributed_interleaved_sampler_padding_lengths_match():
    datasets = [MockDataset(5) for _ in range(2)]
    concat_ds = ConcatDataset(datasets)
    world_size = 3

    rank_indices = [
        list(
            iter(
                DistributedInterleavedShardSampler(
                    concat_ds,
                    world_size=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            )
        )
        for rank in range(world_size)
    ]

    assert [len(indices) for indices in rank_indices] == [4, 4, 4]
    combined = [idx for indices in rank_indices for idx in indices]
    assert len(combined) == 12
    assert set(range(len(concat_ds))).issubset(set(combined))


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("sizes", [[10, 30, 60], [7, 1, 100, 13], [50, 50, 50]])
def test_interleaved_order_every_prefix_holds_the_proportional_shares(sizes, shuffle):
    dataset = _concat(sizes)
    indices = _build_interleaved_shard_indices(dataset, shuffle=shuffle, seed=3)
    assert sorted(indices) == list(range(sum(sizes)))

    shards = _shard_of(dataset, np.array(indices))
    counts = np.zeros(len(sizes))
    total = sum(sizes)
    for position, shard in enumerate(shards, start=1):
        counts[shard] += 1
        expected = position * np.array(sizes) / total
        assert np.all(np.abs(counts - expected) < 1.0 + len(sizes)), f"prefix {position}: {counts} vs {expected}"


def test_interleaved_order_is_shuffled_and_seeded():
    dataset = _concat([10, 30, 60])
    first = _build_interleaved_shard_indices(dataset, shuffle=True, seed=1)
    assert first == _build_interleaved_shard_indices(dataset, shuffle=True, seed=1)
    assert first != _build_interleaved_shard_indices(dataset, shuffle=True, seed=2)

    largest = [index for index in first if index >= 40]
    assert largest != sorted(largest)  # events within a shard are shuffled

    unshuffled = _build_interleaved_shard_indices(dataset, shuffle=False)
    assert [index for index in unshuffled if index >= 40] == list(range(40, 100))


def test_interleaved_order_takes_one_event_per_equal_shard_per_round():
    sizes = [20, 20, 20, 20]
    dataset = _concat(sizes)
    shards = _shard_of(dataset, np.array(_build_interleaved_shard_indices(dataset, shuffle=True, seed=5)))
    rounds = shards.reshape(-1, len(sizes))
    assert all(list(row) == list(range(len(sizes))) for row in rounds)  # equal positions tie, in shard order


def test_interleaved_order_skips_empty_shards():
    assert _build_interleaved_shard_indices(_concat([0, 0]), shuffle=True, seed=0) == []
    dataset = _concat([0, 5, 0, 5])
    assert sorted(_build_interleaved_shard_indices(dataset, shuffle=True, seed=0)) == list(range(10))


def test_distributed_interleaved_ranks_split_the_proportional_order():
    sizes = [10, 30, 60]
    dataset = _concat(sizes)
    ranks = []
    for rank in range(4):
        sampler = DistributedInterleavedShardSampler(dataset, world_size=4, rank=rank, shuffle=True, seed=7)
        ranks.append(list(sampler))
    assert sorted(index for rank_indices in ranks for index in rank_indices) == list(range(sum(sizes)))
    for rank_indices in ranks:
        counts = np.bincount(_shard_of(dataset, np.array(rank_indices)), minlength=len(sizes))
        assert np.all(np.abs(counts - np.array(sizes) / 4) < 2 * (1 + len(sizes)))  # a block = difference of two prefixes


def test_tfds_datasource_adds_source_and_input_type_metadata_and_pads_features():
    example = {
        "X": np.ones((3, 4), dtype=np.float32),
        "ycand": np.zeros((3, 2), dtype=np.float32),
        "ytarget": np.zeros((3, 7), dtype=np.float32),
    }
    example["ytarget"][:, 2] = 1.0
    example["ytarget"][:, 6] = 1.0
    ds = TFDSDataSource(DummyTFDS("clic_edm_ttbar_hits", example), sort=False, feature_dim=6)

    event = ds[0]

    assert event["source_id"] == SOURCE_IDS["clic"]
    assert event["input_type_id"] == INPUT_TYPE_IDS["hits"]
    assert event["X"].shape == (3, 6)
    assert np.all(event["X"][:, :4] == 1.0)
    assert np.all(event["X"][:, 4:] == 0.0)


@pytest.mark.parametrize(
    ("dataset_name", "source_id", "input_type_id"),
    [
        ("cms_pf_ttbar", SOURCE_IDS["cms"], INPUT_TYPE_IDS["pf"]),
        ("clic_edm_ttbar_pf", SOURCE_IDS["clic"], INPUT_TYPE_IDS["pf"]),
        ("clic_edm_ttbar_hits", SOURCE_IDS["clic"], INPUT_TYPE_IDS["hits"]),
        ("cld_edm_ttbar_hits", SOURCE_IDS["cld"], INPUT_TYPE_IDS["hits"]),
        ("idea_edm_qq_pf", SOURCE_IDS["idea"], INPUT_TYPE_IDS["pf"]),
        ("unknown_sample", SOURCE_IDS["unknown"], INPUT_TYPE_IDS["pf"]),
    ],
)
def test_dataset_domain_metadata_mapping(dataset_name, source_id, input_type_id):
    assert dataset_source_id(dataset_name) == source_id
    assert dataset_input_type_id(dataset_name) == input_type_id


def test_collater_preserves_source_and_input_type_metadata():
    collater = Collater(["X", "ytarget"], ["source_id", "input_type_id"])
    batch = collater(
        [
            {
                "X": np.ones((2, 3), dtype=np.float32),
                "ytarget": np.zeros((2, 2), dtype=np.float32),
                "source_id": np.int64(2),
                "input_type_id": np.int64(1),
            },
            {
                "X": np.ones((3, 3), dtype=np.float32),
                "ytarget": np.zeros((3, 2), dtype=np.float32),
                "source_id": np.int64(3),
                "input_type_id": np.int64(2),
            },
        ]
    )

    assert batch.X.shape == (2, 3, 3)
    assert batch.source_id.tolist() == [2, 3]
    assert batch.input_type_id.tolist() == [1, 2]
