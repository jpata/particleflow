"""
Spec: Validates 'InterleavedShardSampler' for mixed-source training.
Key tests: Ensures early batches cover multiple ConcatDataset shards, verifies
deterministic shuffled ordering, and checks distributed ranks cover the same
global interleaved sequence without overlap when no padding is required.
"""

import pytest
from torch.utils.data import ConcatDataset

from mlpf.model.PFDataset import DistributedInterleavedShardSampler, InterleavedShardSampler, ShardConsecutiveSampler
from tests.mock_data import MockDataset


def shard_id(index, shard_size):
    return index // shard_size


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
