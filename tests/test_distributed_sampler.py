import pytest
from torch.utils.data import Dataset, ConcatDataset
from mlpf.model.PFDataset import ShardConsecutiveSampler, DistributedShardConsecutiveSampler


class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_distributed_shard_consecutive_sampler_sync(world_size, drop_last, shuffle):
    # Create a dataset with uneven shard sizes
    shard_sizes = [100, 150, 200, 50, 300]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)

    # Check all ranks
    yielded_counts = []
    for rank in range(world_size):
        sampler = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)

        indices = list(iter(sampler))
        yielded_counts.append(len(indices))

        # 1. Check that yielded length matches reported length
        assert len(indices) == len(sampler), f"Rank {rank} yielded {len(indices)} but reported {len(sampler)}"

    # 2. Check that all ranks yielded the same number of samples
    assert len(set(yielded_counts)) == 1, f"Ranks yielded different number of samples: {yielded_counts}"

    # 3. Check total yielded samples
    expected_per_rank = len(DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, drop_last=drop_last))
    assert yielded_counts[0] == expected_per_rank


def test_distributed_shard_consecutive_sampler_locality():
    # Large number of shards to check locality
    shard_sizes = [100] * 20
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)
    world_size = 2

    sampler0 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, shuffle=False)
    sampler1 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=1, shuffle=False)

    indices0 = list(iter(sampler0))
    indices1 = list(iter(sampler1))

    # Rank 0 should have the first half of indices, Rank 1 the second half
    # Because we slice consecutively: indices = all_indices[rank * num_samples : (rank+1) * num_samples]
    assert indices0[-1] < indices1[0]
    assert max(indices0) < min(indices1)


def test_shard_consecutive_sampler_basic():
    shard_sizes = [10, 20, 30]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)

    sampler = ShardConsecutiveSampler(concat_ds, shuffle=False)
    indices = list(iter(sampler))
    assert indices == list(range(60))

    sampler_shuffled = ShardConsecutiveSampler(concat_ds, shuffle=True, seed=42)
    indices_shuffled = list(iter(sampler_shuffled))
    assert len(indices_shuffled) == 60
    assert sorted(indices_shuffled) == list(range(60))
    assert indices_shuffled != list(range(60))


def test_distributed_shard_consecutive_sampler_determinism():
    shard_sizes = [10, 20, 30]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)
    world_size = 2

    # Check that all_indices is identical across ranks by seeing if they are disjoint and cover the dataset
    sampler0 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, shuffle=True, seed=42)
    sampler1 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=1, shuffle=True, seed=42)

    indices0 = list(iter(sampler0))
    indices1 = list(iter(sampler1))

    assert len(indices0) == 30
    assert len(indices1) == 30

    # They should be disjoint because 60 is divisible by 2 and no padding happened
    assert set(indices0).isdisjoint(set(indices1))
    assert sorted(indices0 + indices1) == list(range(60))


def test_distributed_shard_consecutive_sampler_padding():
    shard_sizes = [5, 5]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)
    world_size = 3  # 10 samples, 3 ranks -> 4 samples each, 12 total

    sampler0 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, shuffle=False, drop_last=False)
    sampler1 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=1, shuffle=False, drop_last=False)
    sampler2 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=2, shuffle=False, drop_last=False)

    indices0 = list(iter(sampler0))
    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    assert len(indices0) == 4
    assert len(indices1) == 4
    assert len(indices2) == 4

    all_indices = indices0 + indices1 + indices2
    assert len(all_indices) == 12
    # Check padding: last 2 should be from the beginning
    # all_indices = [0,1,2,3,4,5,6,7,8,9, 0,1]
    assert all_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]


def test_distributed_shard_consecutive_sampler_drop_last():
    shard_sizes = [5, 5]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)
    world_size = 3  # 10 samples, 3 ranks -> 3 samples each, 9 total

    sampler0 = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, shuffle=False, drop_last=True)
    indices0 = list(iter(sampler0))
    assert len(indices0) == 3


def test_distributed_shard_consecutive_sampler_epoch():
    shard_sizes = [10, 10]
    datasets = [MockDataset(s) for s in shard_sizes]
    concat_ds = ConcatDataset(datasets)
    world_size = 2

    sampler = DistributedShardConsecutiveSampler(concat_ds, world_size=world_size, rank=0, shuffle=True, seed=42)

    sampler.set_epoch(0)
    indices_e0 = list(iter(sampler))

    sampler.set_epoch(1)
    indices_e1 = list(iter(sampler))

    assert indices_e0 != indices_e1
