
import pytest
import torch
from torch.utils.data import Dataset, ConcatDataset
from mlpf.model.PFDataset import DistributedShardConsecutiveSampler

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
    total_elements = sum(shard_sizes)
    
    # Check all ranks
    yielded_counts = []
    for rank in range(world_size):
        sampler = DistributedShardConsecutiveSampler(
            concat_ds, 
            world_size=world_size, 
            rank=rank, 
            shuffle=shuffle, 
            drop_last=drop_last
        )
        
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
