import unittest
from torch.utils.data import Dataset, ConcatDataset
from mlpf.model.PFDataset import DistributedShardConsecutiveSampler


class MockShard(Dataset):
    def __init__(self, start, end):
        self.data = list(range(start, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDistributedShardConsecutiveSampler(unittest.TestCase):
    def test_init_with_world_size(self):
        shards = [MockShard(0, 10), MockShard(10, 20)]
        concat = ConcatDataset(shards)
        # This is expected to succeed now
        sampler = DistributedShardConsecutiveSampler(concat, world_size=2, rank=0, shuffle=False)
        self.assertEqual(sampler.num_replicas, 2)
        self.assertEqual(sampler.rank, 0)

    def test_distributed_sharding(self):
        # 4 shards, 2 ranks
        shards = [MockShard(0, 10), MockShard(10, 20), MockShard(20, 30), MockShard(30, 40)]
        concat = ConcatDataset(shards)

        sampler0 = DistributedShardConsecutiveSampler(concat, world_size=2, rank=0, shuffle=False)
        sampler1 = DistributedShardConsecutiveSampler(concat, world_size=2, rank=1, shuffle=False)

        indices0 = list(sampler0)
        indices1 = list(sampler1)

        # Rank 0 should get shards 0 and 2 (indices 0-9 and 20-29)
        # Rank 1 should get shards 1 and 3 (indices 10-19 and 30-39)
        # Note: the implementation uses my_shard_ranges = shard_ranges[self.rank :: self.num_replicas]

        self.assertEqual(len(indices0), 20)
        self.assertEqual(len(indices1), 20)

        self.assertEqual(set(indices0), set(range(0, 10)) | set(range(20, 30)))
        self.assertEqual(set(indices1), set(range(10, 20)) | set(range(30, 40)))

        # Combined should be all indices
        self.assertEqual(set(indices0) | set(indices1), set(range(40)))
        # No overlap
        self.assertEqual(set(indices0) & set(indices1), set())

    def test_distributed_sharding_shuffle(self):
        # 4 shards, 2 ranks, shuffle=True
        shards = [MockShard(0, 10), MockShard(10, 20), MockShard(20, 30), MockShard(30, 40)]
        concat = ConcatDataset(shards)

        # Use same seed to ensure shard order is same for both ranks
        seed = 42
        sampler0 = DistributedShardConsecutiveSampler(concat, world_size=2, rank=0, shuffle=True, seed=seed)
        sampler1 = DistributedShardConsecutiveSampler(concat, world_size=2, rank=1, shuffle=True, seed=seed)

        indices0 = list(sampler0)
        indices1 = list(sampler1)

        self.assertEqual(len(indices0), 20)
        self.assertEqual(len(indices1), 20)

        # Combined should still be all indices
        self.assertEqual(set(indices0) | set(indices1), set(range(40)))
        # No overlap
        self.assertEqual(set(indices0) & set(indices1), set())

        # Check that it stayed within shards (each block of 10 should be one shard)
        # Shard ranges: 0-9, 10-19, 20-29, 30-39
        def get_shard_id(val):
            return val // 10

        shard_ids0 = [get_shard_id(x) for x in indices0]
        shard_ids1 = [get_shard_id(x) for x in indices1]

        # Each block of 10 should have same shard id
        self.assertEqual(len(set(shard_ids0[:10])), 1)
        self.assertEqual(len(set(shard_ids0[10:])), 1)
        self.assertEqual(len(set(shard_ids1[:10])), 1)
        self.assertEqual(len(set(shard_ids1[10:])), 1)


if __name__ == "__main__":
    unittest.main()
