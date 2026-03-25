import unittest
from torch.utils.data import Dataset, ConcatDataset
from mlpf.model.PFDataset import ShardConsecutiveSampler


class MockShard(Dataset):
    def __init__(self, start, end):
        self.data = list(range(start, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestShardConsecutiveSampler(unittest.TestCase):
    def test_consecutive_no_shuffle(self):
        shards = [MockShard(0, 10), MockShard(10, 20), MockShard(20, 30)]
        concat = ConcatDataset(shards)
        sampler = ShardConsecutiveSampler(concat, shuffle=False)

        indices = list(sampler)
        # Expected: 0, 1, ..., 9, 10, 11, ..., 19, 20, 21, ..., 29
        self.assertEqual(indices, list(range(30)))

    def test_consecutive_shuffle_within_shards(self):
        # We want to check that it stays within a shard until it's done,
        # but the order within the shard is shuffled.
        shards = [MockShard(0, 10), MockShard(10, 20), MockShard(20, 30)]
        concat = ConcatDataset(shards)
        sampler = ShardConsecutiveSampler(concat, shuffle=True, seed=42)

        indices = list(sampler)
        self.assertEqual(len(indices), 30)

        # Split into 3 blocks of 10
        block1 = sorted(indices[0:10])
        block2 = sorted(indices[10:20])
        block3 = sorted(indices[20:30])

        # Each block should contain elements from EXACTLY ONE shard
        # (Since shard order is also shuffled, we don't know which one)
        blocks = [block1, block2, block3]
        expected_shards = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]

        for b in blocks:
            self.assertIn(b, expected_shards)
            expected_shards.remove(b)
        self.assertEqual(len(expected_shards), 0)


if __name__ == "__main__":
    unittest.main()
