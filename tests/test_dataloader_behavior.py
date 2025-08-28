import unittest
import torch
from torch.utils.data import DataLoader, Dataset

# A mock dataset that returns dictionaries, similar to the real one
class MockDictDataset(Dataset):
    def __init__(self, size=20, offset=0):
        self.size = size
        self.offset = offset
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return {"X": torch.tensor([[float(idx + self.offset)]])}

class TestDataLoaderBehavior(unittest.TestCase):
    def test_fast_forward(self):
        """
        Tests if a DataLoader iterator behaves as expected after
        being manually fast-forwarded.
        """
        dataset = MockDictDataset(size=20, offset=0)
        loader = DataLoader(dataset, batch_size=2) # 10 batches

        # Fast-forward by 3 steps
        iterator = iter(loader)
        for _ in range(3):
            batch = next(iterator)
        
        # The next batch should be batch index 3, with values [6, 7]
        next_batch = next(iterator)
        self.assertTrue(torch.equal(next_batch['X'], torch.tensor([[[6.]],[[7.]]])))

        # Now, let's fast-forward until the end
        # We already took 4 batches (3 in loop, 1 manually)
        for _ in range(6): # Take the remaining 6 batches
            next(iterator)

        # The next call should raise StopIteration
        with self.assertRaises(StopIteration):
            next(iterator)
