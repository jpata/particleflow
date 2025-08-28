import unittest
import torch
from torch.utils.data import DataLoader, Dataset

from mlpf.model.PFDataset import InterleavedIterator, EndlessIterator

# A mock dataset that returns dictionaries, similar to the real one
class MockDictDataset(Dataset):
    def __init__(self, size=20, offset=0):
        self.size = size
        self.offset = offset
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        print(idx, self.offset)
        return {"X": torch.tensor([float(idx + self.offset)])}

class TestEndlessInterleavedIterator(unittest.TestCase):
    def test_save_restore_with_endless(self):
        """
        Tests that the combination of EndlessIterator and InterleavedIterator
        can be saved and restored correctly. This isolated test should
        reproduce the bug from test_dataloader.py.
        """
        # 1. Setup DataLoaders
        # Use num_workers=0 to prove the bug is not related to multiprocessing
        loader1 = DataLoader(MockDictDataset(size=20, offset=0), batch_size=2) # 10 batches
        loader2 = DataLoader(MockDictDataset(size=20, offset=100), batch_size=2) # 10 batches

        # --- Ground truth run ---
        # Manually iterate the InterleavedIterator to get the expected sequence
        gt_iter = InterleavedIterator([loader1, loader2])
        # We will run for 10 steps total, so we need the first 10 batches
        gt_data = [batch['X'].clone() for i, batch in enumerate(gt_iter) if i < 10]
        print("gt_data", gt_data)

        # --- Interrupted Run (run1) ---
        # We don't need torch.manual_seed for SequentialSampler, but good practice
        torch.manual_seed(0)
        inter_iter1 = InterleavedIterator([loader1, loader2])
        # samplers and world_size are not used in this simplified case
        endless_iter1 = EndlessIterator(inter_iter1, samplers=[], world_size=1)

        run1_data = []
        for i in range(5):
            batch = next(endless_iter1)
            run1_data.append(batch['X'].clone())
        print("run1_data", run1_data)
        state = endless_iter1.state_dict()

        # --- Restored Run (run2) ---
        torch.manual_seed(0) # Re-seed to ensure loaders are identical before loading state
        inter_iter2 = InterleavedIterator([loader1, loader2])
        endless_iter2 = EndlessIterator(inter_iter2, samplers=[], world_size=1)
        endless_iter2.load_state_dict(state)

        run2_data = []
        for i in range(5):
            batch = next(endless_iter2)
            run2_data.append(batch['X'].clone())
        print("run2_data", run2_data)

        # --- Verification ---
        combined_data = run1_data + run2_data

        self.assertEqual(len(combined_data), len(gt_data))
        for i in range(len(gt_data)):
            if not torch.equal(combined_data[i], gt_data[i]):
                print(f"Mismatch at index {i}:")
                print(f"  Combined: {combined_data[i].numpy().flatten()}")
                print(f"  Ground Truth: {gt_data[i].numpy().flatten()}")
            self.assertTrue(torch.equal(combined_data[i], gt_data[i]))

if __name__ == '__main__':
    unittest.main()
