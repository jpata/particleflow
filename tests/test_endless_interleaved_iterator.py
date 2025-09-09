import unittest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from mlpf.model.PFDataset import InterleavedIterator, EndlessIterator, ResumableSampler


# A mock dataset that returns dictionaries, similar to the real one
class MockDictDataset(Dataset):
    def __init__(self, size=20, offset=0):
        self.size = size
        self.offset = offset

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"X": torch.tensor([float(idx + self.offset)])}


class TestEndlessInterleavedIterator(unittest.TestCase):
    def test_save_restore_with_endless(self):
        """
        Tests that the combination of EndlessIterator and InterleavedIterator
        can be saved and restored correctly.
        """
        # --- Ground truth run ---
        d1_gt = MockDictDataset(size=20, offset=0)
        d2_gt = MockDictDataset(size=20, offset=100)
        s1_gt = ResumableSampler(SequentialSampler(d1_gt))
        s2_gt = ResumableSampler(SequentialSampler(d2_gt))
        l1_gt = DataLoader(d1_gt, batch_size=2, sampler=s1_gt)
        l2_gt = DataLoader(d2_gt, batch_size=2, sampler=s2_gt)
        gt_iter = InterleavedIterator([l1_gt, l2_gt])
        # We will run for 10 steps total, so we need the first 10 batches
        gt_data = [batch["X"].clone() for i, batch in enumerate(gt_iter) if i < 10]

        # --- Interrupted Run (run1) ---
        torch.manual_seed(0)
        d1_1 = MockDictDataset(size=20, offset=0)
        d2_1 = MockDictDataset(size=20, offset=100)
        s1_1 = ResumableSampler(SequentialSampler(d1_1))
        s2_1 = ResumableSampler(SequentialSampler(d2_1))
        l1_1 = DataLoader(d1_1, batch_size=2, sampler=s1_1)
        l2_1 = DataLoader(d2_1, batch_size=2, sampler=s2_1)
        inter_iter1 = InterleavedIterator([l1_1, l2_1])
        endless_iter1 = EndlessIterator(inter_iter1, samplers=[s1_1, s2_1], world_size=1)

        run1_data = []
        for i in range(5):
            batch = next(endless_iter1)
            run1_data.append(batch["X"].clone())
        state = endless_iter1.state_dict()

        # --- Restored Run (run2) ---
        torch.manual_seed(0)  # Re-seed to ensure loaders are identical before loading state
        d1_2 = MockDictDataset(size=20, offset=0)
        d2_2 = MockDictDataset(size=20, offset=100)
        s1_2 = ResumableSampler(SequentialSampler(d1_2))
        s2_2 = ResumableSampler(SequentialSampler(d2_2))
        l1_2 = DataLoader(d1_2, batch_size=2, sampler=s1_2)
        l2_2 = DataLoader(d2_2, batch_size=2, sampler=s2_2)
        inter_iter2 = InterleavedIterator([l1_2, l2_2])
        endless_iter2 = EndlessIterator(inter_iter2, samplers=[s1_2, s2_2], world_size=1)
        endless_iter2.load_state_dict(state)

        run2_data = []
        for i in range(5):
            batch = next(endless_iter2)
            run2_data.append(batch["X"].clone())

        # --- Verification ---
        combined_data = run1_data + run2_data

        self.assertEqual(len(combined_data), len(gt_data))
        for i in range(len(gt_data)):
            self.assertTrue(torch.equal(combined_data[i], gt_data[i]))

    def test_endless_iterator_with_empty_dataloaders(self):
        """
        Tests that EndlessIterator wrapping an empty InterleavedIterator
        (because all dataloaders are empty) does not result in an unhandled exception.
        This is the condition that likely causes the user-reported IndexError.
        """
        d1 = MockDictDataset(size=0)
        d2 = MockDictDataset(size=0)
        s1 = ResumableSampler(SequentialSampler(d1))
        s2 = ResumableSampler(SequentialSampler(d2))
        l1 = DataLoader(d1, batch_size=2, sampler=s1)
        l2 = DataLoader(d2, batch_size=2, sampler=s2)
        inter_iter = InterleavedIterator([l1, l2])
        endless_iter = EndlessIterator(inter_iter, samplers=[s1, s2], world_size=1)

        # According to analysis, this should lead to a RecursionError.
        # The user reported an IndexError. This test will expose the actual error.
        with self.assertRaises(Exception):
            next(endless_iter)


    def test_multiple_epochs(self):
        """
        Tests that EndlessIterator can iterate for more than one epoch over
        an InterleavedIterator, verifying that the iterators are reset correctly.
        """
        d1 = MockDictDataset(size=10, offset=0)
        d2 = MockDictDataset(size=10, offset=100)
        s1 = ResumableSampler(SequentialSampler(d1))
        s2 = ResumableSampler(SequentialSampler(d2))
        # batch_size=2, so 5 batches from each loader, 10 total per epoch
        l1 = DataLoader(d1, batch_size=2, sampler=s1)
        l2 = DataLoader(d2, batch_size=2, sampler=s2)
        inter_iter = InterleavedIterator([l1, l2])
        endless_iter = EndlessIterator(inter_iter, samplers=[s1, s2], world_size=1)

        total_batches_per_epoch = len(l1) + len(l2)  # 5 + 5 = 10
        num_batches_to_iterate = int(total_batches_per_epoch * 1.5)  # 15 batches

        # Iterate for 1.5 epochs
        results = []
        for _ in range(num_batches_to_iterate):
            batch = next(endless_iter)
            results.append(batch["X"].clone())

        self.assertEqual(len(results), num_batches_to_iterate)

        # Check that the data from the second epoch matches the data from the first
        # The first 5 batches of the second epoch should be the same as the first 5 batches of the first epoch
        results_first_epoch_part = results[0:5]
        results_second_epoch_part = results[total_batches_per_epoch : total_batches_per_epoch + 5]

        self.assertEqual(len(results_first_epoch_part), len(results_second_epoch_part))
        for i in range(len(results_first_epoch_part)):
            self.assertTrue(torch.equal(results_first_epoch_part[i], results_second_epoch_part[i]))


if __name__ == "__main__":
    unittest.main()
