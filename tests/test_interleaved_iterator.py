import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from mlpf.model.PFDataset import InterleavedIterator, ResumableSampler


class TestInterleavedIterator(unittest.TestCase):
    def test_length(self):
        """Tests that the length of the InterleavedIterator is correct."""
        d1 = TensorDataset(torch.arange(10))
        d2 = TensorDataset(torch.arange(5))
        loader1 = DataLoader(d1, batch_size=1)
        loader2 = DataLoader(d2, batch_size=1)

        inter_iter = InterleavedIterator([loader1, loader2])
        self.assertEqual(len(inter_iter), 15)

    def test_iteration_order(self):
        """Tests that the iterator yields batches in the correct interleaved order."""
        d1 = TensorDataset(torch.arange(0, 10, 2))  # 0, 2, 4, 6, 8
        d2 = TensorDataset(torch.arange(1, 10, 2))  # 1, 3, 5, 7, 9
        loader1 = DataLoader(d1, batch_size=1)
        loader2 = DataLoader(d2, batch_size=1)

        inter_iter = InterleavedIterator([loader1, loader2])

        results = [item[0].item() for item in inter_iter]
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.assertEqual(results, expected)

    def _run_save_restore_test(self, num_workers):
        """Helper to run the save/restore test with a given number of workers."""
        d1 = TensorDataset(torch.arange(0, 20, 2))  # 10 elements
        d2 = TensorDataset(torch.arange(1, 20, 2))  # 10 elements

        # --- Ground truth run ---
        s1_gt = ResumableSampler(SequentialSampler(d1))
        s2_gt = ResumableSampler(SequentialSampler(d2))
        l1_gt = DataLoader(d1, batch_size=1, sampler=s1_gt, num_workers=num_workers)
        l2_gt = DataLoader(d2, batch_size=1, sampler=s2_gt, num_workers=num_workers)
        inter_iter_gt = InterleavedIterator([l1_gt, l2_gt])
        gt_data = [item[0].item() for item in inter_iter_gt]

        # --- Interrupted run ---

        # Part 1: Iterate part-way and save state
        s1_1 = ResumableSampler(SequentialSampler(d1))
        s2_1 = ResumableSampler(SequentialSampler(d2))
        l1_1 = DataLoader(d1, batch_size=1, sampler=s1_1, num_workers=num_workers)
        l2_1 = DataLoader(d2, batch_size=1, sampler=s2_1, num_workers=num_workers)
        inter_iter1 = InterleavedIterator([l1_1, l2_1])
        run1_data = []
        stop_step = 7
        iterator1 = iter(inter_iter1)
        for _ in range(stop_step):
            item = next(iterator1)
            run1_data.append(item[0].item())

        state = inter_iter1.state_dict()
        self.assertEqual(state["cur_index"], stop_step)

        # Part 2: Create a new iterator, load state, and continue
        s1_2 = ResumableSampler(SequentialSampler(d1))
        s2_2 = ResumableSampler(SequentialSampler(d2))
        l1_2 = DataLoader(d1, batch_size=1, sampler=s1_2, num_workers=num_workers)
        l2_2 = DataLoader(d2, batch_size=1, sampler=s2_2, num_workers=num_workers)
        inter_iter2 = InterleavedIterator([l1_2, l2_2])
        inter_iter2.load_state_dict(state)

        run2_data = [item[0].item() for item in inter_iter2]

        # --- Verification ---
        combined_data = run1_data + run2_data
        self.assertEqual(combined_data, gt_data)

    def test_save_restore_single_worker(self):
        self._run_save_restore_test(num_workers=0)

    def test_save_restore_multi_worker(self):
        self._run_save_restore_test(num_workers=2)
