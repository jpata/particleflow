"""
Spec: Validates 'InterleavedIterator' (validation) and 'WeightedMixIterator' / 'next_group' (training) in 'mlpf.model.PFDataset'. Key tests: interleaving order, weighted step shares spread evenly over training, independent per-group epochs, state-dict saving/loading across epoch boundaries (incl. pre-weighting checkpoints), and empty dataloaders.
"""

import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from mlpf.model.PFDataset import InterleavedIterator, ResumableSampler, WeightedMixIterator, _train_mix_iterator, next_group


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

    def test_reusability(self):
        """Tests that the InterleavedIterator can be iterated over multiple times."""
        d1 = TensorDataset(torch.arange(10))
        d2 = TensorDataset(torch.arange(5))

        s1 = ResumableSampler(SequentialSampler(d1))
        s2 = ResumableSampler(SequentialSampler(d2))

        loader1 = DataLoader(d1, batch_size=1, sampler=s1)
        loader2 = DataLoader(d2, batch_size=1, sampler=s2)

        inter_iter = InterleavedIterator([loader1, loader2])

        # First iteration
        results1 = [item[0].item() for item in inter_iter]
        self.assertEqual(len(results1), 15)

        # Second iteration should yield the same results
        results2 = [item[0].item() for item in inter_iter]
        self.assertEqual(results1, results2)


def _loader(size, offset, batch_size=1, num_workers=0):
    ds = TensorDataset(torch.arange(offset, offset + size))
    sampler = ResumableSampler(SequentialSampler(ds))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers), sampler


def _mix(sizes, weights=None, batch_size=1, num_workers=0):
    pairs = [_loader(n, 1000 * i, batch_size, num_workers) for i, n in enumerate(sizes)]
    return WeightedMixIterator([p[0] for p in pairs], weights=weights)


def _groups(it, n):
    return [next(it)[0][0].item() // 1000 for _ in range(n)]


class TestNextGroup(unittest.TestCase):
    def test_every_prefix_stays_within_one_batch_of_the_target_share(self):
        weights = [3.0, 1.0, 0.5]
        counts = [0, 0, 0]
        for n in range(1, 901):
            counts[next_group(counts, weights)] += 1
            for c, w in zip(counts, weights):
                self.assertLessEqual(abs(c - n * w / sum(weights)), 1.0)
        self.assertEqual(counts, [600, 200, 100])


class TestWeightedMixIterator(unittest.TestCase):
    def test_default_weights_give_one_pass_per_group_per_nominal_epoch(self):
        it = _mix([2, 10])
        self.assertEqual(len(it), 12)
        groups = _groups(it, 12)
        self.assertEqual(groups.count(0), 2)
        self.assertEqual(groups.count(1), 10)
        # the short group is spread over the epoch, not front-loaded
        self.assertEqual([i for i, g in enumerate(groups) if g == 0], [2, 8])

    def test_weights_set_the_step_share_and_short_groups_cycle(self):
        it = _mix([2, 10], weights=[1.0, 1.0])
        groups = _groups(it, 20)
        self.assertEqual(groups, [0, 1] * 10)
        self.assertEqual(it.epochs, [4, 0])  # 0-based pass index; rolls over on the next request
        stats = it.stats()
        self.assertAlmostEqual(stats["0/step_fraction"], 0.5)
        self.assertAlmostEqual(stats["0/passes"], 5.0)
        self.assertAlmostEqual(stats["1/passes"], 1.0)

    def test_empty_groups_are_skipped_and_all_empty_raises(self):
        it = _mix([0, 3], weights=[1.0, 1.0])
        self.assertEqual(_groups(it, 4), [1, 1, 1, 1])
        with self.assertRaises(RuntimeError):
            next(_mix([0, 0]))

    def _run_save_restore(self, num_workers):
        sizes, weights, n_steps, stop = [4, 6], [2.0, 1.0], 25, 11
        uninterrupted = _mix(sizes, weights, 2, num_workers)
        reference = [next(uninterrupted)[0].clone() for _ in range(n_steps)]

        first = _mix(sizes, weights, 2, num_workers)
        part1 = [next(first)[0].clone() for _ in range(stop)]
        resumed = _mix(sizes, weights, 2, num_workers)
        resumed.load_state_dict(first.state_dict())
        part2 = [next(resumed)[0].clone() for _ in range(n_steps - stop)]

        for got, want in zip(part1 + part2, reference):
            self.assertTrue(torch.equal(got, want))

    def test_save_restore_across_epochs_single_worker(self):
        self._run_save_restore(num_workers=0)

    def test_save_restore_across_epochs_multi_worker(self):
        self._run_save_restore(num_workers=2)

    def test_loads_pre_weighting_state(self):
        it = _mix([4, 2])
        it.load_state_dict({"epoch": 1, "loader_state_dict": {"cur_index": 5, "batches_yielded_per_loader": [3, 2]}})
        self.assertEqual(it.batches, [7, 4])
        self.assertEqual(it.epochs, [1, 1])
        self.assertEqual(it.scheduled, [0, 0])  # no weights in the old state: the schedule restarts
        # group 0 resumes at its 4th (last) batch and then wraps; group 1 had used its epoch up,
        # so its next batch is the first of epoch 2
        self.assertEqual([next(it)[0][0].item() for _ in range(3)], [3, 1000, 0])
        self.assertEqual(it.epochs, [2, 2])

    def test_train_groups_pass_their_weights(self):
        loaders = [_loader(4, 0), _loader(4, 1000)]
        args = ([p[0] for p in loaders], 1)
        it = _train_mix_iterator({"a": SimpleNamespace(weight=3.0), "b": SimpleNamespace(weight=1.0)}, *args)
        self.assertEqual((it.names, it.weights), (["a", "b"], [3.0, 1.0]))
        self.assertEqual(_train_mix_iterator({"a": SimpleNamespace(weight=None), "b": SimpleNamespace(weight=None)}, *args).weights, [4.0, 4.0])

    def test_resuming_with_other_weights_restarts_the_schedule(self):
        first = _mix([100, 100], weights=[1.0, 1.0])
        _groups(first, 50)
        resumed = _mix([100, 100], weights=[1.0, 3.0])
        resumed.load_state_dict(first.state_dict())
        self.assertEqual(resumed.scheduled, [0, 0])
        # the 1:3 shares apply at once, rather than a catch-up run of group 1 over the old counts [25, 25]
        self.assertEqual(_groups(resumed, 8), [1, 0, 1, 1, 1, 0, 1, 1])
        self.assertEqual(resumed.batches, [27, 31])  # each group continues from its own sampler position

    def test_mismatched_group_count_raises(self):
        with self.assertRaises(ValueError):
            _mix([2, 2]).load_state_dict({"batches_in_epoch": [1], "epochs": [0], "scheduled": [1], "weights": [2.0, 2.0]})


if __name__ == "__main__":
    unittest.main()
