import unittest
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DistributedSampler, Sampler

from mlpf.model.PFDataset import ResumableSampler


class TestResumableSampler(unittest.TestCase):
    def test_sampler_state_and_len(self):
        """Tests that the sampler can be advanced and its length is correct."""
        d = TensorDataset(torch.arange(10))
        base_sampler = SequentialSampler(d)
        resumable_sampler = ResumableSampler(base_sampler)

        self.assertEqual(len(resumable_sampler), 10)

        # Check initial iteration
        self.assertEqual(list(resumable_sampler), list(range(10)))

        # Advance the sampler
        resumable_sampler.load_state_dict({"start_index": 5})
        self.assertEqual(len(resumable_sampler), 10)  # Should report full length
        self.assertEqual(list(resumable_sampler), list(range(5, 10)))

    def test_distributed_sampler_compatibility(self):
        """Tests that set_epoch is correctly passed to a wrapped DistributedSampler."""

        class MockDistributedSampler(DistributedSampler):
            def __init__(self, dataset):
                # super() requires distributed process group to be initialized
                # We are just mocking the set_epoch method, so we can skip the parent init
                self.dataset = dataset
                self.epoch = 0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                return iter(range(len(self.dataset)))

            def __len__(self):
                return len(self.dataset)

        d = TensorDataset(torch.arange(10))
        dist_sampler = MockDistributedSampler(d)
        resumable_sampler = ResumableSampler(dist_sampler)

        resumable_sampler.set_epoch(5)
        self.assertEqual(resumable_sampler.sampler.epoch, 5)

    def test_shuffling_with_set_epoch(self):
        """
        Tests that set_epoch is correctly used to shuffle data across epochs,
        even when the sampler is reset.
        """

        class MockShufflingSampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source
                self.epoch = 0

            def __iter__(self):
                g = torch.Generator()
                g.manual_seed(self.epoch)
                indices = torch.randperm(len(self.data_source), generator=g).tolist()
                return iter(indices)

            def __len__(self):
                return len(self.data_source)

            def set_epoch(self, epoch):
                self.epoch = epoch

        dataset = TensorDataset(torch.arange(100))
        base_sampler = MockShufflingSampler(dataset)
        resumable_sampler = ResumableSampler(base_sampler)

        # Epoch 1
        resumable_sampler.set_epoch(0)
        epoch1_indices = list(resumable_sampler)

        # Simulate advancing and then starting a new epoch
        resumable_sampler.load_state_dict({"start_index": 50})  # mid-epoch
        resumable_sampler.set_epoch(1)  # new epoch
        resumable_sampler.load_state_dict({"start_index": 0})  # iterator reset

        # Epoch 2
        epoch2_indices = list(resumable_sampler)

        # Check that the order is different
        self.assertNotEqual(epoch1_indices, epoch2_indices)

        # Check that both epochs contain all samples
        self.assertEqual(sorted(epoch1_indices), list(range(100)))
        self.assertEqual(sorted(epoch2_indices), list(range(100)))

        # Check that epoch 1 is reproducible by resetting the epoch
        resumable_sampler.set_epoch(0)
        resumable_sampler.load_state_dict({"start_index": 0})
        epoch1_redux_indices = list(resumable_sampler)
        self.assertEqual(epoch1_indices, epoch1_redux_indices)