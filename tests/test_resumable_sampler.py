import unittest
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DistributedSampler

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
