"""
Spec: 'get_train_epoch' reports the fractional training epoch (completed epochs + fraction of the current epoch's batches consumed)
of the endless train loader built by 'get_interleaved_dataloaders'. Mocks 'PFDataset'. Key assertions: the epoch advances by
1/batches_per_epoch per batch across epoch boundaries, and continues from the right value after a mid-epoch state restore.
"""

import copy
import unittest
from unittest.mock import patch

from tests.mock_data import MockDictDataset

from mlpf.conf import MLPFConfig
from mlpf.model.PFDataset import get_interleaved_dataloaders
from mlpf.model.training import get_train_epoch

NUM_EVENTS = 100
BATCH_SIZE = 10
BATCHES_PER_EPOCH = NUM_EVENTS // BATCH_SIZE


def _make_config():
    dataset_section = {
        "cms": {
            "type1": {
                "batch_size": BATCH_SIZE,
                "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
            }
        }
    }
    return MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "model": {"type": "attention", "attention": {"num_convs": 2}},
            "conv_type": "attention",
            "train_dataset": dataset_section,
            "valid_dataset": dataset_section,
            "ntrain": NUM_EVENTS,
            "nvalid": NUM_EVENTS,
            "num_workers": 0,
            "sort_data": False,
            "pad_to_multiple_elements": None,
            "gpu_batch_multiplier": 1,
        }
    )


def _make_train_loader(config):
    loaders, _ = get_interleaved_dataloaders(1, 0, config, use_cuda=False, use_ray=False, shuffle_train=False)
    return loaders["train"]


class TestTrainEpoch(unittest.TestCase):
    @patch("mlpf.model.PFDataset.PFDataset")
    def test_epoch_progress_and_restore(self, MockPFDataset):
        MockPFDataset.return_value.ds = MockDictDataset(size=NUM_EVENTS, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,)))
        config = _make_config()

        train_loader = _make_train_loader(config)
        self.assertEqual(get_train_epoch(train_loader), 0.0)

        # run across two epoch boundaries, stopping mid-epoch
        num_batches = 2 * BATCHES_PER_EPOCH + 3
        for step in range(1, num_batches + 1):
            next(train_loader)
            self.assertAlmostEqual(get_train_epoch(train_loader), step / BATCHES_PER_EPOCH)

        # a restored loader continues from the same fractional epoch, even though its data loader lengths have shrunk
        state = copy.deepcopy(train_loader.state_dict())
        restored = _make_train_loader(config)
        restored.load_state_dict(state)
        self.assertAlmostEqual(get_train_epoch(restored), num_batches / BATCHES_PER_EPOCH)
        for step in range(num_batches + 1, num_batches + BATCHES_PER_EPOCH + 1):
            next(restored)
            self.assertAlmostEqual(get_train_epoch(restored), step / BATCHES_PER_EPOCH)


if __name__ == "__main__":
    unittest.main()
