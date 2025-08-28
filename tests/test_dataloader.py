import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.optim as optim
from torch.utils.data import Dataset

from mlpf.model.PFDataset import get_interleaved_dataloaders
from mlpf.model.utils import save_checkpoint, load_checkpoint
from mlpf.model.mlpf import MLPF


class MockTorchDataset(Dataset):
    """A mock torch dataset that returns dictionaries."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"X": torch.tensor([[float(idx), 1.0]]), "ytarget": torch.tensor([[float(idx), 1.0]]), "genmet": 0.0}


class TestDataloaderRestoration(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @patch("mlpf.model.PFDataset.PFDataset")
    def test_restoration(self, MockPFDataset):
        """Ensures that the dataloader state is correctly saved and restored."""
        # Configure the mock PFDataset to return our mock torch dataset
        mock_pf_instance = MockPFDataset.return_value
        mock_pf_instance.ds = MockTorchDataset()

        # Set seed once for reproducibility
        torch.manual_seed(0)

        # 1. Create mock config
        config = {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "train_dataset": {
                "cms": {
                    "type1": {
                        "batch_size": 2,
                        "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                    }
                }
            },
            "valid_dataset": {  # needed for get_interleaved_dataloaders
                "cms": {
                    "type1": {
                        "batch_size": 2,
                        "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                    }
                }
            },
            "ntrain": 100,
            "nvalid": 100,
            "num_workers": 2,  # Test with multiple workers
            "prefetch_factor": 2,
            "sort_data": False,
            "pad_to_multiple_elements": None,
            "gpu_batch_multiplier": 1,
        }

        world_size = 1
        rank = 0

        # --- Ground Truth Run: run uninterrupted to get the target data sequence ---
        loaders_gt, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False)
        train_loader_gt = loaders_gt["train"]
        gt_data = []
        for i, batch in enumerate(train_loader_gt):
            if i >= 10:
                break
            gt_data.append(batch.X.clone())
        print(gt_data)

        # --- Interrupted Run ---

        # This run will be stopped mid-way and a checkpoint will be saved.
        # We need to re-seed to ensure this run starts with the same shuffle as the ground truth run.
        torch.manual_seed(0)
        loaders1, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False)
        train_loader1 = loaders1["train"]

        run1_data = []
        for i in range(5):
            batch = next(train_loader1)
            run1_data.append(batch.X.clone())
        print(run1_data)

        model = MLPF(input_dim=2, num_classes=2)
        optimizer = optim.Adam(model.parameters())
        checkpoint_path = os.path.join(self.tempdir, "checkpoint.pth")
        extra_state = {"step": 5, "train_loader_state_dict": train_loader1.state_dict()}
        save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        # --- Restored Run ---
        model2 = MLPF(input_dim=2, num_classes=2)
        optimizer2 = optim.Adam(model2.parameters())

        # This run will load the checkpoint and continue where the previous run left off.
        # It does not need to be re-seeded, as the RNG state is restored from the checkpoint.
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model2, optimizer2)

        loaders2, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False)
        train_loader2 = loaders2["train"]
        train_loader2.load_state_dict(checkpoint["extra_state"]["train_loader_state_dict"])

        run2_data = []
        for i in range(5):
            batch = next(train_loader2)
            run2_data.append(batch.X.clone())
        print(run2_data)

        # --- Verification ---
        combined_data = run1_data + run2_data

        self.assertEqual(len(combined_data), len(gt_data))
        for i in range(len(gt_data)):
            self.assertTrue(torch.equal(combined_data[i], gt_data[i]))
