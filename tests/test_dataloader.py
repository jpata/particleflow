"""
Spec: Ensures the training dataloader state is correctly saved and restored via 'get_interleaved_dataloaders'. Mocks 'PFDataset' to isolate dataloader logic. Key assertions: Verifies that a 'restored' dataloader yields a sequence of data identical to an uninterrupted 'ground truth' run, even across epoch boundaries and partial batch exhaustion.
"""

import copy
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.optim as optim
from tests.mock_data import MockDictDataset

from mlpf.model.PFDataset import get_interleaved_dataloaders
from mlpf.model.utils import save_checkpoint, load_checkpoint
from mlpf.model.mlpf import MLPF
from mlpf.conf import MLPFConfig


class TestDataloaderRestoration(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @patch("mlpf.model.PFDataset.PFDataset")
    def test_validation_keeps_partial_per_rank_batch(self, MockPFDataset):
        """A small nvalid must still produce a batch on every distributed rank."""
        mock_pf_instance = MockPFDataset.return_value
        mock_pf_instance.ds = MockDictDataset(
            size=100,
            keys=("X", "ytarget", "genmet"),
            shapes=((1, 2), (1, 2), (1,)),
        )

        config = MLPFConfig.model_validate(
            {
                "dataset": "cms",
                "data_dir": "/tmp/dummy_data",
                "model": {"type": "attention", "attention": {"num_convs": 2}},
                "conv_type": "attention",
                "train_dataset": {
                    "cms": {
                        "physical": {
                            "batch_size": 1,
                            "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                        }
                    }
                },
                "valid_dataset": {
                    "cms": {
                        "physical": {
                            "batch_size": 1,
                            "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                        }
                    }
                },
                "ntrain": 100,
                "nvalid": 100,
                "num_workers": 1,
                "prefetch_factor": 2,
                "sort_data": False,
                "pad_to_multiple_elements": None,
                "gpu_batch_multiplier": 64,
                "sampler_mode": "interleaved-shards",
            }
        )

        loaders, _ = get_interleaved_dataloaders(
            world_size=8,
            rank=2,
            config=config,
            use_cuda=False,
            use_ray=False,
            shuffle_train=False,
        )

        train_data_loader = loaders["train"].data_loaders[0]
        valid_loader = loaders["valid"]
        valid_data_loader = valid_loader.data_loaders[0]

        self.assertTrue(train_data_loader.drop_last)
        self.assertFalse(valid_data_loader.drop_last)
        self.assertEqual(len(valid_loader), 1)
        self.assertEqual(next(iter(valid_loader)).X.shape[0], 13)

    def _two_group_config(self, sampler_mode):
        def group(batch_size, weight, sample):
            return {"batch_size": batch_size, "weight": weight, "samples": {sample: {"version": "1.0.0", "splits": ["1"]}}}

        return MLPFConfig.model_validate(
            {
                "dataset": "cms",
                "data_dir": "/tmp/dummy_data",
                "model": {"type": "attention", "attention": {"num_convs": 2}},
                "conv_type": "attention",
                "train_dataset": {"cms": {"big": group(2, 2.0, "sample_big"), "small": group(1, 1.0, "sample_small")}},
                "valid_dataset": {"cms": {"big": group(2, None, "sample_big"), "small": group(1, None, "sample_small")}},
                "num_workers": 0,
                "sort_data": False,
                "pad_to_multiple_elements": None,
                "gpu_batch_multiplier": 1,
                "sampler_mode": sampler_mode,
            }
        )

    @staticmethod
    def _fake_pfdataset(data_dir, name, split, **kwargs):
        # sample_big: 48 events (values 0..47), sample_small: 40 events (values 1000..1039)
        size, offset = (48, 0) if name.startswith("sample_big") else (40, 1000)
        return SimpleNamespace(ds=MockDictDataset(size=size, offset=offset, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,))))

    def _check_multi_rank_resume(self, sampler_mode):
        """Every rank resumes its own batch sequence from rank 0's saved loader state, as in DDP."""
        world_size, n_steps, stop = 4, 30, 12
        config = self._two_group_config(sampler_mode)

        def train_loader(rank):
            with patch("mlpf.model.PFDataset.PFDataset", side_effect=self._fake_pfdataset):
                loaders, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False)
            return loaders["train"]

        def take(loader, n):
            return [next(loader).X[:, 0, 0].tolist() for _ in range(n)]

        reference = {rank: take(train_loader(rank), n_steps) for rank in range(world_size)}
        first = {rank: train_loader(rank) for rank in range(world_size)}
        part1 = {rank: take(first[rank], stop) for rank in range(world_size)}
        state = first[0].state_dict()  # only rank 0 writes the checkpoint
        for rank in range(world_size):
            self.assertEqual(first[rank].state_dict(), state)
            resumed = train_loader(rank)
            resumed.load_state_dict(state)
            self.assertEqual(part1[rank] + take(resumed, n_steps - stop), reference[rank])

        # all ranks draw the same group at every step (batch size 2 = big, 1 = small), 2:1 overall
        sizes = [[len(b) for b in reference[rank]] for rank in range(world_size)]
        self.assertTrue(all(s == sizes[0] for s in sizes))
        self.assertEqual(sizes[0].count(2), 20)
        # big has 6 batches per rank and epoch, so 20 steps cover 3+ epochs; ranks see disjoint events
        first_epoch = [{v for b in reference[rank][:12] if len(b) == 2 for v in b} for rank in range(world_size)]
        self.assertEqual(len(set().union(*first_epoch)), 48)

    def test_multi_rank_resume_shard_consecutive(self):
        self._check_multi_rank_resume("shard-consecutive")

    def test_multi_rank_resume_interleaved_shards(self):
        self._check_multi_rank_resume("interleaved-shards")

    def test_validation_reports_group_names(self):
        config = self._two_group_config("shard-consecutive")
        with patch("mlpf.model.PFDataset.PFDataset", side_effect=self._fake_pfdataset):
            loaders, _ = get_interleaved_dataloaders(1, 0, config, use_cuda=False, use_ray=False)
        valid = loaders["valid"]
        self.assertEqual(valid.names, ["big", "small"])
        seen = set()
        for batch in valid:
            seen.add((valid.names[valid.last_loader_index], len(batch.X)))
        self.assertEqual(seen, {("big", 2), ("small", 1)})

    @patch("mlpf.model.PFDataset.PFDataset")
    def test_restoration(self, MockPFDataset):
        """Ensures that the dataloader state is correctly saved and restored."""
        # Configure the mock PFDataset to return our mock torch dataset
        mock_pf_instance = MockPFDataset.return_value
        mock_pf_instance.ds = MockDictDataset(size=100, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,)))

        # Set seed once for reproducibility
        torch.manual_seed(0)

        # 1. Create mock config
        config_dict = {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "model": {
                "type": "attention",
                "attention": {"num_convs": 2},
            },
            "conv_type": "attention",
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
            "num_workers": 2,
            "prefetch_factor": 2,
            "sort_data": False,
            "pad_to_multiple_elements": None,
            "gpu_batch_multiplier": 1,
        }
        config = MLPFConfig.model_validate(config_dict)

        world_size = 1
        rank = 0

        # --- Ground Truth Run: run uninterrupted to get the target data sequence ---
        loaders_gt, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
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
        loaders1, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader1 = loaders1["train"]

        run1_data = []
        for i in range(5):
            batch = next(train_loader1)
            run1_data.append(batch.X.clone())
        print(run1_data)

        model = MLPF(config)
        optimizer = optim.Adam(model.parameters())
        checkpoint_path = os.path.join(self.tempdir, "checkpoint.pth")
        extra_state = {"step": 5, "train_loader_state_dict": train_loader1.state_dict()}
        save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        # --- Restored Run ---

        # This run will load the checkpoint and continue where the previous run left off.
        # It does not need to be re-seeded, as the RNG state is restored from the checkpoint.
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model, optimizer)
        loaders2, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
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

    @patch("mlpf.model.PFDataset.PFDataset")
    def test_restoration_until_exhausted(self, MockPFDataset):
        """
        Tests that a restored dataloader can be iterated until it is
        fully exhausted, and that it correctly transitions to the next epoch.
        """
        mock_pf_instance = MockPFDataset.return_value
        mock_pf_instance.ds = MockDictDataset(size=21, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,)))

        torch.manual_seed(0)
        config_dict = {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "model": {
                "type": "attention",
                "attention": {"num_convs": 2},
            },
            "conv_type": "attention",
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
            "num_workers": 2,
            "prefetch_factor": 2,
            "sort_data": False,
            "pad_to_multiple_elements": None,
            "gpu_batch_multiplier": 1,
        }
        config = MLPFConfig.model_validate(config_dict)
        world_size = 1
        rank = 0

        num_total_batches = 21 // 2  # 10
        num_batches_to_test = int(num_total_batches * 1.5)  # 15 batches
        stop_at_batch = 4

        # --- Ground Truth Run: get the full data sequence for 1.5 epochs ---
        torch.manual_seed(0)
        loaders_gt, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader_gt = loaders_gt["train"]
        gt_iterator = iter(train_loader_gt)
        gt_data = []
        for _ in range(num_batches_to_test):
            gt_data.append(next(gt_iterator).X.clone())
        self.assertEqual(len(gt_data), num_batches_to_test)

        # --- Interrupted Run ---
        torch.manual_seed(0)
        loaders1, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader1 = loaders1["train"]
        run1_data = []
        iterator1 = iter(train_loader1)
        for _ in range(stop_at_batch):
            run1_data.append(next(iterator1).X.clone())

        model = MLPF(config)
        optimizer = optim.Adam(model.parameters())
        checkpoint_path = os.path.join(self.tempdir, "checkpoint.pth")
        extra_state = {"step": stop_at_batch, "train_loader_state_dict": train_loader1.state_dict()}
        save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        # --- Restored Run ---
        checkpoint = torch.load(checkpoint_path)
        loaders2, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader2 = loaders2["train"]
        train_loader2.load_state_dict(checkpoint["extra_state"]["train_loader_state_dict"])
        run2_data = []
        iterator2 = iter(train_loader2)
        for _ in range(num_batches_to_test - stop_at_batch):  # 11 batches
            run2_data.append(next(iterator2).X.clone())

        # --- Verification ---
        combined_data = run1_data + run2_data
        self.assertEqual(len(combined_data), len(gt_data))
        for i in range(len(gt_data)):
            self.assertTrue(torch.equal(combined_data[i], gt_data[i]), f"Batch {i} differs")

    @patch("mlpf.model.PFDataset.PFDataset")
    def test_sampler_from_scratch(self, MockPFDataset):
        """Ensures that when sampler_from_scratch is True, the dataloader state is NOT loaded and starts from scratch."""
        mock_pf_instance = MockPFDataset.return_value
        mock_pf_instance.ds = MockDictDataset(size=100, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,)))

        torch.manual_seed(0)

        # Create config with sampler_from_scratch = True
        config_dict = {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "model": {
                "type": "attention",
                "attention": {"num_convs": 2},
            },
            "conv_type": "attention",
            "train_dataset": {
                "cms": {
                    "type1": {
                        "batch_size": 2,
                        "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                    }
                }
            },
            "valid_dataset": {
                "cms": {
                    "type1": {
                        "batch_size": 2,
                        "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}},
                    }
                }
            },
            "ntrain": 100,
            "nvalid": 100,
            "num_workers": 2,
            "prefetch_factor": 2,
            "sort_data": False,
            "pad_to_multiple_elements": None,
            "gpu_batch_multiplier": 1,
            "sampler_from_scratch": True,
        }
        config = MLPFConfig.model_validate(config_dict)
        self.assertTrue(config.sampler_from_scratch)

        world_size = 1
        rank = 0

        # --- Ground Truth Run (uninterrupted, starting from scratch) ---
        loaders_gt, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader_gt = loaders_gt["train"]
        gt_data = []
        for i, batch in enumerate(train_loader_gt):
            if i >= 5:
                break
            gt_data.append(batch.X.clone())

        # --- Interrupted Run ---
        torch.manual_seed(0)
        loaders1, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader1 = loaders1["train"]

        run1_data = []
        for i in range(3):
            batch = next(train_loader1)
            run1_data.append(batch.X.clone())

        # Mock saving a checkpoint with loader state
        checkpoint = {"extra_state": {"train_loader_state_dict": train_loader1.state_dict()}}

        # --- Restored Run with sampler_from_scratch = True ---
        loaders2, _ = get_interleaved_dataloaders(world_size, rank, config, use_cuda=False, use_ray=False, shuffle_train=False)
        train_loader2 = loaders2["train"]

        # In training.py / distributed_ray.py: load state dict only if not sampler_from_scratch
        if not config.sampler_from_scratch:
            train_loader2.load_state_dict(checkpoint["extra_state"]["train_loader_state_dict"])

        run2_data = []
        for i in range(2):
            batch = next(train_loader2)
            run2_data.append(batch.X.clone())

        # Verification: since we did NOT load the state dict, train_loader2 should start from scratch (first 2 batches of gt_data)
        # instead of continuing from batch 3.
        self.assertTrue(torch.equal(run2_data[0], gt_data[0]))
        self.assertTrue(torch.equal(run2_data[1], gt_data[1]))


def _mock_loader_config(MockPFDataset, size, **kwargs):
    MockPFDataset.return_value.ds = MockDictDataset(size=size, keys=("X", "ytarget", "genmet"), shapes=((1, 2), (1, 2), (1,)))
    split_config = {"cms": {"type1": {"batch_size": 2, "samples": {"sample1": {"version": "1.0.0", "splits": ["split1"]}}}}}
    return MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp/dummy_data",
            "model": {"type": "attention", "attention": {"num_convs": 2}},
            "conv_type": "attention",
            "train_dataset": split_config,
            "valid_dataset": split_config,
            **kwargs,
        }
    )


@pytest.mark.parametrize("sampler_mode", ["shard-consecutive", "interleaved-shards"])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("stop_at_batch", [3, 5, 7, 10])  # 5 batches per epoch: mid/end of epoch 0, mid/end of epoch 1
@patch("mlpf.model.PFDataset.PFDataset")
def test_shuffled_restoration_matches_uninterrupted(MockPFDataset, stop_at_batch, num_workers, sampler_mode):
    """A shuffled train loader resumed after any batch, also mid-epoch or exactly at the end of a later epoch, yields
    the same batches and epoch as the uninterrupted loader (as in run(): the state is restored into a fresh loader)."""
    config = _mock_loader_config(MockPFDataset, 10, num_workers=num_workers, prefetch_factor=2, sampler_mode=sampler_mode)
    num_batches = 18

    def train_loader():
        loaders, _ = get_interleaved_dataloaders(1, 0, config, use_cuda=False, use_ray=False)
        return loaders["train"]

    uninterrupted = train_loader()
    gt_data = [next(uninterrupted).X.clone() for _ in range(num_batches)]

    interrupted = train_loader()
    data = [next(interrupted).X.clone() for _ in range(stop_at_batch)]
    state = copy.deepcopy(interrupted.state_dict())

    resumed = train_loader()
    resumed.load_state_dict(state)
    assert len(resumed) == len(uninterrupted) == 5
    data += [next(resumed).X.clone() for _ in range(num_batches - stop_at_batch)]

    assert not all(torch.equal(a, b) for a, b in zip(gt_data[0:5], gt_data[5:10]))  # the epochs are shuffled differently
    for i, (batch, gt_batch) in enumerate(zip(data, gt_data)):
        assert torch.equal(batch, gt_batch), f"batch {i} differs"
    assert resumed.epochs == uninterrupted.epochs


@pytest.mark.parametrize("size", [10, 11])  # full / partial last validation batch
@patch("mlpf.model.PFDataset.PFDataset")
def test_validation_loader_restored_after_a_validation_pass(MockPFDataset, size):
    """Checkpoints are saved between validation passes, so they hold an exhausted valid loader state. The restored loader
    reports and yields a full pass (its len() was 0, or raised ValueError with a partial last batch)."""
    config = _mock_loader_config(MockPFDataset, size, num_workers=0)
    loader = get_interleaved_dataloaders(1, 0, config, use_cuda=False, use_ray=False)[0]["valid"]
    gt_data = [batch.X.clone() for batch in loader]
    state = copy.deepcopy(loader.state_dict())
    assert state["cur_index"] == len(gt_data)

    restored = get_interleaved_dataloaders(1, 0, config, use_cuda=False, use_ray=False)[0]["valid"]
    restored.load_state_dict(state)
    assert len(restored) == len(gt_data)
    assert len(restored.data_loaders[0]) == 0  # nothing left in the restored pass, even after a partial last batch
    data = [batch.X.clone() for batch in restored]
    assert len(data) == len(gt_data)
    assert all(torch.equal(a, b) for a, b in zip(data, gt_data))
