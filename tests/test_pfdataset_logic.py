import unittest
import torch
import numpy as np
from tests.mock_data import MockTFDS
from mlpf.model.PFDataset import PFBatch, Collater, TFDSDataSource


class TestPFBatch(unittest.TestCase):
    def test_basic(self):
        X = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
        ytarget = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        batch = PFBatch(X=X, ytarget=ytarget)

        self.assertTrue(torch.equal(batch.X, X))
        self.assertTrue(torch.equal(batch.ytarget, ytarget))
        # Mask should be true where X[:, :, 0] != 0
        expected_mask = torch.tensor([[True, False]])
        self.assertTrue(torch.equal(batch.mask, expected_mask))

    def test_to_device(self):
        X = torch.tensor([[[1.0, 2.0]]])
        batch = PFBatch(X=X)
        # Just test it doesn't crash, since we might not have GPU
        device = torch.device("cpu")
        batch_cpu = batch.to(device)
        self.assertEqual(batch_cpu.X.device.type, "cpu")


class TestCollater(unittest.TestCase):
    def test_padding_and_stacking(self):
        per_particle = ["X", "ytarget"]
        per_event = ["genmet"]
        collater = Collater(per_particle, per_event)

        item1 = {
            "X": np.array([[1.0, 1.1], [1.2, 1.3]]),
            "ytarget": np.array([[2.0, 2.1], [2.2, 2.3]]),
            "genmet": np.array([10.0]),
        }
        item2 = {
            "X": np.array([[3.0, 3.1]]),
            "ytarget": np.array([[4.0, 4.1]]),
            "genmet": np.array([20.0]),
        }

        batch = collater([item1, item2])

        # item2 should be padded with zeros to length 2
        self.assertEqual(batch.X.shape, (2, 2, 2))
        self.assertEqual(batch.X[1, 1, 0], 0.0)
        self.assertEqual(batch.X[1, 1, 1], 0.0)

        self.assertEqual(batch.ytarget.shape, (2, 2, 2))
        self.assertEqual(batch.genmet.shape, (2, 1))
        self.assertTrue(torch.equal(batch.genmet, torch.tensor([[10.0], [20.0]])))


class TestTFDSDataSource(unittest.TestCase):
    def test_cms_remapping(self):
        # CMS remapping logic:
        # if name starts with cms_:
        # (X[:, 0] == 1) & (ytarget[:, 0] == 2) -> ytarget[:, 0] = 1
        data = {
            "X": np.array([[1, 10.0, 0.0, 0.0, 0.0, 10.0], [4, 10.0, 0.0, 0.0, 0.0, 10.0]]),
            "ytarget": np.array([[2, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0], [1, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]]),
            "ycand": np.array([[0, 0], [0, 0]]),
        }
        mock_ds = MockTFDS([data], name="cms_pf")
        ds = TFDSDataSource(mock_ds, sort=False)

        ret = ds[0]
        # (X[0,0]==1 and ytarget[0,0]==2) -> ytarget[0,0] becomes 1
        self.assertEqual(ret["ytarget"][0, 0], 1)
        # (X[1,0]==4 and ytarget[1,0]==1) -> ytarget[1,0] becomes 5
        self.assertEqual(ret["ytarget"][1, 0], 5)

    def test_log_transforms(self):
        # pt transform: log(ytarget_pt / X_pt)
        # X_pt is at index 1, ytarget_pt is at index 2
        data = {
            "X": np.array([[1, 10.0, 0.0, 0.0, 0.0, 10.0]]),
            "ytarget": np.array([[1, 0.0, 20.0, 0.0, 0.0, 0.0, 40.0]]),
            "ycand": np.array([[0, 0]]),
        }
        mock_ds = MockTFDS([data], name="generic")
        ds = TFDSDataSource(mock_ds, sort=False)

        ret = ds[0]
        # target_pt = log(20/10) = log(2)
        self.assertAlmostEqual(ret["ytarget"][0, 2], np.log(2.0))
        # target_e = log(40/10) = log(4)
        self.assertAlmostEqual(ret["ytarget"][0, 6], np.log(4.0))
        self.assertEqual(ret["ytarget_pt_orig"][0], 20.0)
        self.assertEqual(ret["ytarget_e_orig"][0], 40.0)

    def test_padding(self):
        data = {
            "X": np.array([[1, 10.0, 0.0, 0.0, 0.0, 10.0]]),
            "ytarget": np.array([[1, 0.0, 20.0, 0.0, 0.0, 0.0, 40.0]]),
            "ycand": np.array([[0, 0]]),
        }
        mock_ds = MockTFDS([data], name="generic")
        # Pad to multiple of 4
        ds = TFDSDataSource(mock_ds, sort=False, pad_to_multiple=4)

        ret = ds[0]
        self.assertEqual(ret["X"].shape[0], 4)
        self.assertEqual(ret["ytarget"].shape[0], 4)
        self.assertEqual(ret["X"][1, 0], 0)  # Padded with zero


if __name__ == "__main__":
    unittest.main()
