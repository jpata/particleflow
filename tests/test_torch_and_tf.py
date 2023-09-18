import unittest
import numpy as np
import tensorflow as tf
import torch


class TestGNNTorchAndTensorflow(unittest.TestCase):
    def test_GHConvDense(self):
        from mlpf.tfmodel.model import GHConvDense
        from mlpf.pyg.model import GHConvDense as GHConvDenseTorch

        nn1 = GHConvDense(output_dim=128, activation="selu")
        nn2 = GHConvDenseTorch(output_dim=128, activation="selu", hidden_dim=64)

        x = np.random.normal(size=(2, 4, 64, 64)).astype(np.float32)
        adj = np.random.normal(size=(2, 4, 64, 64, 1)).astype(np.float32)
        msk = np.random.normal(size=(2, 4, 64, 1)).astype(np.float32)
        msk = (msk > 0).astype(np.float32)

        # run once to initialize weights
        nn1((tf.convert_to_tensor(x), tf.convert_to_tensor(adj), tf.convert_to_tensor(msk))).numpy()
        nn2((torch.tensor(x), torch.tensor(adj), torch.tensor(msk))).detach().numpy()

        # ensure weights of the TF and pytorch models are the same
        sd = nn2.state_dict()
        sd["W_t"] = torch.from_numpy(nn1.weights[0].numpy())
        sd["b_t"] = torch.from_numpy(nn1.weights[1].numpy())
        sd["W_h"] = torch.from_numpy(nn1.weights[2].numpy())
        sd["theta"] = torch.from_numpy(nn1.weights[3].numpy())
        nn2.load_state_dict(sd)

        out1 = nn1((tf.convert_to_tensor(x), tf.convert_to_tensor(adj), tf.convert_to_tensor(msk))).numpy()
        out2 = nn2((torch.tensor(x), torch.tensor(adj), torch.tensor(msk))).detach().numpy()

        # this is only approximate, so it might fail in rare cases
        self.assertLess(np.sum(out1 - out2), 1e-2)

    def test_MessageBuildingLayerLSH(self):
        from mlpf.tfmodel.model import MessageBuildingLayerLSH
        from mlpf.pyg.model import MessageBuildingLayerLSH as MessageBuildingLayerLSHTorch

        nn1 = MessageBuildingLayerLSH(distance_dim=128, bin_size=64)
        nn2 = MessageBuildingLayerLSHTorch(distance_dim=128, bin_size=64)

        x_dist = np.random.normal(size=(2, 256, 128)).astype(np.float32)
        x_node = np.random.normal(size=(2, 256, 32)).astype(np.float32)
        msk = np.random.normal(size=(2, 256)).astype(np.float32)
        msk = (msk > 0).astype(bool)

        # run once to initialize weights
        nn1(tf.convert_to_tensor(x_dist), tf.convert_to_tensor(x_node), tf.convert_to_tensor(msk))
        nn2(torch.tensor(x_dist), torch.tensor(x_node), torch.tensor(msk))

        sd = nn2.state_dict()
        sd["codebook_random_rotations"] = torch.from_numpy(nn1.weights[0].numpy())
        nn2.load_state_dict(sd)

        out1 = nn1(tf.convert_to_tensor(x_dist), tf.convert_to_tensor(x_node), tf.convert_to_tensor(msk))
        out2 = nn2(torch.tensor(x_dist), torch.tensor(x_node), torch.tensor(msk))

        self.assertTrue(np.all(out1[0].numpy() == out2[0].numpy()))
        self.assertLess(np.sum(out1[1].numpy() - out2[1].detach().numpy()), 1e-2)
        self.assertLess(np.sum(out1[2].numpy() - out2[2].detach().numpy()), 1e-2)
        self.assertEqual(np.sum(out1[3].numpy() - out2[3].detach().numpy()), 0.0)

        from mlpf.tfmodel.model import reverse_lsh

        bins_split, x, dm, msk_f = out1
        ret = reverse_lsh(bins_split, x, False)
        self.assertTrue(np.all(x_node == ret.numpy()))

        from mlpf.pyg.model import reverse_lsh as reverse_lsh_torch

        bins_split, x, dm, msk_f = out2
        ret = reverse_lsh_torch(bins_split, x)
        self.assertTrue(np.all(x_node == ret.detach().numpy()))
