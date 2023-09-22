import unittest
import tensorflow as tf


class TestGNN(unittest.TestCase):
    def helper_test_pairwise_dist_shape(self, dist_func):
        A = tf.random.normal((2, 128, 32))
        B = tf.random.normal((2, 128, 32))
        out = dist_func(A, B)
        self.assertEqual(out.shape, (2, 128, 128))

    def test_pairwise_l2_dist_shape(self):
        from mlpf.tfmodel.model import pairwise_l2_dist

        self.helper_test_pairwise_dist_shape(pairwise_l2_dist)

    def test_pairwise_l1_dist_shape(self):
        from mlpf.tfmodel.model import pairwise_l1_dist

        self.helper_test_pairwise_dist_shape(pairwise_l1_dist)

    def test_GHConvDense_shape(self):
        from mlpf.tfmodel.model import GHConvDense

        nn = GHConvDense(output_dim=128, activation="selu")

        x = tf.random.normal((2, 256, 64))
        adj = tf.random.normal((2, 256, 256, 1))
        msk = tf.random.normal((2, 256, 1))

        out = nn((x, adj, msk))
        self.assertEqual(out.shape, (2, 256, 128))

    def test_GHConvDense_binned_shape(self):
        from mlpf.tfmodel.model import GHConvDense

        nn = GHConvDense(output_dim=128, activation="selu")

        x = tf.random.normal((2, 4, 64, 64))
        adj = tf.random.normal((2, 4, 64, 64, 1))
        msk = tf.random.normal((2, 4, 64, 1))

        out = nn((x, adj, msk))
        self.assertEqual(out.shape, (2, 4, 64, 128))

    def test_NodePairGaussianKernel_shape(self):
        from mlpf.tfmodel.model import NodePairGaussianKernel

        nn = NodePairGaussianKernel()

        x = tf.random.normal((2, 256, 32))
        msk = tf.random.normal((2, 256, 1))

        out = nn(x, msk)
        self.assertEqual(out.shape, (2, 256, 256, 1))

    def test_NodePairGaussianKernel_binned_shape(self):
        from mlpf.tfmodel.model import NodePairGaussianKernel

        nn = NodePairGaussianKernel()

        x = tf.random.normal((2, 4, 64, 32))
        msk = tf.random.normal((2, 4, 64, 1))

        out = nn(x, msk)
        self.assertEqual(out.shape, (2, 4, 64, 64, 1))

    def test_MessageBuildingLayerLSH_shape(self):
        from mlpf.tfmodel.model import MessageBuildingLayerLSH

        nn = MessageBuildingLayerLSH(bin_size=64, distance_dim=128)

        x_dist = tf.random.normal((2, 256, 128))
        x_features = tf.random.normal((2, 256, 32))
        msk = tf.random.normal((2, 256)) > 0

        # bin the x_features using the distances in x_dist, check shapes
        bins_split, x_features_binned, dm_binned, msk_f_binned = nn(x_dist, x_features, msk)
        self.assertEqual(bins_split.shape, (2, 4, 64))
        self.assertEqual(x_features_binned.shape, (2, 4, 64, 32))
        self.assertEqual(dm_binned.shape, (2, 4, 64, 64, 1))
        self.assertEqual(msk_f_binned.shape, (2, 4, 64, 1))

        from mlpf.tfmodel.model import reverse_lsh

        # undo the LSH binning, check that the results is as before the binning
        x_features2 = reverse_lsh(bins_split, x_features_binned)
        self.assertEqual(tf.reduce_sum(x_features - x_features2).numpy(), 0)
