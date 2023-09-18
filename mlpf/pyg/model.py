import torch
from torch import nn


def split_indices_to_bins_batch(cmul, nbins, bin_size, msk):
    bin_idx = torch.argmax(cmul, axis=-1) + torch.where(~msk, nbins - 1, 0).to(torch.int64)
    bins_split = torch.reshape(torch.argsort(bin_idx, stable=True), (cmul.shape[0], nbins, bin_size))
    return bins_split


def pairwise_l2_dist(A, B):
    na = torch.sum(torch.square(A), -1)
    nb = torch.sum(torch.square(B), -1)

    # na as a row and nb as a column vectors
    na = torch.unsqueeze(na, -1)
    nb = torch.unsqueeze(nb, -2)

    # return pairwise euclidean difference matrix
    # note that this matrix multiplication can go out of range for float16 in case the absolute values of A and B are large
    D = torch.sqrt(torch.clip(na - 2 * torch.matmul(A, torch.transpose(B, -1, -2)) + nb, 1e-6, 1e6))
    return D


class GHConvDense(nn.Module):
    def __init__(self, *args, **kwargs):
        self.activation = getattr(nn.functional, kwargs.pop("activation"))
        self.output_dim = kwargs.pop("output_dim")
        self.normalize_degrees = kwargs.pop("normalize_degrees", True)
        self.hidden_dim = kwargs.pop("hidden_dim")
        super(GHConvDense, self).__init__(*args, **kwargs)

        self.W_t = torch.nn.Parameter(
            data=torch.zeros(self.hidden_dim, self.output_dim),
            requires_grad=True,
        )
        self.b_t = torch.nn.Parameter(
            data=torch.zeros(self.output_dim),
            requires_grad=True,
        )
        self.W_h = torch.nn.Parameter(
            data=torch.zeros(self.hidden_dim, self.output_dim),
            requires_grad=True,
        )
        self.theta = torch.nn.Parameter(
            data=torch.zeros(self.hidden_dim, self.output_dim),
            requires_grad=True,
        )

    def forward(self, inputs):
        x, adj, msk = inputs
        adj = torch.squeeze(adj, axis=-1)

        # compute the normalization of the adjacency matrix
        if self.normalize_degrees:
            in_degrees = torch.sum(torch.abs(adj), axis=-1)

            # add epsilon to prevent numerical issues from 1/sqrt(x)
            norm = torch.unsqueeze(torch.pow(in_degrees + 1e-6, -0.5), -1) * msk

        f_hom = torch.linalg.matmul(x * msk, self.theta) * msk
        if self.normalize_degrees:
            f_hom = torch.linalg.matmul(adj, f_hom * norm) * norm
        else:
            f_hom = torch.linalg.matmul(adj, f_hom)

        f_het = torch.linalg.matmul(x * msk, self.W_h)
        gate = torch.sigmoid(torch.linalg.matmul(x, self.W_t) + self.b_t)

        out = gate * f_hom + (1.0 - gate) * f_het
        return self.activation(out) * msk


class NodePairGaussianKernel(nn.Module):
    def __init__(self, **kwargs):
        self.clip_value_low = kwargs.pop("clip_value_low", 0.0)
        self.dist_mult = kwargs.pop("dist_mult", 0.1)

        dist_norm = kwargs.pop("dist_norm", "l2")
        if dist_norm == "l2":
            self.dist_norm = pairwise_l2_dist
        else:
            raise Exception("Unkown dist_norm: {}".format(dist_norm))
        super(NodePairGaussianKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, 1) message matrix
    """

    def forward(self, x_msg_binned, msk, training=False):
        x = x_msg_binned * msk
        dm = torch.unsqueeze(self.dist_norm(x, x), axis=-1)
        dm = torch.exp(-self.dist_mult * dm)
        dm = torch.clip(dm, self.clip_value_low, 1)
        return dm


def index_dim(a, b):
    return a[b]


class MessageBuildingLayerLSH(nn.Module):
    def __init__(
        self,
        distance_dim=128,
        max_num_bins=200,
        bin_size=128,
        kernel=NodePairGaussianKernel(),
        small_graph_opt=False,
        **kwargs
    ):
        self.initializer = kwargs.pop("initializer", "random_normal")
        super(MessageBuildingLayerLSH, self).__init__(**kwargs)

        self.distance_dim = distance_dim
        self.max_num_bins = max_num_bins
        self.bin_size = bin_size
        self.kernel = kernel
        self.small_graph_opt = small_graph_opt

        # generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = nn.Parameter(
            torch.randn(self.distance_dim, self.max_num_bins // 2),
            requires_grad=False,
        )

    def forward(self, x_msg, x_node, msk, training=False):
        msk_f = torch.unsqueeze(msk, -1)

        shp = x_msg.shape
        n_points = shp[1]

        # compute the number of LSH bins to divide the input points into on the fly
        # n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = torch.floor_divide(n_points, self.bin_size)

        shp = x_msg.shape
        n_points = shp[1]

        # compute the number of LSH bins to divide the input points into on the fly
        # n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = torch.floor_divide(n_points, self.bin_size)
        if n_bins > 1:
            mul = torch.linalg.matmul(
                x_msg,
                self.codebook_random_rotations[:, : torch.maximum(torch.tensor(1), n_bins // 2)],
            )
            cmul = torch.concatenate([mul, -mul], axis=-1)
            bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk)

            # replace tf.gather with torch.vmap, indexing and reshape
            bins_split_2 = torch.reshape(bins_split, (bins_split.shape[0], bins_split.shape[1] * bins_split.shape[2]))
            x_msg_binned = torch.vmap(index_dim)(x_msg, bins_split_2)
            x_features_binned = torch.vmap(index_dim)(x_node, bins_split_2)
            msk_f_binned = torch.vmap(index_dim)(msk, bins_split_2)
            x_msg_binned = torch.reshape(x_msg_binned, (cmul.shape[0], n_bins, self.bin_size, x_msg_binned.shape[-1]))
            x_features_binned = torch.reshape(
                x_features_binned, (cmul.shape[0], n_bins, self.bin_size, x_features_binned.shape[-1])
            )
            msk_f_binned = torch.reshape(msk_f_binned, (cmul.shape[0], n_bins, self.bin_size, 1))
        else:
            x_msg_binned = torch.unsqueeze(x_msg, axis=1)
            x_features_binned = torch.unsqueeze(x_node, axis=1)
            msk_f_binned = torch.unsqueeze(msk_f, axis=1)
            shp = x_msg_binned.shape
            bins_split = torch.zeros([shp[0], shp[1], shp[2]], dtype=torch.int32)

        # Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg_binned, msk_f_binned, training=training)

        # remove the masked points row-wise and column-wise
        msk_f_binned_squeeze = torch.squeeze(msk_f_binned, axis=-1)
        shp_dm = dm.shape
        rshp_row = [shp_dm[0], shp_dm[1], shp_dm[2], 1, 1]
        rshp_col = [shp_dm[0], shp_dm[1], 1, shp_dm[3], 1]
        msk_row = torch.reshape(msk_f_binned_squeeze, rshp_row).to(dm.dtype)
        msk_col = torch.reshape(msk_f_binned_squeeze, rshp_col).to(dm.dtype)
        dm = torch.multiply(dm, msk_row)
        dm = torch.multiply(dm, msk_col)

        return bins_split, x_features_binned, dm, msk_f_binned
