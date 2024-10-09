import torch
from torch import nn


def point_wise_feed_forward_network(
    d_in,
    d_hidden,
    d_out,
    num_layers=1,
    activation="ELU",
    dropout=0.0,
):
    layers = []
    layers.append(
        nn.Linear(
            d_in,
            d_hidden,
        )
    )
    layers.append(getattr(nn, activation)())
    for ilayer in range(num_layers - 1):
        layers.append(
            nn.Linear(
                d_hidden,
                d_hidden,
            )
        )
        layers.append(getattr(nn, activation)())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


def split_indices_to_bins_batch(cmul, nbins, bin_size, msk, stable_sort=False):
    a = torch.argmax(cmul, axis=-1)

    # This gives a CUDA error for some reason
    # b = torch.where(~msk, nbins - 1, 0)
    # b = b.to(torch.int64)

    b = torch.zeros(msk.shape, dtype=torch.int64, device=cmul.device)
    # JP: check if this should be ~msk or msk (both here and in the TF implementation)
    b[~msk] = nbins - 1

    bin_idx = a + b
    if stable_sort:
        bins_split = torch.argsort(bin_idx, stable=True)
    else:
        # for ONNX export to work, stable must not be provided at all as an argument
        bins_split = torch.argsort(bin_idx)
    bins_split = bins_split.reshape((cmul.shape[0], nbins, bin_size))
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
        self.normalize_degrees = kwargs.pop("normalize_degrees", False)
        self.hidden_dim = kwargs.pop("hidden_dim")
        super(GHConvDense, self).__init__(*args, **kwargs)

        self.W_t = torch.nn.Parameter(
            data=torch.randn(self.hidden_dim, self.output_dim),
            requires_grad=True,
        )
        self.b_t = torch.nn.Parameter(
            data=torch.randn(self.output_dim),
            requires_grad=True,
        )
        self.W_h = torch.nn.Parameter(
            data=torch.randn(self.hidden_dim, self.output_dim),
            requires_grad=True,
        )
        self.theta = torch.nn.Parameter(
            data=torch.randn(self.hidden_dim, self.output_dim),
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


def split_msk_and_msg(bins_split, cmul, x_msg, x_node, msk, n_bins, bin_size):
    bins_split_2 = torch.reshape(bins_split, (bins_split.shape[0], bins_split.shape[1] * bins_split.shape[2]))

    bins_split_3 = torch.unsqueeze(bins_split_2, axis=-1).expand(bins_split_2.shape[0], bins_split_2.shape[1], x_msg.shape[-1])
    x_msg_binned = torch.gather(x_msg, 1, bins_split_3)
    x_msg_binned = torch.reshape(x_msg_binned, (cmul.shape[0], n_bins, bin_size, x_msg_binned.shape[-1]))

    bins_split_3 = torch.unsqueeze(bins_split_2, axis=-1).expand(bins_split_2.shape[0], bins_split_2.shape[1], x_node.shape[-1])
    x_features_binned = torch.gather(x_node, 1, bins_split_3)
    x_features_binned = torch.reshape(x_features_binned, (cmul.shape[0], n_bins, bin_size, x_features_binned.shape[-1]))

    msk_f_binned = torch.gather(msk, 1, bins_split_2)
    msk_f_binned = torch.reshape(msk_f_binned, (cmul.shape[0], n_bins, bin_size, 1))
    return x_msg_binned, x_features_binned, msk_f_binned


def reverse_lsh(bins_split, points_binned_enc):
    shp = points_binned_enc.shape
    batch_dim = shp[0]
    n_points = shp[1] * shp[2]
    n_features = shp[-1]

    bins_split_flat = torch.reshape(bins_split, (batch_dim, n_points))
    points_binned_enc_flat = torch.reshape(points_binned_enc, (batch_dim, n_points, n_features))

    ret = torch.zeros(batch_dim, n_points, n_features, device=points_binned_enc.device)
    for ibatch in range(batch_dim):
        # torch._assert(torch.min(bins_split_flat[ibatch]) >= 0, "reverse_lsh n_points min")
        # torch._assert(torch.max(bins_split_flat[ibatch]) < n_points, "reverse_lsh n_points max")
        ret[ibatch][bins_split_flat[ibatch]] = points_binned_enc_flat[ibatch]
    return ret


class MessageBuildingLayerLSH(nn.Module):
    def __init__(self, distance_dim=128, max_num_bins=200, bin_size=128, kernel=NodePairGaussianKernel(), **kwargs):
        self.initializer = kwargs.pop("initializer", "random_normal")
        super(MessageBuildingLayerLSH, self).__init__(**kwargs)

        self.distance_dim = distance_dim
        self.max_num_bins = max_num_bins
        self.bin_size = bin_size
        self.kernel = kernel
        self.stable_sort = False

        # generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = nn.Parameter(
            torch.randn(self.distance_dim, self.max_num_bins // 2),
            requires_grad=False,
        )

    def forward(self, x_msg, x_node, msk, training=False):
        shp = x_msg.shape
        n_points = torch.tensor(shp[1])

        if n_points % self.bin_size != 0:
            raise Exception("Number of elements per event must be exactly divisible by the bin size")

        # compute the number of LSH bins to divide the input points into on the fly
        # n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = torch.floor_divide(n_points, self.bin_size)

        mul = torch.linalg.matmul(
            x_msg,
            self.codebook_random_rotations[:, : torch.maximum(torch.tensor(1), n_bins // 2)],
        )
        cmul = torch.concatenate([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk, self.stable_sort)

        # replaced tf.gather with torch.vmap, indexing and reshape
        x_msg_binned, x_features_binned, msk_f_binned = split_msk_and_msg(bins_split, cmul, x_msg, x_node, msk, n_bins, self.bin_size)

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


class CombinedGraphLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        self.inout_dim = kwargs.pop("inout_dim")
        self.max_num_bins = kwargs.pop("max_num_bins")
        self.bin_size = kwargs.pop("bin_size")
        self.distance_dim = kwargs.pop("distance_dim")
        self.do_layernorm = kwargs.pop("layernorm")
        self.num_node_messages = kwargs.pop("num_node_messages")
        self.dropout = kwargs.pop("dropout")
        self.ffn_dist_hidden_dim = kwargs.pop("ffn_dist_hidden_dim")
        self.ffn_dist_num_layers = kwargs.pop("ffn_dist_num_layers", 2)
        self.dist_activation = getattr(torch.nn.functional, kwargs.pop("dist_activation", "elu"))
        super(CombinedGraphLayer, self).__init__(**kwargs)

        if self.do_layernorm:
            self.layernorm1 = torch.nn.LayerNorm(
                self.inout_dim,
                eps=1e-6,
            )

        self.ffn_dist = point_wise_feed_forward_network(
            self.inout_dim,
            self.ffn_dist_hidden_dim,
            self.distance_dim,
            num_layers=self.ffn_dist_num_layers,
            dropout=self.dropout,
        )

        self.message_building_layer = MessageBuildingLayerLSH(
            distance_dim=self.distance_dim,
            max_num_bins=self.max_num_bins,
            bin_size=self.bin_size,
            kernel=NodePairGaussianKernel(),
        )

        self.message_passing_layers = nn.ModuleList()
        for iconv in range(self.num_node_messages):
            self.message_passing_layers.append(GHConvDense(output_dim=self.inout_dim, hidden_dim=self.inout_dim, activation="elu"))
        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, x, msk):
        n_elems = x.shape[1]
        bins_to_pad_to = -torch.floor_divide(-n_elems, self.bin_size)

        # pad the element dimension
        pad_size = (0, 0, 0, bins_to_pad_to * self.bin_size - n_elems)
        x = torch.nn.functional.pad(x, pad_size)

        pad_size = (0, bins_to_pad_to * self.bin_size - n_elems)
        msk = torch.nn.functional.pad(msk, pad_size, value=True)

        if self.do_layernorm:
            x = self.layernorm1(x)

        # compute node features for graph building
        x_dist = self.dist_activation(self.ffn_dist(x))

        # compute the element-to-element messages / distance matrix / graph structure
        bins_split, x, dm, msk_f = self.message_building_layer(x_dist, x, msk)

        # run the node update with message passing
        for msg in self.message_passing_layers:
            x_out = msg((x, dm, msk_f))
            x = x_out
            if self.dropout_layer:
                x = self.dropout_layer(x)

        # undo the binning according to the element-to-bin indices
        x = reverse_lsh(bins_split, x)

        return x[:, :n_elems, :]
