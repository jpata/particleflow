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

    # Use torch.where instead of in-place assignment for better ONNX support
    bin_idx = a + torch.where(msk.to(torch.bool), torch.zeros_like(a), (nbins - 1) * torch.ones_like(a))

    # Add a small deterministic offset to the bin indices based on the original node order
    # to stabilize the sort and make it consistent between PyTorch and ONNX
    # without using the 'stable=True' argument which can be problematic for export.
    # We use a tensor-based approach (ones + cumsum) instead of linspace to ensure 
    # the offset generalizes correctly to dynamic axes during ONNX export.
    n_points_active = bin_idx.shape[-1]
    offset = torch.ones(n_points_active, device=bin_idx.device, dtype=cmul.dtype).cumsum(0) * (0.1 / (n_points_active + 1e-6))
    bin_idx_stable = bin_idx.to(cmul.dtype) + offset

    if stable_sort:
        bins_split = torch.argsort(bin_idx_stable, stable=True)
    else:
        # for ONNX export to work, stable must not be provided at all as an argument
        bins_split = torch.argsort(bin_idx_stable)
    bins_split = bins_split.reshape((cmul.shape[0], nbins, bin_size))
    return bins_split


def pairwise_l2_dist(A, B):
    # Ensure computation happens in fp32 to avoid overflow in fp16
    # This is critical for ONNX export with mixed precision
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)

    na = torch.sum(torch.square(A_fp32), -1)
    nb = torch.sum(torch.square(B_fp32), -1)

    # na as a row and nb as a column vectors
    na = torch.unsqueeze(na, -1)
    nb = torch.unsqueeze(nb, -2)

    # return pairwise euclidean difference matrix
    # note that this matrix multiplication can go out of range for float16 in case the absolute values of A and B are large
    dist_sq = na - 2 * torch.matmul(A_fp32, torch.transpose(B_fp32, -1, -2)) + nb
    dist_sq = torch.maximum(dist_sq, torch.tensor(1e-6, device=dist_sq.device, dtype=dist_sq.dtype))
    dist_sq = torch.minimum(dist_sq, torch.tensor(1e6, device=dist_sq.device, dtype=dist_sq.dtype))
    D = torch.sqrt(dist_sq)

    # Cast result back to the original dtype of inputs to maintain consistency
    # This prevents dtype mismatches in downstream operations
    return D.to(A.dtype)


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

        # ensure matrix multiplications are done in fp32 to avoid overflow in fp16
        with torch.autocast(enabled=False, device_type="cuda"):
            x_32 = x.float()
            theta_32 = self.theta.float()
            W_h_32 = self.W_h.float()
            W_t_32 = self.W_t.float()
            b_t_32 = self.b_t.float()
            adj_32 = adj.float()
            msk_32 = msk.float()

            f_hom = torch.linalg.matmul(x_32 * msk_32, theta_32) * msk_32
            if self.normalize_degrees:
                norm_32 = norm.float()
                f_hom = torch.linalg.matmul(adj_32, f_hom * norm_32) * norm_32
            else:
                f_hom = torch.linalg.matmul(adj_32, f_hom)

            f_het = torch.linalg.matmul(x_32 * msk_32, W_h_32)
            gate = torch.sigmoid(torch.linalg.matmul(x_32, W_t_32) + b_t_32)

            out = gate * f_hom + (1.0 - gate) * f_het

        return self.activation(out.to(x.dtype)) * msk


from mlpf.conf import KernelType


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
        # Force FP32 for kernel math to avoid precision loss on GPU
        with torch.autocast(enabled=False, device_type=x_msg_binned.device.type):
            x_32 = x_msg_binned.float() * msk.float()
            dm = torch.unsqueeze(self.dist_norm(x_32, x_32), axis=-1)
            dm = torch.exp(-self.dist_mult * dm)
            dm = torch.maximum(dm, torch.tensor(self.clip_value_low, device=dm.device, dtype=dm.dtype))
            dm = torch.minimum(dm, torch.tensor(1.0, device=dm.device, dtype=dm.dtype))

        return dm.to(x_msg_binned.dtype)


class AttentionKernel(nn.Module):
    def __init__(self, distance_dim, num_heads, **kwargs):
        super(AttentionKernel, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.distance_dim = distance_dim
        self.head_dim = distance_dim // num_heads
        if distance_dim % num_heads != 0:
            raise ValueError(f"distance_dim ({distance_dim}) must be divisible by num_heads ({num_heads})")

        self.W_q = nn.Linear(distance_dim, distance_dim)
        self.W_k = nn.Linear(distance_dim, distance_dim)
        self.scale = self.head_dim**-0.5

    def forward(self, x, msk, training=False):
        # x: (B, n_bins, bin_size, distance_dim)
        # msk: (B, n_bins, bin_size, 1)
        B, n_bins, bin_size, D = x.shape

        # ensure matrix multiplications are done in fp32 to avoid overflow in fp16
        with torch.autocast(enabled=False, device_type="cuda"):
            x_32 = x.float()
            msk_32 = msk.float()

            q = self.W_q(x_32).view(B, n_bins, bin_size, self.num_heads, self.head_dim)
            k = self.W_k(x_32).view(B, n_bins, bin_size, self.num_heads, self.head_dim)

            # (B, n_bins, num_heads, bin_size, head_dim)
            q = q.permute(0, 1, 3, 2, 4)
            k = k.permute(0, 1, 3, 2, 4)

            # scores: (B, n_bins, num_heads, bin_size, bin_size)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            # mask: (B, n_bins, 1, bin_size, 1) -> (B, n_bins, 1, bin_size, bin_size)
            mask = torch.matmul(msk_32, msk_32.transpose(-2, -1)).unsqueeze(2)
            # Use a large negative number instead of -inf for better numerical stability in ONNX
            scores = scores.masked_fill(mask == 0, -1e9)

            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn)

            # Average over heads
            if self.num_heads > 1:
                attn = torch.mean(attn, dim=2)
            else:
                attn = attn.squeeze(2)

            return torch.unsqueeze(attn, axis=-1).to(x.dtype)


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

    # Use torch.scatter instead of a loop for better ONNX export compatibility and performance
    indices = bins_split_flat.unsqueeze(-1).expand(-1, -1, n_features)
    ret = torch.zeros(batch_dim, n_points, n_features, device=points_binned_enc.device, dtype=points_binned_enc.dtype)
    ret.scatter_(1, indices, points_binned_enc_flat)
    return ret


class InterBinAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, msk):
        # x: (B, n_bins, bin_size, D)
        # msk: (B, n_bins, bin_size, 1)

        # Force FP32 for pooling and attention to avoid precision loss on GPU
        with torch.autocast(enabled=False, device_type=x.device.type):
            x_32 = x.float()
            msk_32 = msk.float()

            # Pool nodes in each bin to get bin-level features
            x_bin_sum = torch.sum(x_32 * msk_32, dim=2)  # (B, n_bins, D)
            x_bin_count = torch.sum(msk_32, dim=2)  # (B, n_bins, 1)

            # Avoid division by zero
            x_bin_mean = x_bin_sum / (x_bin_count + 1e-6)  # (B, n_bins, D)

            # Bin-level mask: True for bins with no nodes
            bin_mask = x_bin_count.squeeze(-1) < 1e-6  # (B, n_bins)

            # Multi-head attention between bins
            # Note: Avoid data-dependent control flow (if not bin_mask.all()) for ONNX export
            x_bin_attn, _ = self.mha(x_bin_mean, x_bin_mean, x_bin_mean, key_padding_mask=bin_mask)
            # Residual connection and layer norm
            x_bin_mean = self.layernorm(x_bin_mean + self.dropout(x_bin_attn))
            res = x_32 + x_bin_mean.unsqueeze(2) * msk_32

        return res.to(x.dtype)


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
        # use register_buffer to ensure it's part of the model state but not trainable
        # and it's correctly exported to ONNX
        torch.manual_seed(42)  # ensure deterministic codebook
        self.register_buffer(
            "codebook_random_rotations",
            torch.randn(self.distance_dim, self.max_num_bins // 2),
        )

    def forward(self, x_msg, x_node, msk, training=False):
        shp = x_msg.shape
        # Keep n_points as integer for better ONNX tracing
        n_points = shp[1]

        if n_points % self.bin_size != 0:
            raise Exception("Number of elements per event must be exactly divisible by the bin size")

        # compute the number of LSH bins to divide the input points into on the fly
        # n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = n_points // self.bin_size

        # Use ONNX-friendly operations for codebook slicing
        # We avoid dynamic slicing [:, :codebook_slice] as it can trigger guards in Dynamo.
        # Instead, we multiply with the full codebook and mask the results.
        # perform in FP32 to avoid overflow in FP16 and ensure stable binning on GPU
        with torch.autocast(enabled=False, device_type=x_msg.device.type):
            x_msg_32 = x_msg.float()
            mul = torch.matmul(
                x_msg_32,
                self.codebook_random_rotations.float(),
            )

            # Mask the output to only keep the relevant rotations
            # This is equivalent to slicing the codebook but more tracing-friendly
            n_bins_t = torch.tensor(n_bins, device=x_msg.device)
            codebook_slice = torch.maximum(torch.tensor(1, device=x_msg.device), n_bins_t // 2)
            rotation_idx = torch.arange(self.codebook_random_rotations.shape[1], device=x_msg.device).unsqueeze(0).unsqueeze(0)
            mul = torch.where(rotation_idx < codebook_slice, mul, torch.zeros_like(mul))

            cmul_32 = torch.concatenate([mul, -mul], axis=-1)
            bins_split = split_indices_to_bins_batch(cmul_32, n_bins, self.bin_size, msk, self.stable_sort)

        # replaced tf.gather with torch.vmap, indexing and reshape
        x_msg_binned, x_features_binned, msk_f_binned = split_msk_and_msg(bins_split, cmul_32.to(x_msg.dtype), x_msg, x_node, msk, n_bins, self.bin_size)

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
        self.kernel_type = kwargs.pop("kernel_type", "gaussian")
        self.use_interbin_attention = kwargs.pop("use_interbin_attention", False)
        self.num_interbin_heads = kwargs.pop("num_interbin_heads", 4)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 4)
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

        if self.kernel_type == KernelType.ATTENTION:
            kernel = AttentionKernel(distance_dim=self.distance_dim, num_heads=self.num_attention_heads)
        else:
            kernel = NodePairGaussianKernel()

        self.message_building_layer = MessageBuildingLayerLSH(
            distance_dim=self.distance_dim,
            max_num_bins=self.max_num_bins,
            bin_size=self.bin_size,
            kernel=kernel,
        )

        self.message_passing_layers = nn.ModuleList()
        for iconv in range(self.num_node_messages):
            self.message_passing_layers.append(GHConvDense(output_dim=self.inout_dim, hidden_dim=self.inout_dim, activation="elu"))

        if self.use_interbin_attention:
            self.interbin_attention = InterBinAttentionLayer(self.inout_dim, n_heads=self.num_interbin_heads, dropout=self.dropout)

        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, x, msk, initial_embedding):
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

        if self.use_interbin_attention:
            x = self.interbin_attention(x, msk_f)

        # undo the binning according to the element-to-bin indices
        x = reverse_lsh(bins_split, x)

        return x
