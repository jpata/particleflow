import torch
from torch import nn
import torch.nn.functional as F


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


def split_indices_to_bins_batch(cmul, bin_size, msk, stable_sort=False):
    # Use float32 for binning logic
    cmul_32 = cmul.to(torch.float32)
    a = torch.argmax(cmul_32, axis=-1)

    shp = cmul.shape
    batch_size = shp[0]
    n_points = shp[1]
    
    # Calculate nbins as a tensor
    nbins = torch.div(n_points, bin_size, rounding_mode="floor")
    
    # Assign masked elements to the last bin
    bin_idx = torch.where(msk.to(torch.bool).reshape(batch_size, n_points), a, (nbins - 1))

    # Add a deterministic offset to stabilize the sort
    # In float32, we need a larger offset so it's representable when added to bin indices (up to 200)
    # The epsilon at 200 is approx 2.4e-5. We use an offset in range [0, 0.1]
    n_points_active_t = torch.as_tensor(bin_idx.size(-1), device=bin_idx.device, dtype=torch.float32)
    offset = torch.arange(bin_idx.size(-1), device=bin_idx.device, dtype=torch.float32) * (0.1 / (n_points_active_t + 1.0))
    bin_idx_stable = bin_idx.to(torch.float32) + offset

    # Use torch.sort
    _, bins_split = torch.sort(bin_idx_stable, dim=-1)
    
    # Reshape using -1 for the bins dimension to avoid TopK K-input issues
    bins_split = bins_split.reshape(batch_size, -1, bin_size)
    return bins_split


def pairwise_l2_dist(A, B):
    # Ensure computation happens in fp32
    A_32 = A.to(torch.float32)
    B_32 = B.to(torch.float32)

    na = torch.sum(torch.square(A_32), -1, keepdim=True)
    nb = torch.sum(torch.square(B_32), -1, keepdim=True).transpose(-1, -2)

    dist_sq = na - 2 * torch.matmul(A_32, B_32.transpose(-1, -2)) + nb
    
    dist_sq = torch.clamp(dist_sq, min=1e-6, max=1e6)
    D = torch.sqrt(dist_sq)

    return D.to(A.dtype)


class GHConvDense(nn.Module):
    def __init__(self, *args, **kwargs):
        self.activation = getattr(F, kwargs.pop("activation"))
        self.output_dim = kwargs.pop("output_dim")
        self.normalize_degrees = kwargs.pop("normalize_degrees", False)
        self.hidden_dim = kwargs.pop("hidden_dim")
        super(GHConvDense, self).__init__(*args, **kwargs)

        self.W_t = torch.nn.Parameter(torch.randn(self.output_dim, self.hidden_dim))
        self.b_t = torch.nn.Parameter(torch.randn(self.output_dim))
        self.W_h = torch.nn.Parameter(torch.randn(self.output_dim, self.hidden_dim))
        self.theta = torch.nn.Parameter(torch.randn(self.output_dim, self.hidden_dim))

    def forward(self, inputs):
        x, adj, msk = inputs
        # Force FP32 for consistency
        x_32 = x.to(torch.float32)
        adj_32 = adj.to(torch.float32).squeeze(-1)
        msk_32 = msk.to(torch.float32)
        
        f_hom = F.linear(x_32 * msk_32, self.theta.to(torch.float32)) * msk_32
        
        if self.normalize_degrees:
            in_degrees = torch.sum(torch.abs(adj_32), axis=-1, keepdim=True)
            norm_32 = torch.pow(in_degrees + 1e-6, -0.5) * msk_32
            f_hom = torch.matmul(adj_32, f_hom * norm_32) * norm_32
        else:
            f_hom = torch.matmul(adj_32, f_hom)

        f_het = F.linear(x_32 * msk_32, self.W_h.to(torch.float32))
        gate = torch.sigmoid(F.linear(x_32, self.W_t.to(torch.float32), self.b_t.to(torch.float32)))

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

    def forward(self, x_msg_binned, msk, training=False):
        # Force FP32 for kernel math
        x_32 = x_msg_binned.to(torch.float32) * msk.to(torch.float32)
        dm = torch.unsqueeze(self.dist_norm(x_32, x_32), axis=-1)
        dm = torch.exp(-self.dist_mult * dm)
        dm = torch.clamp(dm, min=self.clip_value_low, max=1.0)

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
        B, n_bins, bin_size, D = x.shape
        x_32 = x.to(torch.float32)
        msk_32 = msk.to(torch.float32)

        q = self.W_q(x_32).view(B, n_bins, bin_size, self.num_heads, self.head_dim)
        k = self.W_k(x_32).view(B, n_bins, bin_size, self.num_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        mask = torch.matmul(msk_32, msk_32.transpose(-2, -1)).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        if self.num_heads > 1:
            attn = torch.mean(attn, dim=2)
        else:
            attn = attn.squeeze(2)

        return torch.unsqueeze(attn, axis=-1).to(x.dtype)


def split_msk_and_msg(bins_split, cmul, x_msg, x_node, msk, bin_size):
    shp = x_msg.shape
    batch_size = shp[0]
    n_points = shp[1]
    
    bins_split_flat = torch.reshape(bins_split, (batch_size, n_points))

    indices_msg = torch.unsqueeze(bins_split_flat, axis=-1).expand(batch_size, n_points, x_msg.shape[-1])
    x_msg_binned = torch.gather(x_msg, 1, indices_msg)
    x_msg_binned = torch.reshape(x_msg_binned, (batch_size, -1, bin_size, x_msg_binned.shape[-1]))

    indices_node = torch.unsqueeze(bins_split_flat, axis=-1).expand(batch_size, n_points, x_node.shape[-1])
    x_features_binned = torch.gather(x_node, 1, indices_node)
    x_features_binned = torch.reshape(x_features_binned, (batch_size, -1, bin_size, x_features_binned.shape[-1]))

    msk_flat = msk.reshape(batch_size, -1)
    msk_f_binned = torch.gather(msk_flat, 1, bins_split_flat)
    msk_f_binned = torch.reshape(msk_f_binned, (batch_size, -1, bin_size, 1))
    
    return x_msg_binned, x_features_binned, msk_f_binned


def reverse_lsh(bins_split, points_binned_enc):
    shp = points_binned_enc.shape
    batch_dim = shp[0]
    n_points = bins_split.numel() // batch_dim
    n_features = shp[-1]

    bins_split_flat = torch.reshape(bins_split, (batch_dim, n_points))
    points_binned_enc_flat = torch.reshape(points_binned_enc, (batch_dim, n_points, n_features))

    indices = bins_split_flat.unsqueeze(-1).expand(-1, -1, n_features)
    ret = torch.zeros(batch_dim, n_points, n_features, device=points_binned_enc.device, dtype=points_binned_enc.dtype)
    ret.scatter_(1, indices, points_binned_enc_flat)
    return ret


class InterBinAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # We'll use the same parameter structure as nn.MultiheadAttention for compatibility
        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mha_dropout_p = dropout
        self.scale = self.head_dim**-0.5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(self, x, msk):
        x_32 = x.to(torch.float32)
        msk_32 = msk.to(torch.float32)

        x_bin_sum = torch.sum(x_32 * msk_32, dim=2)
        x_bin_count = torch.sum(msk_32, dim=2)
        x_bin_mean = x_bin_sum / (x_bin_count + 1e-6)

        bs, seq_len, embed_dim = x_bin_mean.size()
        head_dim = self.head_dim
        num_heads = self.n_heads

        # split stacked in_proj_weight, in_proj_bias to q, k, v matrices
        # this matches nn.MultiheadAttention parameter layout
        wq, wk, wv = torch.split(self.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0)
        bq, bk, bv = torch.split(self.in_proj_bias, [embed_dim, embed_dim, embed_dim], dim=0)

        q = torch.matmul(x_bin_mean, wq.T) + bq
        k = torch.matmul(x_bin_mean, wk.T) + bk
        v = torch.matmul(x_bin_mean, wv.T) + bv

        # Use reshape with symbolic seq_len from size()
        q = q.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)

        bin_mask = x_bin_count.squeeze(-1) < 1e-6
        # Prepare attention mask for F.scaled_dot_product_attention (True = keep, False = ignore)
        attn_mask = ~bin_mask.view(bs, 1, 1, seq_len)

        # F.scaled_dot_product_attention is more tracing-friendly than nn.MultiheadAttention
        x_bin_attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.mha_dropout_p if self.training else 0.0
        )

        x_bin_attn = x_bin_attn.transpose(1, 2).reshape(bs, seq_len, embed_dim)
        x_bin_attn = self.out_proj(x_bin_attn)

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

        self.register_buffer(
            "codebook_random_rotations",
            torch.randn(self.distance_dim, self.max_num_bins // 2),
        )

    def forward(self, x_msg, x_node, msk, training=False):
        shp = x_msg.shape
        batch_size = shp[0]
        n_points = shp[1]
        
        # perform in FP32
        x_msg_32 = x_msg.to(torch.float32)

        mul = torch.matmul(
            x_msg_32,
            self.codebook_random_rotations.to(torch.float32),
        )
        
        n_rotations = self.codebook_random_rotations.shape[1]
        rotation_idx = torch.arange(n_rotations, device=mul.device).unsqueeze(0).unsqueeze(0)
        
        # Calculate n_bins purely as a tensor
        n_points_t = torch.as_tensor(x_msg.size(1), device=x_msg.device)
        n_bins_t = torch.div(n_points_t, self.bin_size, rounding_mode="floor")
        codebook_slice_t = torch.div(n_bins_t, 2, rounding_mode="floor")
        
        # Mask out unused rotations
        mul = torch.where(rotation_idx < codebook_slice_t, mul, torch.full_like(mul, -1e9))

        cmul_32 = torch.concatenate([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul_32, self.bin_size, msk, self.stable_sort)

        x_msg_binned, x_features_binned, msk_f_binned = split_msk_and_msg(bins_split, cmul_32, x_msg, x_node, msk, self.bin_size)

        dm = self.kernel(x_msg_binned, msk_f_binned, training=training)

        shp_dm = dm.shape
        msk_row = msk_f_binned.reshape(shp_dm[0], shp_dm[1], shp_dm[2], 1, 1).to(dm.dtype)
        msk_col = msk_f_binned.reshape(shp_dm[0], shp_dm[1], 1, shp_dm[3], 1).to(dm.dtype)
        dm = dm * msk_row * msk_col

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

        x_dist = self.dist_activation(self.ffn_dist(x))

        bins_split, x, dm, msk_f = self.message_building_layer(x_dist, x, msk)

        for msg in self.message_passing_layers:
            x_out = msg((x, dm, msk_f))
            x = x_out
            if self.dropout_layer:
                x = self.dropout_layer(x)

        if self.use_interbin_attention:
            x = self.interbin_attention(x, msk_f)

        x = reverse_lsh(bins_split, x)

        return x
