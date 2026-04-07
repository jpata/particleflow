import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import sys
import random
import math
import einops
from einops import rearrange

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from mlpf.standalone.dsl import HEPTConfig

# --- HEPT Utilities ---
# Adapted from https://github.com/mova/HEPT
# Paper: "HEPT: Hashed Efficient Particle Transformer", https://arxiv.org/abs/2405.21051
# MIT License
# Copyright (c) 2024 Graph-COM
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Implementation Differences from Official HEPT (HEPT/example/hept.py & HEPT/hept.md):
# 1. Batching & Precision: Supports batched inputs [B, N, D] via flattening. Coordinates
#    and offsets are forced to float32. Note: Large batch sizes (>500) may cause "smearing"
#    of hit coordinates due to float32 precision limits when adding large offsets.
# 2. Query-Key Alignment (Sec 4.3): Implements coordinate-based AND LSH codes via
#    quantile_partition() and get_geo_shift() specifically for HEP 2D eta-phi space.
# 3. Output Projection: HEPTAttention projects to full embedding dim (num_heads * dim_per_head)
#    instead of dim_per_head as seen in some official examples.
# 4. Numerical Stability & Gradient Safety: Core RBF distance and weighted sums are performed
#    in float32. Includes 1e-20 eps and crops outputs back to raw_size before projection
#    to ensure stable training and finite gradients in highly sparse attention patterns.
# 5. Backbone Integration: Uses ELU activations and integrated PositionalEncoding
#    as used in the paper's Tracking experiments.


def uniform(a, b, shape, device="cpu"):
    return (b - a) * torch.rand(shape, device=device) + a


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super(E2LSH, self).__init__()
        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.beta = nn.Parameter(uniform(0, r, shape=(1, n_hashes)))
        self.alpha.requires_grad = False
        self.beta.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1)


def lsh_mapping(e2lsh, queries, keys):
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    max_hash_shift = torch.max(queries_hashed.max(-1, keepdim=True).values, keys_hashed.max(-1, keepdim=True).values)
    min_hash_shift = torch.min(queries_hashed.min(-1, keepdim=True).values, keys_hashed.min(-1, keepdim=True).values)
    hash_shift = max_hash_shift - min_hash_shift
    return queries_hashed, keys_hashed, hash_shift


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)


def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value, s_mask_q=None, s_mask_k=None, return_attn=False):
    # Upcast to float32 for stable RBF distance and weighted sum
    s_query, s_key, s_value = s_query.float(), s_key.float(), s_value.float()
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)
    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)
    if s_mask_k is not None:
        clustered_dists = clustered_dists.masked_fill(s_mask_k.transpose(-1, -2) == 0, float("-inf"))
    clustered_dists = clustered_dists.clamp(max=0.0).exp()
    if s_mask_q is not None:
        clustered_dists = clustered_dists * s_mask_q.float()
    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists
    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    if return_attn:
        return denom, so, qk
    return denom, so


def prep_qk(query, key, w, coords):
    # Ensure coords are float32 for consistent precision
    coords = coords.float()
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]

    # Combine in float32
    q_hat = torch.cat([query.float(), sqrt_w_r], dim=-1)
    k_hat = torch.cat([key.float(), sqrt_w_r], dim=-1)
    return q_hat, k_hat


def pad_to_multiple(tensor, multiple, dims=-1, value=0):
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    dims = [d if d >= 0 else tensor.ndim + d for d in dims]
    padding = [0] * (2 * tensor.ndim)
    for d in dims:
        size = tensor.size(d)
        m = size // multiple
        remainder = size - m * multiple
        if remainder != 0:
            padding[2 * (tensor.ndim - d - 1) + 1] = multiple - remainder
    if all(p == 0 for p in padding):
        return tensor
    return F.pad(tensor, tuple(padding), value=value)

def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(torch.tensor(total_elements, device=sorted_indices.device) / torch.as_tensor(num_regions, device=sorted_indices.device))
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    reassigned_regions = region_indices.gather(-1, inverse_indices.expand(region_indices.shape[0], -1))
    return reassigned_regions

def get_shifts(n_hashes, n_heads):
    return uniform(0, 2*math.pi, (n_hashes*n_heads))[:, None]

def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = []
        for _ in range(num_and_hashes):
            a = random.uniform(lb, ub)
            region.append(a)
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions
    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)


@torch.no_grad()
def get_geo_shift(regions_h, hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices
    q_hash_shift_phi = region_indices_phi * hash_shift
    k_hash_shift_phi = region_indices_phi * hash_shift
    q_hash_shift_eta = region_indices_eta * hash_shift * (torch.ceil(regions_h[1][:, None]) + 1)
    k_hash_shift_eta = region_indices_eta * hash_shift * (torch.ceil(regions_h[1][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.num_heads * self.dim_per_head)
        self.block_size = kwargs.get("block_size", 100)
        self.n_hashes = kwargs.get("n_hashes", 3)
        self.num_w_per_dist = kwargs.get("num_w_per_dist", 10)
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, return_attn=False, **kwargs):
        # query, key, value: [N_padded, num_heads * dim_per_head]
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        raw_size = kwargs["raw_size"]
        padding_mask = kwargs.get("padding_mask", None)

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
        hash_shift = rearrange(hash_shift, "c h d -> (c h) d")

        if padding_mask is not None:
            pm_expanded = padding_mask.view(1, -1, 1)
            q_hat = q_hat.masked_fill(pm_expanded == 0, 0.0)
            k_hat = k_hat.masked_fill(pm_expanded == 0, 0.0)
            value = value.masked_fill(pm_expanded == 0, 0.0)
            q_hashed = q_hashed.masked_fill(padding_mask.view(1, 1, -1) == 0, float("inf"))
            k_hashed = k_hashed.masked_fill(padding_mask.view(1, 1, -1) == 0, float("inf"))
        else:
            q_hat[:, raw_size:] = 0.0
            k_hat[:, raw_size:] = 0.0
            value[:, raw_size:] = 0.0
            q_hashed[..., raw_size:] = float("inf")
            k_hashed[..., raw_size:] = float("inf")

        q_shifts, k_shifts = get_geo_shift(kwargs["regions_h"], hash_shift, kwargs["region_indices"], self.n_hashes)
        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        s_mask_q = None
        s_mask_k = None
        if padding_mask is not None:
            pm_exp = padding_mask.view(1, -1, 1).expand(self.num_heads, -1, 1)
            s_mask_q = sort_to_buckets(pm_exp, q_positions, self.block_size)
            s_mask_k = sort_to_buckets(pm_exp, k_positions, self.block_size)
        else:
            pm = torch.ones(q_hat.shape[1], device=q_hat.device)
            pm[raw_size:] = 0.0
            pm_exp = pm.view(1, -1, 1).expand(self.num_heads, -1, 1)
            s_mask_q = sort_to_buckets(pm_exp, q_positions, self.block_size)
            s_mask_k = sort_to_buckets(pm_exp, k_positions, self.block_size)

        if return_attn:
            denom, so, qk = qkv_res(s_query, s_key, s_value, s_mask_q=s_mask_q, s_mask_k=s_mask_k, return_attn=True)
        else:
            denom, so = qkv_res(s_query, s_key, s_value, s_mask_q=s_mask_q, s_mask_k=s_mask_k)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)

        # Crop back to raw_size to match queries/keys and ensure gradients are safe
        o = o[:, :, :raw_size]
        logits = logits[:, :, :raw_size]

        out = o.sum(dim=0) / (logits.sum(dim=0) + 1e-20)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        if return_attn:
            return out, (qk, q_positions, k_positions)
        return out


# --- Model Definition ---


def get_activation(activation):
    if activation == "elu":
        return nn.ELU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    return nn.ReLU


class SimpleMultiheadAttention(nn.Module):
    """
    A simplified linear-time attention mechanism that pools sequence information into a global context.
    Related to: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", https://arxiv.org/abs/2006.16236

    Notation:
    - X: Input sequence of shape [B, N, D]
    - Q, K, V: Projected query, key, value matrices of shape [B, N, H, D_h] (H=heads, D_h=head_dim)
    - W_alpha, W_beta: Learnable parameter weights of shape [H, D_h]
    - N: Sequence length

    Mechanism:
    1. α = Softmax_N( (Q * W_alpha) / √D_h )  -> shape [B, N, H]
    2. Q_global = Σ_N (Q * α)                -> shape [B, 1, H, D_h]
    3. P = K * Q_global                      -> shape [B, N, H, D_h]
    4. β = Softmax_N( (P * W_beta) / √D_h )   -> shape [B, N, H]
    5. V_global = Σ_N (V * β)                -> shape [B, 1, H, D_h]
    6. Out = Q + V_global                    -> shape [B, N, H, D_h]
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.scale_factor = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_alpha = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.k_beta = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout_p}"

    def forward(self, q, k, v, key_padding_mask=None, return_attn=False):
        bs, n, d = q.size()

        Q = self.q_proj(q).reshape(bs, n, self.num_heads, self.head_dim)
        K = self.k_proj(k).reshape(bs, n, self.num_heads, self.head_dim)
        V = self.v_proj(v).reshape(bs, n, self.num_heads, self.head_dim)

        alpha = (Q * self.q_alpha).sum(dim=-1) * self.scale_factor  # (bs, n, num_heads)
        if key_padding_mask is not None:
            mask_inf = (key_padding_mask == 0).unsqueeze(-1)  # (bs, n, 1)
            alpha = alpha.masked_fill(mask_inf, float("-inf"))
        alpha = torch.softmax(alpha, dim=1)  # (bs, n, num_heads)

        global_q = (Q * alpha.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (bs, 1, num_heads, head_dim)

        P = K * global_q  # (bs, n, num_heads, head_dim)

        beta = (P * self.k_beta).sum(dim=-1) * self.scale_factor  # (bs, n, num_heads)
        if key_padding_mask is not None:
            beta = beta.masked_fill(mask_inf, float("-inf"))
        beta = torch.softmax(beta, dim=1)

        global_v = (V * beta.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (bs, 1, num_heads, head_dim)

        out = Q + global_v  # (bs, n, num_heads, head_dim)
        out = out.reshape(bs, n, d)

        out = self.out_proj(out)
        out = self.dropout(out)
        if return_attn:
            return out, (alpha, beta)
        return out, None


class HEPTAttentionLayer(nn.Module):
    """
    HEPT (Hashed Efficient Particle Transformer) Layer.
    Paper: "HEPT: Hashed Efficient Particle Transformer", https://arxiv.org/abs/2405.21051

    Notation:
    - X: Input sequence of shape [B, N, D]
    - Q, K, V: Projected query, key, value matrices of shape [B, N, H, D_h] (H=heads, D_h=head_dim)
    - W_rpe: Relative positional encoding weight matrix
    - N: Sequence length

    Mechanism (approximate/sparse attention):
    1. LSH (Locality Sensitive Hashing) + Spatial Sorting clusters Q, K based on features and (eta, phi) coords
    2. Sequence N is divided into contiguous buckets of size `block_size`
    3. Q, K are enriched with continuous positional distance embeddings (RBF Kernel) via W_rpe
    4. Distance = exp(-0.5 ||Q_bucket - K_bucket||^2)       -> shape [B, H, Bucket, Bucket]
    5. Attention = Distance / (Sum(Distance) + ε)           -> Softmax-like sparse distribution
    6. Context = Attention * V_bucket
    7. Sequence unsorted back to original permutation
    """

    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, pos=False, **kwargs):
        super(HEPTAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
        self.dropout_p = dropout
        self.pos = pos
        self.dim_per_head = embedding_dim // num_heads

        self.block_size = kwargs.get("block_size", 100)
        self.n_hashes = kwargs.get("n_hashes", 3)
        self.num_regions = kwargs.get("num_regions", 140)
        self.num_w_per_dist = kwargs.get("num_w_per_dist", 10)

        # coords_dim = 2 (eta, phi)
        self.coords_dim = 2
        self.attn = HEPTAttention(self.dim_per_head + self.coords_dim, h_dim=self.dim_per_head, num_heads=num_heads, **kwargs)

        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.w_rpe = nn.Linear(self.num_w_per_dist * (self.coords_dim - 1), self.num_heads * self.dim_per_head)

        self.regions = nn.Parameter(
            get_regions(self.num_regions, self.n_hashes, self.num_heads),
            requires_grad=False,
        )

        self.rand_phi_shifts = nn.Parameter(
                get_shifts(self.n_hashes, self.num_heads),
                requires_grad=False
        )

        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width), nn.ELU(), nn.Linear(width, embedding_dim), nn.ELU())
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = PositionalEncoding(embedding_dim) if pos else None

    def extra_repr(self):
        return (
            f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, width={self.width}, "
            f"dropout={self.dropout_p}, pos={self.pos}, block_size={self.block_size}, "
            f"n_hashes={self.n_hashes}, num_regions={self.num_regions}, num_w_per_dist={self.num_w_per_dist}"
        )

    def forward(self, x, mask, X, return_attn=False):
        # x: [B, N, D]
        # mask: [B, N]
        # X: [B, N, 25] or [B, N, 55] (original features)
        if self.pos_embed:
            x = self.pos_embed(x, X)

        if mask is not None:
            mask_ = mask.unsqueeze(-1)

        B, N, D = x.shape
        device = x.device

        if mask is not None:
            x = x * mask_

        # Extract eta and phi for hashing, explicitly in float32 for safety
        eta = X[..., 2:3].float()
        phi = torch.atan2(X[..., 3:4].float(), X[..., 4:5].float())
        coords = torch.cat([eta, phi], dim=-1)  # [B, N, 2]

        # Flatten batch to support HEPT which expects [N_total, D]
        # Use large float32 offsets to ensure sample isolation
        offsets = torch.arange(B, device=device).view(B, 1, 1).float() * 100.0
        coords = coords + offsets

        x_flat = x.view(B * N, D)
        coords_flat = coords.view(B * N, 2)

        raw_size = B * N
        x_flat_padded = pad_to_multiple(x_flat, self.block_size, dims=0)
        coords_flat_padded = pad_to_multiple(coords_flat, self.block_size, dims=0, value=float("inf"))

        # Precompute regions and indices
        with torch.no_grad():
            coords_for_sort = coords_flat_padded.clone()
            if mask is not None:
                mask_flat = mask.view(-1)
                mask_flat_padded = pad_to_multiple(mask_flat, self.block_size, dims=0, value=0.0)
                coords_for_sort[mask_flat_padded == 0] = float("inf")
            else:
                mask_flat_padded = torch.ones_like(coords_flat_padded[..., 0])
                mask_flat_padded[raw_size:] = 0.0

            offsets = offsets.expand(-1, N, -1).reshape(-1)
            offsets = pad_to_multiple(offsets, self.block_size, dims=0)

            phi_for_sort_shifted = coords_for_sort[..., 1] - offsets
            phi_for_sort_shifted = (phi_for_sort_shifted + math.pi + self.rand_phi_shifts) % (2*math.pi) + offsets[None, :]

            sorted_eta_idx = torch.argsort(coords_for_sort[..., 0], dim=-1)
            sorted_phi_idx = torch.argsort(phi_for_sort_shifted, dim=-1)
            regions_h = rearrange(self.regions, "c a h -> a (c h)")
            region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
            region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])

            region_indices = [region_indices_eta, region_indices_phi]

            coords_flat_padded[mask_flat_padded == 0] = 0.0

        # Run attention
        x_norm = self.norm0(x_flat_padded)
        q, k, v = self.w_q(x_norm), self.w_k(x_norm), self.w_v(x_norm)

        if return_attn:
            attn_out, attn_data = self.attn(
                q,
                k,
                v,
                coords=coords_flat_padded,
                w_rpe=self.w_rpe,
                regions_h=regions_h,
                region_indices=region_indices,
                raw_size=raw_size,
                padding_mask=mask_flat_padded,
                return_attn=True,
            )
        else:
            attn_out = self.attn(
                q,
                k,
                v,
                coords=coords_flat_padded,
                w_rpe=self.w_rpe,
                regions_h=regions_h,
                region_indices=region_indices,
                raw_size=raw_size,
                padding_mask=mask_flat_padded,
            )

        # Reshape to [B, N, D]
        attn_out = attn_out.view(B, N, D)

        # Residual and FFN
        x = x + self.dropout(attn_out)

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out

        if mask is not None:
            x = x * mask_

        if return_attn:
            return x, attn_data
        return x


class GlobalAttentionLayer(nn.Module):
    """
    Wrapper layer for the SimpleMultiheadAttention.
    Uses linear-time additive global attention.

    See `SimpleMultiheadAttention` for the mathematical formulation.
    """

    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, pos=False, **kwargs):
        super(GlobalAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
        self.dropout_p = dropout
        self.pos = pos

        self.mha = SimpleMultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width), nn.GELU(), nn.Linear(width, embedding_dim), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = PositionalEncoding(embedding_dim) if pos else None

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, width={self.width}, dropout={self.dropout_p}, pos={self.pos}"

    def forward(self, x, mask, X=None, return_attn=False):
        if self.pos_embed and X is not None:
            x = self.pos_embed(x, X)

        # x: [B, N, D]
        # mask: [B, N]
        if mask is not None:
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = self.norm0(x)
        if mask is not None:
            x_norm = x_norm * mask_

        mha_out, attn = self.mha(x_norm, x_norm, x_norm, key_padding_mask=mask, return_attn=return_attn)
        x = residual + mha_out

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out
        if mask is not None:
            x = x * mask_
        if return_attn:
            return x, attn
        return x


class StandardAttentionLayer(nn.Module):
    """
    Standard $O(N^2)$ attention using fused kernels (FlashAttention-2 when available).
    Reference: "Attention Is All You Need", https://arxiv.org/abs/1706.03762

    Notation:
    - X: Input sequence of shape [B, N, D]
    - Q, K, V: Projected query, key, value matrices of shape [B, N, H, D_h] (H=heads, D_h=head_dim)
    - W_q, W_k, W_v, W_o: Learnable projection weight matrices
    - N: Sequence length

    Mechanism (per head):
    1. Q, K, V = X * W_q, X * W_k, X * W_v          -> shape [B, N, H, D_h]
    2. Attention = Softmax( (Q * K^T) / √D_h )      -> shape [B, H, N, N]
    3. Context = Attention * V                      -> shape [B, H, N, D_h]
    4. Out = Context * W_o                          -> shape [B, N, D]
    """

    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, pos=False, **kwargs):
        super(StandardAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
        self.dropout_p = dropout
        self.pos = pos
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width), nn.GELU(), nn.Linear(width, embedding_dim), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = PositionalEncoding(embedding_dim) if pos else None

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, width={self.width}, dropout={self.dropout_p}, pos={self.pos}"

    def forward(self, x, mask, X=None, return_attn=False):
        if self.pos_embed and X is not None:
            x = self.pos_embed(x, X)

        B, N, D = x.shape
        if mask is not None:
            # Ensure elements with zero features are also masked
            mask = mask * (x.abs().sum(dim=-1) > 1e-6).float()
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = self.norm0(x)
        if mask is not None:
            x_norm = x_norm * mask_

        q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # If mask is provided, use it as attn_mask for proper padding handling
        attn_mask = mask.bool().unsqueeze(1).unsqueeze(2) if mask is not None else None

        if return_attn:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            mha_out = torch.matmul(attn_weights, v)
        else:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                mha_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        mha_out = mha_out.transpose(1, 2).reshape(B, N, D)
        mha_out = self.out_proj(mha_out)

        x = residual + self.dropout(mha_out)

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out

        if mask is not None:
            x = x * mask_
        if return_attn:
            return x, attn_weights
        return x


class Fastformer(nn.Module):
    """
    Fastformer: Additive Attention is All You Need.
    Paper: https://arxiv.org/abs/2108.09084
    Code adapted from: https://github.com/wilile26811249/Fastformer-PyTorch
    License: MIT

    Notation:
    - X: Input sequence of shape [B, N, D]
    - Q, K, V: Projected query, key, value matrices of shape [B, N, D_decode]
    - W_alpha, W_beta: Learnable parameter weights of shape [D_decode]
    - W_r: Learnable linear projection of shape [D_decode, D_decode]
    - N: Sequence length

    Mechanism (Single Headed context):
    1. α = Softmax_N( (Q * W_alpha) / √D_decode )  -> shape [B, N, D_decode]
    2. Q_global = Σ_N (Q * α)                      -> shape [B, D_decode]
    3. P = K * Q_global                            -> shape [B, N, D_decode]
    4. β = Softmax_N( (P * W_beta) / √D_decode )   -> shape [B, N, D_decode]
    5. K_global = Σ_N (P * β)                      -> shape [B, D_decode]
    6. KV_interaction = K_global * V               -> shape [B, N, D_decode]
    7. Out = W_r(KV_interaction) + Q               -> shape [B, N, D_decode]
    """

    def __init__(self, dim=3, decode_dim=16):
        super(Fastformer, self).__init__()
        self.dim = dim
        self.decode_dim = decode_dim
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim**-0.5

    def extra_repr(self):
        return f"dim={self.dim}, decode_dim={self.decode_dim}"

    def forward(self, x, mask=None, return_attn=False):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        if mask is not None:
            mask = rearrange(mask, "b n -> b n ()").bool()

        # Caculate the global query
        alpha_weight = torch.mul(query, self.weight_alpha) * self.scale_factor
        if mask is not None:
            alpha_weight = alpha_weight.masked_fill(~mask, float("-inf"))
        # Fix: Softmax over sequence dimension (dim=1)
        alpha_weight = torch.softmax(alpha_weight, dim=1)
        global_query = query * alpha_weight
        global_query = torch.einsum("b n d -> b d", global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, "b d -> b copy d", copy=n)
        p = repeat_global_query * key
        beta_weight = torch.mul(p, self.weight_beta) * self.scale_factor
        if mask is not None:
            beta_weight = beta_weight.masked_fill(~mask, float("-inf"))

        # Fix: Softmax over sequence dimension (dim=1)
        beta_weight = torch.softmax(beta_weight, dim=1)
        global_key = p * beta_weight
        global_key = torch.einsum("b n d -> b d", global_key)

        # key-value
        key_value_interaction = torch.einsum("b j, b n j -> b n j", global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        if return_attn:
            return result, (alpha_weight, beta_weight)
        return result


class FastformerAttentionLayer(nn.Module):
    """
    Wrapper layer for the Fastformer mechanism.
    Uses linear-time additive global attention.

    See `Fastformer` for the mathematical formulation.
    """

    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, pos=False, **kwargs):
        super(FastformerAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
        self.dropout_p = dropout
        self.pos = pos

        self.attn = Fastformer(dim=embedding_dim, decode_dim=embedding_dim)
        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width), nn.GELU(), nn.Linear(width, embedding_dim), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = PositionalEncoding(embedding_dim) if pos else None

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, width={self.width}, dropout={self.dropout_p}, pos={self.pos}"

    def forward(self, x, mask, X=None, return_attn=False):
        if self.pos_embed and X is not None:
            x = self.pos_embed(x, X)

        # x: [B, N, D]
        # mask: [B, N]
        if mask is not None:
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = self.norm0(x)
        if mask is not None:
            x_norm = x_norm * mask_

        if return_attn:
            attn_out, attn = self.attn(x_norm, mask=mask, return_attn=True)
        else:
            attn_out = self.attn(x_norm, mask=mask)

        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out
        if mask is not None:
            x = x * mask_
        if return_attn:
            return x, attn
        return x


# Registry of available attention layers
SUPPORTED_ATTENTION_LAYERS = {
    "hept": HEPTAttentionLayer,
    "global": GlobalAttentionLayer,
    "standard": StandardAttentionLayer,
    "fastformer": FastformerAttentionLayer,
}


def ffn(input_dim, output_dim, width, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.GELU(),
        nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.proj = nn.Linear(2, embedding_dim)

    def forward(self, x, X):
        # Extract eta and phi
        eta = X[..., 2:3]
        phi = torch.atan2(X[..., 3:4], X[..., 4:5])
        coords = torch.cat([eta, phi], dim=-1)  # [B, N, 2]
        return x + self.proj(coords)


class RegressionOutput(nn.Module):
    def __init__(self, mode, embed_dim, width, elemtypes, dropout=0.1):
        super(RegressionOutput, self).__init__()
        self.mode = mode  # "direct", "additive", "multiplicative", "linear"
        self.embed_dim = embed_dim
        self.width = width
        self.dropout_p = dropout
        self.elemtypes = elemtypes
        if mode == "linear":
            self.nn = ffn(embed_dim, 2 * len(elemtypes), width, dropout=dropout)
        else:
            self.nn = ffn(embed_dim, len(elemtypes), width, dropout=dropout)

    def extra_repr(self):
        return f"mode={self.mode}, embed_dim={self.embed_dim}, width={self.width}, dropout={self.dropout_p}"

    def forward(self, X, x, orig_value):
        # X: [B, N, 25] or [B, N, 55] (original features)
        # x: [B, N, D] (latent representation)
        # orig_value: [B, N, 1] (the feature to regress from)

        nn_out = self.nn(x)
        elemtype_mask = torch.stack([X[..., 0] == elemtype for elemtype in self.elemtypes], dim=-1)

        if self.mode == "linear":
            # nn_out: [B, N, 2 * len(elemtypes)]
            num_types = len(self.elemtypes)
            a_all = nn_out[..., :num_types]
            b_all = nn_out[..., num_types:]
            a = torch.sum(elemtype_mask * a_all, dim=-1, keepdim=True)
            b = torch.sum(elemtype_mask * b_all, dim=-1, keepdim=True)
            return orig_value * a + b
        else:
            res = torch.sum(elemtype_mask * nn_out, dim=-1, keepdim=True)
            if self.mode == "direct":
                return res
            elif self.mode == "additive":
                return orig_value + res
            elif self.mode == "multiplicative":
                return orig_value * res
            else:
                return res


class MLPF(nn.Module):
    def __init__(self, config=None, **kwargs):
        super(MLPF, self).__init__()

        if config is None:
            # Backward compatibility or manual construction
            from mlpf.standalone.dsl import i, h, g, o, ModelConfig

            input_dim = kwargs.get("input_dim", 55)
            embedding_dim = kwargs.get("embedding_dim", 128)
            width = kwargs.get("width", 128)
            num_convs = kwargs.get("num_convs", 6)
            num_heads = kwargs.get("num_heads", 16)
            attention_type = kwargs.get("attention_type", "global")

            if attention_type == "hept":
                layer = h(num_heads, embedding_dim, width * 4)
            elif attention_type == "standard":
                from mlpf.standalone.dsl import s

                layer = s(num_heads, embedding_dim, width * 4)
            elif attention_type == "fastformer":
                from mlpf.standalone.dsl import f

                layer = f(embedding_dim, width * 4)
            else:
                layer = g(num_heads, embedding_dim, width * 4)

            config = ModelConfig(
                input=i(input_dim, embedding_dim, width * 2), backbone=layer * num_convs, output=o(kwargs.get("num_classes", 8), width * 2)
            )

        self.config = config
        self.elemtypes = [1, 2, 3, 4, 5, 8, 9, 10, 11]

        # 1. Input encoding
        if config.input.type == "default":
            self.nn0 = ffn(config.input.input_dim, config.input.embedding_dim, config.input.width, dropout=config.input.dropout)
            self.type_emb = nn.Embedding(12, config.input.embedding_dim)
        elif config.input.type == "projection_only":
            self.nn0 = nn.Linear(config.input.input_dim, config.input.embedding_dim)
            self.type_emb = None
        else:
            self.nn0 = ffn(config.input.input_dim, config.input.embedding_dim, config.input.width, dropout=config.input.dropout)
            self.type_emb = nn.Embedding(12, config.input.embedding_dim)

        # 2. Backbone
        def build_layers(layer_configs):
            layers = nn.ModuleList()
            for lc in layer_configs:
                layer_cls = SUPPORTED_ATTENTION_LAYERS[lc.type]
                # Map lc to layer_cls kwargs
                kwargs = {
                    "embedding_dim": lc.embedding_dim,
                    "num_heads": lc.num_heads,
                    "width": lc.width,
                    "dropout": lc.dropout,
                    "pos": lc.pos,
                }
                if isinstance(lc, HEPTConfig):
                    kwargs.update(
                        {
                            "block_size": lc.block_size,
                            "n_hashes": lc.n_hashes,
                            "num_regions": lc.num_regions,
                            "num_w_per_dist": lc.num_w_per_dist,
                        }
                    )
                layers.append(layer_cls(**kwargs))
            return layers

        if isinstance(config.backbone, list):
            self.shared_backbone = build_layers(config.backbone)
            self.pid_backbone = None
            self.reg_backbone = None
            self.binary_backbone = None
            self.PU_backbone = None
        else:
            self.shared_backbone = build_layers(config.backbone.get("shared", []))
            self.pid_backbone = build_layers(config.backbone.get("pid", []))
            self.reg_backbone = build_layers(config.backbone.get("reg", []))
            self.binary_backbone = build_layers(config.backbone.get("binary", []))
            self.PU_backbone = build_layers(config.backbone.get("PU", []))

        # 3. Output heads
        self.nn_binary_particle = ffn(
            config.output.embedding_dim if config.output.embedding_dim is not None else config.input.embedding_dim,
            2,
            config.output.width,
            dropout=config.output.dropout,
        )
        self.nn_pid = ffn(
            config.output.embedding_dim if config.output.embedding_dim is not None else config.input.embedding_dim,
            config.output.num_classes,
            config.output.width,
            dropout=config.output.dropout,
        )
        self.nn_PU = ffn(
            config.output.embedding_dim if config.output.embedding_dim is not None else config.input.embedding_dim,
            2,
            config.output.width,
            dropout=config.output.dropout,
        )

        out_dim = config.output.embedding_dim if config.output.embedding_dim is not None else config.input.embedding_dim

        # Get regression modes from config
        rg = config.output.rg_mode
        if isinstance(rg, str):
            modes = {k: rg for k in ["pt", "eta", "sin_phi", "cos_phi", "energy"]}
        else:
            modes = {"pt": "direct", "eta": "additive", "sin_phi": "additive", "cos_phi": "additive", "energy": "direct"}
            modes.update(rg)

        self.nn_pt = RegressionOutput(modes["pt"], out_dim, config.output.width, self.elemtypes, dropout=config.output.dropout)
        self.nn_eta = RegressionOutput(modes["eta"], out_dim, config.output.width, self.elemtypes, dropout=config.output.dropout)
        self.nn_sin_phi = RegressionOutput(modes["sin_phi"], out_dim, config.output.width, self.elemtypes, dropout=config.output.dropout)
        self.nn_cos_phi = RegressionOutput(modes["cos_phi"], out_dim, config.output.width, self.elemtypes, dropout=config.output.dropout)
        self.nn_energy = RegressionOutput(modes["energy"], out_dim, config.output.width, self.elemtypes, dropout=config.output.dropout)

    def forward(self, X, mask, return_attn=False):
        B, N, _ = X.shape

        # Shared input encoding
        if self.type_emb is not None:
            type_idx = X[..., 0].long().clamp(0, 11)
            emb = self.nn0(X) + self.type_emb(type_idx)
        else:
            emb = self.nn0(X)

        # Backbone
        all_attns = []
        for layer in self.shared_backbone:
            if return_attn:
                emb, attn = layer(emb, mask, X, return_attn=True)
                all_attns.append(attn)
            else:
                emb = layer(emb, mask, X)

        emb_shared = emb

        # PID branch
        emb_pid = emb_shared
        if self.pid_backbone is not None:
            for layer in self.pid_backbone:
                emb_pid = layer(emb_pid, mask, X)

        # Binary branch
        emb_binary = emb_shared
        if self.binary_backbone is not None:
            for layer in self.binary_backbone:
                emb_binary = layer(emb_binary, mask, X)

        # Regression branch
        emb_reg = emb_shared
        if self.reg_backbone is not None:
            for layer in self.reg_backbone:
                emb_reg = layer(emb_reg, mask, X)

        # Outputs
        logits_binary = self.nn_binary_particle(emb_binary)

        logits_pid = self.nn_pid(emb_pid)

        preds_pt = self.nn_pt(X, emb_reg, X[..., 1:2])
        preds_energy = self.nn_energy(X, emb_reg, X[..., 5:6])
        preds_eta = self.nn_eta(X, emb_reg, X[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X, emb_reg, X[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X, emb_reg, X[..., 4:5])

        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], dim=-1)

        logits_binary = torch.nan_to_num(logits_binary, nan=0.0, posinf=0.0, neginf=0.0)
        logits_pid = torch.nan_to_num(logits_pid, nan=0.0, posinf=0.0, neginf=0.0)
        preds_momentum = torch.nan_to_num(preds_momentum, nan=0.0, posinf=0.0, neginf=0.0)

        # Neutral PU branch
        emb_PU = emb_shared
        #       modification
        #        real_parts = (torch.argmax(logits_pid, dim=-1) != 0) & (torch.argmax(logits_binary, dim=-1) == 1)
        #        for layer in self.shared_backbone:
        #            emb_PU = layer(emb_PU, mask & real_parts, X)
        #       end
        if self.PU_backbone is not None:
            for layer in self.PU_backbone:
                emb_PU = layer(emb_PU, mask, X)

        logits_PU = self.nn_PU(emb_PU)
        logits_PU = torch.nan_to_num(logits_PU, nan=0.0, posinf=0.0, neginf=0.0)

        if return_attn:
            return logits_binary, logits_pid, logits_PU, preds_momentum, all_attns
        return logits_binary, logits_pid, logits_PU, preds_momentum


# --- Loss Function ---


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        # x: [B, C, N]
        # y: [B, N]
        log_p = F.log_softmax(x, dim=1)
        pt = torch.exp(torch.gather(log_p, 1, y.unsqueeze(1))).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * torch.gather(log_p, 1, y.unsqueeze(1)).squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        return loss


def mlpf_loss(y, ypred, mask, X):
    """
    y: dict with "cls_id", "pt", "eta", "sin_phi", "cos_phi", "energy", "ispu"
    ypred: (logits_binary, logits_pid, logits_ispu, preds_momentum)
    mask: [B, N]
    X: [B, N, 25]
    """
    logits_binary, logits_pid, logits_ispu, preds_momentum = ypred

    # y["cls_id"] is [B, N]
    npart = torch.sum(y["cls_id"] != 0)
    npart_neutral = torch.sum((y["cls_id"] == 4) | (y["cls_id"] == 5))
    nelem = torch.sum(mask)

    # Binary loss
    loss_binary = F.cross_entropy(logits_binary.permute(0, 2, 1), (y["cls_id"] != 0).long(), reduction="none")

    # PU loss
    loss_PU = F.cross_entropy(logits_ispu.permute(0, 2, 1), y["ispu"].long(), reduction="none")

    # PID loss
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")
    loss_pid = loss_obj_id(logits_pid.permute(0, 2, 1), y["cls_id"].long())
    loss_pid[y["cls_id"] == 0] *= 0

    # Regression loss
    pred_pt = preds_momentum[..., 0]
    pred_eta = preds_momentum[..., 1]
    pred_sin_phi = preds_momentum[..., 2]
    pred_cos_phi = preds_momentum[..., 3]
    pred_energy = preds_momentum[..., 4]

    loss_pt = F.mse_loss(pred_pt, y["pt"], reduction="none")
    loss_eta = 1e-2 * F.mse_loss(pred_eta, y["eta"], reduction="none")
    loss_sin_phi = 1e-2 * F.mse_loss(pred_sin_phi, y["sin_phi"], reduction="none")
    loss_cos_phi = 1e-2 * F.mse_loss(pred_cos_phi, y["cos_phi"], reduction="none")
    loss_energy = F.mse_loss(pred_energy, y["energy"], reduction="none")

    # Weight regression loss by target pT
    sqrt_target_pt = torch.sqrt(torch.exp(y["pt"]) * X[:, :, 1])
    loss_pt *= sqrt_target_pt
    loss_energy *= sqrt_target_pt

    # Masking
    for _l in [loss_pt, loss_eta, loss_sin_phi, loss_cos_phi, loss_energy]:
        _l[y["cls_id"] == 0] *= 0
        _l[mask == 0] *= 0

    loss_binary[mask == 0] *= 0
    loss_pid[mask == 0] *= 0

    loss_PU[(mask == 0) | ~((y["cls_id"] == 4) | (y["cls_id"] == 5))] *= 0

    tot_loss = (
        loss_binary.sum() / nelem
        + loss_pid.sum() / nelem
        + loss_pt.sum() / npart
        + loss_eta.sum() / npart
        + loss_sin_phi.sum() / npart
        + loss_cos_phi.sum() / npart
        + loss_energy.sum() / npart
        + loss_PU.sum() / npart_neutral
    )

    return (
        tot_loss,
        loss_binary.sum() / nelem,
        loss_pid.sum() / nelem,
        loss_pt.sum() / npart + loss_eta.sum() / npart + loss_sin_phi.sum() / npart + loss_cos_phi.sum() / npart + loss_energy.sum() / npart,
        loss_PU.sum() / npart_neutral,
    )


# --- Training Loop ---


def train(model, train_loader, optimizer, device, duration_seconds=120, experiment=None):
    model.train()
    start_time = time.time()
    num_steps = 0
    total_loss = 0
    total_loss_binary, total_loss_pid, total_loss_kinematics, total_loss_PU = 0, 0, 0, 0

    print(f"Starting training for {duration_seconds} seconds...")

    while (time.time() - start_time) < duration_seconds:
        for batch in train_loader:
            if (time.time() - start_time) >= duration_seconds:
                break
            X = batch.X.to(device)
            mask = batch.mask.to(device)
            # Prepare targets
            y = {
                "cls_id": batch.ytarget[:, :, 0].to(device),
                "pt": batch.ytarget[:, :, 2].to(device),
                "eta": batch.ytarget[:, :, 3].to(device),
                "sin_phi": batch.ytarget[:, :, 4].to(device),
                "cos_phi": batch.ytarget[:, :, 5].to(device),
                "energy": batch.ytarget[:, :, 6].to(device),
                "ispu": batch.ytarget[:, :, 7].to(device),
            }

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                ypred = model(X, mask)
                loss, loss_binary, loss_pid, loss_kinematics, loss_PU = mlpf_loss(y, ypred, mask, X)
            loss.backward()
            optimizer.step()

            if experiment:
                experiment.log_metric("train_loss", loss.item(), step=num_steps)
                experiment.log_metric("train_loss_binary", loss_binary.item(), step=num_steps)
                experiment.log_metric("train_loss_pid", loss_pid.item(), step=num_steps)
                experiment.log_metric("train_loss_kinematics", loss_kinematics.item(), step=num_steps)
                experiment.log_metric("train_loss_PU", loss_PU.item(), step=num_steps)

            total_loss += loss.item()
            total_loss_binary += loss_binary.item()
            total_loss_pid += loss_pid.item()
            total_loss_kinematics += loss_kinematics.item()
            total_loss_PU += loss_PU.item()
            num_steps += 1

            if num_steps % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {num_steps}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s")

    if num_steps > 0:
        return (
            total_loss / num_steps,
            total_loss_binary / num_steps,
            total_loss_pid / num_steps,
            total_loss_kinematics / num_steps,
            total_loss_PU / num_steps,
            num_steps,
        )

    return 0, num_steps


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_loss_binary, total_loss_pid, total_loss_kinematics, total_loss_PU = 0, 0, 0, 0
    num_steps = 0

    for batch in loader:
        X = batch.X.to(device)
        mask = batch.mask.to(device)

        # Prepare targets
        y = {
            "cls_id": batch.ytarget[:, :, 0].to(device),
            "pt": batch.ytarget[:, :, 2].to(device),
            "eta": batch.ytarget[:, :, 3].to(device),
            "sin_phi": batch.ytarget[:, :, 4].to(device),
            "cos_phi": batch.ytarget[:, :, 5].to(device),
            "energy": batch.ytarget[:, :, 6].to(device),
            "ispu": batch.ytarget[:, :, 7].to(device),
        }

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ypred = model(X, mask)
            loss, loss_binary, loss_pid, loss_kinematics, loss_PU = mlpf_loss(y, ypred, mask, X)

        total_loss += loss.item()
        total_loss_binary += loss_binary.item()
        total_loss_pid += loss_pid.item()
        total_loss_kinematics += loss_kinematics.item()
        total_loss_PU += loss_PU.item()
        num_steps += 1
        # if num_steps > 10:  # Limit validation for speed
        #    break

    if num_steps > 0:
        return (
            total_loss / num_steps,
            total_loss_binary / num_steps,
            total_loss_pid / num_steps,
            total_loss_kinematics / num_steps,
            total_loss_PU / num_steps,
        )

    return 0, num_steps
