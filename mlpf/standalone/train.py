import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import os
import sys
import math
import einops
from einops import rearrange

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from mlpf.standalone.dsl import HEPTConfig

# --- HEPTv2 Implementation ---
# Copied from mlpf/model/heptv2.py.
# Upstream source:
#     https://github.com/Graph-COM/HEPTv2
# Paper: https://arxiv.org/abs/2606.20437
#
# Copyright (c) 2024 Graph-COM
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
try:
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention

    # The eager flex_attention kernel prints a warning that it "may produce
    # incorrect results" — wrap with torch.compile to get the fused path.
    flex_attention = torch.compile(_raw_flex_attention, dynamic=False)
except ImportError:
    flex_attention = None


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


# Defaults favor stable hashing and equivalent lower-overhead tensor paths.
# Each switch can still be overridden with HEPTV2_*={0,1,true,false,...}.
_HEPTV2_DEFAULTS = {
    "HEPTV2_ENCODER_HASH_FP32": True,
    "HEPTV2_COMBINED_BUCKET_GATHER": True,
    "HEPTV2_COMBINED_UNSORT": True,
    "HEPTV2_MANUAL_HASH_SOFTMAX": False,
    "HEPTV2_DIRECT_BUCKET_GATHER": True,
    "HEPTV2_E2LSH_COORDS_EINSUM": True,
    "HEPTV2_DEBUG_SHAPES": False,
}

_HEPTV2_ENCODER_HASH_FP32 = _env_bool("HEPTV2_ENCODER_HASH_FP32", _HEPTV2_DEFAULTS["HEPTV2_ENCODER_HASH_FP32"])
_HEPTV2_COMBINED_BUCKET_GATHER = _env_bool("HEPTV2_COMBINED_BUCKET_GATHER", _HEPTV2_DEFAULTS["HEPTV2_COMBINED_BUCKET_GATHER"])
_HEPTV2_COMBINED_UNSORT = _env_bool("HEPTV2_COMBINED_UNSORT", _HEPTV2_DEFAULTS["HEPTV2_COMBINED_UNSORT"])
_HEPTV2_MANUAL_HASH_SOFTMAX = _env_bool("HEPTV2_MANUAL_HASH_SOFTMAX", _HEPTV2_DEFAULTS["HEPTV2_MANUAL_HASH_SOFTMAX"])
_HEPTV2_DIRECT_BUCKET_GATHER = _env_bool("HEPTV2_DIRECT_BUCKET_GATHER", _HEPTV2_DEFAULTS["HEPTV2_DIRECT_BUCKET_GATHER"])
_HEPTV2_E2LSH_COORDS_EINSUM = _env_bool("HEPTV2_E2LSH_COORDS_EINSUM", _HEPTV2_DEFAULTS["HEPTV2_E2LSH_COORDS_EINSUM"])
_HEPTV2_DEBUG_SHAPES = _env_bool("HEPTV2_DEBUG_SHAPES", _HEPTV2_DEFAULTS["HEPTV2_DEBUG_SHAPES"])


def _debug_shape(name, tensor):
    if _HEPTV2_DEBUG_SHAPES:
        if isinstance(tensor, torch.Tensor):
            print(f"[HEPTv2 Shape Debug] {name}: {list(tensor.shape)} (dtype: {tensor.dtype})")
        elif isinstance(tensor, (list, tuple)):
            shapes = [list(t.shape) if isinstance(t, torch.Tensor) else type(t) for t in tensor]
            print(f"[HEPTv2 Shape Debug] {name}: {shapes}")


def _env_int_limit(name, default, maximum):
    raw = os.environ.get(name, "")
    if not raw:
        return int(default)
    value = int(raw)
    if value <= 0:
        return int(default)
    return max(1, min(int(value), int(maximum)))


def _env_int_layer_limit(name, layer_idx, default, maximum):
    raw = os.environ.get(name, "")
    if raw and layer_idx is not None:
        values = [part.strip() for part in raw.split(",")]
        if int(layer_idx) < len(values) and values[int(layer_idx)]:
            value = int(values[int(layer_idx)])
            if value > 0:
                return max(1, min(value, int(maximum)))
    return _env_int_limit("HEPTV2_ENCODER_NUM_HASHES", default, maximum)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act_fn=F.silu):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.beta = nn.Parameter((r - 0) * torch.rand((1, n_hashes)) + 0)
        self.alpha.requires_grad = False
        self.beta.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha.to(vecs.dtype))
        return projection.permute(2, 0, 1)


def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1

    if sorted_indices.dim() == 2:
        region_indices = region_indices.unsqueeze(1).expand(-1, sorted_indices.shape[0], -1)
        inverse_indices_expanded = inverse_indices.unsqueeze(0).expand(region_indices.shape[0], -1, -1)
        return torch.gather(region_indices, -1, inverse_indices_expanded)
    return region_indices[:, inverse_indices]


def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = [torch.rand(1).item() * (ub - lb) + lb for _ in range(num_and_hashes)]
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions
    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    if os.environ.get("HEPTV2_SCATTER_INVERT_PERM", "0").lower() in {"1", "true", "yes", "on"}:
        arange = torch.arange(perm.shape[-1], device=perm.device, dtype=perm.dtype).expand_as(perm)
        return torch.zeros_like(perm).scatter(-1, perm, arange)
    if os.environ.get("HEPTV2_FAST_INVERT_PERM", "0").lower() in {"1", "true", "yes", "on"}:
        arange = torch.arange(perm.shape[-1], device=perm.device, dtype=perm.dtype).expand_as(perm)
        return torch.empty_like(perm).scatter_(-1, perm, arange)
    return torch.argsort(perm, dim=-1)


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)


def sort_to_buckets(x, perm, bucketsz):
    if _HEPTV2_DIRECT_BUCKET_GATHER:
        num_hashes = int(perm.shape[0])
        num_heads = int(perm.shape[1])
        seq_len = int(perm.shape[2])
        head_dim = int(x.shape[-1])
        gathered = (
            x.unsqueeze(0)
            .expand(num_hashes, -1, -1, -1)
            .gather(
                2,
                perm[..., None].expand(num_hashes, num_heads, seq_len, head_dim),
            )
        )
        return gathered.reshape(num_hashes, num_heads, seq_len // bucketsz, bucketsz, head_dim)
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def sort_qkv_to_buckets(q, k, v, perm, bucketsz):
    if not _HEPTV2_COMBINED_BUCKET_GATHER:
        return (
            sort_to_buckets(q, perm, bucketsz),
            sort_to_buckets(k, perm, bucketsz),
            sort_to_buckets(v, perm, bucketsz),
        )
    qkv = torch.cat((q, k, v), dim=-1)
    s_qkv = sort_to_buckets(qkv, perm, bucketsz)
    return s_qkv.split(q.shape[-1], dim=-1)


def unsort_from_buckets(s_x, perm_inverse):
    if _HEPTV2_DIRECT_BUCKET_GATHER:
        flat = s_x.reshape(s_x.shape[0], s_x.shape[1], -1, s_x.shape[-1])
        return flat.gather(2, perm_inverse[..., None].expand(*perm_inverse.shape, s_x.shape[-1]))
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value):
    if flex_attention is not None and s_query.is_cuda:
        try:
            t_query = rearrange(s_query, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
            t_key = rearrange(s_key, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
            t_value = rearrange(s_value, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
            out, lse = flex_attention(t_query, t_key, t_value, return_lse=True)
            out = rearrange(out, "(c h) nbuckets b d -> c h nbuckets b d", h=s_query.shape[1])
            lse = rearrange(lse, "(c h) nbuckets b -> c h nbuckets b 1", h=s_query.shape[1])
            return lse, out
        except Exception:
            pass

    # Fallback scaled dot-product attention
    # s_query shape: [c, h, nbuckets, bucketsz, d]
    # s_key shape: [c, h, nbuckets, bucketsz, d]
    d = s_query.shape[-1]
    scores = torch.matmul(s_query, s_key.transpose(-1, -2)) / math.sqrt(d)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, s_value)
    return lse, out


def lsh_coords(e2lsh, coords2, num_heads):
    if _HEPTV2_E2LSH_COORDS_EINSUM:
        return torch.einsum("nd,hdc->chn", coords2, e2lsh.alpha.to(coords2.dtype))
    pos = coords2.repeat(num_heads, 1, 1)
    return e2lsh(pos)


@torch.no_grad()
def get_geo_shift_single(regions_h, hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices
    hash_shift_eta = region_indices_eta * hash_shift
    if region_indices_eta.dim() == 3:
        hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None, None]) + 1)
        return rearrange(hash_shift_phi + hash_shift_eta, "(c h) b n -> c h b n", c=num_or_hashes)
    else:
        hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
        return rearrange(hash_shift_phi + hash_shift_eta, "(c h) n -> c h n", c=num_or_hashes)


class HEPTv2Attention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.num_heads = kwargs["num_heads"]
        self.dim_per_head = kwargs["head_dim"]
        self.model_dim = kwargs.get("model_dim", self.num_heads * self.dim_per_head)
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.model_dim)

        self.block_size = kwargs.get("block_size", 100)
        self.n_hashes = kwargs.get("n_hashes", 3)
        self.num_w_per_dist = kwargs.get("num_w_per_dist", 10)
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)
        self.e2lsh_new = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=2)

    def forward(self, query, key, value, **kwargs):
        _debug_shape("HEPTv2Attention in query", query)
        _debug_shape("HEPTv2Attention in key", key)
        _debug_shape("HEPTv2Attention in value", value)

        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        q_hat = rearrange(query, "n h d -> h n d")
        k_hat = rearrange(key, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        valid_mask = kwargs.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(q_hat.shape[1], dtype=torch.bool, device=q_hat.device)
        else:
            valid_mask = valid_mask.bool()
        coords2 = kwargs["coords"][..., :2]
        if coords2.dim() == 3:
            B, S, _ = coords2.shape
            coords2_flat = coords2.view(B * S, 2)
            is_batched = True
        else:
            coords2_flat = coords2
            is_batched = False

        invalid = ~valid_mask
        active_hashes = _env_int_layer_limit(
            "HEPTV2_ENCODER_NUM_HASHES_BY_LAYER",
            kwargs.get("_encoder_layer_idx"),
            self.n_hashes,
            self.n_hashes,
        )
        active_width = int(active_hashes) * int(self.num_heads)

        # Hashing / projection
        if _HEPTV2_ENCODER_HASH_FP32:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                coords2_f = coords2_flat.float()
                hashed_flat = lsh_coords(self.e2lsh_new, coords2_f, self.num_heads)
                if active_hashes < self.n_hashes:
                    hashed_flat = hashed_flat[:active_hashes]

                if is_batched:
                    hashed = hashed_flat.view(active_hashes, self.num_heads, B, S)
                    max_v = hashed.max(-1, keepdim=True).values
                    min_v = hashed.min(-1, keepdim=True).values
                    hash_shift = max_v - min_v
                    hash_shift = rearrange(hash_shift, "c h b d -> (c h) b d")
                    invalid_expanded = invalid.view(1, 1, B, S).expand(active_hashes, self.num_heads, -1, -1)
                    hashed[invalid_expanded] = float("inf")
                else:
                    hashed = hashed_flat
                    max_v = hashed.max(-1, keepdim=True).values
                    min_v = hashed.min(-1, keepdim=True).values
                    hash_shift = max_v - min_v
                    hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
                    hashed[..., invalid] = float("inf")

                regions_h = kwargs["regions_h"].float()[:, :active_width]
                region_indices = [idx[:active_width] for idx in kwargs["region_indices"]]
                shifts = get_geo_shift_single(regions_h, hash_shift, region_indices, active_hashes)

                hashed = hashed + shifts
                positions = hashed.argsort(dim=-1)
        else:
            hashed_flat = lsh_coords(self.e2lsh_new, coords2_flat, self.num_heads)
            if active_hashes < self.n_hashes:
                hashed_flat = hashed_flat[:active_hashes]

            if is_batched:
                hashed = hashed_flat.view(active_hashes, self.num_heads, B, S)
                max_v = hashed.max(-1, keepdim=True).values
                min_v = hashed.min(-1, keepdim=True).values
                hash_shift = max_v - min_v
                hash_shift = rearrange(hash_shift, "c h b d -> (c h) b d")
                invalid_expanded = invalid.view(1, 1, B, S).expand(active_hashes, self.num_heads, -1, -1)
                hashed[invalid_expanded] = float("inf")
            else:
                hashed = hashed_flat
                max_v = hashed.max(-1, keepdim=True).values
                min_v = hashed.min(-1, keepdim=True).values
                hash_shift = max_v - min_v
                hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
                hashed[..., invalid] = float("inf")

            regions_h = kwargs["regions_h"][:, :active_width]
            region_indices = [idx[:active_width] for idx in kwargs["region_indices"]]
            shifts = get_geo_shift_single(regions_h, hash_shift, region_indices, active_hashes)

            hashed = hashed + shifts
            positions = hashed.argsort(dim=-1)

        if is_batched:
            batch_offsets = torch.arange(B, device=positions.device).view(1, 1, B, 1) * S
            positions = (positions + batch_offsets).view(active_hashes, self.num_heads, B * S)

        _debug_shape("HEPTv2Attention hash positions", positions)
        s_query, s_key, s_value = sort_qkv_to_buckets(q_hat, k_hat, value, positions, self.block_size)
        _debug_shape("HEPTv2Attention buckets s_query", s_query)

        denom, so = qkv_res(s_query, s_key, s_value)
        q_rev_positions = invert_permutation(positions)

        if _HEPTV2_COMBINED_UNSORT:
            combined = torch.cat([so, denom], dim=-1)
            combined = unsort_from_buckets(combined, q_rev_positions)
            o = combined[..., : self.dim_per_head]
            logits = combined[..., self.dim_per_head :]
        else:
            o = unsort_from_buckets(so, q_rev_positions)
            logits = unsort_from_buckets(denom, q_rev_positions)

        if _HEPTV2_MANUAL_HASH_SOFTMAX:
            weights = torch.exp(logits - logits.max(dim=0, keepdim=True).values)
            out = torch.sum(o * weights, dim=0) / torch.sum(weights, dim=0)
        else:
            probs = torch.softmax(logits, dim=0)
            out = torch.sum(o * probs, dim=0)

        final_out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        _debug_shape("HEPTv2Attention final_out", final_out)
        return final_out


class PELearned(nn.Module):
    def __init__(self, input_channel, h_dim):
        super().__init__()
        num_pos_feats = h_dim * 4
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.SiLU(),
            nn.Linear(num_pos_feats, h_dim),
        )

    def forward(self, xyz):
        return self.position_embedding_head(xyz)


class HEPTv2Layer(nn.Module):
    def __init__(self, name=None, embedding_dim=128, num_heads=16, width=512, dropout=0.1, pe_type="learned", **kwargs):
        super().__init__()
        self.name = name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
        self.dropout_p = dropout
        self.dim_per_head = embedding_dim // num_heads

        self.block_size = kwargs.get("block_size", 100)
        self.n_hashes = kwargs.get("n_hashes", 3)
        self.num_regions = kwargs.get("num_regions", 140)
        self.num_w_per_dist = kwargs.get("num_w_per_dist", 10)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)

        # coords_dim = 2 (eta, phi)
        self.coords_dim = 2
        self.attn = HEPTv2Attention(
            self.dim_per_head + self.coords_dim,
            head_dim=self.dim_per_head,
            model_dim=self.embedding_dim,
            num_heads=num_heads,
            **kwargs,
        )

        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.w_rpe = nn.Linear(self.num_w_per_dist * (self.coords_dim - 1), self.num_heads * self.dim_per_head)

        self.pe_func = PELearned(input_channel=self.coords_dim, h_dim=embedding_dim) if pe_type == "learned" else None

        self.regions = nn.Parameter(
            get_regions(self.num_regions, self.n_hashes, self.num_heads),
            requires_grad=False,
        )

        self.norm1 = Qwen3RMSNorm(embedding_dim)
        self.norm2 = Qwen3RMSNorm(embedding_dim)
        self.q_norm = Qwen3RMSNorm(self.dim_per_head)
        self.k_norm = Qwen3RMSNorm(self.dim_per_head)

        mlp_hidden_dim = max(1, int(embedding_dim * self.mlp_ratio))
        self.ff = Qwen3MLP(embedding_dim, mlp_hidden_dim, act_fn=F.silu)

        self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        return (
            f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, width={self.width}, "
            f"dropout={self.dropout_p}, block_size={self.block_size}, "
            f"n_hashes={self.n_hashes}, num_regions={self.num_regions}, num_w_per_dist={self.num_w_per_dist}"
        )

    def forward(self, x, mask, X_features, return_attn=False):
        _debug_shape("HEPTv2Layer in x", x)
        if mask is not None:
            _debug_shape("HEPTv2Layer in mask", mask)
            mask_ = mask.unsqueeze(-1)

        B, S, D = x.shape
        device = x.device

        if mask is not None:
            x = x * mask_

        # Extract eta and phi for hashing, explicitly in float32 for safety
        eta = X_features[..., 2:3].float()
        phi = torch.atan2(X_features[..., 3:4].float(), X_features[..., 4:5].float())
        coords = torch.cat([eta, phi], dim=-1)  # [B, S, 2]

        x_flat = x.view(B * S, D)
        coords_flat = coords.view(B * S, 2)

        raw_size = B * S

        # Hashing preparation
        with torch.no_grad():
            coords_for_sort = coords.clone()
            if mask is not None:
                coords_for_sort = torch.where(
                    mask.unsqueeze(-1) == 0,
                    torch.tensor(float("inf"), device=device),
                    coords_for_sort,
                )
                valid_mask = mask

            else:
                valid_mask = torch.ones_like(coords[..., 0], dtype=torch.bool)

            sorted_eta_idx = torch.argsort(coords_for_sort[..., 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords_for_sort[..., 1], dim=-1)
            regions_h = rearrange(self.regions, "c a h -> a (c h)")
            region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
            region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
            region_indices = [region_indices_eta, region_indices_phi]

        # Positional embedding
        if self.pe_func is not None:
            pe_flat = self.pe_func(coords_flat)
            x_pe = x_flat + pe_flat
        else:
            pe_flat = None
            x_pe = x_flat

        x_normed = self.norm1(x_pe)
        q = self.w_q(x_normed)
        k = self.w_k(x_normed)
        v = self.w_v(x_normed)

        # Head normalization
        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)

        q = self.q_norm(q).view(-1, self.num_heads * self.dim_per_head)
        k = self.k_norm(k).view(-1, self.num_heads * self.dim_per_head)
        v = v.view(-1, self.num_heads * self.dim_per_head)

        # Call attention
        attn_out = self.attn(
            q,
            k,
            v,
            coords=coords,
            w_rpe=self.w_rpe,
            regions_h=regions_h,
            region_indices=region_indices,
            raw_size=raw_size,
            valid_mask=valid_mask,
        )

        attn_out = attn_out.view(B, S, D)

        # Residual + MLP (FNN)
        x_flat_pe = x_flat + self.dropout(attn_out.view(B * S, D))
        ff_out = self.ff(self.norm2(x_flat_pe))
        x_out = x_flat_pe + self.dropout(ff_out)

        x_out = x_out.view(B, S, D)

        if mask is not None:
            x_out = x_out * mask_

        return x_out


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


# Backward-compatible standalone name; implementation is HEPTv2.
class HEPTAttentionLayer(HEPTv2Layer):
    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, pos=False, **kwargs):
        super().__init__(embedding_dim=embedding_dim, num_heads=num_heads, width=width, dropout=dropout, **kwargs)

    def forward(self, x, mask, X_features, return_attn=False):
        B, S, _ = x.shape
        pad_len = 0
        while (B * (S + pad_len)) % self.block_size != 0:
            pad_len += 1

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            X_features = F.pad(X_features, (0, 0, 0, pad_len), value=0.0)
            if mask is None:
                mask = torch.ones(B, S, dtype=x.dtype, device=x.device)
            mask = F.pad(mask, (0, pad_len), value=0.0)

        out = super().forward(x, mask, X_features, return_attn=False)
        if pad_len:
            out = out[:, :S]

        if return_attn:
            return out, None
        return out


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
            from mlpf.standalone.dsl import i, h, g, s, f, o, ModelConfig

            input_dim = kwargs.get("input_dim", 55)
            embedding_dim = kwargs.get("embedding_dim", 128)
            width = kwargs.get("width", 128)
            num_convs = kwargs.get("num_convs", 6)
            num_heads = kwargs.get("num_heads", 16)
            attention_type = kwargs.get("attention_type", "global")

            if attention_type == "hept":
                layer = h(num_heads, embedding_dim, width * 4)
            elif attention_type == "standard":
                layer = s(num_heads, embedding_dim, width * 4)
            elif attention_type == "fastformer":
                layer = f(num_heads, embedding_dim, width * 4)
            else:
                layer = g(num_heads, embedding_dim, width * 4)

            config = ModelConfig(
                input=i(input_dim, embedding_dim, width * 2),
                backbone={"shared": layer * num_convs},
                output=o(kwargs.get("num_classes", 8), width * 2),
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
