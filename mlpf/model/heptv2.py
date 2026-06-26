"""HEPTv2 point encoder layer.

Upstream source:
    https://github.com/Graph-COM/HEPTv2
Paper: https://arxiv.org/abs/2606.20437

Copyright (c) 2024 Graph-COM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Local modifications in this repository:
- Batched `[B, S, D]` handling in `HEPTv2Layer.forward`, including mask-aware
  per-event hashing and flattened batch-offset gather indices.
- CPU fallback for bucket attention when `flex_attention` is unavailable or not
  used, plus optional shape/debug logging.
- Integration with the MLPF stack through the repo-specific feature layout
  (`eta`, `sin_phi`, `cos_phi` -> eta-phi coordinates).
- Several `HEPTV2_*` environment switches are preserved from the upstream
  inference code; this file also supports those switches in the batched path.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
        projection = torch.bmm(vecs, self.alpha)
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
        return torch.einsum("nd,hdc->chn", coords2, e2lsh.alpha)
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
