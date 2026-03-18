import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import numpy as np
import argparse
import sys
import random
from torch.utils.data import DataLoader
from einops import rearrange

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import library modules
try:
    from mlpf.model.PFDataset import Collater, PFDataset
    from mlpf.logger import _configLogger
except ImportError:
    # Fallback for cases where the library is not in the path
    pass

# --- HEPT Utilities ---


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


def qkv_res(s_query, s_key, s_value):
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)
    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()
    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists
    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
    
    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
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
    region_size = torch.ceil(torch.tensor(total_elements / num_regions, device=sorted_indices.device))
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    reassigned_regions = region_indices[:, inverse_indices]
    return reassigned_regions


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
    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift
    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.num_heads * self.dim_per_head)
        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, **kwargs):
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
        q_hat[:, raw_size:] = 0.0
        k_hat[:, raw_size:] = 0.0
        value[:, raw_size:] = 0.0

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
        hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
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

        denom, so = qkv_res(s_query, s_key, s_value)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        out = o.sum(dim=0) / (logits.sum(dim=0) + 1e-20)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
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
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.q_alpha = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.k_beta = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        bs, n, d = q.size()
        
        Q = self.q_proj(q).reshape(bs, n, self.num_heads, self.head_dim)
        K = self.k_proj(k).reshape(bs, n, self.num_heads, self.head_dim)
        V = self.v_proj(v).reshape(bs, n, self.num_heads, self.head_dim)

        alpha = (Q * self.q_alpha).sum(dim=-1) # (bs, n, num_heads)
        if key_padding_mask is not None:
            mask_inf = (key_padding_mask == 0).unsqueeze(-1) # (bs, n, 1)
            alpha = alpha.masked_fill(mask_inf, float('-inf'))
        alpha = torch.softmax(alpha, dim=1) # (bs, n, num_heads)
        
        global_q = (Q * alpha.unsqueeze(-1)).sum(dim=1, keepdim=True) # (bs, 1, num_heads, head_dim)
        
        P = K * global_q # (bs, n, num_heads, head_dim)
        
        beta = (P * self.k_beta).sum(dim=-1) # (bs, n, num_heads)
        if key_padding_mask is not None:
            beta = beta.masked_fill(mask_inf, float('-inf'))
        beta = torch.softmax(beta, dim=1)
        
        global_v = (V * beta.unsqueeze(-1)).sum(dim=1, keepdim=True) # (bs, 1, num_heads, head_dim)
        
        out = Q + global_v # (bs, n, num_heads, head_dim)
        out = out.reshape(bs, n, d)
        
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, None


class HEPTAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, width, dropout=0.1, **kwargs):
        super(HEPTAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.width = width
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

        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width), nn.ELU(), nn.Linear(width, embedding_dim), nn.ELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, X):
        # x: [B, N, D]
        # mask: [B, N]
        # X: [B, N, 25] or [B, N, 55] (original features)

        B, N, D = x.shape
        device = x.device

        # Extract eta and phi for hashing
        eta = X[..., 2:3]
        phi = torch.atan2(X[..., 3:4], X[..., 4:5])
        coords = torch.cat([eta, phi], dim=-1)  # [B, N, 2]

        # Flatten batch to support HEPT which expects [N_total, D]
        offsets = torch.arange(B, device=device).view(B, 1, 1) * 100.0
        coords = coords + offsets

        x_flat = x.view(B * N, D)
        coords_flat = coords.view(B * N, 2)
        
        raw_size = B * N
        x_flat_padded = pad_to_multiple(x_flat, self.block_size, dims=0)
        coords_flat_padded = pad_to_multiple(coords_flat, self.block_size, dims=0, value=float("inf"))

        # Precompute regions and indices
        with torch.no_grad():
            sorted_eta_idx = torch.argsort(coords_flat_padded[..., 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords_flat_padded[..., 1], dim=-1)
            regions_h = rearrange(self.regions, "c a h -> a (c h)")
            region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
            region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
            region_indices = [region_indices_eta, region_indices_phi]

            coords_flat_padded[raw_size:] = 0.0

        # Run attention
        x_norm = self.norm0(x_flat_padded)
        q, k, v = self.w_q(x_norm), self.w_k(x_norm), self.w_v(x_norm)

        attn_out = self.attn(
            q,
            k,
            v,
            coords=coords_flat_padded,
            w_rpe=self.w_rpe,
            regions_h=regions_h,
            region_indices=region_indices,
            raw_size=raw_size,
        )

        # Crop back to raw_size and reshape to [B, N, D]
        attn_out = attn_out[:raw_size].view(B, N, D)

        # Residual and FFN
        x = x + self.dropout(attn_out)

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


class PreLnSelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, width, dropout=0.1):
        super(PreLnSelfAttentionLayer, self).__init__()
        self.mha = SimpleMultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(nn.Linear(embedding_dim, width * 4), nn.GELU(), nn.Linear(width * 4, embedding_dim), nn.GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, N]
        if mask is not None:
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = self.norm0(x)

        q = x_norm
        if mask is not None:
            q = q * mask_

        mha_out, _ = self.mha(q, x_norm, x_norm, key_padding_mask=mask)
        x = residual + mha_out

        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        x = residual + ffn_out
        if mask is not None:
            x = x * mask_
        return x


def ffn(input_dim, output_dim, width, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.GELU(),
        nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class RegressionOutput(nn.Module):
    def __init__(self, embed_dim, width, elemtypes):
        super(RegressionOutput, self).__init__()
        self.elemtypes = elemtypes
        self.nn = ffn(embed_dim, len(elemtypes), width)

    def forward(self, X, x):
        # X: [B, N, 25] (original features)
        # x: [B, N, D] (latent representation)

        nn_out = self.nn(x)  # [B, N, num_elemtypes]

        # Create mask for each element type
        elemtype_mask = torch.stack([X[..., 0] == elemtype for elemtype in self.elemtypes], dim=-1)  # [B, N, num_elemtypes]

        # Select the output corresponding to the element type
        res = torch.sum(elemtype_mask * nn_out, dim=-1, keepdim=True)

        return res


class MLPF(nn.Module):
    def __init__(self, input_dim=25, num_classes=8, embedding_dim=128, width=128, num_convs=6, num_heads=16, use_hept=False, hept_kwargs=None):
        super(MLPF, self).__init__()

        self.use_hept = use_hept
        self.elemtypes = [1, 2, 3, 4, 5, 8, 9, 10, 11]
        num_types = len(self.elemtypes)

        # Input encoding
        self.nn0 = ffn(input_dim, embedding_dim, width * 2)
        self.type_emb = nn.Embedding(12, embedding_dim)

        # Attention layers
        if use_hept:
            if hept_kwargs is None:
                hept_kwargs = {
                    "block_size": 100,
                    "n_hashes": 3,
                    "num_regions": 140,
                    "num_w_per_dist": 10,
                }
            self.conv = nn.ModuleList([HEPTAttentionLayer(embedding_dim, num_heads, width*4, **hept_kwargs) for _ in range(num_convs)])
        else:
            self.conv = nn.ModuleList([PreLnSelfAttentionLayer(embedding_dim, num_heads, width*4) for _ in range(num_convs)])

        # Final output heads
        self.nn_binary_particle = ffn(embedding_dim, 2, width * 2)
        self.nn_pid = ffn(embedding_dim, num_classes, width * 2)

        self.nn_pt = RegressionOutput(embedding_dim, width * 2, self.elemtypes)
        self.nn_eta = RegressionOutput(embedding_dim, width * 2, self.elemtypes)
        self.nn_sin_phi = RegressionOutput(embedding_dim, width * 2, self.elemtypes)
        self.nn_cos_phi = RegressionOutput(embedding_dim, width * 2, self.elemtypes)
        self.nn_energy = RegressionOutput(embedding_dim, width * 2, self.elemtypes)

    def forward(self, X, mask):
        B, N, _ = X.shape

        # Shared input encoding
        type_idx = X[..., 0].long().clamp(0, 11)
        emb = self.nn0(X) + self.type_emb(type_idx)

        # Attention layers
        for conv in self.conv:
            if self.use_hept:
                emb = conv(emb, mask, X)
            else:
                emb = conv(emb, mask)

        # Outputs
        logits_binary = self.nn_binary_particle(emb)
        logits_pid = self.nn_pid(emb)

        preds_pt = self.nn_pt(X, emb)
        preds_energy = self.nn_energy(X, emb)
        preds_eta = X[..., 2:3] + self.nn_eta(X, emb)
        preds_sin_phi = X[..., 3:4] + self.nn_sin_phi(X, emb)
        preds_cos_phi = X[..., 4:5] + self.nn_cos_phi(X, emb)

        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], dim=-1)

        logits_binary = torch.nan_to_num(logits_binary, nan=0.0, posinf=0.0, neginf=0.0)
        logits_pid = torch.nan_to_num(logits_pid, nan=0.0, posinf=0.0, neginf=0.0)
        preds_momentum = torch.nan_to_num(preds_momentum, nan=0.0, posinf=0.0, neginf=0.0)

        return logits_binary, logits_pid, preds_momentum


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
    y: dict with "cls_id", "pt", "eta", "sin_phi", "cos_phi", "energy"
    ypred: (logits_binary, logits_pid, preds_momentum)
    mask: [B, N]
    X: [B, N, 25]
    """
    logits_binary, logits_pid, preds_momentum = ypred

    # y["cls_id"] is [B, N]
    npart = torch.sum(y["cls_id"] != 0)
    nelem = torch.sum(mask)

    # Binary loss
    loss_binary = F.cross_entropy(logits_binary.permute(0, 2, 1), (y["cls_id"] != 0).long(), reduction="none")

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

    tot_loss = (
        loss_binary.sum() / nelem
        + loss_pid.sum() / nelem
        + loss_pt.sum() / npart
        + loss_eta.sum() / npart
        + loss_sin_phi.sum() / npart
        + loss_cos_phi.sum() / npart
        + loss_energy.sum() / npart
    )

    return tot_loss


# --- Training Loop ---


def train(model, train_loader, optimizer, device, duration_seconds=120):
    model.train()
    start_time = time.time()
    num_steps = 0
    total_loss = 0

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
            }

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                ypred = model(X, mask)
                loss = mlpf_loss(y, ypred, mask, X)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            if num_steps % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {num_steps}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s")

    return total_loss / num_steps if num_steps > 0 else 0, num_steps
