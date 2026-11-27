"""Experimental indexed Triton attention for the BF16 HEPTv2 release geometry.

Buckets read token-major Q/K/V, and write the output and LSE directly to token
order. The integer routing is a permutation for every hash and head.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

_FORWARD_TILE = 64


@triton.jit
def _bucket_addresses(Perm, Tokens, Heads: tl.constexpr, Block: tl.constexpr, Dim: tl.constexpr, HashOffset):
    bucket = tl.program_id(0)
    head = tl.program_id(1)
    hash_id = tl.program_id(2) + HashOffset
    group = (hash_id * Heads + head).to(tl.int64) * Tokens
    rows = tl.load(Perm + group + bucket * Block + tl.arange(0, Block)).to(tl.int32)
    channels = tl.arange(0, Dim)
    input_elements = (rows[:, None].to(tl.int64) * Heads + head) * Dim + channels[None, :]
    output_rows = group + rows
    output_elements = output_rows[:, None] * Dim + channels[None, :]
    return input_elements, output_elements, output_rows


@triton.jit
def _forward(Q, K, V, Perm, Out, LSE, Tokens, Heads: tl.constexpr, Block: tl.constexpr, Dim: tl.constexpr, Scale: tl.constexpr, Tile: tl.constexpr):
    input_elements, output_elements, output_rows = _bucket_addresses(Perm, Tokens, Heads, Tile, Dim, 0)
    head = tl.program_id(1)
    group = (tl.program_id(2) * Heads + head).to(tl.int64) * Tokens
    bucket_start = tl.program_id(0) // (Block // Tile) * Block
    q = tl.load(Q + input_elements)
    max_score = tl.full((Tile,), -float("inf"), tl.float32)
    total = tl.full((Tile,), 0.0, tl.float32)
    out = tl.full((Tile, Dim), 0.0, tl.float32)

    # Match the dense bucket backend's online 64-key normalization and retain
    # the cast to BF16 before the value dot product in each tile.
    for start in tl.static_range(0, Block, Tile):
        rows = tl.load(Perm + group + bucket_start + start + tl.arange(0, Tile)).to(tl.int64)
        elements = (rows[:, None] * Heads + head) * Dim + tl.arange(0, Dim)[None, :]
        k = tl.load(K + elements)
        v = tl.load(V + elements)
        scores = (tl.dot(q, tl.trans(k)) * Scale) * 1.4426950408889634
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        rescale = tl.exp2(max_score - next_max)
        p = tl.exp2(scores - next_max[:, None])
        total = total * rescale + tl.sum(p, axis=1)
        out = tl.dot(p.to(v.dtype), v, out * rescale[:, None])
        max_score = next_max

    tl.store(Out + output_elements, out / total[:, None])
    tl.store(LSE + output_rows, (max_score + tl.log2(total)) * 0.6931471805599453)


@triton.jit
def _write_grad(ptr, offsets, value, DType: tl.constexpr, Atomic: tl.constexpr, Accumulate: tl.constexpr):
    # Match the materialized path's BF16 rounding before reducing across hashes.
    value = value.to(DType).to(tl.float32)
    if Atomic:
        tl.atomic_add(ptr + offsets, value, sem="relaxed")
    elif Accumulate:
        tl.store(ptr + offsets, tl.load(ptr + offsets) + value)
    else:
        tl.store(ptr + offsets, value)


@triton.jit
def _backward(
    Q,
    K,
    V,
    Perm,
    Out,
    LSE,
    DOut,
    DLSE,
    DQ,
    DK,
    DV,
    Tokens,
    HashOffset,
    Heads: tl.constexpr,
    Block: tl.constexpr,
    Dim: tl.constexpr,
    Scale: tl.constexpr,
    HasOutGrad: tl.constexpr,
    HasLSEGrad: tl.constexpr,
    Atomic: tl.constexpr,
    Accumulate: tl.constexpr,
):
    input_elements, output_elements, output_rows = _bucket_addresses(Perm, Tokens, Heads, Block, Dim, HashOffset)
    q = tl.load(Q + input_elements)
    k = tl.load(K + input_elements)
    lse = tl.load(LSE + output_rows)
    log2_scale = Scale * 1.4426950408889634
    scores = tl.dot(q, tl.trans(k)) * log2_scale
    p = tl.exp2(scores - lse[:, None] * 1.4426950408889634)
    ds = tl.full((Block, Block), 0.0, tl.float32)

    if HasOutGrad:
        v = tl.load(V + input_elements)
        out = tl.load(Out + output_elements)
        dout = tl.load(DOut + output_elements)
        correction = tl.sum(dout.to(tl.float32) * out.to(tl.float32), axis=1)
        ds += tl.dot(dout, tl.trans(v)) - correction[:, None]
        dv = tl.dot(tl.trans(p.to(q.dtype)), dout)
        _write_grad(DV, input_elements, dv, q.dtype, Atomic, Accumulate)

    if HasLSEGrad:
        ds += tl.load(DLSE + output_rows)[:, None]

    # d(logsumexp(s))/ds = softmax(s). Include its upstream gradient
    # because the outer HEPTv2 softmax uses LSE to weight the hashes.
    ds = (ds * p).to(q.dtype)
    dq = tl.dot(ds, k) * Scale
    dk = tl.dot(tl.trans(ds), q) * Scale
    _write_grad(DQ, input_elements, dq, q.dtype, Atomic, Accumulate)
    _write_grad(DK, input_elements, dk, q.dtype, Atomic, Accumulate)


class _IndexedBucketAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, positions, block_size):
        hashes, heads, tokens = positions.shape
        dim = query.shape[-1]
        out = query.new_empty((hashes, heads, tokens, dim))
        lse = query.new_empty((hashes, heads, tokens, 1), dtype=torch.float32)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query, key, value, positions, out, lse)
        ctx.block_size = block_size
        grid = (tokens // _FORWARD_TILE, heads, hashes)
        with torch.cuda.device(query.device):
            _forward[grid](query, key, value, positions, out, lse, tokens, heads, block_size, dim, 1 / math.sqrt(dim), _FORWARD_TILE, num_warps=4)
        return lse, out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_lse, grad_out):
        q, k, v, positions, out, lse = ctx.saved_tensors
        hashes, heads, tokens = positions.shape
        block_size = ctx.block_size
        dim = q.shape[-1]
        grid = (tokens // block_size, heads)
        grad_out_ptr = out if grad_out is None else grad_out.contiguous()
        grad_lse_ptr = lse if grad_lse is None else grad_lse.contiguous()
        deterministic = torch.are_deterministic_algorithms_enabled()
        grads = [
            torch.empty_like(tensor, dtype=torch.float32) if deterministic else torch.zeros_like(tensor, dtype=torch.float32) for tensor in (q, k, v)
        ]
        # A single hash is a permutation, so its buckets have disjoint stores.
        # Process hashes in order when deterministic algorithms are requested.
        launches = [(h, 1) for h in range(hashes)] if deterministic else [(0, hashes)]
        with torch.cuda.device(q.device):
            for start, count in launches:
                _backward[(*grid, count)](
                    q,
                    k,
                    v,
                    positions,
                    out,
                    lse,
                    grad_out_ptr,
                    grad_lse_ptr,
                    *grads,
                    tokens,
                    start,
                    heads,
                    block_size,
                    dim,
                    1 / math.sqrt(dim),
                    grad_out is not None,
                    grad_lse is not None,
                    not deterministic,
                    bool(start),
                    num_warps=8,
                )
        dq, dk, dv = grads
        return dq.to(q.dtype), dk.to(k.dtype), None if grad_out is None else dv.to(v.dtype), None, None


def indexed_bucket_attention(query, key, value, positions, block_size):
    """Return per-hash LSE and attention output in original token order."""
    return _IndexedBucketAttention.apply(query.contiguous(), key.contiguous(), value.contiguous(), positions.contiguous(), block_size)
