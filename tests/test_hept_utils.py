import torch
from mlpf.standalone.train import (
    quantile_partition,
    invert_permutation,
    batched_index_select,
    sort_to_buckets,
    unsort_from_buckets,
    pad_to_multiple,
    E2LSH,
    lsh_mapping,
    get_regions,
    get_geo_shift,
    prep_qk,
    qkv_res,
)


def test_quantile_partition():
    # Test if quantile_partition divides elements into regions correctly
    num_regions = 4
    # Create sorted_indices (e.g., from argsort)
    # For 12 elements, region_size = ceil(12/4) = 3
    sorted_indices = torch.arange(12)  # [12] (1D)

    regions = quantile_partition(sorted_indices, num_regions)
    # 0,1,2 -> 1
    # 3,4,5 -> 2
    # 6,7,8 -> 3
    # 9,10,11 -> 4
    expected = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]])
    assert torch.equal(regions, expected)

    # Test with non-divisible size
    # For 10 elements, region_size = ceil(10/4) = 3
    sorted_indices_10 = torch.arange(10)
    regions_10 = quantile_partition(sorted_indices_10, num_regions)
    # 0,1,2 -> 1
    # 3,4,5 -> 2
    # 6,7,8 -> 3
    # 9 -> 4
    expected_10 = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3, 4]])
    assert torch.equal(regions_10, expected_10)


def test_invert_permutation():
    perm = torch.randperm(10).unsqueeze(0)
    inv_perm = invert_permutation(perm)

    # Check if perm[inv_perm] is arange
    arange = torch.arange(10).unsqueeze(0)
    result = torch.gather(perm, -1, inv_perm)
    assert torch.equal(result, arange)

    # Check if inv_perm[perm] is arange
    result2 = torch.gather(inv_perm, -1, perm)
    assert torch.equal(result2, arange)


def test_batched_index_select():
    values = torch.randn(2, 5, 10, 4)  # [h, b, s, d]
    indices = torch.randint(0, 10, (2, 5, 10))  # [h, b, s]
    selected = batched_index_select(values, indices)

    assert selected.shape == (2, 5, 10, 4)
    # Check a random element
    h, b, s = 1, 3, 7
    idx = indices[h, b, s]
    assert torch.equal(selected[h, b, s], values[h, b, idx])


def test_sort_unsort_buckets():
    h, b, s, d = 2, 3, 16, 8
    bucketsz = 4
    x = torch.randn(b, s, d)
    perm = torch.stack([torch.randperm(s) for _ in range(h * b)]).view(h, b, s)

    s_x = sort_to_buckets(x, perm, bucketsz)
    assert s_x.shape == (h, b, s // bucketsz, bucketsz, d)

    perm_inverse = invert_permutation(perm)
    unsorted_x = unsort_from_buckets(s_x, perm_inverse)

    assert unsorted_x.shape == (h, b, s, d)
    # Check if each head's unsorted x matches the original x
    for i in range(h):
        assert torch.allclose(unsorted_x[i], x, atol=1e-6)


def test_pad_to_multiple():
    x = torch.randn(10, 5)
    padded = pad_to_multiple(x, 4, dims=0)
    assert padded.shape == (12, 5)
    assert torch.equal(padded[:10], x)
    assert torch.all(padded[10:] == 0)

    y = torch.randn(10, 5)
    padded_y = pad_to_multiple(y, 4, dims=1)
    assert padded_y.shape == (10, 8)
    assert torch.equal(padded_y[:, :5], y)

    z = torch.randn(10, 5)
    padded_z = pad_to_multiple(z, 4, dims=[0, 1])
    assert padded_z.shape == (12, 8)


def test_e2lsh_mapping():
    n_hashes = 3
    n_heads = 2
    dim = 8
    seq_len = 16
    e2lsh = E2LSH(n_hashes, n_heads, dim)

    queries = torch.randn(n_heads, seq_len, dim)
    keys = torch.randn(n_heads, seq_len, dim)

    q_hashed, k_hashed, hash_shift = lsh_mapping(e2lsh, queries, keys)

    assert q_hashed.shape == (n_hashes, n_heads, seq_len)
    assert k_hashed.shape == (n_hashes, n_heads, seq_len)
    assert hash_shift.shape == (n_hashes, n_heads, 1)

    # Check hash_shift property
    max_val = torch.max(q_hashed.max(-1, keepdim=True).values, k_hashed.max(-1, keepdim=True).values)
    min_val = torch.min(q_hashed.min(-1, keepdim=True).values, k_hashed.min(-1, keepdim=True).values)
    assert torch.allclose(hash_shift, max_val - min_val)


def test_lsh_property():
    # Close points should have closer hash values
    n_hashes = 10
    n_heads = 1
    dim = 64
    e2lsh = E2LSH(n_hashes, n_heads, dim)

    x1 = torch.randn(n_heads, 1, dim)
    x2 = x1 + torch.randn(n_heads, 1, dim) * 0.01  # very close
    x3 = torch.randn(n_heads, 1, dim) * 10.0  # very far

    h1 = e2lsh(x1)  # [n_hashes, n_heads, 1]
    h2 = e2lsh(x2)
    h3 = e2lsh(x3)

    dist_12 = (h1 - h2).abs().mean()
    dist_13 = (h1 - h3).abs().mean()

    assert dist_12 < dist_13


def test_get_regions():
    num_regions = 140
    num_or_hashes = 3
    num_heads = 4
    regions = get_regions(num_regions, num_or_hashes, num_heads)

    # Expected shape: [num_or_hashes, num_and_hashes=2, num_heads]
    assert regions.shape == (num_or_hashes, 2, num_heads)

    # Check if products of regions approximate num_regions
    prod = regions[:, 0, :] * regions[:, 1, :]
    assert torch.allclose(prod, torch.tensor(float(num_regions)), rtol=0.1)


def test_get_geo_shift():
    num_or_hashes = 2
    num_heads = 2
    num_regions = 100
    seq_len = 10

    regions = get_regions(num_regions, num_or_hashes, num_heads)
    # regions_h: [2, num_or_hashes * num_heads]
    from einops import rearrange

    regions_h = rearrange(regions, "c a h -> a (c h)")

    hash_shift = torch.ones(num_or_hashes * num_heads, 1)

    # region_indices: [num_regions_h, num_or_hashes * num_heads, seq_len]
    # In train.py it's a list of 2 tensors
    region_indices_eta = torch.randint(1, 10, (num_or_hashes * num_heads, seq_len))
    region_indices_phi = torch.randint(1, 10, (num_or_hashes * num_heads, seq_len))
    region_indices = [region_indices_eta, region_indices_phi]

    res = get_geo_shift(regions_h, hash_shift, region_indices, num_or_hashes)

    # Expected shape: [2 (q/k), num_or_hashes, num_heads, seq_len]
    assert res.shape == (2, num_or_hashes, num_heads, seq_len)


def test_prep_qk():
    # In HEPTAttentionLayer, prep_qk is called with coords_flat_padded: [N_padded, 2]
    # query/key/value are also flattened to [N_padded, num_heads, dim_per_head]
    num_heads = 2
    dim_per_head = 8
    seq_len = 16
    num_w_per_dist = 10

    query = torch.randn(seq_len, num_heads, dim_per_head)
    key = torch.randn(seq_len, num_heads, dim_per_head)
    # w: [num_heads, dim_per_head, coords_dim-1=1, num_w_per_dist]
    w = torch.randn(num_heads, dim_per_head, 1, num_w_per_dist)
    coords = torch.randn(seq_len, 2)  # [N_total, 2]

    q_hat, k_hat = prep_qk(query, key, w, coords)

    # coords_dim = 2, so sqrt_w_r has dim 2 (one per coordinate)
    # prep_qk appends these 2 dims to each head.
    # expected dim = dim_per_head + 2
    assert q_hat.shape == (seq_len, num_heads, dim_per_head + 2)
    assert k_hat.shape == (seq_len, num_heads, dim_per_head + 2)


def test_qkv_res():
    h, b, nbuckets, bucketsz, d = 2, 1, 4, 8, 16
    s_query = torch.randn(h, b, nbuckets, bucketsz, d)
    s_key = torch.randn(h, b, nbuckets, bucketsz, d)
    s_value = torch.randn(h, b, nbuckets, bucketsz, d)

    denom, so = qkv_res(s_query, s_key, s_value)

    assert denom.shape == (h, b, nbuckets, bucketsz, 1)
    assert so.shape == (h, b, nbuckets, bucketsz, d)
    assert torch.all(denom > 0)


def test_batch_offset_logic():
    # Test the logic used in HEPTAttentionLayer.forward for batch flattening
    B, N = 2, 10
    eta = torch.randn(B, N, 1)
    phi = torch.randn(B, N, 1)
    coords = torch.cat([eta, phi], dim=-1)  # [B, N, 2]

    offsets = torch.arange(B).view(B, 1, 1) * 100.0
    coords_offset = coords + offsets

    # Flatten
    coords_flat = coords_offset.view(B * N, 2)

    # Samples in batch 0 should be far from samples in batch 1 in coordinate space
    batch0 = coords_flat[:N]
    batch1 = coords_flat[N:]

    dist = torch.cdist(batch0, batch1)
    assert torch.all(dist > 50.0)  # Should be around 100.0


def test_large_batch_isolation():
    # Simulate a large batch to test floating point precision limits in float32
    # Magnitude limit for 0.01 resolution: X < 0.005 * 2^23 = 41943.
    # With 100.0 offset per batch, B_max ~ 419.

    # 1. Safe Batch Index (B=100)
    B_safe = 100
    offset_safe = B_safe * 100.0
    coords_safe = torch.tensor([[1.0, 1.0], [1.01, 1.0]]).float()
    isolated_safe = coords_safe + offset_safe
    diff_safe = (isolated_safe[0] - isolated_safe[1]).abs().sum()

    assert diff_safe > 0.0
    # Should be close to 0.01
    assert torch.allclose(diff_safe, torch.tensor(0.01), atol=1e-3)

    # 2. Unsafe Batch Index (B=1000)
    # At B=1000, offset is 100,000.
    # float32 eps at 100,000 is ~0.0078 (2^-23 * 2^17)
    B_unsafe = 1000
    offset_unsafe = B_unsafe * 100.0
    coords_unsafe = torch.tensor([[1.0, 1.0], [1.01, 1.0]]).float()
    isolated_unsafe = coords_unsafe + offset_unsafe
    diff_unsafe = (isolated_unsafe[0] - isolated_unsafe[1]).abs().sum()

    # This demonstrates the 'smearing' failure:
    # 100001.01 in float32 might round to something that makes the diff != 0.01
    # or even 0.0 if the hits are close enough.
    precision_error = (diff_unsafe - 0.01).abs()
    assert precision_error > 1e-3, f"Float32 surprisingly accurate at B=1000: error={precision_error}"

    # 3. Isolation check still works because massive distance is still massive
    dist_sq = torch.sum((isolated_safe[0] - coords_unsafe[0]) ** 2)
    kernel_val = torch.exp(-0.5 * dist_sq)
    assert kernel_val == 0.0
