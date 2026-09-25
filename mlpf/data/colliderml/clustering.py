# Deterministic, truth-blind spatial clusterer for ColliderML release 1.
#
# Region codes are the six OpenDataDetector calorimeter subsystems stored in the
# `detector` field of the calo_hits table:
#   9  ECAL endcap -      (z in -5440..-3200 mm, r in ~300..1500)
#   10 ECAL barrel        (|z| <= 3200,      r in ~1300..1500)
#   11 ECAL endcap +      (z in 3200..5440, r in ~300..1500)
#   12 HCAL endcap -
#   13 HCAL barrel
#   14 HCAL endcap +
#
# Two algorithms, selected by `algorithm=` on `cluster_event`:
#
#   union_find (default): connected components over the radius graph, with an optional
#     energy-asymmetry edge gate (merge_frac) — no seeding, no splitting at valleys. Each
#     connected blob of linked hits is one cluster. Pandora-like in that a shower is one
#     energy-energy block; Pandora does more with seeds, this is a pure geometric clustering.
#
#   bfs (repository's original): seeded multi-source BFS watershed per region. A strict
#     local energy maximum is a cluster seed; every hit is assigned to the seed that reaches
#     it first by hop count, with ties broken deterministically by seed priority then stable
#     rank. Unlinked components (no seed above them) become singletons. Pandora's step 1
#     topological clustering shape: seeding handles the "valley between maxima" case
#     directly.
#
# Properties enforced by both implementations:
#   * deterministic per-event: identical inputs always give the same output
#   * truth-blind: contrib_particle_ids/energies/times never read
#   * region-crossing hits allowed via a single tree over all hits + radius max per pair
#     (ECAL cells can join HCAL cells when a shower crosses the detector boundary)
#   * energy-conserving: every retained hit belongs to exactly one cluster
from typing import Dict, List, Tuple

import numba
import numpy as np
from scipy.spatial import cKDTree

from mlpf.data.target_building import SparseMatrixCOO

# NOTE: radii are shared between union_find and bfs_merge; merge_frac differs by algorithm.
# The radii may benefit from tuning
DEFAULT_REGION_RADII_MM = {
    9: 25.0,  # ECAL endcap - (5.1 mm cells -> ~5 cells wide)
    10: 25.0,  # ECAL barrel
    11: 25.0,  # ECAL endcap +
    12: 90.0,  # HCAL endcap - (30 mm cells -> 3 cells wide)
    13: 90.0,  # HCAL barrel
    14: 90.0,  # HCAL endcap +
}
DEFAULT_MERGE_FRAC = 0.25


@numba.njit
def _build_csr_adjacency(src: np.ndarray, dst: np.ndarray, deg: np.ndarray, n_hit: int):
    """Counting-sort fill of CSR adjacency; within-group order = original edge order,
    content-identical to `dst[np.argsort(src, kind="stable")]` (deterministic, O(n))."""
    indptr = np.empty(n_hit + 1, np.int64)
    indptr[0] = 0
    for i in range(n_hit):
        indptr[i + 1] = indptr[i] + deg[i]
    fill = indptr[:-1].copy()
    adj = np.empty(len(dst), np.int64)
    for k in range(len(dst)):
        adj[fill[src[k]]] = dst[k]
        fill[src[k]] += 1
    return adj, indptr


@numba.njit
def _wavefront_bfs(seed_idx: np.ndarray, adj: np.ndarray, indptr: np.ndarray, hops: np.ndarray, seed_of: np.ndarray) -> None:
    """Multi-source BFS by hop count; within a hop a hit takes the minimum-priority seed that
    reaches it (priority = seed order by stable rank). Imperative version of the per-hop
    lexsort/dedupe numpy wavefront, with identical hop/seed assignments."""
    n = hops.shape[0]
    hop_inf = n + 2
    hops[seed_idx] = 0
    for k in range(len(seed_idx)):
        seed_of[seed_idx[k]] = k
    cand_min = np.empty(n, dtype=np.int64)  # per-candidate min seed-priority this hop
    cand_min[:] = hop_inf
    frontier = seed_idx.copy()
    frontier_prio = np.arange(len(seed_idx), dtype=np.int64)
    hop = 0
    while len(frontier) > 0:
        cand_min[:] = hop_inf
        for f in range(len(frontier)):
            node = frontier[f]
            prio = frontier_prio[f]
            for e in range(indptr[node], indptr[node + 1]):
                c = adj[e]
                if hops[c] == hop_inf and prio < cand_min[c]:
                    cand_min[c] = prio
        m = cand_min != hop_inf
        cand = np.nonzero(m)[0]
        if len(cand) == 0:
            break
        hops[cand] = hop + 1
        seed_of[cand] = cand_min[cand]
        frontier = cand
        frontier_prio = cand_min[cand]
        hop += 1


def _stable_rank(E: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # stable per-event sort rank: (-E, x, y, z, original index)
    order = np.lexsort((np.arange(len(x)), z, y, x, -E))
    stable_rank = np.empty(len(x), dtype=np.int64)
    stable_rank[order] = np.arange(len(x))
    return stable_rank


def _pair_graph(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pi, pj): hit indexes of linked pairs under the max(r_i, r_j) rule.

    Per detector-region pair (a, b), a dedicated query at radius max(r_a, r_b) yields exactly
    the pairs surviving the max(r_i, r_j) test. Querying a single tree at the global maximum
    radius instead would generate and store a much larger candidate set first, which does not
    fit in memory for high-multiplicity (pileup) events.
    """
    pts = np.column_stack([x, y, z])
    det = np.asarray(detector)
    regions = np.unique(det)
    index_of = {int(r): np.where(det == r)[0] for r in regions}
    trees = {r: cKDTree(pts[idx]) for r, idx in index_of.items()}
    # per-region coordinate bounds, for skipping provably-empty cross-region queries
    bounds = {r: (pts[idx].min(axis=0), pts[idx].max(axis=0)) for r, idx in index_of.items()}

    pis, pjs = [], []
    for ai, a in enumerate(regions):
        a = int(a)
        ra = float(radii_mm[a])
        ia = index_of[a]
        # scipy queries are inclusive ("at most r"); the max(r_i, r_j) rule is a strict <,
        # so drop boundary hits at exactly the radius (rare, but real for grid-aligned cells)
        segments = []
        # same-region pairs: exact test is dist < r_a
        pairs = trees[a].query_pairs(ra, output_type="ndarray")
        if pairs.size:
            segments.append((ia[pairs[:, 0]], ia[pairs[:, 1]], ra))
        # cross-region pairs (a, b): exact test is dist < max(r_a, r_b)
        for b in (int(r) for r in regions[ai + 1 :]):
            rab = max(ra, float(radii_mm[b]))
            ib = index_of[b]
            # exact skip: if the regions' coordinate bounding boxes are rab-disjoint, no pair
            # can lie within rab (ODD's regions are radially well-separated, so this skips
            # most ball-tree walks, which returned ~0 pairs anyway)
            (amin, amax), (bmin, bmax) = bounds[a], bounds[b]
            if np.any(bmin > amax + rab) or np.any(amin > bmax + rab):
                continue
            for li, ljs in enumerate(trees[a].query_ball_tree(trees[b], rab)):
                if ljs:
                    segments.append((np.full(len(ljs), ia[li], dtype=np.int64), ib[np.asarray(ljs, dtype=np.int64)], rab))
        for gi, gj, rmax in segments:
            keep = np.linalg.norm(pts[gi] - pts[gj], axis=1) < rmax
            if keep.any():
                pis.append(gi[keep])
                pjs.append(gj[keep])

    if not pis:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(pis), np.concatenate(pjs)


def _cluster_event_union_find(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float],
    merge_frac: float,
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """Connected components of the radius graph, with an optional edge gate.

    Each link (i, j) is actually activated only if `E_small <= merge_frac * E_big`, where
    E_small = min(E_i, E_j) and E_big is the maximum *hit* energy of the larger hit's current
    cluster. The gate is the "merge small into large" rule of a post-hoc Pandora-like merge,
    baked into union-find at the edge level.
    """
    n_hit = len(x)
    pi, pj = _pair_graph(x, y, z, detector, radii_mm)

    # deterministic pair order: sort by (stable_rank pi, stable_rank pj) within each
    # (pi, pj) so we process pairs in a stable order
    stable_rank = _stable_rank(E, x, y, z)
    pair_order = np.lexsort((stable_rank[pj], stable_rank[pi]))
    pi = pi[pair_order]
    pj = pj[pair_order]

    if len(pi) == 0:
        # no links at all -> every hit is its own cluster
        cluster_of = np.arange(n_hit, dtype=np.int64)
        return cluster_of, _build_features(x, y, z, E, detector, cluster_of), _make_hit_to_cluster(cluster_of), _cluster_region(cluster_of, detector)

    parent = np.arange(n_hit, dtype=np.int64)
    comp_max_E = E.astype(np.float64).copy()

    def _find(v: int) -> int:
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root

    for a, b in zip(pi.tolist(), pj.tolist()):
        ra, rb = _find(int(a)), _find(int(b))
        if ra == rb:
            continue
        if merge_frac > 0.0:
            E_a, E_b = E[a], E[b]
            Ebig_a, Ebig_b = comp_max_E[ra], comp_max_E[rb]
            E_small, E_big = (E_a, Ebig_b) if E_a <= E_b else (E_b, Ebig_a)
            if E_big > 0.0 and E_small > merge_frac * E_big:
                continue  # too balanced -> leave as two clusters
        if stable_rank[ra] < stable_rank[rb]:
            parent[rb] = ra
            if comp_max_E[rb] > comp_max_E[ra]:
                comp_max_E[ra] = comp_max_E[rb]
        else:
            parent[ra] = rb
            if comp_max_E[ra] > comp_max_E[rb]:
                comp_max_E[rb] = comp_max_E[ra]

    labels = np.array([_find(v) for v in range(n_hit)], dtype=np.int64)
    # map root label -> dense cluster id; order clusters by best (smallest) stable rank
    root_to_id = {}
    cluster_of = np.empty(n_hit, dtype=np.int64)
    for v in range(n_hit):
        root = int(labels[v])
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_of[v] = root_to_id[root]
    n_clusters = len(root_to_id)
    comp_best_rank = np.full(n_clusters, n_hit + 10, dtype=np.int64)
    np.minimum.at(comp_best_rank, cluster_of, stable_rank)
    order_cluster = np.argsort(comp_best_rank, kind="stable")
    cluster_relabel = np.empty(n_clusters, dtype=np.int64)
    cluster_relabel[order_cluster] = np.arange(n_clusters, dtype=np.int64)
    cluster_of = cluster_relabel[cluster_of]
    return cluster_of, _build_features(x, y, z, E, detector, cluster_of), _make_hit_to_cluster(cluster_of), _cluster_region(cluster_of, detector)


def _cluster_event_bfs(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float],
    merge_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """Multi-source BFS with local-max seeds + leftover components, one pass per event.

    Like Pandora's initial topological clustering: a strict local energy maximum is a seed,
    and every hit it claims is assigned to the nearest seed by hop count; ties are broken by
    seed priority then by stable rank. Unreached components (no seed above them) become
    singleton clusters.

    With `merge_frac > 0`, a Pandora-style fragment merge runs afterwards: for every pair of
    clusters with a hit link between them whose small hit satisfies
    `E_small <= merge_frac × E_large_cluster_total`, the smaller cluster is absorbed into
    the larger. This recovers shower fragments the tight seeding split at local maxima
    without reclaiming a second comparably-loaded shower.
    """
    n_hit = len(x)
    pi, pj = _pair_graph(x, y, z, detector, radii_mm)
    stable_rank = _stable_rank(E, x, y, z)

    if len(pi) == 0:
        cluster_of = np.arange(n_hit, dtype=np.int64)
        return cluster_of, _build_features(x, y, z, E, detector, cluster_of), _make_hit_to_cluster(cluster_of), _cluster_region(cluster_of, detector)

    # CSR adjacency of the link graph, built by counting sort (fill order = edge order, same
    # content as a stable argsort). Within a hop the BFS assigns each hit the minimum-priority
    # seed that reaches it, so neighbor *order* cannot change the outcome.
    src = np.concatenate([pi, pj])
    dst = np.concatenate([pj, pi])
    deg = np.bincount(src, minlength=n_hit)
    adj, indptr = _build_csr_adjacency(src, dst, deg, n_hit)

    # seeds = strict local maxima within this link graph: no neighbor with LARGER energy
    # (equal-energy neighbors are both seeds; isolated hits are seeds)
    nbr_max_E = np.full(n_hit, -np.inf, dtype=np.float64)
    np.maximum.at(nbr_max_E, pi, E[pj].astype(np.float64))
    np.maximum.at(nbr_max_E, pj, E[pi].astype(np.float64))
    seed = np.asarray(E, dtype=np.float64) >= nbr_max_E
    seed_idx = np.where(seed)[0].tolist()
    # deterministic seed order: by stable rank, so the most energetic / lowest-rank goes first
    seed_idx.sort(key=lambda i: stable_rank[i])

    # multi-source BFS by hop count; within a hop a hit takes the minimum-priority seed that
    # reaches it (priority = seed order by stable rank). Vectorized wavefront: each hop is a
    # handful of numpy ops over the frontier's edges instead of a python loop + per-hop sort.
    hop_inf = n_hit + 2
    hops = np.full(n_hit, hop_inf, dtype=np.int64)
    seed_of = -np.ones(n_hit, dtype=np.int64)
    _wavefront_bfs(np.asarray(seed_idx, dtype=np.int64), adj, indptr, hops, seed_of)

    # lazy neighbor slices: only the leftover-component path needs them (usually empty)
    def _nbs(i):
        return adj[indptr[i] : indptr[i + 1]]

    # unlinked components (hops == inf due to isolated subgraphs): build connected components
    unreached = np.where(hops == hop_inf)[0]
    components: List[List[int]] = []
    seen = np.zeros(n_hit, dtype=bool)
    for u in unreached:
        if seen[u]:
            continue
        comp = [int(u)]
        seen[u] = True
        stack = [int(u)]
        while stack:
            cur = stack.pop()
            for nb in _nbs(cur):
                if hops[nb] == hop_inf and not seen[nb]:
                    seen[nb] = True
                    comp.append(nb)
                    stack.append(nb)
        components.append(sorted(comp, key=lambda i: stable_rank[i]))

    # final cluster labels: seeds 0..n_seeds-1, then leftover components
    cluster_of = -np.ones(n_hit, dtype=np.int64)
    for si in range(len(seed_idx)):
        members = np.where(seed_of == si)[0]
        cluster_of[members] = si
    for k, comp in enumerate(components):
        for i in comp:
            cluster_of[i] = len(seed_idx) + k
    if merge_frac > 0.0:
        cluster_of = _merge_bfs_clusters(cluster_of, E, x, y, z, pi, pj, stable_rank, merge_frac)
    return cluster_of, _build_features(x, y, z, E, detector, cluster_of), _make_hit_to_cluster(cluster_of), _cluster_region(cluster_of, detector)


@numba.njit
def _uf_find(uf: np.ndarray, v: int) -> int:
    root = v
    while uf[root] != root:
        root = uf[root]
    while uf[v] != root:
        uf[v], v = root, uf[v]
    return root


@numba.njit
def _merge_union_loop(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
    order_edges: np.ndarray,
    cluster_E: np.ndarray,
    best_rank: np.ndarray,
    uf: np.ndarray,
    merge_frac: float,
) -> None:
    """Cluster-pair union-find loop for the fragment merge; identical to the python loop it
    replaces (same candidate order, same two-by-two energy/rank tie-breaks), just compiled."""
    for oi in order_edges:
        ra = _uf_find(uf, edges_a[oi])
        rb = _uf_find(uf, edges_b[oi])
        if ra == rb:
            continue
        if cluster_E[ra] <= cluster_E[rb]:
            lo, hi = ra, rb
        else:
            lo, hi = rb, ra
        if cluster_E[lo] > merge_frac * cluster_E[hi]:
            continue  # fragment too heavy: probably a distinct comparably-energetic shower
        # attach by stable rank for determinism
        if best_rank[lo] < best_rank[hi]:
            root, child = lo, hi
        else:
            root, child = hi, lo
        uf[child] = root
        cluster_E[root] += cluster_E[child]
        cluster_E[child] = 0.0


def _merge_bfs_clusters(
    cluster_of: np.ndarray,
    E: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
    stable_rank: np.ndarray,
    merge_frac: float,
) -> np.ndarray:
    """Pandora-style fragment merge over BFS clusters.

    Any two clusters linked by ≥1 hit pair become merge candidates. The fragment (smaller
    cluster, by total energy) is absorbed into the larger only if its *total* energy
    satisfies `E_fragment <= merge_frac × E_large_cluster` — Pandora's rule is cluster-level,
    not hit-level: it asks "is this whole fragment a tail of that shower?", not "is this one
    bridge hit weak?".

    Iterated on a cluster-level union find: when A absorbs B, later candidates are evaluated
    against the combined cluster energy. Candidates are processed in a fixed order
    (descending larger-cluster energy, then ascending stable ranks) so the output is
    insensitive to link-list ordering. `merge_frac=1.0` would reunite every component into
    its strongest seed — use bounded values (Pandora's analogue is ~0.1-0.3).

    Note: Pandora's subdetector-sharing (region-match) and centroid-direction cone gates were
    implemented and evaluated here (2026-09), and vetoed zero candidate merges on ttbar_pu0
    with the tuned radii — adjacent BFS clusters always share a boundary layer and point
    along the parent shower axis. They were removed again; revisit only if radii grow enough
    to create spurious cross-shower links.
    """
    if len(pi) == 0:
        return cluster_of
    n_hit = len(x)
    n_clusters = int(cluster_of.max()) + 1
    cluster_E = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(cluster_E, cluster_of, E)
    best_rank = np.full(n_clusters, n_hit + 10, dtype=np.int64)
    np.minimum.at(best_rank, cluster_of, stable_rank)

    ci = cluster_of[pi]
    cj = cluster_of[pj]
    cross = ci != cj
    if not np.any(cross):
        return cluster_of
    ca = np.minimum(ci[cross], cj[cross])
    cb = np.maximum(ci[cross], cj[cross])
    edge_key = ca.astype(np.int64) * n_clusters + cb
    uniq_keys = np.unique(edge_key)
    edges_a = (uniq_keys // n_clusters).astype(np.int64)
    edges_b = (uniq_keys % n_clusters).astype(np.int64)
    E_pair_max = np.maximum(cluster_E[edges_a], cluster_E[edges_b])
    order_edges = np.lexsort((best_rank[edges_b], best_rank[edges_a], -E_pair_max))

    uf = np.arange(n_clusters, dtype=np.int64)
    _merge_union_loop(edges_a, edges_b, order_edges, cluster_E, best_rank, uf, merge_frac)

    labels = cluster_of
    while True:  # vectorized pointer-jump to roots (no path compression; final roots identical)
        root = uf[labels]
        if np.array_equal(root, labels):
            break
        labels = root
    uniq, inv = np.unique(labels, return_inverse=True)
    new_cluster_of = inv.astype(np.int64)
    n_new = len(uniq)
    new_rank = np.full(n_new, n_hit + 10, dtype=np.int64)
    np.minimum.at(new_rank, new_cluster_of, stable_rank)
    order = np.argsort(new_rank, kind="stable")
    remap = np.empty(n_new, dtype=np.int64)
    remap[order] = np.arange(n_new, dtype=np.int64)
    return remap[new_cluster_of]


def _build_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    cluster_of: np.ndarray,
) -> np.ndarray:
    """17-wide cluster feature matrix (EDM4hep cluster layout)."""
    n_clusters = int(cluster_of.max()) + 1 if len(cluster_of) else 0
    feats = np.zeros((n_clusters, 17), dtype=np.float32)

    # per-cluster totals; also per-hit E/total
    tot_per_cluster = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(tot_per_cluster, cluster_of, E)
    safe_tot = np.maximum(tot_per_cluster, 1e-12)
    w_per_hit = E / safe_tot[cluster_of]

    # energy-weighted centroids, width from weighted RMS around centroid
    head_coords = {}
    for coord_name, coord_arr in (("x", x), ("y", y), ("z", z)):
        s = np.zeros(n_clusters, dtype=np.float64)
        np.add.at(s, cluster_of, w_per_hit * coord_arr)
        head_coords[coord_name] = s
    cx_arr = head_coords["x"]
    cy_arr = head_coords["y"]
    cz_arr = head_coords["z"]

    n_hits_per_cluster = np.bincount(cluster_of, minlength=n_clusters)

    head_sig = {}
    for coord_name, coord_arr, ctr_arr in (("x", x, cx_arr), ("y", y, cy_arr), ("z", z, cz_arr)):
        dev2 = (coord_arr - ctr_arr[cluster_of]) ** 2
        s = np.zeros(n_clusters, dtype=np.float64)
        np.add.at(s, cluster_of, w_per_hit * dev2)
        head_sig[coord_name] = np.sqrt(s)
    sigma_x_arr = head_sig["x"]
    sigma_y_arr = head_sig["y"]
    sigma_z_arr = head_sig["z"]

    is_ecal = np.isin(detector, (9, 10, 11))
    is_hcal = np.isin(detector, (12, 13, 14))
    e_ecal_arr = np.zeros(n_clusters, dtype=np.float64)
    e_hcal_arr = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(e_ecal_arr, cluster_of, np.where(is_ecal, E, 0.0))
    np.add.at(e_hcal_arr, cluster_of, np.where(is_hcal, E, 0.0))
    e_other_arr = tot_per_cluster - e_ecal_arr - e_hcal_arr

    # direction (eta, phi) from centroid; guard degenerate directions
    dir_norm = np.sqrt(cx_arr**2 + cy_arr**2 + cz_arr**2)
    safe_dir = np.maximum(dir_norm, 1e-30)
    nz = np.clip(cz_arr / safe_dir, -1 + 1e-9, 1 - 1e-9)
    eta_arr = -np.log(np.sqrt((1 - nz) / (1 + nz)))
    eta_arr = np.where(dir_norm == 0, 0.0, eta_arr)
    phi_arr = np.where(dir_norm > 0, np.arctan2(cy_arr, cx_arr), 0.0)

    r_arr = np.hypot(cx_arr, cy_arr)
    denom = np.maximum(np.hypot(cz_arr, r_arr), 1e-30)
    pt_arr = tot_per_cluster * r_arr / denom

    feats[:, 0] = 2.0  # elemtype: cluster
    feats[:, 1] = pt_arr
    feats[:, 2] = eta_arr
    feats[:, 3] = np.sin(phi_arr)
    feats[:, 4] = np.cos(phi_arr)
    feats[:, 5] = tot_per_cluster
    feats[:, 6] = cx_arr
    feats[:, 7] = cy_arr
    feats[:, 8] = cz_arr
    feats[:, 9] = 0.0  # iTheta placeholder
    feats[:, 10] = e_ecal_arr
    feats[:, 11] = e_hcal_arr
    feats[:, 12] = e_other_arr
    feats[:, 13] = n_hits_per_cluster.astype(np.float64)
    feats[:, 14] = sigma_x_arr
    feats[:, 15] = sigma_y_arr
    feats[:, 16] = sigma_z_arr
    return feats


def _cluster_region(cluster_of: np.ndarray, detector: np.ndarray) -> np.ndarray:
    """Dominant region code per cluster (audit; for neutral-hadron PID forcing)."""
    n_clusters = int(cluster_of.max()) + 1 if len(cluster_of) else 0
    if n_clusters == 0:
        return np.zeros(0, dtype=np.int64)
    region_codes_sorted = np.unique(detector)
    region_compact = np.searchsorted(region_codes_sorted, detector)
    es_by_creg = np.zeros((n_clusters, len(region_codes_sorted)), dtype=np.float64)
    np.add.at(es_by_creg, (cluster_of, region_compact), np.ones(len(detector), dtype=np.float64))
    membership_eps = np.zeros_like(es_by_creg)
    np.add.at(membership_eps, (cluster_of, region_compact), 1e-12)
    return region_codes_sorted[np.argmax(es_by_creg + membership_eps, axis=1)]


def _make_hit_to_cluster(cluster_of: np.ndarray) -> SparseMatrixCOO:
    """COO (hit_idx, cluster_idx, 1.0) — dense assignment, no partial hits."""
    return (
        np.arange(len(cluster_of), dtype=np.int64),
        cluster_of.astype(np.int64),
        np.ones(len(cluster_of), dtype=np.float32),
    )


def cluster_event(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float] | None = None,
    merge_frac: float | None = None,
    algorithm: str = "bfs_merge",
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """Cluster one event's calorimeter hits.

    Three algorithms, selected by `algorithm`:

    - **bfs_merge** (default): seeded multi-source BFS watershed (strict local maxima claim
      hits by hop count) + a Pandora-style fragment merge over the seed clusters that touch
      via a link. The merge rule is cluster-level: a fragment is absorbed into a bigger
      neighbouring cluster when its total energy E_frag <= merge_frac × E_big. Default
      merge_frac is 0.25 (tuned on 5 ttbar events); turning it off (`merge_frac=0`) recovers
      the raw BFS output (**bfs**).
    - **union_find**: connected components of the whole radius graph (with an optional edge
      gate based on hit energy asymmetry if merge_frac > 0). No seeding. Higher R2 on the
      5-event sweep before fragment merging, but melts separate showers when they sit close.
    - **bfs**: raw seeded BFS on the radius graph, no merging.

    Args:
      x, y, z:     float32/64 positions in mm, length N
      E:           reconstructed hit energy (calibrated GeV), same length
      detector:    uint8 region code 9..14 (ECAL 9-11, HCAL 12-14)
      radii_mm:    override of the per-region link radius in mm
      merge_frac:  None (per-algorithm defaults), 0.0 (disable merging), or a fraction.
                  For bfs_merge this is the fragment-absorption threshold; for union_find
                  this is the link-refusal gate.
      algorithm:   "bfs_merge" | "union_find" | "bfs"
    """

    # ColliderML parquet columns are option-typed, so ak.to_numpy hands us numpy.ma
    # MaskedArrays whose per-element getitem is far slower than plain ndarrays. Nothing is
    # ever masked in these inputs — assert that (a masked entry would corrupt energy
    # comparisons silently) and strip the mask.
    def _plain(a):
        if isinstance(a, np.ma.MaskedArray):
            m = np.ma.getmask(a)
            if np.any(m):
                raise ValueError("cluster_event received masked entries")
            return np.asarray(a.data)
        return a

    x, y, z, E, detector = _plain(x), _plain(y), _plain(z), _plain(E), _plain(detector)

    if merge_frac is None:
        merge_frac = DEFAULT_MERGE_FRAC if algorithm == "bfs_merge" else 0.0
    if radii_mm is None:
        radii_mm = DEFAULT_REGION_RADII_MM

    if algorithm == "union_find":
        return _cluster_event_union_find(x, y, z, E, detector, radii_mm, merge_frac)
    elif algorithm == "bfs":
        return _cluster_event_bfs(x, y, z, E, detector, radii_mm, merge_frac=0.0)
    elif algorithm == "bfs_merge":
        return _cluster_event_bfs(x, y, z, E, detector, radii_mm, merge_frac=merge_frac)
    else:
        raise ValueError(f"algorithm must be 'bfs_merge', 'union_find' or 'bfs', got {algorithm!r}")
