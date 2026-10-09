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
# Algorithms, selected by `algorithm=` on `cluster_event` (default bfs_merge): the
# radius-graph clusterers below, plus CLUE (`algorithm="clue"`), which lives in clue.py
# (presets and defaults: clue.CLUE_PRESETS / DEFAULT_CLUE_PARAMS / DEFAULT_CLUE_OPTIONS).
#
#   union_find: connected components over the radius graph, with an optional
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
#   bfs_merge: bfs followed by a Pandora-style fragment merge (see _merge_bfs_clusters).
#
# Properties enforced by all clusterers (CLUE included):
#   * deterministic per-event: identical inputs always give the same output
#   * truth-blind: contrib_particle_ids/energies/times never read
#   * region-crossing hits allowed via a single tree over all hits + radius max per pair
#     (ECAL cells can join HCAL cells when a shower crosses the detector boundary)
#   * energy-conserving: every retained hit belongs to exactly one cluster (only CLUE's
#     optional drop_E / drop_1hit leave hits unclustered, with cluster -1 and no
#     hit_to_cluster entry)
from typing import Any, Dict, List, Tuple

import numba
import numpy as np
from scipy.spatial import cKDTree

from mlpf.data.colliderml.clue import _cluster_event_clue, resolve_clue_config
from mlpf.data.colliderml.cluster_common import build_features, cluster_region, make_hit_to_cluster, relabel_by_rank, stable_rank, uf_find
from mlpf.data.target_building import SparseMatrixCOO

# NOTE: radii and merge_frac below are tuned on ttbar_pu0. At pu200 the defaults percolate:
# any fragment-merge criterion chains transitively through the pileup-dense link graph, so
# O(10^5)-hit / multi-TeV mega-clusters form even from capped or frozen-threshold merge
# variants (measured 2026-10-05 on ttbar_pu200 shards 0-1). The pu200 conversion therefore
# runs algorithm="bfs" (no merge) with tightened radii ECAL 16.25 / HCAL 36 mm via
# scripts/flatiron/colliderml_convert{,_array}.sh; keep these defaults pu0-shaped.
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

    # deterministic pair order: sort by (rank pi, rank pj) within each
    # (pi, pj) so we process pairs in a stable order
    rank = stable_rank(E, x, y, z)
    pair_order = np.lexsort((rank[pj], rank[pi]))
    pi = pi[pair_order]
    pj = pj[pair_order]

    if len(pi) == 0:
        # no links at all -> every hit is its own cluster
        cluster_of = np.arange(n_hit, dtype=np.int64)
        return cluster_of, build_features(x, y, z, E, detector, cluster_of), make_hit_to_cluster(cluster_of), cluster_region(cluster_of, detector)

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
        if rank[ra] < rank[rb]:
            parent[rb] = ra
            if comp_max_E[rb] > comp_max_E[ra]:
                comp_max_E[ra] = comp_max_E[rb]
        else:
            parent[ra] = rb
            if comp_max_E[ra] > comp_max_E[rb]:
                comp_max_E[rb] = comp_max_E[ra]

    labels = np.array([_find(v) for v in range(n_hit)], dtype=np.int64)
    cluster_of = relabel_by_rank(labels, rank)
    return cluster_of, build_features(x, y, z, E, detector, cluster_of), make_hit_to_cluster(cluster_of), cluster_region(cluster_of, detector)


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
    rank = stable_rank(E, x, y, z)

    if len(pi) == 0:
        cluster_of = np.arange(n_hit, dtype=np.int64)
        return cluster_of, build_features(x, y, z, E, detector, cluster_of), make_hit_to_cluster(cluster_of), cluster_region(cluster_of, detector)

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
    seed_idx.sort(key=lambda i: rank[i])

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
        components.append(sorted(comp, key=lambda i: rank[i]))

    # final cluster labels: seeds 0..n_seeds-1, then leftover components
    # (seed_of already holds the seed priority 0..n_seeds-1, or -1 for unreached hits)
    cluster_of = seed_of.copy()
    for k, comp in enumerate(components):
        for i in comp:
            cluster_of[i] = len(seed_idx) + k
    if merge_frac > 0.0:
        cluster_of = _merge_bfs_clusters(cluster_of, E, x, y, z, pi, pj, rank, merge_frac)
    return cluster_of, build_features(x, y, z, E, detector, cluster_of), make_hit_to_cluster(cluster_of), cluster_region(cluster_of, detector)


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
        ra = uf_find(uf, edges_a[oi])
        rb = uf_find(uf, edges_b[oi])
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
    rank: np.ndarray,
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
    np.minimum.at(best_rank, cluster_of, rank)

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
    return relabel_by_rank(labels, rank)


def cluster_event(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float] | None = None,
    merge_frac: float | None = None,
    algorithm: str = "bfs_merge",
    clue_params: Dict[int, Dict[str, float]] | None = None,
    clue_options: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """Cluster one event's calorimeter hits.

    Four algorithms, selected by `algorithm`:

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
    - **clue**: CLUE density-peak clustering per region (clue.py); radii_mm
      and merge_frac are ignored, clue_params/clue_options configure it.

    Args:
      x, y, z:     float32/64 positions in mm, length N
      E:           reconstructed hit energy (calibrated GeV), same length
      detector:    uint8 region code 9..14 (ECAL 9-11, HCAL 12-14)
      radii_mm:    override of the per-region link radius in mm
      merge_frac:  None (per-algorithm defaults), 0.0 (disable merging), or a fraction.
                  For bfs_merge this is the fragment-absorption threshold; for union_find
                  this is the link-refusal gate.
      algorithm:   "bfs_merge" | "union_find" | "bfs" | "clue"
      clue_params: per-region overrides of DEFAULT_CLUE_PARAMS for "clue", e.g.
                   {9: {"dc": 8.0}}; keys dc, rhoc, dm, seed_dc, alpha (see clue.py's header)
      clue_options: overrides of DEFAULT_CLUE_OPTIONS for "clue" (keys in clue.py's header;
                   clue.CLUE_PRESETS holds the named per-pileup settings)
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
    elif algorithm == "clue":
        params, options = resolve_clue_config(params=clue_params, options=clue_options)
        return _cluster_event_clue(x, y, z, E, detector, params, **options)
    else:
        raise ValueError(f"algorithm must be 'bfs_merge', 'union_find', 'bfs' or 'clue', got {algorithm!r}")
