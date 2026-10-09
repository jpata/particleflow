# A re-implementation of CLUE using numba.
# Reference: CLUE (CLUstering of Energy, Rovere et al., Front. Big Data 3 (2020) 591315; the algorithm of
# the CLUEstering package and CMS HGCAL):
#
# How CLUE works
# 1. Density: for each hit, add up its own energy plus half the energy of hits within dc.
#    Shower cores get high density.
# 2. Point uphill: each hit finds the nearest hit within seed_dc that has higher density.
# 3. Seeds: a hit with density ≥ rhoc and no higher-density hit within seed_dc is a density
#    peak, and starts a cluster.
# 4. Followers: every other hit adopts the cluster of its uphill neighbour — a watershed on the
#    density field, so each hill becomes one cluster. The exception: low-density hits
#    (rho < rhoc) follow only if their uphill neighbour is within seed_dc.
# 5. Outliers: low-density hits with no higher-density hit within seed_dc. We group them
#    (connected components at link radius dm), absorb clusters below min_cluster_E into the
#    nearest large cluster, and optionally drop those still below drop_E.
# Density and clustering run independently in each calorimeter region, with distances measured
# on a per-region reference surface (hits projected along their ray, depth differences divided
# by alpha) rather than in plain euclidean space.
#
# Distances are measured in projective coordinates: each hit is projected along its ray from
# the origin onto a per-region reference surface (CLUE_REF_DEPTH_MM: a cylinder for the
# barrels, a z-plane for the endcaps), which gives the transverse position, and its depth
# (r for barrels, |z| for endcaps) is divided by alpha. So the distance is
# sqrt(d_perp^2 + (d_depth / alpha)^2) with d_perp measured at the reference depth: alpha > 1
# links hits along a (pointing) shower more readily than across showers. Neighbour search is
# a tile grid in numba with O(n_hit) memory: no pair list is stored, which matters at pu200
# (~0.7M hits, ~240k in one ECAL endcap).
#
# Per-region parameters (keys of each region dict):
#   dc       density radius, mm
#   rhoc     seed density threshold, GeV
#   dm       link radius for grouping outliers into components (outlier_mode="components"), mm.
#            It also caps the nearest-higher search, but has no other effect on seeds and
#            followers: a hit whose nearest higher-density hit is farther than seed_dc becomes a
#            seed or an outlier anyway, so the search runs within min(dm, seed_dc)
#   seed_dc  minimum distance to a higher-density hit for a seed, mm; also the reach of a
#            follower (no hit follows a nearest-higher farther away than this)
#   alpha    longitudinal de-weighting of depth differences (1 = isotropic)
# Global options:
#   attach_mm       outliers join the cluster of the nearest clustered hit of the same region
#                   within this (projective) distance — Pandora's isolated-hit merging
#                   analogue; 0 disables
#   outlier_mode    outliers that were not attached: "singleton" (own cluster each),
#                   "components" (connected components of the leftovers at link radius dm, built
#                   on the tile grid without a pair list; can percolate through pileup-dense
#                   leftovers, see max_component_hits), or "clue" (a second CLUE
#                   pass over the leftovers alone, seed threshold rhoc * outlier_rhoc_frac;
#                   what is still left becomes singletons — bounded like CLUE itself)
#   outlier_rhoc_frac  see outlier_mode="clue"
#   max_component_hits outlier_mode="components" guard against percolation through dense
#                   leftovers (pileup): a component with more hits than this is re-clustered
#                   by a CLUE pass at seed threshold rhoc * outlier_rhoc_frac instead (its
#                   remaining outliers become singletons); 0 disables
#   min_cluster_E   small-cluster absorption (after outlier handling): every cluster with
#   absorb_mm       energy < min_cluster_E GeV joins the cluster of the nearest hit (same
#                   region, projective distance < absorb_mm) of a cluster with energy >=
#                   min_cluster_E. Single pass, only small -> large, so it cannot chain or
#                   percolate; small clusters with no large neighbour stay. 0 disables
#   drop_E          clusters still below this energy (GeV) after absorption — in practice
#                   isolated soft hits > absorb_mm from any larger cluster — are dropped: their
#                   hits get cluster -1 and are in no cluster (Pandora-like; loses their
#                   energy, ~1% at pu0). 0 (default) keeps them as their own small clusters
#   drop_1hit       likewise drop single-hit clusters (isolated, usually soft hits) regardless
#                   of energy. False (default) keeps them
#   core            "numba" (ours) or "cluestering" (the CLUEstering package, for
#                   cross-checks; see _clue_labels_cluestering)
#   cross_region_mm ECAL->HCAL linking: an HCAL cluster joins the ECAL cluster that holds the
#                   ECAL hit closest to any of its hits, if that is within this (euclidean)
#                   distance and the ECAL cluster is the more energetic; 0 disables
# Density and nearest-higher are computed within a region only (ECAL and HCAL energy densities
# are not comparable); barrel/endcap regions of the same calorimeter are distinct regions.
from typing import Any, Dict, Tuple

import numba
import numpy as np
from scipy.spatial import cKDTree

from mlpf.data.colliderml.cluster_common import (
    build_features,
    cluster_region,
    make_hit_to_cluster,
    relabel_by_rank,
    resolve_roots,
    stable_rank,
    uf_find,
)
from mlpf.data.target_building import SparseMatrixCOO

CLUE_ECAL = dict(dc=10.0, rhoc=0.5, dm=30.0, seed_dc=25.0, alpha=3.0)
CLUE_HCAL = dict(dc=60.0, rhoc=0.5, dm=150.0, seed_dc=120.0, alpha=3.0)
DEFAULT_CLUE_PARAMS = {9: CLUE_ECAL, 10: CLUE_ECAL, 11: CLUE_ECAL, 12: CLUE_HCAL, 13: CLUE_HCAL, 14: CLUE_HCAL}
DEFAULT_CLUE_OPTIONS = dict(
    attach_mm=0.0,
    outlier_mode="components",
    outlier_rhoc_frac=0.1,
    max_component_hits=300,
    min_cluster_E=0.5,
    absorb_mm=200.0,
    cross_region_mm=0.0,
    drop_E=0.0,
    drop_1hit=False,
    core="numba",
)
# Named settings (converter --clue-preset). "pu0" = the defaults above. "pu200" = the 2026-10-08
# "middle setting": lower seed threshold, smaller seed_dc (seed separation and follower reach),
# smaller outlier link radius dm and weaker absorption,
# ~2.5x the default's clusters at pu200 — halves the allocator merges (hard scatter 30% -> 10%)
# at the cost of more split particles; it hurts pu0, so it is pu200-only.
_PU200_ECAL = dict(CLUE_ECAL, rhoc=0.2, seed_dc=15.0, dm=20.0)
_PU200_HCAL = dict(CLUE_HCAL, rhoc=0.2, seed_dc=80.0, dm=100.0)
CLUE_PRESETS = {
    "pu0": dict(params={}, options={}),
    "pu200": dict(
        params={**{r: _PU200_ECAL for r in (9, 10, 11)}, **{r: _PU200_HCAL for r in (12, 13, 14)}},
        options=dict(min_cluster_E=0.1),
    ),
}
_PARAM_KEYS = frozenset(("dc", "rhoc", "dm", "seed_dc", "alpha"))
_CHOICES = {"outlier_mode": ("singleton", "components", "clue"), "core": ("numba", "cluestering")}


def resolve_clue_config(
    preset: str | None = None, params: Dict[int, Dict[str, float]] | None = None, options: Dict[str, Any] | None = None
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """The full CLUE setting: DEFAULT_CLUE_PARAMS/OPTIONS, then the named preset, then the
    per-region `params` and the `options` overrides. The single place this merge happens
    (cluster_event, the converter CLI and the tuning scripts all call it).

    Validates everything up front, so a typo fails immediately instead of silently running the
    defaults: region keys must be the int codes 9-14, parameter names in dc/rhoc/dm/seed_dc/alpha,
    option names those of DEFAULT_CLUE_OPTIONS, and outlier_mode/core among their choices.
    """
    if preset is not None and preset not in CLUE_PRESETS:
        raise ValueError(f"unknown CLUE preset {preset!r}; known: {sorted(CLUE_PRESETS)}")
    pre = CLUE_PRESETS[preset] if preset is not None else dict(params={}, options={})
    layers_p = [pre["params"], params or {}]
    layers_o = [pre["options"], options or {}]
    for layer in layers_p:
        for r, d in layer.items():
            if not isinstance(r, (int, np.integer)) or int(r) not in DEFAULT_CLUE_PARAMS:
                raise ValueError(f"CLUE params: region key {r!r} is not one of the int codes {sorted(DEFAULT_CLUE_PARAMS)}")
            bad = set(d) - _PARAM_KEYS
            if bad:
                raise ValueError(f"CLUE params for region {r}: unknown parameter(s) {sorted(bad)}; known: {sorted(_PARAM_KEYS)}")
    for layer in layers_o:
        bad = set(layer) - set(DEFAULT_CLUE_OPTIONS)
        if bad:
            raise ValueError(f"CLUE options: unknown option(s) {sorted(bad)}; known: {sorted(DEFAULT_CLUE_OPTIONS)}")
    full_p = {r: dict(d) for r, d in DEFAULT_CLUE_PARAMS.items()}
    for layer in layers_p:
        for r, d in layer.items():
            full_p[int(r)].update(d)
    full_o = dict(DEFAULT_CLUE_OPTIONS)
    for layer in layers_o:
        full_o.update(layer)
    for k, allowed in _CHOICES.items():
        if full_o[k] not in allowed:
            raise ValueError(f"CLUE option {k}={full_o[k]!r}; must be one of {allowed}")
    return full_p, full_o


# reference depth per region (mm): mid-depth of the ODD calorimeter volumes (barrel radius,
# endcap |z|), where projective transverse distances equal physical ones
CLUE_REF_DEPTH_MM = {9: 3320.0, 10: 1385.0, 11: 3320.0, 12: 4540.0, 13: 2450.0, 14: 4540.0}
_BARREL_REGIONS = (10, 13)


def _projective_coords(pts: np.ndarray, region: int, alpha: float) -> np.ndarray:
    """(n, 4) barrel / (n, 3) endcap coordinates in which euclidean distance is the CLUE metric."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ref = CLUE_REF_DEPTH_MM[region]
    if region in _BARREL_REGIONS:
        r = np.maximum(np.hypot(x, y), 1e-6)
        s = ref / r
        return np.column_stack([x * s, y * s, z * s, r / alpha])
    az = np.maximum(np.abs(z), 1e-6)
    s = ref / az
    return np.column_stack([x * s, y * s, az / alpha])


def _tile_grid(q: np.ndarray, cell: float):
    """Bucket points into a grid of cubic tiles of size `cell`.

    Returns (order, tile_start, tile_end, nb_ptr, nb_idx): points sorted by tile
    (`order`), per occupied tile the [start, end) slice of that sorted order, and a CSR list
    of the occupied neighbour tiles (3^ndim block, including itself) of every occupied tile.
    """
    t = np.floor((q - q.min(axis=0)) / cell).astype(np.int64) + 1  # +1 margin for offsets
    span = t.max(axis=0) + 2
    if np.prod(span.astype(np.float64)) > 2.0**62:
        raise ValueError(f"tile grid too large for int64 keys (span {span.tolist()}, cell {cell} mm)")
    strides = np.ones(q.shape[1], np.int64)
    for k in range(q.shape[1] - 2, -1, -1):
        strides[k] = strides[k + 1] * span[k + 1]
    key = t @ strides
    order = np.argsort(key, kind="stable")
    ukey, tile_start, counts = np.unique(key[order], return_index=True, return_counts=True)
    nb_ptr, nb_idx = _tile_neighbours(ukey, _neighbour_offsets(q.shape[1], strides))
    return order, tile_start.astype(np.int64), (tile_start + counts).astype(np.int64), nb_ptr, nb_idx


@numba.njit
def _neighbour_offsets(ndim: int, strides: np.ndarray) -> np.ndarray:
    n_off = 3**ndim
    out = np.empty(n_off, np.int64)
    for o in range(n_off):
        v = o
        s = 0
        for k in range(ndim):
            s += ((v % 3) - 1) * strides[k]
            v //= 3
        out[o] = s
    return np.sort(out)


@numba.njit
def _tile_neighbours(ukey: np.ndarray, offsets: np.ndarray):
    n = len(ukey)
    nb_ptr = np.zeros(n + 1, np.int64)
    buf = np.empty(n * len(offsets), np.int64)
    m = 0
    for a in range(n):
        for o in range(len(offsets)):
            k = ukey[a] + offsets[o]
            b = np.searchsorted(ukey, k)
            if b < n and ukey[b] == k:
                buf[m] = b
                m += 1
        nb_ptr[a + 1] = m
    return nb_ptr, buf[:m].copy()


@numba.njit
def _clue_density(q, E, tile_start, tile_end, nb_ptr, nb_idx, dc):
    """rho_i = E_i + 0.5 * sum_{d_ij < dc} E_j; q/E in tile order, tiles of size >= dc."""
    n, nd = q.shape
    dc2 = dc * dc
    rho = E.astype(np.float64).copy()
    for a in range(len(tile_start)):
        for i in range(tile_start[a], tile_end[a]):
            acc = 0.0
            for u in range(nb_ptr[a], nb_ptr[a + 1]):
                b = nb_idx[u]
                for j in range(tile_start[b], tile_end[b]):
                    if j == i:
                        continue
                    d2 = 0.0
                    for c in range(nd):
                        t = q[i, c] - q[j, c]
                        d2 += t * t
                    if d2 < dc2:
                        acc += E[j]
            rho[i] += 0.5 * acc
    return rho


@numba.njit
def _clue_nearest_higher(q, rho, rank, tile_start, tile_end, nb_ptr, nb_idx, dm):
    """nh_i = closest j with d_ij < dm that is higher in the strict total order (rho desc,
    stable rank asc), delta_i its distance (inf if none); inputs in tile order, tiles >= dm."""
    n, nd = q.shape
    dm2 = dm * dm
    nh = np.full(n, -1, np.int64)
    delta = np.full(n, np.inf)
    for a in range(len(tile_start)):
        for i in range(tile_start[a], tile_end[a]):
            best = np.inf
            bj = -1
            for u in range(nb_ptr[a], nb_ptr[a + 1]):
                b = nb_idx[u]
                for j in range(tile_start[b], tile_end[b]):
                    if not (rho[j] > rho[i] or (rho[j] == rho[i] and rank[j] < rank[i])):
                        continue
                    d2 = 0.0
                    for c in range(nd):
                        t = q[i, c] - q[j, c]
                        d2 += t * t
                    # equal distances (common on the cell grid): prefer the denser neighbour, as
                    # CLUEstering does, then the better stable rank
                    if d2 < dm2 and (d2 < best or (d2 == best and (rho[j] > rho[bj] or (rho[j] == rho[bj] and rank[j] < rank[bj])))):
                        best = d2
                        bj = j
            nh[i] = bj
            delta[i] = np.sqrt(best)
    return nh, delta


def _clue_region(q: np.ndarray, E: np.ndarray, rank: np.ndarray, dc: float, dm: float):
    """CLUE density + nearest-higher for one region's hits (original order in and out)."""
    o1, s1, e1, p1, i1 = _tile_grid(q, dc)
    rho = np.empty(len(E))
    rho[o1] = _clue_density(np.ascontiguousarray(q[o1]), E[o1], s1, e1, p1, i1, dc)
    o2, s2, e2, p2, i2 = _tile_grid(q, dm)
    nh_s, delta_s = _clue_nearest_higher(np.ascontiguousarray(q[o2]), rho[o2], rank[o2], s2, e2, p2, i2, dm)
    nh = np.where(nh_s >= 0, o2[np.maximum(nh_s, 0)], -1)
    out_nh = np.empty(len(E), np.int64)
    out_delta = np.empty(len(E))
    out_nh[o2] = nh
    out_delta[o2] = delta_s
    return rho, out_nh, out_delta


@numba.njit
def _union_within(q, tile_start, tile_end, nb_ptr, nb_idx, r, parent):
    """Union-find over all point pairs within distance r (inclusive, as cKDTree.query_pairs);
    q in tile order, tiles of size >= r. No pair list: memory stays O(n)."""
    nd = q.shape[1]
    r2 = r * r
    for a in range(len(tile_start)):
        for i in range(tile_start[a], tile_end[a]):
            for u in range(nb_ptr[a], nb_ptr[a + 1]):
                b = nb_idx[u]
                for j in range(tile_start[b], tile_end[b]):
                    if j <= i:
                        continue
                    d2 = 0.0
                    for c in range(nd):
                        t = q[i, c] - q[j, c]
                        d2 += t * t
                    if d2 <= r2:
                        ri = uf_find(parent, i)
                        rj = uf_find(parent, j)
                        if ri != rj:
                            if ri < rj:
                                parent[rj] = ri
                            else:
                                parent[ri] = rj


def _components(q: np.ndarray, r: float) -> np.ndarray:
    """Connected components of the points at link radius r: dense ids 0..n_comp-1."""
    order, s, e, nb_ptr, nb_idx = _tile_grid(q, r)
    parent = np.arange(len(q), dtype=np.int64)
    _union_within(np.ascontiguousarray(q[order]), s, e, nb_ptr, nb_idx, r, parent)
    roots = np.empty(len(q), np.int64)
    roots[order] = resolve_roots(parent, np.arange(len(q), dtype=np.int64))
    return np.unique(roots, return_inverse=True)[1]


@numba.njit
def _clue_assign(order: np.ndarray, is_seed: np.ndarray, nh: np.ndarray, label: np.ndarray) -> None:
    """Followers take their nearest-higher's label, walking hits in decreasing density so the
    nearest-higher is always labelled first; -1 propagates (followers of outliers are outliers)."""
    for t in range(len(order)):
        i = order[t]
        if not is_seed[i] and nh[i] >= 0:
            label[i] = label[nh[i]]


def _clue_labels_cluestering(q: np.ndarray, E: np.ndarray, p: Dict[str, float], rhoc: float) -> np.ndarray:
    """The same pass by the CLUEstering package (cross-check only; serial CPU backend).

    Needs `pip install CLUEstering` (built against Boost; on Rusty `module load boost` at build
    AND run time). Same inputs and parameter meaning; since ours adopted its outlier rule and
    its equal-distance tie-break (2026-10-08), the two give the same seeds and agree on all but
    a handful of hits (single-precision near-ties), and identical metrics to ~1e-3.
    """
    import CLUEstering

    c = CLUEstering.clusterer(float(p["dc"]), float(rhoc), float(p["dm"]), float(p.get("seed_dc", p["dm"])))
    c.read_data([np.ascontiguousarray(q[:, k], dtype=np.float64) for k in range(q.shape[1])] + [np.asarray(E, dtype=np.float64)])
    c.run_clue(backend="cpu serial")
    lab = np.asarray(c.cluster_ids, dtype=np.int64)
    return np.where(lab >= 0, lab, -1)


def _clue_labels(q: np.ndarray, E: np.ndarray, rank: np.ndarray, p: Dict[str, float], rhoc: float) -> np.ndarray:
    """One CLUE pass: labels 0..n_seeds-1 (seeds in decreasing-density order), -1 = outlier."""
    if p.get("core", "numba") == "cluestering":
        return _clue_labels_cluestering(q, E, p, rhoc)
    seed_dc = float(p.get("seed_dc", p["dm"]))
    # a nearest-higher beyond seed_dc is never followed (the hit becomes a seed or an outlier),
    # so searching up to dm > seed_dc is wasted work; nextafter keeps a hit at exactly seed_dc a
    # follower, as with the search up to dm
    r_nh = float(p["dm"]) if p["dm"] <= seed_dc else float(np.nextafter(seed_dc, np.inf))
    rho, nh, delta = _clue_region(q, E, rank, float(p["dc"]), r_nh)
    is_seed = (rho >= rhoc) & (delta > seed_dc)
    # outliers (CLUEstering's rule): low density and no higher-density hit within seed_dc; a
    # low-density hit only follows a nearest-higher closer than seed_dc (dm bounds the search)
    nh = np.where((rho < rhoc) & (delta > seed_dc), -1, nh)
    order = np.lexsort((rank, -rho))
    seeds = order[is_seed[order]]
    lab = np.full(len(E), -1, np.int64)
    lab[seeds] = np.arange(len(seeds))
    _clue_assign(order, is_seed, nh, lab)
    return lab


def _split_big_components(
    comp: np.ndarray, q: np.ndarray, E: np.ndarray, rank: np.ndarray, p: Dict[str, float], rhoc: float, max_hits: int
) -> np.ndarray:
    """Re-cluster components with more than max_hits hits by a CLUE pass (leftovers -> singletons)."""
    sizes = np.bincount(comp)
    big = np.nonzero(sizes > max_hits)[0]
    if len(big) == 0:
        return comp
    out = comp.copy()
    nxt = len(sizes)
    for c in big:
        m = np.nonzero(comp == c)[0]
        lab = _clue_labels(q[m], E[m], rank[m], p, rhoc)
        n_cl = int(lab.max()) + 1 if (lab >= 0).any() else 0
        lab[lab < 0] = n_cl + np.arange(int(np.sum(lab < 0)))
        out[m] = nxt + lab
        nxt += int(lab.max()) + 1
    return np.unique(out, return_inverse=True)[1]


def _cluster_event_clue(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    params: Dict[int, Dict[str, float]],
    *,
    attach_mm: float,
    outlier_mode: str,
    outlier_rhoc_frac: float,
    max_component_hits: int,
    core: str,
    min_cluster_E: float,
    absorb_mm: float,
    cross_region_mm: float,
    drop_E: float,
    drop_1hit: bool,
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """CLUE per region, then outlier handling, small-cluster absorption, optional
    ECAL->HCAL linking and optional dropping of soft leftovers (see the header block above).
    With drop_E > 0, cluster_of is -1 for dropped hits and hit_to_cluster omits them.

    The options have no defaults here on purpose: resolve_clue_config builds the full setting
    (DEFAULT_CLUE_OPTIONS + preset + overrides), called from clustering.cluster_event."""
    n_hit = len(x)
    pts = np.column_stack([x, y, z]).astype(np.float64)
    E64 = np.asarray(E, dtype=np.float64)
    det = np.asarray(detector).astype(np.int64)
    rank = stable_rank(E64, x, y, z)
    regions = [int(r) for r in np.unique(det)]

    label = np.full(n_hit, -1, np.int64)
    n_lab = 0
    coords = {}
    for r in regions:
        p = {**params[r], "core": core}
        ia = np.nonzero(det == r)[0]
        q = _projective_coords(pts[ia], r, float(p.get("alpha", 1.0)))
        coords[r] = (ia, q)
        rank_r = rank[ia]
        lab = _clue_labels(q, E64[ia], rank_r, p, p["rhoc"])
        lab[lab >= 0] += n_lab
        n_lab = int(lab.max()) + 1 if (lab >= 0).any() else n_lab

        # outliers -> nearest clustered hit of the region within attach_mm
        out_l = np.nonzero(lab < 0)[0]
        in_l = np.nonzero(lab >= 0)[0]
        if attach_mm > 0.0 and len(out_l) and len(in_l):
            dist, nn = cKDTree(q[in_l]).query(q[out_l], k=1, distance_upper_bound=attach_mm)
            ok = np.isfinite(dist)
            lab[out_l[ok]] = lab[in_l[nn[ok]]]
            out_l = out_l[~ok]
        # leftovers: singletons, or connected components at link radius dm
        if len(out_l):
            if outlier_mode == "singleton":
                comp = np.arange(len(out_l))
            elif outlier_mode == "components":
                comp = _components(q[out_l], float(p["dm"]))
                if max_component_hits > 0:
                    comp = _split_big_components(comp, q[out_l], E64[ia][out_l], rank_r[out_l], p, p["rhoc"] * outlier_rhoc_frac, max_component_hits)
            elif outlier_mode == "clue":
                comp = _clue_labels(q[out_l], E64[ia][out_l], rank_r[out_l], p, p["rhoc"] * outlier_rhoc_frac)
                left2 = comp < 0
                comp[left2] = (comp.max() + 1 if (~left2).any() else 0) + np.arange(int(left2.sum()))
            else:
                raise ValueError(f"outlier_mode must be 'singleton', 'components' or 'clue', got {outlier_mode!r}")
            lab[out_l] = n_lab + comp
            n_lab += int(comp.max()) + 1
        label[ia] = lab

    if min_cluster_E > 0.0 and absorb_mm > 0.0:
        label = _absorb_small_clusters(label, coords, E64, rank, min_cluster_E, absorb_mm)

    if cross_region_mm > 0.0:
        label = _link_ecal_hcal(label, pts, E64, det, rank, cross_region_mm)

    cluster_of = relabel_by_rank(label, rank)
    if (drop_E > 0.0 or drop_1hit) and n_hit:
        # drop clusters below drop_E and/or single-hit clusters (isolated, usually soft
        # leftovers): their hits become unclustered (-1), like Pandora's
        cl_E = np.bincount(cluster_of, weights=E64)
        keep = cl_E >= drop_E if drop_E > 0.0 else np.ones(len(cl_E), bool)
        if drop_1hit:
            keep &= np.bincount(cluster_of, minlength=len(cl_E)) > 1
        new_id = np.full(len(cl_E), -1, np.int64)
        new_id[keep] = np.arange(int(keep.sum()))  # relabelling keeps the rank order
        cluster_of = new_id[cluster_of]
    inc = cluster_of >= 0
    if inc.all():
        return cluster_of, build_features(x, y, z, E, detector, cluster_of), make_hit_to_cluster(cluster_of), cluster_region(cluster_of, detector)
    co = cluster_of[inc]
    feats = build_features(x[inc], y[inc], z[inc], np.asarray(E)[inc], det[inc], co)
    coo = (np.nonzero(inc)[0].astype(np.int64), co, np.ones(len(co), dtype=np.float32))
    return cluster_of, feats, coo, cluster_region(co, det[inc])


def _absorb_small_clusters(
    label: np.ndarray, coords: Dict[int, Tuple[np.ndarray, np.ndarray]], E: np.ndarray, rank: np.ndarray, min_E: float, r_mm: float
) -> np.ndarray:
    """Move every cluster below min_E into the cluster of its nearest large-cluster hit."""
    cl_E = np.bincount(label, weights=E)
    out = label.copy()
    for ia, q in coords.values():
        lab = label[ia]
        small = cl_E[lab] < min_E
        if not small.any() or small.all():
            continue
        s_i, b_i = np.nonzero(small)[0], np.nonzero(~small)[0]
        dist, nn = cKDTree(q[b_i]).query(q[s_i], k=1, distance_upper_bound=r_mm)
        ok = np.isfinite(dist)
        if not ok.any():
            continue
        s_i, dist, tgt = s_i[ok], dist[ok], lab[b_i[nn[ok]]]
        # per small cluster: the closest hit pair (ties: best stable rank of the small hit)
        o = np.lexsort((rank[ia[s_i]], dist, lab[s_i]))
        first = np.ones(len(o), dtype=bool)
        first[1:] = lab[s_i[o]][1:] != lab[s_i[o]][:-1]
        src, dst = lab[s_i[o[first]]], tgt[o[first]]
        remap = np.arange(len(cl_E), dtype=np.int64)
        remap[src] = dst
        out[ia] = remap[lab]
    return out


def _link_ecal_hcal(label: np.ndarray, pts: np.ndarray, E: np.ndarray, det: np.ndarray, rank: np.ndarray, r_mm: float) -> np.ndarray:
    """Merge each HCAL cluster into the ECAL cluster nearest to it (closest hit pair within
    r_mm), if that ECAL cluster carries more energy. One ECAL cluster may absorb several HCAL
    clusters; HCAL clusters never chain through each other."""
    is_ecal = np.isin(det, (9, 10, 11))
    is_hcal = np.isin(det, (12, 13, 14))
    ie, ih = np.nonzero(is_ecal)[0], np.nonzero(is_hcal)[0]
    if len(ie) == 0 or len(ih) == 0:
        return label
    dist, nn = cKDTree(pts[ie]).query(pts[ih], k=1, distance_upper_bound=r_mm)
    ok = np.isfinite(dist)
    if not ok.any():
        return label
    h_lab = label[ih[ok]]
    e_lab = label[ie[nn[ok]]]
    d_ok = dist[ok]
    # per HCAL cluster: the closest (hit-pair distance) ECAL cluster
    o = np.lexsort((rank[ih[ok]], d_ok, h_lab))
    first = np.ones(len(o), dtype=bool)
    first[1:] = h_lab[o][1:] != h_lab[o][:-1]
    pick = o[first]
    uniq, inv = np.unique(label, return_inverse=True)
    cl_E = np.bincount(inv, weights=E)
    pos_h = np.searchsorted(uniq, h_lab[pick])
    pos_e = np.searchsorted(uniq, e_lab[pick])
    take = cl_E[pos_e] > cl_E[pos_h]
    remap = dict(zip(h_lab[pick][take].tolist(), e_lab[pick][take].tolist()))
    if not remap:
        return label
    out = label.copy()
    src = np.fromiter(remap.keys(), np.int64)
    dst = np.fromiter(remap.values(), np.int64)
    srt = np.argsort(src)
    src, dst = src[srt], dst[srt]
    pos = np.clip(np.searchsorted(src, out), 0, len(src) - 1)
    hit = (src[pos] == out) & is_hcal
    out[hit] = dst[pos[hit]]
    return out
