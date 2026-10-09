# Shared helpers of the ColliderML calorimeter clusterers: clustering.py's radius-graph
# algorithms (union_find, bfs, bfs_merge) and clue.py's CLUE. Holds the deterministic
# stable hit rank, union-find primitives, rank-ordered relabelling, and the cluster output
# assembly (17-wide feature matrix, dominant region, hit->cluster COO). The properties every
# clusterer enforces (determinism, truth-blindness, energy conservation, ...) are listed in
# clustering.py.
import numba
import numpy as np

from mlpf.data.target_building import SparseMatrixCOO


def stable_rank(E: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # stable per-event sort rank: (-E, x, y, z, original index)
    order = np.lexsort((np.arange(len(x)), z, y, x, -E))
    stable_rank = np.empty(len(x), dtype=np.int64)
    stable_rank[order] = np.arange(len(x))
    return stable_rank


@numba.njit
def uf_find(uf: np.ndarray, v: int) -> int:
    root = v
    while uf[root] != root:
        root = uf[root]
    while uf[v] != root:
        uf[v], v = root, uf[v]
    return root


@numba.njit
def resolve_roots(parent: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Union-find root of every entry of idx."""
    out = np.empty(len(idx), np.int64)
    for k in range(len(idx)):
        out[k] = uf_find(parent, idx[k])
    return out


def relabel_by_rank(label: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Dense 0..n-1 cluster ids ordered by each cluster's best (smallest) stable rank."""
    uniq, inv = np.unique(label, return_inverse=True)
    best = np.full(len(uniq), len(label) + 10, np.int64)
    np.minimum.at(best, inv, rank)
    order = np.argsort(best, kind="stable")
    remap = np.empty(len(uniq), np.int64)
    remap[order] = np.arange(len(uniq))
    return remap[inv].astype(np.int64)


def build_features(
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


def cluster_region(cluster_of: np.ndarray, detector: np.ndarray) -> np.ndarray:
    """Region code holding most of each cluster's hits (by hit count, not energy; ties go to
    the lowest region code). Audit output, used for neutral-hadron PID forcing."""
    n_clusters = int(cluster_of.max()) + 1 if len(cluster_of) else 0
    if n_clusters == 0:
        return np.zeros(0, dtype=np.int64)
    region_codes_sorted = np.unique(detector)
    region_compact = np.searchsorted(region_codes_sorted, detector)
    nhits_by_creg = np.zeros((n_clusters, len(region_codes_sorted)), dtype=np.float64)
    np.add.at(nhits_by_creg, (cluster_of, region_compact), np.ones(len(detector), dtype=np.float64))
    membership_eps = np.zeros_like(nhits_by_creg)
    np.add.at(membership_eps, (cluster_of, region_compact), 1e-12)
    return region_codes_sorted[np.argmax(nhits_by_creg + membership_eps, axis=1)]


def make_hit_to_cluster(cluster_of: np.ndarray) -> SparseMatrixCOO:
    """COO (hit_idx, cluster_idx, 1.0) — dense assignment, no partial hits."""
    return (
        np.arange(len(cluster_of), dtype=np.int64),
        cluster_of.astype(np.int64),
        np.ones(len(cluster_of), dtype=np.float32),
    )
