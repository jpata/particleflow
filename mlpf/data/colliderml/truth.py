# Truth selection and per-event attribution for ColliderML release 1.
#
# Policy (written in code, not user-facing; there is deliberately no runtime switch):
#
# TARGETS (gen_features)
#   Leaf primary particles: primary == True with no primary children, except that a leaf pi0
#   which Geant4 decayed is replaced by its direct decay products (gamma gamma, or gamma e+ e-),
#   as in key4hep, where the generator decays the pi0 and the photons are status 1 (the
#   release's hard-scatter generator leaves pi0 stable; its pileup generator already decays
#   them). Neutrinos (|PDG| in {12, 14, 16}) are excluded. A leaf must additionally be
#   *visible*, i.e.:
#     - track-visible: it owns >= 20% of the hits of a reconstructed track
#       (TRACK_HIT_FRACTION_MIN; tracker_hits.particle_id joined through tracks.hit_ids and
#       walked to the leaf — the analogue of key4hep's SiTracksMCTruthLink fractions), or
#     - calo-visible: it passes the shared key4hep rule (target_building.visibility_mask):
#       attributed calibrated deposit / particle energy > 0.10, or — charged only, the
#       MIP catcher — attributed calibrated deposit > 0.5 GeV.
#
# GEN REFERENCE (genref_features, feeding genmet/genjet)
#   The *measurable* leaf primaries: any attributed calibrated deposit or any track hit share,
#   minus neutrinos. Deliberately pre-visibility and pre-allocator, so the gen jets/MET are an
#   independent truth-level reference instead. Analogue of key4hep's status-1 AND propagated set.
#
# TRUTH LINKS (attribution)
#   Calo deposit owner ids (contrib_particle_ids) and tracker-hit owner ids
#   (tracker_hits.particle_id) of *any* generator or simulated particle are walked up
#   parent_id until a leaf is reached, and attributed entirely to that leaf — so a visible
#   parent and its visible descendants are the *same* target. Contributions the release
#   credits directly to a split pi0 (unrecorded soft decay photons) go to its most energetic
#   decay product.
#
# MERGES
#   Handled by the shared allocator (target_building.assign_genparticles_to_obj_and_merge);
#   nothing is dropped silently: surviving + dropped reproduces the selected input energy
#   inside the documented tolerance.
#
# PILEUP
#   ispu = vertex_primary != 1.
#
# STATUS FIELDS (constants, not release-table columns)
#   generatorStatus = 1 for every target (they are generator-level final particles);
#   simulatorStatus = 0x01000000 (bit 24, "endpoint": every target was Geant4-simulated,
#   matching key4hep's flagging — see the comment at gen_features below).
from typing import Any, Dict, Tuple

import awkward as ak
import numpy as np

from mlpf.data.target_building import SparseMatrixCOO, visibility_mask

NEUTRINO_PDGS = {12, 14, 16}

# Region-specific multiplicative constants that take the raw ColliderML `contrib_energies`
# (sampling-fraction deposits, = `total_energy` in the source) to the calibrated-GeV scale
# that the rest of the pipeline expects. These are the ODD default scales shipped by
# colliderml.physics.calibration (see https://opendatadetector.github.io/ColliderML/library/physics.html).
DEFAULT_CALIBRATION = {
    9: 38.7,  # ECAL endcap -
    10: 37.5,  # ECAL barrel
    11: 38.7,  # ECAL endcap +
    12: 46.9,  # HCAL endcap -
    13: 45.0,  # HCAL barrel
    14: 46.9,  # HCAL endcap +
}

# A leaf must own at least this share of a track's hits to be regarded as track-visible,
# mirroring the key4hep gp_in_tracker cut (`gp_to_track >= 0.2` in key4hep/postprocessing.py).
TRACK_HIT_FRACTION_MIN = 0.2

# Gen-reference ("measurable truth") admission: a leaf primary enters the gen jet/MET
# reference if the detector registered it at all — any attributed calibrated calo deposit, or
# any share of a reconstructed track's hits. This is the ColliderML analogue of the key4hep
# `simulatorStatus` propagated bits (24-27).
GENREF_DEPOSIT_MIN = 0.0  # GeV, calibrated attributed calo deposit
GENREF_TRACK_FRACTION_MIN = 0.0  # share of a track's hits


def _empty_coo() -> SparseMatrixCOO:
    return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))


def _build_parent_maps(particles_ev: Dict[str, Any]):
    # ak.to_numpy on these (schema-nullable) parquet columns returns MaskedArrays; the release
    # never has masked entries, so we strip to plain ndarrays once here and keep every downstream
    # path (numpy ops, awkward/fastjet in the converter) mask-free.
    pid = np.asarray(ak.to_numpy(particles_ev["particle_id"]))
    pdg = np.asarray(ak.to_numpy(particles_ev["pdg_id"]))
    parent = np.asarray(ak.to_numpy(particles_ev["parent_id"]))
    # `primary` is used only here (leaf = primary with no primary children); everything
    # downstream keys off `is_leaf`.
    primary = np.asarray(ak.to_numpy(particles_ev["primary"])).astype(bool)
    index_of: Dict[int, int] = {int(p): i for i, p in enumerate(pid)}
    parents_of_primary = {int(parent[i]) for i in np.where(primary)[0] if int(parent[i]) in index_of and primary[index_of[int(parent[i])]]}
    is_leaf = np.array([primary[i] and int(pid[i]) not in parents_of_primary for i in range(len(pid))], dtype=bool)
    redirect = _split_leaf_pi0(pid, pdg, parent, primary, np.asarray(ak.to_numpy(particles_ev["energy"])), is_leaf)
    return pid, pdg, parent, is_leaf, redirect


def _split_leaf_pi0(
    pid: np.ndarray, pdg: np.ndarray, parent: np.ndarray, primary: np.ndarray, energy: np.ndarray, is_leaf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace each leaf pi0 that Geant4 decayed by its decay products, in place on is_leaf.

    The release's generator leaves hard-scatter pi0 stable, so Geant4 decays them and the
    photons (or gamma e+ e- for Dalitz decays) are non-primary children of a leaf pi0. In
    key4hep the generator decays the pi0 (status 2) and the photons are the status-1 targets,
    so promote the direct children to leaves and unmark the pi0. A pi0 without recorded
    children (not simulated, or both photons below the recording threshold) stays a leaf.

    Returns the redirect (split pi0 ids -> id of the pi0's most energetic child, sorted by pi0
    id): the release credits deposits of unrecorded soft decay photons directly to the pi0 id,
    and the walk from a non-leaf pi0 would lose them.
    """
    no_redirect = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    pid64 = pid.astype(np.int64)
    pi0_leaf = is_leaf & (np.abs(pdg) == 111)
    if len(pid) == 0 or not pi0_leaf.any():
        return no_redirect
    order = np.argsort(pid64, kind="stable")
    pid_sorted = pid64[order]
    pos = np.clip(np.searchsorted(pid_sorted, parent.astype(np.int64)), 0, len(pid_sorted) - 1)
    parent_row = np.where(pid_sorted[pos] == parent.astype(np.int64), order[pos], -1)
    child = ~primary & (parent_row >= 0)
    child[child] = pi0_leaf[parent_row[child]]
    if not child.any():
        return no_redirect
    split_rows = np.unique(parent_row[child])
    is_leaf[split_rows] = False
    is_leaf[child] = True

    # most energetic child per split pi0 (ties: first child in table order)
    child_rows = np.nonzero(child)[0]
    by_energy = child_rows[np.lexsort((child_rows, -energy[child_rows], parent_row[child_rows]))]
    first = np.ones(len(by_energy), dtype=bool)
    first[1:] = parent_row[by_energy[1:]] != parent_row[by_energy[:-1]]
    best = by_energy[first]  # one per split pi0, ordered by parent row
    src = pid64[parent_row[best]]
    dst = pid64[best]
    o = np.argsort(src)
    return src[o], dst[o]


def _redirect_ids(ids: np.ndarray, redirect: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Replace every id found in redirect[0] by the matching redirect[1] entry."""
    src, dst = redirect
    ids = np.asarray(ids)
    if len(src) == 0 or len(ids) == 0:
        return ids
    ids64 = ids.astype(np.int64)
    pos = np.clip(np.searchsorted(src, ids64), 0, len(src) - 1)
    hit = src[pos] == ids64
    if not hit.any():
        return ids
    out = ids64.copy()
    out[hit] = dst[pos[hit]]
    return out


def walk_to_leaf_many(particle_ids: np.ndarray, pid: np.ndarray, is_leaf: np.ndarray, parent: np.ndarray, max_depth: int = 256) -> np.ndarray:
    """Vectorized walk_to_leaf over many ids: per id, the source row whose parent chain ends
    in a leaf (row index), else -1. The leaf is a leaf primary or a promoted pi0 decay product."""
    ids = np.asarray(particle_ids)
    if len(ids) == 0:
        return np.empty(0, np.int64)
    order = np.argsort(pid, kind="stable")
    pid_s = pid[order]

    def row_of(v):
        pos = np.clip(np.searchsorted(pid_s, v), 0, len(pid_s) - 1)
        return np.where(pid_s[pos] == v, order[pos], -1)

    cur = row_of(ids)
    res = np.full(len(ids), -1, np.int64)
    alive = cur >= 0
    for _ in range(max_depth):
        if not alive.any():
            break
        idx_alive = np.nonzero(alive)[0]
        cur_alive = cur[alive]
        leaf_now = is_leaf[cur_alive]
        res[idx_alive[leaf_now]] = cur_alive[leaf_now]
        cont = idx_alive[~leaf_now]
        if len(cont) == 0:
            break
        cur[cont] = row_of(parent[cur_alive[~leaf_now]])
        alive[cont[cur[cont] < 0]] = False
    return res


def calibration_factors(detector: np.ndarray, calibration: Dict[int, float]) -> np.ndarray:
    """Per-hit float32 calibration factor for an array of calo region codes.

    Raises on a region code missing from `calibration`, rather than silently giving its
    deposits weight 0 (the truth attribution and the clusterer input must agree).
    """
    detector = np.asarray(detector, dtype=np.int64)
    calib_map = np.full(max(calibration) + 1, np.nan, dtype=np.float32)
    for d, k in calibration.items():
        calib_map[d] = k
    if len(detector) == 0:
        return np.zeros(0, dtype=np.float32)
    known = (detector >= 0) & (detector < len(calib_map))
    known[known] = ~np.isnan(calib_map[detector[known]])
    if not np.all(known):
        raise ValueError(f"calo region code(s) {np.unique(detector[~known]).tolist()} have no calibration factor (known: {sorted(calibration)})")
    return calib_map[detector]


def _track_hit_fraction_links(
    tracks_ev: Dict[str, Any],
    tracker_ev: Dict[str, Any],
    leaf_row_to_local: np.ndarray,
    pid: np.ndarray,
    is_leaf: np.ndarray,
    parent: np.ndarray,
    n_leaf: int,
    redirect: Tuple[np.ndarray, np.ndarray],
) -> SparseMatrixCOO:
    """COO (leaf, track, fraction of the track's hits attributed to the leaf).

    Joins tracks.hit_ids (tracker-hit row per track) to tracker_hits.particle_id and walks each
    hit's particle up to its leaf primary; a (leaf, track) link weight is the share of the
    track's hits owned by that leaf — the ColliderML analogue of the key4hep
    SiTracksMCTruthLink hit-fraction weights used for gp_to_track.
    """
    n_track = len(ak.to_numpy(tracks_ev["majority_particle_id"]))
    n_hits_trk = np.asarray(ak.to_numpy(ak.num(tracks_ev["hit_ids"])), dtype=np.int64)
    flat_hids = np.asarray(ak.to_numpy(ak.flatten(tracks_ev["hit_ids"], axis=None)), dtype=np.int64)
    hit_pid = _redirect_ids(np.asarray(ak.to_numpy(tracker_ev["particle_id"])).astype(np.int64), redirect)

    # ignore out-of-range hit-row references defensively (release data should not contain any)
    in_range = (flat_hids >= 0) & (flat_hids < len(hit_pid))
    trk_of_hit = np.repeat(np.arange(n_track, dtype=np.int64), n_hits_trk)[in_range]
    pid_of_hit = hit_pid[flat_hids[in_range]]
    if len(pid_of_hit) == 0:
        return _empty_coo()

    # walk each unique hit particle once, broadcast over all hit entries
    uniq, inv = np.unique(pid_of_hit, return_inverse=True)
    leaf_of_uniq = walk_to_leaf_many(uniq, pid, is_leaf, parent)
    leaf_global = leaf_of_uniq[inv]
    tracked = leaf_global != -1
    leaf_local = leaf_row_to_local[leaf_global[tracked]]
    trk = trk_of_hit[tracked]

    # per (leaf, track) hit share; hits without a reachable leaf dilute the fractions. Count
    # the (leaf, track) pairs sparsely — a dense n_leaf x n_track matrix is ~GB at pileup —
    # in the same row-major order the dense np.nonzero produced.
    if len(trk) == 0:
        return _empty_coo()
    key = np.asarray(leaf_local, dtype=np.int64) * np.int64(n_track) + trk
    ukey, counts = np.unique(key, return_counts=True)
    li, ti = ukey // n_track, ukey % n_track
    frac = counts.astype(np.float64) / np.maximum(n_hits_trk[ti], 1).astype(np.float64)
    return li.astype(np.int64), ti.astype(np.int64), frac.astype(np.float32)


def compute_gen_tables(
    particles_ev: Dict[str, Any],
    calo_ev: Dict[str, Any],
    tracks_ev: Dict[str, Any],
    calibration: Dict[int, float],
    tracker_ev: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], SparseMatrixCOO, SparseMatrixCOO, Dict[str, np.ndarray]]:
    """Return gen_features, gp_to_hit, gp_to_track, genref_features for one event.

    gen_features keys follow the MLPF target layout (pt/eta/phi + charge etc.). Only visible leaves
    are retained (leaf primaries, with Geant4-decayed pi0 replaced by their decay products).
    Contributions whose parent chain never reaches a leaf cannot be attributed and are dropped
    (mostly Geant4 secondaries of generator-decayed K0S/Lambda/Sigma/Xi that interacted before
    their preassigned decay — key4hep's surrogate-ancestor case, not handled here).

    genref_features has the same layout but spans the *measurable* leaf primaries (any
    attributed calibrated deposit or track hit share, minus neutrinos) — the truth-level
    set used for the gen jet/MET reference.

    gp_to_track weights are per-track hit ownership fractions (key4hep SiTracksMCTruthLink
    analogue). gp_to_hit additionally carries zero-weight tracker-hit links (hit index is in the
    extended space: calo hits first, then tracker hits) feeding the hits-view PN marks.
    """

    pid, pdg, parent, is_leaf, redirect = _build_parent_maps(particles_ev)
    leaf_indices = np.where(is_leaf)[0]
    n_leaf = len(leaf_indices)
    # neutrino leaf primaries: excluded from both gen/genref so they can
    # never be assigned an element
    nu_leaf = np.isin(np.abs(pdg[leaf_indices]), list(NEUTRINO_PDGS))
    # global particle row -> leaf-local index (visible-only); -1 for non-leaf or not-visible rows
    leaf_row_to_local = np.full(len(is_leaf), -1, dtype=np.int64)
    leaf_row_to_local[leaf_indices] = np.arange(n_leaf, dtype=np.int64)

    # ---------------- calo hits ----------------
    n_hit = len(ak.to_numpy(calo_ev["x"]))
    cid_flat = _redirect_ids(np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_particle_ids"], axis=None))).ravel(), redirect)
    ce_flat = np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_energies"], axis=None))).ravel()
    nctr_flat = np.asarray(ak.to_numpy(ak.num(calo_ev["contrib_particle_ids"]))).ravel()
    hit_idx_flat = np.repeat(np.arange(n_hit, dtype=np.int64), nctr_flat)

    # attribute each contribution to a leaf local index; walk each *unique* source id once and
    # broadcast the result to all of its occurrences (contributions number ~10^4-10^5 per event,
    # distinct source ids a few hundred, so per-unique dict caching replaced repeated walks).
    leaf_of_contrib = np.full(len(cid_flat), -1, dtype=np.int64)
    if len(cid_flat):
        uniq, inv = np.unique(cid_flat, return_inverse=True)
        uniq_leaf_id = walk_to_leaf_many(uniq, pid, is_leaf, parent)
        leaf_id_flat = uniq_leaf_id[inv]
        valid_flat = leaf_id_flat != -1
        leaf_of_contrib[valid_flat] = leaf_row_to_local[leaf_id_flat[valid_flat]]

    det = np.asarray(ak.to_numpy(calo_ev["detector"]))
    det_flat = np.repeat(det, nctr_flat)
    # weight = raw contrib energy * region calibration (vectorized over the 6 named regions)
    calib_flat = calibration_factors(det_flat, calibration)
    weight_flat = ce_flat.astype(np.float32, copy=False) * calib_flat

    m_valid = leaf_of_contrib >= 0
    gp_to_hit: SparseMatrixCOO = (
        leaf_of_contrib[m_valid].astype(np.int64),
        hit_idx_flat[m_valid].astype(np.int64),
        weight_flat[m_valid].astype(np.float32),
    )

    # ---------------- tracks ----------------
    # Link weights are per-track hit-ownership fractions (matching the key4hep
    # SiTracksMCTruthLink semantics). The visibility cut below applies the
    # key4hep gp_in_tracker rule (>= TRACK_HIT_FRACTION_MIN of a track's hits).
    gp_to_track: SparseMatrixCOO = _track_hit_fraction_links(tracks_ev, tracker_ev, leaf_row_to_local, pid, is_leaf, parent, n_leaf, redirect)

    # ---------------- per-leaf features ----------------
    # np.asarray strips nominal parquet-nullable masks (see _build_parent_maps note)
    px = np.asarray(ak.to_numpy(particles_ev["px"]))
    py = np.asarray(ak.to_numpy(particles_ev["py"]))
    pz = np.asarray(ak.to_numpy(particles_ev["pz"]))
    E = np.asarray(ak.to_numpy(particles_ev["energy"]))
    charge = np.asarray(ak.to_numpy(particles_ev["charge"])).astype(np.float32)
    pt = np.hypot(px, py)
    pz_safe = np.where(np.abs(pz) < 1e-9, 1e-9, pz)
    eta = np.arcsinh(pz_safe / np.where(pt == 0.0, 1e-9, pt))
    phi = np.arctan2(py, px)
    vertex_primary = np.asarray(ak.to_numpy(particles_ev["vertex_primary"]))

    # visibility: either owns >= TRACK_HIT_FRACTION_MIN of a track's hits (the key4hep
    # gp_in_tracker cut), or passes the shared calo rule (visibility_mask: attributed-deposit
    # fraction / MIP absolute term)
    trk_frac_max = np.zeros(n_leaf, dtype=np.float64)
    if len(gp_to_track[0]):
        np.maximum.at(trk_frac_max, np.asarray(gp_to_track[0], dtype=np.int64), np.asarray(gp_to_track[2], dtype=np.float64))
    vis_track = trk_frac_max >= TRACK_HIT_FRACTION_MIN
    agg = np.zeros(n_leaf, dtype=np.float64)
    if len(gp_to_hit[0]):
        np.add.at(agg, gp_to_hit[0], np.asarray(gp_to_hit[2], dtype=np.float64))
    vis_calo = visibility_mask(agg, E[leaf_indices], np.abs(charge[leaf_indices]) > 0)
    keep = (vis_track | vis_calo) & ~nu_leaf & (E[leaf_indices] > 0.0)

    # remap memberships from leaf-local -> filtered (target) ordering
    local_to_filtered = -np.ones(n_leaf, dtype=np.int64)
    local_to_filtered[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    gp_to_hit_filtered = ([], [], [])
    if len(gp_to_hit[0]):
        remapped = local_to_filtered[np.asarray(gp_to_hit[0])]
        m = remapped >= 0
        gp_to_hit_filtered = (remapped[m], np.asarray(gp_to_hit[1], dtype=np.int64)[m], np.asarray(gp_to_hit[2], dtype=np.float32)[m])
    gp_to_track_filtered = ([], [], [])
    if len(gp_to_track[0]):
        remapped = local_to_filtered[np.asarray(gp_to_track[0])]
        m = remapped >= 0
        gp_to_track_filtered = (remapped[m], np.asarray(gp_to_track[1], dtype=np.int64)[m], np.asarray(gp_to_track[2], dtype=np.float32)[m])

    # ---------------- tracker hits (for the hits view) ----------------
    # Zero-weight links in the extended hit index space (calo hits first, then tracker hits),
    # so the shared allocator's post-merge filtering delivers target-owned tracker hits in the
    # cleaned row order — the hits view's PN marks ride on those. Zero weight means they never
    # enter the exclusive-claim race (CSR eliminate_zeros drops them). In key4hep tracker-hit
    # eDeps simply lose that race to GeV-scale calo deposits.
    th_pid = _redirect_ids(np.asarray(ak.to_numpy(tracker_ev["particle_id"])).astype(np.int64), redirect)
    n_th = len(th_pid)
    if n_th:
        uniq, inv = np.unique(th_pid, return_inverse=True)
        leaf_global = walk_to_leaf_many(uniq, pid, is_leaf, parent)[inv]
        gp_rows = np.full(n_th, -1, dtype=np.int64)
        ok = leaf_global != -1
        gp_rows[ok] = local_to_filtered[leaf_row_to_local[leaf_global[ok]]]
        m = gp_rows >= 0
        if np.any(m):
            add_cols = np.nonzero(m)[0] + n_hit  # tracker hits live after the calo hits
            gp_to_hit_filtered = (
                np.concatenate([np.asarray(gp_to_hit_filtered[0], dtype=np.int64), gp_rows[m]]),
                np.concatenate([np.asarray(gp_to_hit_filtered[1], dtype=np.int64), add_cols]),
                np.concatenate([np.asarray(gp_to_hit_filtered[2], dtype=np.float32), np.zeros(len(add_cols), dtype=np.float32)]),
            )

    # One row per leaf primary on the leaf axis; gen/genref are slices of this table under
    # their respective masks. The gp columns are the leaf-axis aggregates:
    #   gp_to_track  = max fraction of a track's hits owned by this leaf (0 if untracked)
    #   gp_to_cluster = total calibrated calo deposit attributed to this leaf (GeV)
    # (mirror key4hep/postprocessing.py's computation of the same two quantities, next to its
    # mask_visible_hit call; slicing them by `keep` equals the filtered-COO reductions, since
    # the filtering only drops rows that were cut anyway).
    # Simulator status: all ColliderML targets are Geant4-simulated (either the leaf itself
    # deposits, or a secondary from it does and we attribute it up the parent chain), so we
    # set bit 24 ("endpoint") to match how the key4hep postprocessing flags its targets.
    leaf_feats = {
        "PDG": pdg[leaf_indices].astype(np.float32),
        "charge": charge[leaf_indices],
        "pt": pt[leaf_indices].astype(np.float32),
        "eta": eta[leaf_indices].astype(np.float32),
        "phi": phi[leaf_indices].astype(np.float32),
        "energy": E[leaf_indices].astype(np.float32),
        "ispu": (np.asarray(vertex_primary[leaf_indices]) != 1).astype(np.float32),
        "generatorStatus": np.ones(n_leaf, dtype=np.float32),
        "simulatorStatus": np.full(n_leaf, 0x01000000, dtype=np.float32),
        "gp_to_track": trk_frac_max.astype(np.float32),
        "gp_to_cluster": agg.astype(np.float32),
        "jet_idx": np.full(n_leaf, -1.0, dtype=np.float32),
    }
    gen_features = {k: v[keep] for k, v in leaf_feats.items()}

    # ---------------- gen reference (measurable truth) ------------------------
    # Any registration at all, before the visibility thresholds; a strict superset of `keep`
    # with the default constants (visibility implies a deposit or track share above these
    # minima), minus the same neutrino / E>0 exclusions.
    measurable = ((agg > GENREF_DEPOSIT_MIN) | (trk_frac_max > GENREF_TRACK_FRACTION_MIN)) & ~nu_leaf & (E[leaf_indices] > 0.0)
    genref_features = {k: v[measurable] for k, v in leaf_feats.items()}
    return gen_features, gp_to_hit_filtered, gp_to_track_filtered, genref_features
