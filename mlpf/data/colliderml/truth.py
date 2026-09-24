# Truth selection and per-event attribution for ColliderML release 1.
#
# Policy (written in code, not user-facing; there is deliberately no runtime switch):
#
# TARGETS (gen_features)
#   Leaf primary particles: primary == True with no primary children. Neutrinos
#   (|PDG| in {12, 14, 16}) are excluded. A leaf must additionally be *visible*, i.e.:
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
#   parent_id until a leaf primary is reached, and attributed entirely to that leaf — so a
#   visible parent and its visible descendants are the *same* target.
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
    index_of = {int(p): i for i, p in enumerate(pid)}
    parents_of_primary = {int(parent[i]) for i in np.where(primary)[0] if int(parent[i]) in index_of and primary[index_of[int(parent[i])]]}
    is_leaf = np.array([primary[i] and int(pid[i]) not in parents_of_primary for i in range(len(pid))], dtype=bool)
    return pid, pdg, parent, is_leaf, index_of


def walk_to_leaf(particle_id: int, index_of: Dict[int, int], pid: np.ndarray, is_leaf: np.ndarray, parent: np.ndarray, max_depth: int = 256) -> int:
    """Source index of the leaf primary that contains `particle_id`, or -1."""
    depth = 0
    i = index_of.get(int(particle_id), -1)
    while i != -1 and depth < max_depth:
        if is_leaf[i]:
            return i
        i = index_of.get(int(parent[i]), -1)
        depth += 1
    return -1


def _track_hit_fraction_links(
    tracks_ev: Dict[str, Any],
    tracker_ev: Dict[str, Any],
    leaf_row_to_local: np.ndarray,
    index_of: Dict[int, int],
    pid: np.ndarray,
    is_leaf: np.ndarray,
    parent: np.ndarray,
    n_leaf: int,
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
    hit_pid = np.asarray(ak.to_numpy(tracker_ev["particle_id"])).astype(np.int64)

    # ignore out-of-range hit-row references defensively (release data should not contain any)
    in_range = (flat_hids >= 0) & (flat_hids < len(hit_pid))
    trk_of_hit = np.repeat(np.arange(n_track, dtype=np.int64), n_hits_trk)[in_range]
    pid_of_hit = hit_pid[flat_hids[in_range]]
    if len(pid_of_hit) == 0:
        return _empty_coo()

    # walk each unique hit particle once, broadcast over all hit entries
    uniq, inv = np.unique(pid_of_hit, return_inverse=True)
    leaf_of_uniq = np.asarray([walk_to_leaf(int(u), index_of, pid, is_leaf, parent) for u in uniq.tolist()], dtype=np.int64)
    leaf_global = leaf_of_uniq[inv]
    tracked = leaf_global != -1
    leaf_local = leaf_row_to_local[leaf_global[tracked]]
    trk = trk_of_hit[tracked]

    # per (leaf, track) hit share; hits without a reachable leaf dilute the fractions
    counts = np.zeros((n_leaf, n_track), dtype=np.float64)
    np.add.at(counts, (leaf_local, trk), 1.0)
    frac = counts / np.maximum(n_hits_trk, 1).astype(np.float64)[None, :]
    li, ti = np.nonzero(frac)
    return li.astype(np.int64), ti.astype(np.int64), frac[li, ti].astype(np.float32)


def compute_gen_tables(
    particles_ev: Dict[str, Any],
    calo_ev: Dict[str, Any],
    tracks_ev: Dict[str, Any],
    calibration: Dict[int, float],
    tracker_ev: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], SparseMatrixCOO, SparseMatrixCOO]:
    """Return gen_features, gp_to_hit, gp_to_track, genref_features for one event.

    gen_features keys follow the MLPF target layout (pt/eta/phi + charge etc.). Only visible leaf
    primaries are retained. Contributions whose parent chain never reaches a leaf primary (stray
    Geant4 secondaries) cannot be attributed and are dropped.

    genref_features has the same layout but spans the *measurable* leaf primaries (any
    attributed calibrated deposit or track hit share, minus neutrinos) — the truth-level
    set used for the gen jet/MET reference.

    gp_to_track weights are per-track hit ownership fractions (key4hep SiTracksMCTruthLink
    analogue).
    """

    pid, pdg, parent, is_leaf, index_of = _build_parent_maps(particles_ev)
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
    cid_flat = np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_particle_ids"], axis=None))).ravel()
    ce_flat = np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_energies"], axis=None))).ravel()
    nctr_flat = np.asarray(ak.to_numpy(ak.num(calo_ev["contrib_particle_ids"]))).ravel()
    hit_idx_flat = np.repeat(np.arange(n_hit, dtype=np.int64), nctr_flat)

    # attribute each contribution to a leaf local index; walk each *unique* source id once and
    # broadcast the result to all of its occurrences (contributions number ~10^4-10^5 per event,
    # distinct source ids a few hundred, so per-unique dict caching replaced repeated walks).
    leaf_of_contrib = np.full(len(cid_flat), -1, dtype=np.int64)
    if len(cid_flat):
        uniq, inv = np.unique(cid_flat, return_inverse=True)
        uniq = uniq.tolist()
        uniq_leaf_id = [walk_to_leaf(int(u), index_of, pid, is_leaf, parent) for u in uniq]
        leaf_id_flat = np.asarray(uniq_leaf_id, dtype=np.int64)[inv]
        valid_flat = leaf_id_flat != -1
        leaf_of_contrib[valid_flat] = leaf_row_to_local[leaf_id_flat[valid_flat]]

    det = np.asarray(ak.to_numpy(calo_ev["detector"]))
    det_flat = np.repeat(det, nctr_flat)
    # weight = raw contrib energy * region calibration (vectorized over the 6 named regions)
    calib_map = np.zeros(max(calibration) + 1, dtype=np.float32)
    for d, k in calibration.items():
        calib_map[d] = k
    calib_flat = calib_map[det_flat]
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
    gp_to_track: SparseMatrixCOO = _track_hit_fraction_links(tracks_ev, tracker_ev, leaf_row_to_local, index_of, pid, is_leaf, parent, n_leaf)

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
