# ColliderML release 1 -> MLPF parquet converter (clustered view).
#
# For each source shard this produces one MLPF-format parquet with the same record layout as
# the key4hep converters write:
#   X_track, X_cluster, ytarget_track, ytarget_cluster,
#   X_hit_tracker, ytarget_hit_tracker  (raw tracker hits; PN marks for target-owned hits),
#   X_hit_calo,  ytarget_hit_calo       (raw calo hits; also hosts the exclusive target rows),
#   genmet, genjet, targetjet, event_id
# genmet/genjet are clustered from the *measurable truth* set (pre-visibility, pre-allocator,
# see truth.py genref_features); targetjet from the ytarget representatives.
# (ycand is zero-filled by the TFDS builder because ColliderML has no PF-candidate baseline.)
#
# The converter runs the shared allocator (mlpf.data.target_building) over one EventData built
# from the raw ColliderML tables, so truth merge/drop accounting and class forcing are exactly
# the ones key4hep already uses.
import argparse
import gc
import os
import resource
from pathlib import Path
from typing import Any, Dict, List

import awkward as ak
import fastjet
import numpy as np
import pyarrow.parquet as pq
import tqdm
import vector

from mlpf.conf import EDM4HEP
from mlpf.data.colliderml.clustering import cluster_event
from mlpf.data.colliderml.reader import iter_event_records, shard_paths
from mlpf.data.colliderml.tracks import track_features_cml
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, calibration_factors, compute_gen_tables
from mlpf.data.target_building import (
    EventData,
    assign_genparticles_to_obj_and_merge,
    assign_to_recoobj,
    compute_jets,
    compute_met,
    map_charged_to_neutral,
    map_neutral_to_charged,
    map_pdgid_to_candid,
    sanitize,
)

# pp @ 14 TeV: anti-kt R=0.4 with a 3 GeV jet seed cut — the CMS convention (their
# postprocessing uses jet_ptcut=3); key4hep uses 5 GeV (ee has much less soft energy).
# Must match the cluster/hit JET_CONFIG entries for colliderml in mlpf/conf.py.
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
jet_ptcut = 3.0

particle_feature_order = [
    "PDG",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "ispu",
    "generatorStatus",
    "simulatorStatus",
    "gp_to_track",
    "gp_to_cluster",
    "jet_idx",
    "particle_number",
]
track_feature_order = EDM4HEP.TrackFeatures.get_names()  # 16 names; the union layout fills 13
cluster_feature_order = EDM4HEP.ClusterFeatures.get_names()  # 17 names
# EDM4hep hit layout used elsewhere in the codebase (X_hit already includes `type`), shared by
# X_hit_calo (the raw calo hits we also cluster over) and X_hit_tracker (the raw tracker hits).
_hit_feature_order = EDM4HEP.HitFeatures.get_names()  # 15 names


def _event_record_one(
    event_id: int,
    particles_ev: ak.Record,
    tracks_ev: ak.Record,
    calo_ev: ak.Record,
    tracker_ev: ak.Record,
    algorithm: str = "bfs_merge",
    merge_frac: float | None = None,
    event_index: int = -1,
    shard_label: str = "",
) -> Dict[str, Any]:
    """Convert one ColliderML event into the MLPF record fields.

    merge_frac=None takes cluster_event's per-algorithm default (DEFAULT_MERGE_FRAC for
    bfs_merge), the same resolution as the CLI.
    """
    # raw inputs
    hit_x = ak.to_numpy(calo_ev["x"]).astype(np.float32)
    hit_y = ak.to_numpy(calo_ev["y"]).astype(np.float32)
    hit_z = ak.to_numpy(calo_ev["z"]).astype(np.float32)
    hit_e = ak.to_numpy(calo_ev["total_energy"]).astype(np.float32)
    hit_det = ak.to_numpy(calo_ev["detector"]).astype(np.int64)
    n_hit = len(hit_x)
    n_track = len(ak.to_numpy(tracks_ev["d0"]))

    # -- truth + attribution -------------------------------------------------
    gen_features, gp_to_hit, gp_to_track, genref_features = compute_gen_tables(
        particles_ev, calo_ev, tracks_ev, DEFAULT_CALIBRATION, tracker_ev=tracker_ev
    )

    # -- gen reference (measurable truth): genmet / genjet ---------------------
    # Built from genref_features — the pre-visibility, pre-allocator measurable set (any
    # attributed calibrated deposit or track hit share, minus neutrinos; the key4hep
    # status-1-and-propagated analogue, see truth.py). Deliberately independent of the
    # allocator so the gen jets/MET stay a truth-level reference instead of restating the
    # target.
    genref_p4 = vector.awk(
        ak.zip(
            {
                "px": genref_features["pt"] * np.cos(genref_features["phi"]),
                "py": genref_features["pt"] * np.sin(genref_features["phi"]),
                # pz from eta and pt
                "pz": genref_features["pt"] * np.sinh(genref_features["eta"]),
                "energy": genref_features["energy"],
            }
        )
    )
    genmet = float(compute_met(ak.unflatten(genref_p4, [len(genref_p4)], axis=0))[0])
    if len(genref_p4):
        genjet_np = compute_jets(genref_p4, min_pt=jet_ptcut, jetdef=jetdef)
        genjet_np = np.array([genjet_np.pt, genjet_np.eta, genjet_np.phi, genjet_np.energy]).T.astype(np.float32)
    else:
        genjet_np = np.zeros((0, 4), dtype=np.float32)

    # -- tracks -> features ---------------------------------------------------
    tfeat = track_features_cml(tracks_ev, tracker_ev)

    # -- clusters via the truth-blind spatial clusterer ------------------------
    hit_e_calibrated = hit_e * calibration_factors(hit_det, DEFAULT_CALIBRATION)
    cluster_of, cl_feats, hit_to_cluster, cluster_region = cluster_event(
        hit_x, hit_y, hit_z, hit_e_calibrated, hit_det, algorithm=algorithm, merge_frac=merge_frac
    )
    n_cluster = len(cl_feats)

    # -- build EventData and run the shared allocator --------------------------
    # The allocator indexes hits in the extended calo-then-tracker space (truth.py appends
    # zero-weight tracker links with hit index offset by n_hit); the feature filler just needs
    # that total length.
    n_tracker = len(ak.to_numpy(tracker_ev["x"]))
    gpdata = EventData(
        gen_features,
        {"type": np.zeros(n_hit + n_tracker, dtype=np.float32)},  # filler; only the length is read
        {"type": np.zeros(n_cluster, dtype=np.float32)},
        {"type": tfeat["type"]},
        gp_to_hit,
        gp_to_track,
        hit_to_cluster,
        (np.array([]), np.array([])),
    )
    gpdata_cleaned, gp_to_obj, gp_to_hit_idx, track_to_gp_inclusive, cluster_to_gp_inclusive, hit_to_gp_inclusive = (
        assign_genparticles_to_obj_and_merge(gpdata)
    )
    n_gp = len(gpdata_cleaned.gen_features["PDG"])

    # exclusive-average + fill the canonical particle features, exactly like the key4hep
    # stage (lines ~1643-1694 in the pre-extraction postprocessing.py).
    trk_to_gp_excl_map = {itr: igp for igp, itr in enumerate(gp_to_obj[:, 0]) if itr != -1}
    clst_to_gp_excl_map = {icl: igp for igp, icl in enumerate(gp_to_obj[:, 1]) if icl != -1}

    used_gps = np.zeros(n_gp, dtype=np.int64)
    track_to_gp_exclusive = assign_to_recoobj(n_track, trk_to_gp_excl_map, used_gps)
    cluster_to_gp_exclusive = assign_to_recoobj(n_cluster, clst_to_gp_excl_map, used_gps)
    assert np.all(used_gps == 1), "every selected truth particle must own a track or a cluster"

    # gp_to_cluster follows the key4hep semantic: the particle's total attributed calibrated
    # deposit (key4hep computes (gp_to_hit * calohit_to_cluster).sum(axis=1); here every hit
    # lives in exactly one cluster, so the row sum is just the total — which truth.py already
    # carries as gen_features["gp_to_cluster"]).
    gps_canonical = np.zeros((n_gp, len(particle_feature_order)), dtype=np.float32)
    for igp in range(n_gp):
        p = np.abs(float(gpdata_cleaned.gen_features["PDG"][igp]))
        c = float(gpdata_cleaned.gen_features["charge"][igp])
        pid = map_pdgid_to_candid(p, c)
        if gp_to_obj[igp, 0] != -1:
            pid = map_neutral_to_charged(pid)
        elif gp_to_obj[igp, 1] != -1:
            pid = map_charged_to_neutral(pid)
        gp_phi = float(
            np.arctan2(
                gpdata_cleaned.gen_features["sin_phi"][igp],
                gpdata_cleaned.gen_features["cos_phi"][igp],
            )
        )
        gps_canonical[igp] = np.array(
            [
                pid,
                c,
                float(gpdata_cleaned.gen_features["pt"][igp]),
                float(gpdata_cleaned.gen_features["eta"][igp]),
                float(np.sin(gp_phi)),
                float(np.cos(gp_phi)),
                float(gpdata_cleaned.gen_features["energy"][igp]),
                float(gpdata_cleaned.gen_features["ispu"][igp]),
                float(gpdata_cleaned.gen_features["generatorStatus"][igp]),
                float(gpdata_cleaned.gen_features["simulatorStatus"][igp]),
                float(gpdata_cleaned.gen_features["gp_to_track"][igp]),
                float(gpdata_cleaned.gen_features["gp_to_cluster"][igp]),
                float(gpdata_cleaned.gen_features["jet_idx"][igp]),
                float(gpdata_cleaned.gen_features["particle_number"][igp]),
            ],
            dtype=np.float32,
        )
    # gp_to_track carries the max track-hit fraction owned (key4hep SiTracksMCTruthLink
    # analogue); gp_to_cluster carries the total calibrated deposit attributed to this particle
    # (key4hep parity).

    PN_IDX = particle_feature_order.index("particle_number")

    # -- scatter between exclusive/inclusive rows ------------------------------
    gps_track = np.zeros((n_track, gps_canonical.shape[1]), dtype=np.float32)
    m_excl = track_to_gp_exclusive != -1
    gps_track[m_excl] = gps_canonical[track_to_gp_exclusive[m_excl]]
    m_incl_only = (track_to_gp_inclusive != -1) & (~m_excl)
    gps_track[m_incl_only, PN_IDX] = gps_canonical[track_to_gp_inclusive[m_incl_only], PN_IDX]

    gps_cluster = np.zeros((n_cluster, gps_canonical.shape[1]), dtype=np.float32)
    m_excl = cluster_to_gp_exclusive != -1
    gps_cluster[m_excl] = gps_canonical[cluster_to_gp_exclusive[m_excl]]
    gps_cluster[:, 1] = 0.0  # charged targets belong to tracks, clusters carry charge-0
    m_incl_only = (cluster_to_gp_inclusive != -1) & (~m_excl)
    gps_cluster[m_incl_only, PN_IDX] = gps_canonical[cluster_to_gp_inclusive[m_incl_only], PN_IDX]

    # two-sided target-energy accounting (exclusives only, avoiding double count). Sum in
    # float64: float32 accumulation noise grows past the 1e-2 tolerance for high-multiplicity
    # (pileup) events.
    assert (
        abs(
            np.sum(gps_track[:, 6].astype(np.float64))
            + np.sum(gps_cluster[:, 6].astype(np.float64))
            - np.sum(np.asarray(gpdata_cleaned.gen_features["energy"], dtype=np.float64))
        )
        < 1e-2
    )

    # -- X features -------------------------------------------------------------
    # tracks carry the same perigee-derived parameters the key4hep EDM4hep converter reads
    # from the first track state: tanLambda (= 1/tan(theta)) and omega (= q/pT computed from
    # qop with the ODD 3 T solenoid, key4hep's track_pt inverted) in addition to d0/z0, plus
    # radiusOfInnermostHit (the AtFirstHit reference-point radius in key4hep) from the
    # track's own tracker hits. Width = 17 (union layout).
    X_track = np.zeros((n_track, 17), dtype=np.float32)
    X_track[:, 0] = 1.0
    X_track[:, 1] = np.asarray(tfeat["pt"], dtype=np.float32)
    X_track[:, 2] = np.asarray(tfeat["eta"], dtype=np.float32)
    X_track[:, 3] = np.asarray(tfeat["sin_phi"], dtype=np.float32)  # sin_phi
    X_track[:, 4] = np.asarray(tfeat["cos_phi"], dtype=np.float32)  # cos_phi
    X_track[:, 5] = np.asarray(tfeat["p"], dtype=np.float32)
    X_track[:, 6] = np.asarray(tfeat["d0"], dtype=np.float32)
    X_track[:, 7] = np.asarray(tfeat["z0"], dtype=np.float32)
    X_track[:, 8] = np.asarray(tfeat["theta"], dtype=np.float32)
    X_track[:, 9] = np.asarray(tfeat["qop"], dtype=np.float32)
    X_track[:, 10] = np.asarray(tfeat["tanLambda"], dtype=np.float32)  # = sinh(eta) = 1/tan(theta)
    X_track[:, 11] = np.asarray(tfeat["omega"], dtype=np.float32)  # [1/mm], key4hep convention
    X_track[:, 12] = np.asarray(tfeat["radiusOfInnermostHit"], dtype=np.float32)  # [mm]
    # cols 13..16 are the cluster-only slots of the union layout (num_hits, sigma_x, sigma_y,
    # sigma_z) and stay 0 for tracks, except n_meas in col 13 (the num_hits slot) as a proxy
    # for ndf
    X_track[:, 13] = np.asarray(tfeat["n_meas"], dtype=np.float32)

    X_cluster = np.asarray(cl_feats, dtype=np.float32) if n_cluster else np.zeros((0, 17), dtype=np.float32)
    ytarget_track = gps_track
    ytarget_cluster = gps_cluster
    sanitize(X_track)
    sanitize(X_cluster)
    sanitize(ytarget_track)
    sanitize(ytarget_cluster)

    # -- calo-hit view -----------------------------------------------------------
    # X_hit_calo rows follow EDM4hep hit layout:
    #   [ elemtype(2), et, eta, sin_phi, cos_phi, energy(=calibrated total_energy), x, y, z, time(0),
    #     subdetector(=region), type(0), system(0), side(0), layer(0) ]
    # ytarget_hit_calo rows use the canonical particle_feature_order layout with:
    #   * exactly one full-target (exclusive) row per truth particle on its max-deposit
    #     attributed hit (via the allocator's gp_to_hit_idx; key4hep parity), and
    #   * particle_number marks on every hit whose largest contributor is a kept target.
    X_hit_calo = np.zeros((n_hit, len(_hit_feature_order)), dtype=np.float32)
    X_hit_calo[:, 0] = 2.0
    pos_mag = np.sqrt(hit_x**2 + hit_y**2 + hit_z**2)
    pos_mag_safe = np.where(pos_mag < 1e-9, 1.0, pos_mag)
    px_h = (hit_x / pos_mag_safe) * hit_e_calibrated
    py_h = (hit_y / pos_mag_safe) * hit_e_calibrated
    pz_h = (hit_z / pos_mag_safe) * hit_e_calibrated
    X_hit_calo[:, 1] = np.sqrt(px_h**2 + py_h**2)
    eps = 1e-9
    X_hit_calo[:, 2] = 0.5 * np.log((hit_e_calibrated + pz_h + eps) / np.maximum(hit_e_calibrated - pz_h, eps))
    X_hit_calo[:, 3] = py_h / np.maximum(hit_e_calibrated, eps)
    X_hit_calo[:, 4] = px_h / np.maximum(hit_e_calibrated, eps)
    X_hit_calo[:, 5] = hit_e_calibrated
    X_hit_calo[:, 6] = hit_x
    X_hit_calo[:, 7] = hit_y
    X_hit_calo[:, 8] = hit_z
    X_hit_calo[:, 9] = 0.0  # time placeholder
    X_hit_calo[:, 10] = hit_det.astype(np.float32)
    X_hit_calo[:, 11] = 0.0

    ytarget_hit_calo = np.zeros((n_hit, len(particle_feature_order)), dtype=np.float32)
    # inclusive: a calo hit carries the particle_number of its largest contributor, if that
    # contributor is a kept target (the allocator's hit_to_gp_inclusive, already in cleaned
    # row order; -1 when the top contributor was merged away or dropped).
    # The extended hit axis puts calo hits first, so the calo slice is [:n_hit].
    calo_to_gp_incl = np.asarray(hit_to_gp_inclusive[:n_hit], dtype=np.int64)
    m_incl = calo_to_gp_incl != -1
    ytarget_hit_calo[m_incl, PN_IDX] = gps_canonical[calo_to_gp_incl[m_incl], PN_IDX]
    # One exclusive full-target row per particle on its max-deposit attributed calo hit,
    # which also makes the hit-level table the host of the target set for the
    # hits view (set-prediction's build_target_set reads exactly these rows). The one
    # exception: a target whose every hit was claim-stolen in the allocator's greedy race and
    # that owns no tracker hits (a fully-stolen neutral, e.g. a soft PU photon on a single
    # calo cell) gets no row and is absent from the hits-view target set.
    n_excl_written = 0
    for igp in range(n_gp):
        h = int(gp_to_hit_idx[igp])
        if h != -1:
            ytarget_hit_calo[h, :] = gps_canonical[igp]
            n_excl_written += 1
    X_hit_calo_sanitized = X_hit_calo.copy()
    ytarget_hit_calo_sanitized = ytarget_hit_calo.copy()
    sanitize(X_hit_calo_sanitized)
    sanitize(ytarget_hit_calo_sanitized)

    # raw tracker hits from the release (tracker_hits table, subdetector=3, "tracker" in the
    # key4hep HitFeatures convention). Hits view parity: tracker hits carry no energy, so they
    # only get inclusive particle_number marks (from the truth walk in truth.py); the one
    # exception is the exclusive fallback row below.
    tx = ak.to_numpy(tracker_ev["x"]).astype(np.float32)
    ty = ak.to_numpy(tracker_ev["y"]).astype(np.float32)
    tz = ak.to_numpy(tracker_ev["z"]).astype(np.float32)
    te = ak.to_numpy(tracker_ev["time"]).astype(np.float32)
    # n_tracker was already computed for the allocator's EventData
    assert n_tracker == len(tx)
    X_hit_tracker = np.zeros((n_tracker, len(_hit_feature_order)), dtype=np.float32)
    X_hit_tracker[:, 0] = 1.0  # elemtype: tracker
    # columns 1..5 (et/eta/sin_phi/cos_phi/E) stay 0: release-1 tracker hits carry no energy.
    # Consequence for the elementwise hits model: a target hosted on a tracker hit (the exclusive
    # fallback below, ~0.05% of targets) has no element pt/E to anchor log(target / element),
    # so its pt/E regression is lost there; set mode regresses absolute log(pt), log(E).
    X_hit_tracker[:, 6] = tx
    X_hit_tracker[:, 7] = ty
    X_hit_tracker[:, 8] = tz
    X_hit_tracker[:, 9] = te
    X_hit_tracker[:, 10] = 3.0  # subdetector = tracker
    ytarget_hit_tracker = np.zeros((n_tracker, len(particle_feature_order)), dtype=np.float32)
    if n_tracker:
        # Inclusive marks: tag every tracker hit a kept target made with the owner's
        # particle_number (tag only; the row stays background for training). Owners come from
        # the cleaned adjacency's tracker tail (hit index >= n_hit, offset back by n_hit; at
        # most one owner per hit, so this scatter cannot collide). hit_to_gp_inclusive can't
        # serve here because eliminate_zeros dropped the zero-weight tracker links.
        hit_to_gp_incl = np.asarray(gpdata_cleaned.genparticle_to_hit[0], dtype=np.int64)
        hit_to_hit_idx = np.asarray(gpdata_cleaned.genparticle_to_hit[1], dtype=np.int64)
        own_cleaned = np.full(n_tracker, -1, dtype=np.int64)
        m_trk = hit_to_hit_idx >= n_hit
        own_cleaned[hit_to_hit_idx[m_trk] - n_hit] = hit_to_gp_incl[m_trk]
        good = own_cleaned >= 0
        ytarget_hit_tracker[good, PN_IDX] = gps_canonical[own_cleaned[good], PN_IDX]

        # Exclusive fallback row: a target with no calo host (gp_to_hit_idx == -1, e.g. a MIP
        # muon or a soft particle whose deposits were all claim-stolen) would vanish from the
        # hits-view target set, so its target row is written on the innermost tracker
        # hit it owns. Sorting by (owner, radius, hit index) and keeping each owner's first hit
        # picks that innermost hit.
        needs_fallback = np.asarray(gp_to_hit_idx, dtype=np.int64) == -1
        cand = np.nonzero(good)[0]
        cand = cand[needs_fallback[own_cleaned[cand]]]
        if len(cand):
            order = np.lexsort((cand, np.hypot(tx[cand], ty[cand]), own_cleaned[cand]))
            cand_sorted = cand[order]
            owner_sorted = own_cleaned[cand_sorted]
            first = np.ones(len(cand_sorted), dtype=bool)
            first[1:] = owner_sorted[1:] != owner_sorted[:-1]
            ytarget_hit_tracker[cand_sorted[first], :] = gps_canonical[owner_sorted[first]]
            n_excl_written += int(first.sum())

    n_orphan = n_gp - n_excl_written
    if n_orphan:
        where = f" [{shard_label} event {event_index} (event_id {event_id})]" if shard_label else ""
        print(
            f"{n_orphan} of {n_gp} target(s) own no exclusive hit row (fully claim-stolen, no tracker hits);"
            f" they are absent from the hits-view target set{where}"
        )

    # -- target jets, gen jets, gen met ---------------------------------------
    ytarget_all = np.concatenate([ytarget_track, ytarget_cluster], axis=0)
    valid = ytarget_all[:, 0] != 0
    ytarget_valid = ytarget_all[valid]
    ytarget_constituents = -np.ones(n_track + n_cluster, dtype=np.int64)
    if len(ytarget_valid):
        y_p4 = vector.awk(
            ak.zip(
                {
                    "pt": ytarget_valid[:, 2],
                    "eta": ytarget_valid[:, 3],
                    "phi": np.arctan2(ytarget_valid[:, 4], ytarget_valid[:, 5]),
                    "energy": ytarget_valid[:, 6],
                }
            )
        )
        target_jets, target_jets_const = compute_jets(y_p4, min_pt=jet_ptcut, jetdef=jetdef, with_indices=True)
        target_jets_const = target_jets_const.to_list()
        sorted_jet_idx = ak.argsort(target_jets.pt, axis=-1, ascending=False).to_list()
        # map constituent->valid-array index; then project back to the full element axis
        idx_valid = np.where(valid)[0]
        for j_idx in sorted_jet_idx:
            for k in target_jets_const[j_idx]:
                ytarget_constituents[idx_valid[k]] = j_idx
        targetjet_np = np.array([target_jets.pt, target_jets.eta, target_jets.phi, target_jets.energy]).T.astype(np.float32)
    else:
        targetjet_np = np.zeros((0, 4), dtype=np.float32)
    ytarget_track[:, particle_feature_order.index("jet_idx")] = ytarget_constituents[: len(ytarget_track)]
    ytarget_cluster[:, particle_feature_order.index("jet_idx")] = ytarget_constituents[len(ytarget_track) :]

    return {
        "event_id": int(event_id),
        "X_track": X_track,
        "X_cluster": X_cluster,
        "ytarget_track": ytarget_track,
        "ytarget_cluster": ytarget_cluster,
        "X_hit_tracker": X_hit_tracker,
        "X_hit_calo": X_hit_calo_sanitized,
        # one int64 per calo hit: the owning cluster id (row in X_cluster), -1 if unassigned.
        # It's what the converter's clustering already computed; saving it unlocks per-cluster
        # hit visibility in the notebook and lets us audit the V2 rule without re-running it.
        "hit_to_cluster": cluster_of.astype(np.int64),
        "ytarget_hit_tracker": ytarget_hit_tracker,
        "ytarget_hit_calo": ytarget_hit_calo_sanitized,
        "genmet": np.float32(genmet),
        "genjet": genjet_np,
        "targetjet": targetjet_np,
    }


# Explicit per-event scalar fields (the only two in the record dict). Routing is by name, not
# by shape: a 1-D field whose first event has length 1 must not be mistaken for a scalar.
_SCALAR_FIELDS = {"event_id", "genmet"}


def _from_events(key: str, vals):
    """vals is a list of per-event numpy arrays; produce the arrow Array for this field."""
    import pyarrow as pa

    if key in _SCALAR_FIELDS:
        # stack scalars into a 1-D array so each row of the parquet column is one scalar,
        # not a wrapped list — readers then see `record[field][i]` as the number
        a = np.asarray([np.asarray(v).reshape(-1)[0] for v in vals])
        return pa.array(a)
    a0 = np.asarray(vals[0])
    if a0.ndim == 1:
        # ragged 1-D (e.g. hit_to_cluster): large_list per event
        counts = np.asarray([int(len(v)) for v in vals], dtype=np.int64)
        flat = np.concatenate([np.asarray(v).reshape(-1) for v in vals], axis=0)
        offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return pa.ListArray.from_arrays(pa.array(offs, type=pa.int64()), pa.array(flat))
    # 2-D rows per event: large_list of fixed_size_list[F]
    F = a0.shape[1]
    counts = np.asarray([int(len(v)) for v in vals], dtype=np.int64)
    flat = np.concatenate([np.asarray(v) for v in vals], axis=0)
    values = pa.array(flat.reshape(-1))
    inner = pa.FixedSizeListArray.from_arrays(values, F)
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return pa.ListArray.from_arrays(pa.array(offs, type=pa.int64()), inner)


def _batch_arrow_arrays(events_batch: List[Dict[str, Any]]):
    """Convert one batch of events into arrow arrays matching the existing schema.

    The shard is stored as a table whose columns are one value per event for scalar fields,
    or large_list<...<fixed_size_list<float>[F]>> over the variable-length rows. Each
    per-event row in an output field has shape (rows_n, F) with F fixed; we flatten the rows
    axis and accumulate counts for the ragged level, per field."""
    return [(k, _from_events(k, [r[k] for r in events_batch])) for k in events_batch[0]]


def _parquet_usable(ofn: Path, expected_rows: int | None = None) -> bool:
    """True iff the named parquet file exists, pyarrow can open it and (if given) it holds
    expected_rows events. A truncated write crashes pyarrow on the footer, so corrupted outputs
    from a killed mid-shard job are reconvertible instead of being mistaken for done by the
    resume logic; the row count does the same for a --num-events partial output."""
    if not os.path.isfile(ofn):
        return False
    try:
        import pyarrow.parquet as pq

        n_rows = pq.ParquetFile(ofn).metadata.num_rows
    except Exception:
        return False
    return expected_rows is None or n_rows == expected_rows


def _shard_done(ofn: Path, particles_fn: Path) -> bool:
    """A full-shard output is done iff it is readable and holds every source event."""
    return _parquet_usable(ofn, expected_rows=pq.ParquetFile(particles_fn).metadata.num_rows)


def process_one_file(
    particles_fn: Path,
    tracks_fn: Path,
    calo_fn: Path,
    tracker_fn: Path,
    ofn: Path,
    num_events: int = -1,
    shard_index: int = 0,
    job_total_shards: int = 1,
    algorithm: str = "bfs_merge",
    merge_frac: float | None = None,
) -> None:
    if num_events == -1 and _shard_done(ofn, particles_fn):
        print(f"[shard {shard_index + 1}/{job_total_shards}] {Path(ofn).name} already exists, skipping")
        return
    if os.path.isfile(ofn) and num_events == -1:
        # a corrupted leftover (e.g. from an OOM-killed job) or a partial --num-events output
        # just falls through to the convert + rename path below, which overwrites it atomically
        print(f"[shard {shard_index + 1}/{job_total_shards}] {Path(ofn).name} exists but is corrupted or incomplete; reconverting")

    # events per shard from the parquet metadata (1000 for pu0, 100 for pu200); tqdm uses it
    # for the ETA. num_events (debug cap) overrides.
    total = num_events if num_events != -1 else pq.ParquetFile(particles_fn).metadata.num_rows
    desc = f"[shard {shard_index + 1}/{job_total_shards}] {Path(ofn).name}"
    iter_events = iter_event_records(particles_fn, tracks_fn, calo_fn, tracker_fn)
    os.makedirs(os.path.dirname(ofn), exist_ok=True)
    # streaming write: one row group per BATCH events via ParquetWriter, then rename. Peak
    # memory is bounded by one buffered batch + Arrow's encode buffers, so BATCH must stay
    # small with high-multiplicity samples (pileup event outputs are ~100 MB each).
    # Row-group size has no effect on readback content, only on parquet layout.
    BATCH = 25

    writer = None
    inflight = Path(str(ofn) + ".inflight")
    if os.path.isfile(inflight):
        os.remove(inflight)  # leftover from a killed earlier attempt at this shard
    try:
        out: List[Dict[str, Any]] = []

        def _flush_batch() -> int:
            nonlocal writer
            n = len(out)
            if n == 0:
                return 0
            import pyarrow as pa
            import pyarrow.parquet as pq

            cols = _batch_arrow_arrays(out)
            table = pa.Table.from_arrays([c for _, c in cols], names=[k for k, _ in cols])
            if writer is None:
                writer = pq.ParquetWriter(inflight, table.schema, compression="snappy")
            writer.write_table(table)
            out.clear()
            # memory breadcrumb: per-rank peak RSS at each write point, so memory creep over
            # events can be attributed from the rank logs
            print(f"{desc}: flushed {n} events, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
            return n

        n_written = 0
        for i, ev in enumerate(
            tqdm.tqdm(
                iter_events,
                total=total,
                desc=desc,
                unit="event",
                ncols=100,
            )
        ):
            event = _event_record_one(
                ev["event_id"],
                ev["particles"],
                ev["tracks"],
                ev["calo_hits"],
                ev["tracker_hits"],
                algorithm=algorithm,
                merge_frac=merge_frac,
                event_index=i,
                shard_label=Path(ofn).name,
            )
            out.append(event)
            del event
            if len(out) >= BATCH:
                n_written += _flush_batch()
            if num_events != -1 and n_written + len(out) >= num_events:
                break
        n_written += _flush_batch()

        # drop the generator + its captured shard inputs before the footer write — the same
        # iterators-carry-tables pattern that the previous code released at the same point
        if "ev" in locals():
            del ev
        del iter_events
        gc.collect()

        if n_written == 0:
            print("no events; skipping")
            if writer is not None:
                writer.close()
            inflight.unlink(missing_ok=True)
            return

        print(f"{desc}: wrote {n_written} events to {ofn}")
        if writer is not None:
            writer.close()
        os.replace(inflight, ofn)
    except BaseException:
        if writer is not None:
            writer.close()
        # remove the partial .inflight so the resume logic (which globs *.parquet) does not
        # mistake it for a finished shard; a failed shard re-runs in full, rebuilt from the
        # same event data
        inflight.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="source directory with the ColliderML release tables")
    parser.add_argument(
        "--sample",
        type=str,
        default="ttbar",
        help="sample label prefix, e.g. ttbar_pu0 or ttbar_pu200 (tables are <sample>_{particles,tracks,calo_hits,tracker_hits})",
    )
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--shards", type=str, default="0:1", help="shard slice a:b (python slice semantics)")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=("union_find", "bfs", "bfs_merge"),
        default="bfs_merge",
        help="clustering algorithm (bfs_merge is the current default after tuning)",
    )
    parser.add_argument(
        "--merge-frac",
        type=float,
        default=None,
        help="merge threshold; None uses DEFAULT_MERGE_FRAC (0.25) for bfs_merge and 0.0 for the others",
    )
    parser.add_argument("--num-events", type=int, default=-1, help="cap number of events per shard (for tests)")
    args = parser.parse_args()

    shard_ids = args.shards.split(":")
    a, b = int(shard_ids[0]), int(shard_ids[1])
    pa = shard_paths(Path(args.input), args.sample, "particles")[a:b]
    tr = shard_paths(Path(args.input), args.sample, "tracks")[a:b]
    ch = shard_paths(Path(args.input), args.sample, "calo_hits")[a:b]
    # tracker_hits is required input: its per-hit truth feeds the gp->track hit-fraction links
    # (truth.py) and the X_hit_tracker geometry written for the validator.
    th = shard_paths(Path(args.input), args.sample, "tracker_hits")[a:b]
    if len(th) != len(pa):
        raise RuntimeError(
            f"tracker_hits shard count mismatch in the requested range: {len(th)} tracker_hits shards vs {len(pa)} particles shards "
            f"(expected matching four-table layout under {Path(args.input) / f'{args.sample}_tracker_hits'})"
        )

    from mlpf.data.colliderml.clustering import DEFAULT_MERGE_FRAC

    merge_frac = args.merge_frac if args.merge_frac is not None else (DEFAULT_MERGE_FRAC if args.algorithm == "bfs_merge" else 0.0)

    n_files = len(pa)
    print(f"planning {n_files} shards (range {args.shards}) -> {args.outpath} with algorithm={args.algorithm} merge_frac={merge_frac}")
    n_done = 0
    for i, (p, t, c) in enumerate(zip(pa, tr, ch)):
        out_name = Path(args.outpath) / (p.stem + ".parquet")
        # --num-events is a debug cap: always (re)write, and never count a capped output as done
        if args.num_events == -1 and _shard_done(out_name, Path(p)):
            n_done += 1
            print(f"[shard {i + 1}/{n_files}] {out_name.name} already exists, skipping")
            continue
        tracker_fn = Path(th[i])
        process_one_file(
            Path(p),
            Path(t),
            Path(c),
            tracker_fn,
            out_name,
            num_events=args.num_events,
            shard_index=i,
            job_total_shards=n_files,
            algorithm=args.algorithm,
            merge_frac=merge_frac,
        )
    print(f"all {n_files} shards accounted for ({n_done} already present, {n_files - n_done} newly converted)")


if __name__ == "__main__":
    main()
