# ColliderML release 1 -> MLPF parquet converter (clustered view).
#
# For each source shard this produces one MLPF-format parquet with the same record layout as
# the key4hep converters write:
#   X_track, X_cluster, ytarget_track, ytarget_cluster,
#   X_hit_tracker, ytarget_hit_tracker  (empty; ACTS track information lives in X_track),
#   X_hit_calo,  ytarget_hit_calo       (one per raw colliderml calo hit),
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
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables
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
track_feature_order = EDM4HEP.TrackFeatures.get_names()  # 16 names; only the first 11 are filled
cluster_feature_order = EDM4HEP.ClusterFeatures.get_names()  # 17 names
# EDM4hep hit layout used elsewhere in the codebase (X_hit already includes `type`); we write
# only the hits that the current view actually contains. For the clustered view on ColliderML
# the calo hits are the raw inputs we cluster over; the tracker hits exist in the release but
# we model them indirectly via the tracks.
_hit_feature_order = EDM4HEP.HitFeatures.get_names()  # 12 names


def _event_record_one(
    event_id: int,
    particles_ev: ak.Record,
    tracks_ev: ak.Record,
    calo_ev: ak.Record,
    tracker_ev: ak.Record,
    algorithm: str = "bfs_merge",
    merge_frac: float = 0.0,
) -> Dict[str, Any]:
    """Convert one ColliderML event into the MLPF record fields."""
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
    tfeat = track_features_cml(tracks_ev)

    # -- clusters via the truth-blind spatial clusterer ------------------------
    cl_detector_k = np.array([DEFAULT_CALIBRATION[int(r)] for r in hit_det], dtype=np.float32)
    hit_e_calibrated = hit_e * cl_detector_k
    cluster_of, cl_feats, hit_to_cluster, cluster_region = cluster_event(
        hit_x, hit_y, hit_z, hit_e_calibrated, hit_det, algorithm=algorithm, merge_frac=merge_frac
    )
    n_cluster = len(cl_feats)

    # -- build EventData and run the shared allocator --------------------------
    gpdata = EventData(
        gen_features,
        {"type": np.zeros(n_hit, dtype=np.float32)},  # filler; key4hep fills a real matrix but the
        # allocator only uses the length of the features
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
    del hit_to_gp_inclusive  # unused in the clustered view

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
    # tracks: 11 features, plus a copy of sin_phi stored in the tanLambda slot (existing joint
    # EDM4hep layout does the same: track sin_phi = tanLambda). Width = 17 (union layout).
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
    X_track[:, 10] = np.asarray(tfeat["sin_phi"], dtype=np.float32)  # tanLambda slot (same value)
    # cols 11..16 are the cluster-only slots of the union layout (energy_hcal, energy_other,
    # num_hits, sigma_x, sigma_y, sigma_z) and stay 0 for tracks, except n_meas in col 13
    # (the num_hits slot) as a proxy for ndf
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
    #   [ elemtype(2), et, eta, sin_phi, cos_phi, energy(=total_energy), x, y, z, time(0), subdetector(=region), type(0) ]
    # ytarget_hit_calo rows use the canonical particle_feature_order layout with:
    #   * exactly one full-target row per cluster-represented truth particle (via gp_to_hit_idx from
    #     the allocator); all other hits may carry particle_number but zero kinematics, so the
    #     clustered and hit views agree on which elements carry truth information.
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
    # use the *post-merge* filtered hit adjacency (rows dropped by the allocator have
    # already been filtered out of this COO list)
    hit_to_gp_incl = np.array(gpdata_cleaned.genparticle_to_hit[0], dtype=np.int64)
    hit_to_hit_idx = np.array(gpdata_cleaned.genparticle_to_hit[1], dtype=np.int64)
    if len(hit_to_gp_incl):
        # inclusive: every hit of a (target-owned) genparticle picks up that particle's
        # particle_number so shower fragments are grouped
        pns = np.asarray(gps_canonical[hit_to_gp_incl, PN_IDX], dtype=np.float32)
        ytarget_hit_calo[hit_to_hit_idx, PN_IDX] = pns
    for igp in range(n_gp):
        h = int(gp_to_hit_idx[igp])
        if h != -1:
            # exclusive hit if the particle is cluster-represented
            if gp_to_obj[igp, 1] != -1:
                ytarget_hit_calo[h, :] = gps_canonical[igp]
    X_hit_calo_sanitized = X_hit_calo.copy()
    ytarget_hit_calo_sanitized = ytarget_hit_calo.copy()
    sanitize(X_hit_calo_sanitized)
    sanitize(ytarget_hit_calo_sanitized)

    # raw tracker hits from the release (tracker_hits table, subdetector=3, "tracker" in the
    # key4hep HitFeatures convention). We keep only what the validator needs to run; the
    # truth-label column is not filled because tracker hits are not PF targets (targets live
    # on tracks/clusters), so PN on tracker hits stays 0 by construction. This block exists so
    # tests/validate_parquet.py sees a "normal" HDF-hit layout on all detectors.
    tx = ak.to_numpy(tracker_ev["x"]).astype(np.float32)
    ty = ak.to_numpy(tracker_ev["y"]).astype(np.float32)
    tz = ak.to_numpy(tracker_ev["z"]).astype(np.float32)
    te = ak.to_numpy(tracker_ev["time"]).astype(np.float32)
    n_tracker = len(tx)
    X_hit_tracker = np.zeros((n_tracker, len(_hit_feature_order)), dtype=np.float32)
    X_hit_tracker[:, 0] = 1.0  # elemtype: tracker
    # columns 1..5 (et/eta/sin_phi/cos_phi/E) are not meaningful for un-tracked tracker hits;
    # zero them so no physics expectation is silently attached.
    X_hit_tracker[:, 6] = tx
    X_hit_tracker[:, 7] = ty
    X_hit_tracker[:, 8] = tz
    X_hit_tracker[:, 9] = te
    X_hit_tracker[:, 10] = 3.0  # subdetector = tracker
    # ytarget_hit_tracker is a zero format placeholder: unlike key4hep (cld/clic), ColliderML
    # has no hits-view training dataset yet, so nothing consumes this array. Filling it would
    # first require a gp->tracker-hit truth adjacency so charged particles can claim an
    # exclusive tracker-hit representative (the key4hep hits-view pattern); release-1 tracker
    # hits do carry per-hit truth particle_id, so the wiring is possible when needed.
    ytarget_hit_tracker = np.zeros((X_hit_tracker.shape[0], len(particle_feature_order)), dtype=np.float32)

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


def _parquet_usable(ofn: Path) -> bool:
    """True iff the named parquet file exists and pyarrow can open it. A truncated write
    crashes pyarrow on the footer, so corrupted outputs from a killed mid-shard job are
    reconvertible instead of being mistaken for done by the resume logic."""
    if not os.path.isfile(ofn):
        return False
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(ofn)
    except Exception:
        return False
    return True


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
    merge_frac: float = 0.0,
) -> None:
    if num_events == -1 and _parquet_usable(ofn):
        print(f"[shard {shard_index + 1}/{job_total_shards}] {Path(ofn).name} already exists, skipping")
        return
    if os.path.isfile(ofn) and not _parquet_usable(ofn):
        # a corrupted leftover (e.g. from an OOM-killed job) just falls through to the
        # convert + rename path below, which overwrites it atomically
        print(f"[shard {shard_index + 1}/{job_total_shards}] {Path(ofn).name} exists but is corrupted; reconverting")

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
        for ev in tqdm.tqdm(
            iter_events,
            total=total,
            desc=desc,
            unit="event",
            ncols=100,
        ):
            event = _event_record_one(
                ev["event_id"], ev["particles"], ev["tracks"], ev["calo_hits"], ev["tracker_hits"], algorithm=algorithm, merge_frac=merge_frac
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
    parser.add_argument(
        "--tfds",
        action="store_true",
        help="after converting (or confirming present) every shard in --shards, run the TFDS build for the "
        "dataset of the given --sample. TFDS itself is skip-aware: an already-prepared config under "
        "--tfds-data-dir is reused, otherwise the build resumes from scratch. Skipped if any shard still "
        "needs to be produced by this invocation (so a resumable partial run cannot trigger a build "
        "on incomplete parquet).",
    )
    parser.add_argument(
        "--tfds-data-dir",
        type=str,
        default=None,
        help="TFDS data directory for the --tfds step (required when --tfds is set)",
    )
    args = parser.parse_args()
    if args.tfds and not args.tfds_data_dir:
        parser.error("--tfds requires --tfds-data-dir")

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
        if _parquet_usable(out_name):
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

    if args.tfds:
        # refuse to build from an incomplete parquet set: an interrupted earlier run leaves the range
        # half-converted and would produce a silently truncated TFDS dataset.
        missing = [p.stem + ".parquet" for p in pa if not _parquet_usable(Path(args.outpath) / (p.stem + ".parquet"))]
        if missing:
            raise RuntimeError(
                f"--tfds requested but {len(missing)} shard(s) in --shards are not yet converted "
                f"(first missing: {missing[0]}); rerun without --tfds (converter resumes where it "
                "left off) or extend the range once all shards exist."
            )
        if args.num_events != -1:
            raise RuntimeError("--tfds refuses num_events != -1: the TFDS split logic requires the full shard.")

        # TFDS dataset name per sample (two separate datasets, one per pileup level)
        TFDS_DATASET_BY_SAMPLE = {
            "ttbar_pu0": ("mlpf.heptfds.colliderml_pf.ttbar", "colliderml_ttbar_nopu_pf"),
            "ttbar_pu200": ("mlpf.heptfds.colliderml_pf.ttbar_pu200", "colliderml_ttbar_pu200_pf"),
        }
        if args.sample not in TFDS_DATASET_BY_SAMPLE:
            raise RuntimeError(f"no TFDS builder registered for sample {args.sample!r} (known: {sorted(TFDS_DATASET_BY_SAMPLE)})")
        builder_module, builder_name = TFDS_DATASET_BY_SAMPLE[args.sample]
        import importlib

        importlib.import_module(builder_module)  # TFDS builder registration
        from mlpf.heptfds.colliderml_utils.utils import NUM_SPLITS

        import tensorflow_datasets as tfds

        # the builder splits the manual_dir shard list into NUM_SPLITS TFDS configs; each is its own
        # prepared dataset so all must be prepared explicitly. TFDS preparatn is atomic per config:
        # writes go to a tmp dir and only the final rename publishes, so an interrupted build leaves
        # the data_dir clean and a rerun redoes just that config; an already-prepared config is
        # skipped by tfds itself.
        for group in range(1, NUM_SPLITS + 1):
            name = f"{builder_name}/{group}"
            print(f"TFDS build: {name} -> {args.tfds_data_dir}")
            builder = tfds.builder(name, data_dir=args.tfds_data_dir)
            builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=args.outpath))
            print(f"TFDS ready: {builder.info.full_name}")


if __name__ == "__main__":
    main()
