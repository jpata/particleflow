"""ColliderML targets and per-element labels for the general (full) dataset view.

Producers come straight from the release tables: tracker_hits.particle_id and
calo_hits.contrib_particle_ids name the simulated particle that made each hit,
including Geant4 secondaries, and tracks join their hits through hit_ids. Ownership
uses the same leaf walk as truth.compute_gen_tables, so the owning target of every
element is exactly the target that the existing attribution credits.

Differences from key4hep (documented in docs/datasets/full.md): tracker hits carry no
simulated momentum (inward legs are found from the time-ordered hit radius, and the
true_px/py/pz labels stay zero), there is no produced-by-unsaved-secondary flag, and
decay products are identified from species and production vertices alone.
"""

import awkward as ak
import numpy as np

from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, _redirect_ids, build_leaf_maps, calibration_factors, walk_to_leaf_many
from mlpf.data.full_targets import TruthLinks, build_full_targets, decay_products

# Production vertices beyond the tracker envelope (hits end at r = 1032 mm, |z| = 3026 mm) and
# before the calorimeters (r >= 1252 mm, |z| >= 3202 mm) mark particles born in a calorimeter shower.
CALO_REGION_R_MM = 1150.0
CALO_REGION_Z_MM = 3100.0


def calorimeter_entry_rows(rows, parent_row, vertex, target_index):
    """Map calorimeter-shower particles to the ancestor that entered the calorimeter.

    ColliderML stores Geant4 shower particles and credits deposits to them; key4hep credits
    the particle that entered the calorimeter. Walking up while the production vertex lies in
    the calorimeter region (stopping at a target) gives the same convention.
    """
    in_calo = (np.hypot(vertex[:, 0], vertex[:, 1]) > CALO_REGION_R_MM) | (np.abs(vertex[:, 2]) > CALO_REGION_Z_MM)
    entry = np.arange(len(parent_row), dtype=np.int64)
    for row in np.flatnonzero(in_calo):
        current = row
        while in_calo[current] and target_index[current] < 0 and parent_row[current] >= 0:
            current = parent_row[current]
        entry[row] = current
    rows = np.asarray(rows, dtype=np.int64)
    return np.where(rows >= 0, entry[np.clip(rows, 0, max(len(entry) - 1, 0))], rows)


def _rows_of(ids, pid):
    """Particle-table row of each id (-1 if the id is not in the table)."""
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(pid, kind="stable")
    sorted_pid = pid[order].astype(np.int64)
    pos = np.clip(np.searchsorted(sorted_pid, ids), 0, max(len(sorted_pid) - 1, 0))
    if len(sorted_pid) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    return np.where(sorted_pid[pos] == ids, order[pos], -1).astype(np.int64)


def colliderml_full_targets(
    particles_ev, calo_ev, tracks_ev, tracker_ev, gen_features, target_rows, hit_to_cluster, n_cluster, source_energy, jetdef, jet_ptcut
):
    """Full-view fields for one ColliderML event.

    ``gen_features`` and ``target_rows`` are compute_gen_tables' targets (return_rows=True).
    Calorimeter hits come first in the hit index space, followed by tracker hits, as in
    the converter's EventData.
    """
    pid, pdg, parent_id, is_leaf, redirect, _ = build_leaf_maps(particles_ev, calo_ev, tracker_ev)
    pid = np.asarray(pid, dtype=np.int64)
    n_mc = len(pid)
    parent_row = _rows_of(np.asarray(parent_id, dtype=np.int64), pid)

    # owning target of every particle row: the leaf its chain reaches, if that leaf is a target
    leaf_row = walk_to_leaf_many(pid, pid, is_leaf, parent_id) if n_mc else np.zeros(0, dtype=np.int64)
    target_index = np.full(n_mc, -1, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    target_index[target_rows] = np.arange(len(target_rows))
    target_row = np.where(leaf_row >= 0, target_index[np.clip(leaf_row, 0, max(n_mc - 1, 0))], -1)

    get = lambda table, name: np.asarray(ak.to_numpy(table[name]))  # noqa: E731
    # Release tables always carry production vertices; minimal test events may omit them.
    vertex = np.stack([get(particles_ev, axis) if axis in particles_ev.fields else np.zeros(n_mc) for axis in ("vx", "vy", "vz")], axis=-1).astype(
        np.float64
    )
    decay = decay_products(parent_row, pdg, vertex)

    # calorimeter contributions (calibrated GeV) and tracker hits (unit weight), both by producer row
    n_calo = len(get(calo_ev, "x"))
    n_contrib = np.asarray(ak.to_numpy(ak.num(calo_ev["contrib_particle_ids"]))).astype(np.int64)
    contrib_ids = _redirect_ids(np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_particle_ids"], axis=None))).astype(np.int64), redirect)
    contrib_energy = np.asarray(ak.to_numpy(ak.flatten(calo_ev["contrib_energies"], axis=None)), dtype=np.float64)
    contrib_hit = np.repeat(np.arange(n_calo, dtype=np.int64), n_contrib)
    contrib_energy = contrib_energy * calibration_factors(get(calo_ev, "detector")[contrib_hit], DEFAULT_CALIBRATION)
    contrib_row = calorimeter_entry_rows(_rows_of(contrib_ids, pid), parent_row, vertex, target_index)
    tracker_ids = _redirect_ids(get(tracker_ev, "particle_id").astype(np.int64), redirect)
    tracker_row = _rows_of(tracker_ids, pid)
    n_tracker = len(tracker_row)
    calo_ok = (contrib_row >= 0) & (contrib_energy > 0)
    tracker_ok = tracker_row >= 0
    mc_to_hit = (
        np.concatenate([contrib_row[calo_ok], tracker_row[tracker_ok]]),
        np.concatenate([contrib_hit[calo_ok], n_calo + np.flatnonzero(tracker_ok)]),
        np.concatenate([contrib_energy[calo_ok], np.ones(int(tracker_ok.sum()))]),
    )

    # tracks: share of each track's hits made by each producer
    n_track = len(get(tracks_ev, "majority_particle_id"))
    hits_per_track = np.asarray(ak.to_numpy(ak.num(tracks_ev["hit_ids"])), dtype=np.int64)
    hit_ids = np.asarray(ak.to_numpy(ak.flatten(tracks_ev["hit_ids"], axis=None)), dtype=np.int64)
    track_of_hit = np.repeat(np.arange(n_track, dtype=np.int64), hits_per_track)
    in_range = (hit_ids >= 0) & (hit_ids < n_tracker)
    producer = np.full(len(hit_ids), -1, dtype=np.int64)
    producer[in_range] = tracker_row[hit_ids[in_range]]
    ok = producer >= 0
    weight = 1.0 / np.maximum(hits_per_track[track_of_hit[ok]], 1)
    mc_to_track = (producer[ok], track_of_hit[ok], weight)

    positions = np.concatenate(
        [
            np.stack([get(calo_ev, axis) for axis in ("x", "y", "z")], axis=-1),
            np.stack([get(tracker_ev, axis) for axis in ("x", "y", "z")], axis=-1),
        ]
    ).astype(np.float64)
    times = np.concatenate([np.zeros(n_calo), get(tracker_ev, "time").astype(np.float64)])
    px, py, pz = (get(particles_ev, axis).astype(np.float64) for axis in ("px", "py", "pz"))
    mc = {
        "PDG": np.asarray(pdg, dtype=np.float64),
        "charge": get(particles_ev, "charge").astype(np.float64),
        "simulatorStatus": np.zeros(n_mc, dtype=np.int64),
        "px": px,
        "py": py,
        "pz": pz,
        "energy": get(particles_ev, "energy").astype(np.float64),
        "vertex_x": vertex[:, 0],
        "vertex_y": vertex[:, 1],
        "vertex_z": vertex[:, 2],
    }
    n_hit = n_calo + n_tracker
    truth = TruthLinks(
        mc=mc,
        parent=parent_row,
        targets=target_rows,
        mc_to_hit=mc_to_hit,
        mc_to_track=mc_to_track,
        hit_to_cluster=hit_to_cluster,
        tracker_momentum=np.zeros((n_hit, 3), dtype=np.float32),
        tracker_unsaved_secondary=np.zeros(n_hit, dtype=bool),
        is_tracker_hit=np.arange(n_hit) >= n_calo,
        hit_positions=positions,
        target_row=target_row,
        decay_product=decay,
        tracker_time=times,
        momentum_available=False,
    )
    gen = dict(gen_features)
    # A pi0 left unsplit is a photon target (truth.py), not the shared mapping's neutral hadron.
    gen["PDG"] = np.where(np.abs(np.asarray(gen["PDG"])) == 111, 22, np.asarray(gen["PDG"]))
    return build_full_targets(truth, gen, n_track, n_cluster, source_energy=source_energy, jetdef=jetdef, jet_ptcut=jet_ptcut)
