"""Targets and per-element truth labels for the general ("full") reconstruction dataset.

The full view keeps every reconstruction-level input (tracks, clusters, tracker
hits, calorimeter hits) and one cluster-independent target particle table per
event. Each input element carries compact integer labels instead of a full target
row, so that any subset of element kinds can be used for training without
changing the target definition:

``particle_number``
    One-based row of the owning target particle, 0 for background. The owner is
    the target ancestor that contributes the largest truth weight (inclusive:
    hits of descendants count for their ancestor).
``instance_number``
    One-based row in the instance table: the simulated MC particle that actually
    produced the dominant part of the element. For hits made by the target itself
    this is the target's own instance; for decay products and other secondaries
    it is a distinct instance with its own kinematics.
``origin``
    How the element relates to its target (see ``ORIGIN_NAMES``). Primary codes
    (1-3) mean the target particle itself produced the element.
``purity``
    Fraction of the element's truth weight contributed by its owning target.
``true_px, true_py, true_pz``
    Simulated momentum of the producing particle at a tracker hit (GeV), zero for
    other elements.

The target particle is the visible status-1 (or surrogate) generator particle,
exactly as for the track/cluster dataset, but no reconstructed object is needed
to represent it: a visible particle is a target if any element is labelled with it.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import awkward as ak
import numpy as np
import vector
from scipy.sparse import coo_matrix, csc_matrix  # noqa: F401

from mlpf.conf import ParticleFeatures
from mlpf.data.target_building import compute_jets, map_pdgid_to_candid

ORIGIN_NONE = 0
ORIGIN_PRIMARY = 1
ORIGIN_PRIMARY_LOOPING = 2
ORIGIN_PRIMARY_UNSAVED_SECONDARY = 3
ORIGIN_DECAY = 4
ORIGIN_SECONDARY = 5
ORIGIN_NAMES = {
    ORIGIN_NONE: "none",
    ORIGIN_PRIMARY: "primary",
    ORIGIN_PRIMARY_LOOPING: "primary_looping",
    ORIGIN_PRIMARY_UNSAVED_SECONDARY: "primary_unsaved_secondary",
    ORIGIN_DECAY: "decay_product",
    ORIGIN_SECONDARY: "secondary",
}
PRIMARY_ORIGINS = (ORIGIN_PRIMARY, ORIGIN_PRIMARY_LOOPING, ORIGIN_PRIMARY_UNSAVED_SECONDARY)

LABEL_FEATURES = ["particle_number", "instance_number", "origin", "purity", "true_px", "true_py", "true_pz"]
INSTANCE_FEATURES = [
    "instance_number",
    "particle_number",
    "origin",
    "PDG",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "vertex_x",
    "vertex_y",
    "vertex_z",
]
ACCOUNTING_FEATURES = ["source_energy", "invisible_energy", "unlabelled_energy", "target_energy"]
ELEMENT_KINDS = ("track", "cluster", "hit_tracker", "hit_calo")

# edm4hep::MCParticle simulatorStatus bits
BIT_VERTEX_IS_NOT_ENDPOINT_OF_PARENT = 28
BIT_DECAYED_IN_TRACKER = 27
BIT_DECAYED_IN_CALORIMETER = 26
# edm4hep::SimTrackerHit quality bit: produced by a secondary that was not stored as an MCParticle
BIT_PRODUCED_BY_SECONDARY = 30


@dataclass
class TruthLinks:
    """Direct (not ancestor-propagated) MC links of one event, in unfiltered MC indices."""

    mc: Dict[str, np.ndarray]  # PDG, charge, simulatorStatus, px, py, pz, energy, vertex_x/y/z
    parent: np.ndarray  # [n_mc] first parent index, -1 for none
    targets: np.ndarray  # [n_target] MC index of every visible target candidate, in EventData gen order
    mc_to_hit: tuple  # (mc, hit, weight) direct links, weight in GeV
    mc_to_track: tuple  # (mc, track, weight) direct SiTracksMCTruthLink
    hit_to_cluster: tuple  # (hit, cluster, weight)
    tracker_momentum: np.ndarray  # [n_hit, 3] simulated momentum at tracker hits, zero elsewhere
    tracker_unsaved_secondary: np.ndarray  # [n_hit] SimTrackerHit produced-by-secondary flag
    is_tracker_hit: np.ndarray  # [n_hit]
    hit_positions: np.ndarray  # [n_hit, 3]


def parents_from_daughters(n_mc, daughters_begin, daughters_end, daughter_index):
    parent = np.full(n_mc, -1, dtype=np.int64)
    if daughter_index is None:
        return parent
    daughter_index = np.asarray(daughter_index, dtype=np.int64)
    for mother in range(n_mc):
        for daughter in daughter_index[int(daughters_begin[mother]) : int(daughters_end[mother])]:
            if 0 <= daughter < n_mc and parent[daughter] == -1:
                parent[daughter] = mother
    return parent


def target_ancestors(parent, targets):
    """For every MC particle, the row of its nearest ancestor-or-self in ``targets`` (-1 for none)."""
    n_mc = len(parent)
    row_of = np.full(n_mc, -1, dtype=np.int64)
    row_of[np.asarray(targets, dtype=np.int64)] = np.arange(len(targets))
    result = np.full(n_mc, -2, dtype=np.int64)  # -2: not yet resolved
    for start in range(n_mc):
        path = []
        current = start
        while current >= 0 and result[current] == -2 and row_of[current] < 0 and len(path) <= n_mc:
            path.append(current)
            current = parent[current]
        if current < 0:
            value = -1
        elif row_of[current] >= 0:
            value = row_of[current]
            result[current] = value
        else:
            value = result[current]
        for node in path:
            result[node] = value
    return result


def instance_origins(parent, target_row, targets, simulator_status):
    """Origin of each MC particle relative to its target: primary, decay product or other secondary."""
    status = np.asarray(simulator_status, dtype=np.int64)
    decayed = (status & ((1 << BIT_DECAYED_IN_TRACKER) | (1 << BIT_DECAYED_IN_CALORIMETER))) != 0
    continues = (status & (1 << BIT_VERTEX_IS_NOT_ENDPOINT_OF_PARENT)) != 0
    origin = np.full(len(parent), ORIGIN_NONE, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    for mc in np.flatnonzero(target_row >= 0):
        if mc == targets[target_row[mc]]:
            origin[mc] = ORIGIN_PRIMARY
            continue
        mother = parent[mc]
        origin[mc] = ORIGIN_DECAY if mother >= 0 and decayed[mother] and not continues[mc] else ORIGIN_SECONDARY
    return origin


def _csc(edges, shape):
    rows, cols, weights = (np.asarray(x) for x in edges)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) and (not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Full-view truth weights must be finite and nonnegative")
    matrix = coo_matrix((weights, (rows.astype(np.int64), cols.astype(np.int64))), shape=shape).tocsc()
    matrix.sum_duplicates()
    return matrix


def element_labels(weights: csc_matrix, target_row, origin_of_mc):
    """Owner, producing instance (MC index), origin and purity per element (column of ``weights``)."""
    n_element = weights.shape[1]
    owner = np.full(n_element, -1, dtype=np.int64)
    producer = np.full(n_element, -1, dtype=np.int64)
    purity = np.zeros(n_element, dtype=np.float64)
    origin = np.zeros(n_element, dtype=np.int64)
    for element in range(n_element):
        start, stop = weights.indptr[element : element + 2]
        if start == stop:
            continue
        mcs = weights.indices[start:stop]
        values = weights.data[start:stop]
        total = values.sum()
        rows = target_row[mcs]
        valid = rows >= 0
        if total <= 0 or not valid.any():
            continue
        per_target = {}
        for row, value in zip(rows[valid], values[valid]):
            per_target[int(row)] = per_target.get(int(row), 0.0) + float(value)
        best = min(per_target, key=lambda row: (-per_target[row], row))
        owned = valid & (rows == best)
        # Ties are broken by the smaller MC index so labels are deterministic.
        candidates = np.flatnonzero(owned)
        winner = candidates[np.lexsort((mcs[candidates], -values[candidates]))[0]]
        owner[element] = best
        producer[element] = mcs[winner]
        purity[element] = per_target[best] / total
        origin[element] = origin_of_mc[producer[element]]
    return owner, producer, origin, purity


def _field(record, name):
    fields = record.fields if hasattr(record, "fields") else record.keys()
    return np.asarray(record[name], dtype=np.float64) if name in fields else None


def build_full_targets(truth: TruthLinks, gen_features, n_track: int, n_cluster: int, *, source_energy: float, jetdef, jet_ptcut):
    """Return the Parquet fields of the full view for one event.

    ``gen_features`` are the visible target candidates in the order of ``truth.targets``
    (EventData.gen_features). Hit order follows EventData.hit_features; the returned
    hit labels are split into tracker and calorimeter rows with the same masks used
    for ``X_hit_tracker`` and ``X_hit_calo``.
    """
    n_mc = len(truth.parent)
    n_hit = len(truth.is_tracker_hit)
    targets = np.asarray(truth.targets, dtype=np.int64)
    target_row = target_ancestors(truth.parent, targets)
    origin_of_mc = instance_origins(truth.parent, target_row, targets, truth.mc["simulatorStatus"])

    hit_weights = _csc(truth.mc_to_hit, (n_mc, n_hit))
    track_weights = _csc(truth.mc_to_track, (n_mc, n_track))
    hit_to_cluster = _csc(truth.hit_to_cluster, (n_hit, n_cluster))
    # Clusters inherit the calorimeter-hit truth of their hits, weighted by the hit fraction in the cluster.
    cluster_weights = (hit_weights @ hit_to_cluster).tocsc()

    kinds = {}
    for kind, weights in (("track", track_weights), ("cluster", cluster_weights), ("hit", hit_weights)):
        kinds[kind] = element_labels(weights, target_row, origin_of_mc)

    # Refine primary tracker hits: returning legs of loopers and hits made by unsaved secondaries.
    owner, producer, origin, purity = kinds["hit"]
    tracker = np.asarray(truth.is_tracker_hit, dtype=bool)
    primary = tracker & (origin == ORIGIN_PRIMARY)
    radial = np.einsum("ij,ij->i", truth.tracker_momentum[:, :2], truth.hit_positions[:, :2])
    looping = primary & (radial < 0)
    unsaved = primary & np.asarray(truth.tracker_unsaved_secondary, dtype=bool)
    origin = origin.copy()
    origin[looping] = ORIGIN_PRIMARY_LOOPING
    origin[unsaved] = ORIGIN_PRIMARY_UNSAVED_SECONDARY
    kinds["hit"] = (owner, producer, origin, purity)

    # Targets are visible particles with at least one labelled element of any kind.
    labelled = np.zeros(len(targets), dtype=bool)
    for owner, _, _, _ in kinds.values():
        labelled[owner[owner >= 0]] = True
    kept = np.flatnonzero(labelled)
    new_row = np.full(len(targets), -1, dtype=np.int64)
    new_row[kept] = np.arange(len(kept))

    names = ParticleFeatures.get_names()
    col = {name: i for i, name in enumerate(names)}
    table = np.zeros((len(kept), len(names)), dtype=np.float32)
    for name in names:
        if name in ("particle_number", "jet_idx", "PDG"):
            continue
        values = _field(gen_features, name)
        if values is not None:
            table[:, col[name]] = values[kept]
    pdg = np.asarray(gen_features["PDG"], dtype=np.int64)[kept]
    charge = np.asarray(gen_features["charge"], dtype=np.float64)[kept]
    # Truth-charge candidate class; the loader maps charged targets without a selected
    # tracking element to their neutral counterpart.
    table[:, col["PDG"]] = [map_pdgid_to_candid(abs(int(p)), c) for p, c in zip(pdg, charge)]
    table[:, col["particle_number"]] = np.arange(1, len(kept) + 1)
    table[:, col["jet_idx"]] = -1

    jets = np.zeros((0, 4), dtype=np.float32)
    if len(kept):
        phi = np.arctan2(table[:, col["sin_phi"]], table[:, col["cos_phi"]])
        vectors = vector.awk(ak.zip(dict(pt=table[:, col["pt"]], eta=table[:, col["eta"]], phi=phi, energy=table[:, col["energy"]])))
        found, constituents = compute_jets(vectors, min_pt=jet_ptcut, jetdef=jetdef, with_indices=True)
        order = np.argsort(-ak.to_numpy(found.pt), kind="stable")
        jets = np.column_stack([ak.to_numpy(x)[order] for x in (found.pt, found.eta, found.phi, found.energy)]).astype(np.float32)
        constituents = ak.to_list(constituents)
        for jet, source in enumerate(order):
            table[constituents[source], col["jet_idx"]] = jet

    # Instance table: every MC producer referenced by a kept element.
    instance_of = {}
    instance_rows = []
    mc = truth.mc
    for kind in ("track", "cluster", "hit"):
        owner, producer, origin, _ = kinds[kind]
        for element in np.flatnonzero(owner >= 0):
            source = int(producer[element])
            if source in instance_of:
                continue
            instance_of[source] = len(instance_rows) + 1
            px, py, pz = (float(mc[k][source]) for k in ("px", "py", "pz"))
            pt = float(np.hypot(px, py))
            phi = float(np.arctan2(py, px))
            instance_rows.append(
                [
                    instance_of[source],
                    new_row[target_row[source]] + 1,
                    origin_of_mc[source],
                    float(mc["PDG"][source]),
                    float(mc["charge"][source]),
                    pt,
                    float(np.arcsinh(pz / pt)) if pt > 0 else 0.0,
                    float(np.sin(phi)),
                    float(np.cos(phi)),
                    float(mc["energy"][source]),
                    float(mc["vertex_x"][source]),
                    float(mc["vertex_y"][source]),
                    float(mc["vertex_z"][source]),
                ]
            )
    instances = np.asarray(instance_rows, dtype=np.float32).reshape(-1, len(INSTANCE_FEATURES))

    def label_matrix(kind, select=None):
        owner, producer, origin, purity = kinds[kind]
        if select is not None:
            owner, producer, origin, purity = owner[select], producer[select], origin[select], purity[select]
        labels = np.zeros((len(owner), len(LABEL_FEATURES)), dtype=np.float32)
        good = owner >= 0
        labels[good, 0] = new_row[owner[good]] + 1
        labels[good, 1] = [instance_of[int(source)] for source in producer[good]]
        labels[good, 2] = origin[good]
        labels[good, 3] = purity[good]
        return labels

    hit_labels_tracker = label_matrix("hit", tracker)
    hit_labels_tracker[:, 4:7] = truth.tracker_momentum[tracker]
    hit_labels_tracker[hit_labels_tracker[:, 0] == 0, 4:7] = 0

    visible_energy = float(np.asarray(gen_features["energy"], dtype=np.float64).sum())
    target_energy = float(table[:, col["energy"]].astype(np.float64).sum())
    accounting = np.array(
        [source_energy, max(source_energy - visible_energy, 0.0), visible_energy - target_energy, target_energy],
        dtype=np.float64,
    )
    return {
        "ylabel_track": label_matrix("track"),
        "ylabel_cluster": label_matrix("cluster"),
        "ylabel_hit_tracker": hit_labels_tracker,
        "ylabel_hit_calo": label_matrix("hit", ~tracker),
        "target_particles": table,
        "target_instances": instances,
        "target_energy_accounting": accounting,
        "targetjet_full": jets,
    }


def element_class_counts(labels: np.ndarray) -> Optional[Dict[str, int]]:
    """Counts of each origin code among labelled elements (diagnostics)."""
    if len(labels) == 0:
        return {}
    origin = labels[labels[:, 0] > 0, 2].astype(int)
    return {ORIGIN_NAMES[code]: int((origin == code).sum()) for code in ORIGIN_NAMES if code != ORIGIN_NONE}
