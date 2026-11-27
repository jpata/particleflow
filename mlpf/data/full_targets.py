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
    # Optional detector-specific truth. Defaults reproduce the key4hep convention.
    # target_row: [n_mc] owning target row per MC particle (default: nearest ancestor-or-self in targets)
    target_row: Optional[np.ndarray] = None
    # decay_product: [n_mc] produced in a decay of its parent (default: decay_products_from_mc)
    decay_product: Optional[np.ndarray] = None
    # mc_to_cluster: direct (mc, cluster, weight) links, used instead of hit truth propagated through hit_to_cluster
    mc_to_cluster: Optional[tuple] = None
    # tracker_time: [n_hit] hit times; with momentum_available=False, inward legs are found from the
    # time-ordered hit radius of each producing particle instead of the simulated momentum direction
    tracker_time: Optional[np.ndarray] = None
    momentum_available: bool = True


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


NEUTRINOS = {12, 14, 16}
# Allowed decay products (|PDG|) of long-lived species that the simulation decays.
# Neutrinos and photons (radiative decays) are allowed everywhere.
DECAY_DAUGHTERS = {
    211: {13, 11},
    321: {13, 11, 211, 111},
    130: {211, 111, 11, 13},
    310: {211, 111, 11},
    3122: {2212, 2112, 211, 111},
    3222: {2212, 2112, 211, 111},
    3112: {2112, 211},
    3212: {3122},
    3312: {3122, 211},
    3322: {3122, 111},
    3334: {3122, 321, 3312, 3322, 211, 111},
    13: {11},
    111: {11},
}
# Decays that can leave a single stored product (the neutrino is not always stored).
# Any other single product (for example a delta-ray electron from a pion) is not a decay.
SINGLE_PRODUCT_DECAYS = {211: {13}, 321: {13, 211}}
VERTEX_TOLERANCE_MM = 1.0e-3


def decay_products(parent, pdg, vertex, continues=None):
    """Mark particles produced in a decay of their parent.

    The simulator's "decayed in tracker/calorimeter" status is set whenever the parent's track
    ends there, also for inelastic interactions. A product is therefore a decay product only if
    its parent is a long-lived decaying species and every particle produced at the same vertex is
    an allowed decay product of that species. ``continues`` (optional) marks products created
    along the parent's path, such as delta rays and bremsstrahlung photons, which are never decays.
    """
    parent = np.asarray(parent, dtype=np.int64)
    pdg = np.abs(np.asarray(pdg, dtype=np.int64))
    vertex = np.asarray(vertex, dtype=np.float64).reshape(len(parent), 3)
    result = np.zeros(len(parent), dtype=bool)
    children = {}
    for child in np.flatnonzero(parent >= 0):
        children.setdefault(int(parent[child]), []).append(int(child))
    for mother, kids in children.items():
        allowed = DECAY_DAUGHTERS.get(int(pdg[mother]))
        if allowed is None:
            continue
        kids = np.asarray(kids)
        if continues is not None:
            kids = kids[~np.asarray(continues, dtype=bool)[kids]]
        remaining = list(kids)
        while remaining:
            seed = remaining[0]
            same = [k for k in remaining if np.all(np.abs(vertex[k] - vertex[seed]) <= VERTEX_TOLERANCE_MM)]
            remaining = [k for k in remaining if k not in same]
            species = {int(pdg[k]) for k in same}
            visible_species = species - NEUTRINOS - {22}
            if not visible_species <= allowed:
                continue
            charged_or_massive = [k for k in same if int(pdg[k]) not in NEUTRINOS | {22}]
            if len(same) == 1 and not (species & SINGLE_PRODUCT_DECAYS.get(int(pdg[mother]), set())):
                continue
            if not charged_or_massive and not (species & NEUTRINOS):
                continue
            result[same] = True
    return result


def decay_products_from_mc(parent, mc):
    """key4hep: decay products from species and vertices; edm4hep bit 28 marks along-path products."""
    status = np.asarray(mc["simulatorStatus"], dtype=np.int64)
    continues = (status & (1 << BIT_VERTEX_IS_NOT_ENDPOINT_OF_PARENT)) != 0
    vertex = np.stack([mc["vertex_x"], mc["vertex_y"], mc["vertex_z"]], axis=-1)
    return decay_products(parent, mc["PDG"], vertex, continues)


def instance_origins(target_row, targets, decay_product):
    """Origin of each MC particle relative to its target: primary, decay product or other secondary."""
    origin = np.full(len(target_row), ORIGIN_NONE, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    for mc in np.flatnonzero(target_row >= 0):
        if mc == targets[target_row[mc]]:
            origin[mc] = ORIGIN_PRIMARY
        else:
            origin[mc] = ORIGIN_DECAY if decay_product[mc] else ORIGIN_SECONDARY
    return origin


def inward_hits_by_time(producer, positions, times, candidates):
    """Hits on an inward leg: smaller radius than an earlier hit of the same producer (no momentum needed)."""
    inward = np.zeros(len(producer), dtype=bool)
    rows = np.flatnonzero(candidates)
    if len(rows) == 0:
        return inward
    radius = np.hypot(positions[rows, 0], positions[rows, 1])
    order = np.lexsort((times[rows], producer[rows]))
    rows, radius, owners = rows[order], radius[order], producer[rows][order]
    start = 0
    # A margin keeps measurement noise on one surface from flagging the outgoing leg.
    margin_mm = 1.0
    while start < len(rows):
        stop = start
        while stop < len(rows) and owners[stop] == owners[start]:
            stop += 1
        running_max = np.maximum.accumulate(radius[start:stop])
        inward[rows[start:stop]] = radius[start:stop] < running_max - margin_mm
        start = stop
    return inward


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
    target_row = target_ancestors(truth.parent, targets) if truth.target_row is None else np.asarray(truth.target_row, dtype=np.int64)
    decay_product = decay_products_from_mc(truth.parent, truth.mc) if truth.decay_product is None else truth.decay_product
    origin_of_mc = instance_origins(target_row, targets, np.asarray(decay_product, dtype=bool))

    hit_weights = _csc(truth.mc_to_hit, (n_mc, n_hit))
    track_weights = _csc(truth.mc_to_track, (n_mc, n_track))
    hit_to_cluster = _csc(truth.hit_to_cluster, (n_hit, n_cluster))
    if truth.mc_to_cluster is not None:
        cluster_weights = _csc(truth.mc_to_cluster, (n_mc, n_cluster))
    else:
        # Clusters inherit the calorimeter-hit truth of their hits, weighted by the hit fraction in the cluster.
        cluster_weights = (hit_weights @ hit_to_cluster).tocsc()

    kinds = {}
    for kind, weights in (("track", track_weights), ("cluster", cluster_weights), ("hit", hit_weights)):
        kinds[kind] = element_labels(weights, target_row, origin_of_mc)

    # Refine primary tracker hits: returning legs of loopers and hits made by unsaved secondaries.
    owner, producer, origin, purity = kinds["hit"]
    tracker = np.asarray(truth.is_tracker_hit, dtype=bool)
    primary = tracker & (origin == ORIGIN_PRIMARY)
    if truth.momentum_available:
        radial = np.einsum("ij,ij->i", truth.tracker_momentum[:, :2], truth.hit_positions[:, :2])
        looping = primary & (radial < 0)
    else:
        if truth.tracker_time is None:
            raise ValueError("tracker_time is required when the simulated hit momentum is unavailable")
        looping = inward_hits_by_time(producer, truth.hit_positions, np.asarray(truth.tracker_time, dtype=np.float64), primary)
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
