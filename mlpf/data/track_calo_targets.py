"""Cluster-independent targets for reconstructed tracks plus calorimeter hits.

Run on EventData *before* the legacy cluster allocator. Storage is sparse in
truth associations; no particle-by-hit dense matrices are constructed. Original
PF ancestry/visibility selection remains the responsibility of the source reader.
"""

from collections import deque

import awkward as ak
import numpy as np
import vector
from scipy.sparse import coo_matrix

from mlpf.conf import ParticleFeatures
from mlpf.data.target_building import compute_jets, map_charged_to_neutral, map_neutral_to_charged, map_pdgid_to_candid, visibility_mask


def _matrix(edges, shape):
    rows, cols, weights = edges
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Track/calo truth association weights must be finite and nonnegative")
    matrix = coo_matrix((weights, (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=shape).tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _ranked(matrix, row, minimum=0.0):
    start, stop = matrix.indptr[row : row + 2]
    indices, weights = matrix.indices[start:stop], matrix.data[start:stop]
    good = weights > minimum if minimum == 0 else weights >= minimum
    indices, weights = indices[good], weights[good]
    return indices[np.lexsort((indices, -weights))].tolist()


def _match(candidates, order, n_elements):
    """Maximum-cardinality matching; deterministic BFS avoids recursion limits.

    Prefer free elements in candidate order (tracks before hits), then augment
    through arbitrarily long alternating paths. Earlier matched particles stay
    represented, though their representative can move.
    """
    owner = np.full(n_elements, -1, dtype=np.int64)
    representative = np.full(len(candidates), -1, dtype=np.int64)
    for root in order:
        queue = deque([int(root)])
        parent = {int(root): None}
        free = None
        while queue and free is None:
            particle = queue.popleft()
            for element in candidates[particle]:
                other = int(owner[element])
                if other == -1:
                    free = (particle, element)
                    break
                if other not in parent:
                    parent[other] = particle
                    queue.append(other)
        if free is None:
            continue
        particle, element = free
        while particle is not None:
            previous = int(representative[particle])
            owner[element] = particle
            representative[particle] = element
            element = previous
            particle = parent[particle]
    return representative, owner


def build_track_calo_targets(gpdata, calo_mask, *, jetdef, jet_ptcut, pi0_as_photon=False):
    """Return additional Parquet fields; leave legacy EventData and targets intact.

    One full target row is placed on a track or a calorimeter hit. Other elements
    carry only particle_number. Unrepresentable visible particles merge only into
    a retained particle owning a calorimeter hit they actually contributed to;
    unsupported particles are explicitly dropped. Merged associations are summed
    before inclusive ownership is computed. IDs are event-local, one-based (0 is
    background); original_to_target uses zero-based rows, with -1 for removals.
    """
    names = ParticleFeatures.get_names()
    col = {name: i for i, name in enumerate(names)}
    n_gp = len(gpdata.gen_features["PDG"])
    n_track = len(gpdata.track_features["type"])
    n_hit = len(gpdata.hit_features["type"])
    calo_mask = np.asarray(calo_mask, dtype=bool)
    if calo_mask.shape != (n_hit,):
        raise ValueError("calo_mask must align with EventData hits")
    n_calo = int(calo_mask.sum())
    tracks = _matrix(gpdata.genparticle_to_track, (n_gp, n_track))
    hits = _matrix(gpdata.genparticle_to_hit, (n_gp, n_hit))[:, calo_mask].tocsr()
    energy = np.asarray(gpdata.gen_features["energy"], dtype=np.float64)
    charge = np.asarray(gpdata.gen_features["charge"], dtype=np.float64)
    track_support = np.asarray(tracks.max(axis=1).toarray()).ravel() >= 0.2 if n_track and n_gp else np.zeros(n_gp, dtype=bool)
    visible = visibility_mask(np.asarray(hits.sum(axis=1)).ravel(), energy, charge != 0, track_support)
    candidates = [[] for _ in range(n_gp)]
    for particle in np.flatnonzero(visible):
        candidates[particle] = _ranked(tracks, particle, 0.2) + [n_track + h for h in _ranked(hits, particle)]
    order = sorted(np.flatnonzero(visible), key=lambda i: (-energy[i], i))
    representative, owner = _match(candidates, order, n_track + n_calo)
    kept = np.flatnonzero(representative >= 0)
    destination = np.full(n_gp, -1, dtype=np.int64)
    destination[kept] = kept
    # No reconstructed clusters enter retention or merging. Pick the strongest
    # shared-hit host by summed deposited energy, breaking ties by original ID.
    for particle in order:
        if destination[particle] >= 0:
            continue
        hosts = {}
        start, stop = hits.indptr[particle : particle + 2]
        for hit, weight in zip(hits.indices[start:stop], hits.data[start:stop]):
            host = int(owner[n_track + hit])
            if host >= 0:
                hosts[host] = hosts.get(host, 0.0) + weight
        if hosts:
            destination[particle] = min(hosts, key=lambda host: (-hosts[host], host))
    old_to_new = np.full(n_gp, -1, dtype=np.int64)
    old_to_new[kept] = np.arange(len(kept))
    final_id = np.full(n_gp, -1, dtype=np.int64)
    mapped = destination >= 0
    final_id[mapped] = old_to_new[destination[mapped]]

    target = np.zeros((len(kept), len(names)), dtype=np.float32)
    for name in names:
        if name in ("particle_number", "sin_phi", "cos_phi"):
            continue
        target[:, col[name]] = np.asarray(gpdata.gen_features[name])[kept]
    pt = np.asarray(gpdata.gen_features["pt"], dtype=np.float64)
    eta = np.asarray(gpdata.gen_features["eta"], dtype=np.float64)
    phi = np.asarray(gpdata.gen_features["phi"], dtype=np.float64)
    p4 = np.column_stack((pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy))
    summed = np.zeros((len(kept), 4), dtype=np.float64)
    np.add.at(summed, final_id[mapped], p4[mapped])
    transverse = np.hypot(summed[:, 0], summed[:, 1])
    target[:, col["pt"]] = transverse
    target[:, col["eta"]] = np.arcsinh(summed[:, 2] / np.maximum(transverse, 1e-12))
    angle = np.arctan2(summed[:, 1], summed[:, 0])
    target[:, col["sin_phi"]], target[:, col["cos_phi"]] = np.sin(angle), np.cos(angle)
    target[:, col["energy"]] = summed[:, 3]
    target[:, col["particle_number"]] = np.arange(1, len(kept) + 1)
    target[:, col["jet_idx"]] = -1
    for row, particle in enumerate(kept):
        pdg = abs(int(target[row, col["PDG"]]))
        if pi0_as_photon and pdg == 111:
            pdg = 22
        pid = map_pdgid_to_candid(pdg, charge[particle])
        tracked = representative[particle] < n_track
        target[row, col["PDG"]] = map_neutral_to_charged(pid) if tracked else map_charged_to_neutral(pid)
        if not tracked:
            target[row, col["charge"]] = 0

    jets_array = np.zeros((0, 4), dtype=np.float32)
    if len(kept):
        vectors = vector.awk(ak.zip(dict(px=summed[:, 0], py=summed[:, 1], pz=summed[:, 2], energy=summed[:, 3])))
        jets, constituents = compute_jets(vectors, min_pt=jet_ptcut, jetdef=jetdef, with_indices=True)
        jets_array = np.column_stack([ak.to_numpy(x) for x in (jets.pt, jets.eta, jets.phi, jets.energy)]).astype(np.float32)
        for jet, indices in enumerate(ak.to_list(constituents)):
            target[indices, col["jet_idx"]] = jet

    def scatter(matrix, offset):
        edges = matrix.tocoo()
        valid = final_id[edges.row] >= 0
        merged = coo_matrix((edges.data[valid], (final_id[edges.row[valid]], edges.col[valid])), shape=(len(kept), matrix.shape[1])).tocsc()
        merged.sum_duplicates()
        merged.sort_indices()
        result = np.zeros((matrix.shape[1], len(names)), dtype=np.float32)
        for element in range(matrix.shape[1]):
            start, stop = merged.indptr[element : element + 2]
            if start != stop:
                winner = merged.indices[start + np.argmax(merged.data[start:stop])]
                result[element, col["particle_number"]] = winner + 1
        for row, particle in enumerate(kept):
            element = representative[particle] - offset
            if 0 <= element < len(result):
                result[element] = target[row]
        return result

    ytrack = scatter(tracks, 0)
    ycalo = scatter(hits, n_track)
    invisible_energy = float(energy[~visible].sum())
    dropped_energy = float(energy[visible & ~mapped].sum())
    accounting = np.array([energy.sum(), invisible_energy, dropped_energy, summed[:, 3].sum()], dtype=np.float64)
    if not np.isclose(accounting[0], accounting[1:].sum(), rtol=1e-10, atol=1e-8):
        raise AssertionError("Track/calo target energy accounting failed")
    return {
        "ytarget_track_calo_track": ytrack,
        "ytarget_track_calo_hit": ycalo,
        "track_calo_target_particles": target,
        "track_calo_original_to_target": final_id,
        "track_calo_energy_accounting": accounting,
        "targetjet_track_calo": jets_array,
    }
