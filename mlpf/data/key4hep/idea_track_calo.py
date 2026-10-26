"""IDEA track/calo inputs: truth-seeded tracks and separate dual-readout channels."""

import awkward as ak
import numpy as np
from scipy.sparse import coo_matrix

from mlpf.conf import EDM4HEP, ParticleFeatures
from mlpf.data.target_building import EventData, filter_adj
from mlpf.data.track_calo_targets import build_track_calo_targets

CHANNELS = {"DRcaloSiPMreadoutDigiHit": 1, "DRcaloSiPMreadoutDigiHit_scint": 2}
LINK = "CaloHitMCParticleLinks"


def required_branches():
    return (
        list(CHANNELS)
        + [f"{LINK}.weight"]
        + [f"_{LINK}_{end}/_{LINK}_{end}.{field}" for end in ("from", "to") for field in ("index", "collectionID")]
    )


def hit_inputs_and_links(prop_data, iev, collection_ids, n_particles):
    """Keep positive measured signals, including unlinked background hits.

    Stored truth weights are per-hit response fractions. Multiply by the measured
    channel energy before ancestry propagation; do not normalize ancestor copies.
    Collection IDs, not local indices or cell IDs, distinguish the two readouts.
    """
    arrays, remaps = [], {}
    offset = 0
    for collection, channel in CHANNELS.items():
        hits = prop_data[collection][iev]
        energy = np.asarray(hits[f"{collection}.energy"], dtype=np.float64)
        if not np.isfinite(energy).all() or np.any(energy < 0):
            raise ValueError(f"Invalid IDEA {collection} energy")
        keep = energy > 0
        remap = np.full(len(energy), -1, dtype=np.int64)
        remap[keep] = offset + np.arange(keep.sum())
        remaps[int(collection_ids[collection])] = remap
        offset += int(keep.sum())
        xyz = np.column_stack([np.asarray(hits[f"{collection}.position.{axis}"])[keep] for axis in "xyz"])
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        norm = np.linalg.norm(xyz, axis=1)
        if np.any(norm == 0) or not np.isfinite(xyz).all():
            raise ValueError("Invalid positive-energy IDEA hit position")
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        x = np.zeros((int(keep.sum()), len(EDM4HEP.HitFeatures.get_names())), dtype=np.float32)
        x[:, 0] = 2
        x[:, 1] = energy[keep] * radius / norm
        x[:, 2] = np.arcsinh(xyz[:, 2] / np.maximum(radius, 1e-12))
        x[:, 3], x[:, 4], x[:, 5] = np.sin(phi), np.cos(phi), energy[keep]
        x[:, 6:9] = xyz
        x[:, 9] = np.asarray(hits[f"{collection}.time"])[keep]
        x[:, 10] = 2  # Dual-readout calorimeter; neither ECAL nor HCAL.
        x[:, 11] = channel  # 1=Cherenkov, 2=scintillation (metadata documents this).
        # system/side/layer remain unknown; do not guess a DD4hep encoding.
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite IDEA hit features")
        arrays.append(x)
    x = np.concatenate(arrays)
    endpoints = {}
    for end in ("from", "to"):
        for field in ("index", "collectionID"):
            name = f"_{LINK}_{end}"
            endpoints[end, field] = np.asarray(prop_data[f"{name}/{name}.{field}"][iev])
    weights = np.asarray(prop_data[f"{LINK}.weight"][iev], dtype=np.float64)
    if any(len(value) != len(weights) for value in endpoints.values()) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Malformed IDEA hit truth links")
    gp, hit, response = [], [], []
    for i, weight in enumerate(weights):
        collection = int(endpoints["from", "collectionID"][i])
        local_index = int(endpoints["from", "index"][i])
        particle = int(endpoints["to", "index"][i])
        if collection not in remaps or int(endpoints["to", "collectionID"][i]) != int(collection_ids["MCParticles"]):
            raise ValueError("Unexpected collection ID in IDEA hit truth links")
        if not 0 <= local_index < len(remaps[collection]) or not 0 <= particle < n_particles:
            raise ValueError("Out-of-range IDEA hit truth link")
        global_index = remaps[collection][local_index]
        if global_index >= 0 and weight > 0:
            gp.append(particle)
            hit.append(global_index)
            response.append(weight * float(x[global_index, 5]))
    return x, (np.asarray(gp, dtype=int), np.asarray(hit, dtype=int), np.asarray(response, dtype=float))


def build_idea_track_calo(prop_data, iev, collection_ids, gen, tracks, direct_tracks, jetdef, jet_ptcut):
    from mlpf.data.key4hep.postprocessing import add_daughters_to_status1

    n_gp, n_track = len(gen["PDG"]), len(tracks["type"])
    x, hit_links = hit_inputs_and_links(prop_data, iev, collection_ids, n_gp)
    hit_links, inclusive_tracks = add_daughters_to_status1(gen, hit_links, direct_tracks)
    keep = (np.asarray(gen["generatorStatus"]) == 1) & ~np.isin(np.abs(np.asarray(gen["PDG"])), [12, 14, 16])
    mapping = {old: new for new, old in enumerate(np.flatnonzero(keep))}
    hits, links, representatives = (filter_adj(edges, mapping) for edges in (hit_links, inclusive_tracks, direct_tracks))
    count = int(keep.sum())
    features = {
        name: np.asarray(gen[name])[keep]
        for name in ParticleFeatures.get_names() + ["phi"]
        if name not in ("gp_to_track", "gp_to_cluster", "particle_number")
    }
    features["gp_to_cluster"] = np.asarray(coo_matrix((hits[2], (hits[0], hits[1])), shape=(count, len(x))).sum(axis=1)).ravel()
    direct = coo_matrix((representatives[2], (representatives[0], representatives[1])), shape=(count, n_track)).tocsr()
    features["gp_to_track"] = direct.max(axis=1).toarray().ravel() if count and n_track else np.zeros(count)
    features["particle_number"] = np.zeros(count)
    data = EventData(ak.Record(features), ak.Record({"type": np.zeros(len(x))}), ak.Record({"type": []}), tracks, hits, links, ([], [], []), ([], []))
    targets = build_track_calo_targets(
        data, np.ones(len(x), dtype=bool), jetdef=jetdef, jet_ptcut=jet_ptcut, representative_track_edges=representatives
    )
    # Use a view-specific input column: legacy IDEA raw-hit columns stay empty.
    return {"X_track_calo_hit": x, **targets}
