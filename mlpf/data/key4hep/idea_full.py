"""IDEA inputs and targets for the general (full) dataset view.

Tracker hits come from the vertex detector, the silicon wrapper and the drift chamber
(DCH). Calorimeter hits are the separate dual-readout Cherenkov and scintillation
channels. Tracks are the truth-seeded TracksFromGenParticles. Targets and per-element
labels follow mlpf/data/full_targets.py; producers come from the direct links:
digi -> SimTrackerHit -> MCParticle for tracker hits, CaloHitMCParticleLinks for
calorimeter hits, TracksFromGenParticlesAssociation for tracks and
ClusterMCParticleLinks for clusters.
"""

import numpy as np
from scipy.sparse import coo_matrix

from mlpf.conf import EDM4HEP
from mlpf.data.full_targets import build_full_targets
from mlpf.data.target_building import visibility_mask

CHANNELS = {"DRcaloSiPMreadoutDigiHit": 1, "DRcaloSiPMreadoutDigiHit_scint": 2}
LINK = "CaloHitMCParticleLinks"


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


# digi collection -> (SimTrackerHit collection, digi->sim link collection, system code).
# System codes follow the CLD tracker convention where possible (1 vertex barrel, 2 vertex
# endcap, 5 outer barrel, 6 outer endcap); 3 marks the drift chamber.
TRACKER_DIGIS = {
    "VTXBDigis": ("VertexBarrelCollection", "VTXBSimDigiLinks", 1),
    "VTXDDigis": ("VertexEndcapCollection", "VTXDSimDigiLinks", 2),
    "DCH_DigiCollection": ("DCHCollection", "DCH_DigiSimAssociationCollection", 3),
    "SiWrBDigis": ("SiWrBCollection", "SiWrBSimDigiLinks", 5),
    "SiWrDDigis": ("SiWrDCollection", "SiWrDSimDigiLinks", 6),
}
DCH_SYSTEM = 3
HIT_FEATURES = EDM4HEP.HitFeatures.get_names()


def required_branches():
    branches = list(CHANNELS) + [f"{LINK}.weight"]
    branches += [f"_{LINK}_{end}/_{LINK}_{end}.{field}" for end in ("from", "to") for field in ("index", "collectionID")]
    for digi, (sim, link, _) in TRACKER_DIGIS.items():
        branches += [f"{digi}/{digi}.{field}" for field in ("eDep", "time", "position.x", "position.y", "position.z")]
        branches += [f"{sim}/{sim}.{field}" for field in ("quality", "momentum.x", "momentum.y", "momentum.z")]
        branches += [f"_{sim}_particle/_{sim}_particle.{field}" for field in ("index", "collectionID")]
        branches += [f"_{link}_{end}/_{link}_{end}.index" for end in ("from", "to")]
    return branches


def tracker_hit_features(prop_data, iev, mc_collection_id):
    """Tracker hit feature rows plus the direct producer, simulated momentum and unsaved-secondary flag."""
    rows, producers, momenta, unsaved = [], [], [], []
    for digi, (sim, link, system) in TRACKER_DIGIS.items():
        get = lambda name: np.asarray(prop_data[name][iev])  # noqa: E731
        xyz = np.column_stack([get(f"{digi}/{digi}.position.{axis}") for axis in "xyz"]).astype(np.float64)
        n = len(xyz)
        energy = get(f"{digi}/{digi}.eDep").astype(np.float64)
        x = np.zeros((n, len(HIT_FEATURES)), dtype=np.float32)
        if n:
            # Same conventions as key4hep hits_to_features: direction cosines scaled by the deposit.
            norm = np.linalg.norm(xyz, axis=1)
            if np.any(norm == 0) or not np.isfinite(xyz).all():
                raise ValueError(f"Invalid IDEA {digi} hit position")
            direction = xyz / norm[:, None]
            eps = 1e-12
            px, py, pz = (direction[:, i] * energy for i in range(3))
            x[:, 1] = np.hypot(px, py)
            x[:, 2] = 0.5 * np.log((energy + pz + eps) / (energy - pz + eps))
            x[:, 3] = py / (energy + eps)
            x[:, 4] = px / (energy + eps)
        x[:, 0] = 1
        x[:, 5] = energy
        x[:, 6:9] = xyz
        x[:, 9] = get(f"{digi}/{digi}.time")
        x[:, 10] = 3  # tracker
        x[:, 11] = 1 if system == DCH_SYSTEM else 0  # 1 = drift-chamber wire hit (position on the sense wire)
        x[:, 12] = system
        # side and layer stay zero: the IDEA files carry no CellIDEncoding metadata.

        digi_index = get(f"_{link}_from/_{link}_from.index").astype(np.int64)
        sim_index = get(f"_{link}_to/_{link}_to.index").astype(np.int64)
        sim_of_digi = np.full(n, -1, dtype=np.int64)
        valid = (digi_index >= 0) & (digi_index < n)
        sim_of_digi[digi_index[valid]] = sim_index[valid]
        sim_particle = get(f"_{sim}_particle/_{sim}_particle.index").astype(np.int64)
        sim_particle_collection = get(f"_{sim}_particle/_{sim}_particle.collectionID").astype(np.int64)
        sim_momentum = np.column_stack([get(f"{sim}/{sim}.momentum.{axis}") for axis in "xyz"]).astype(np.float32)
        sim_quality = get(f"{sim}/{sim}.quality").astype(np.int64)
        linked = (sim_of_digi >= 0) & (sim_of_digi < len(sim_particle))
        producer = np.full(n, -1, dtype=np.int64)
        momentum = np.zeros((n, 3), dtype=np.float32)
        flag = np.zeros(n, dtype=bool)
        sims = sim_of_digi[linked]
        to_mc = sim_particle_collection[sims] == mc_collection_id
        producer[np.flatnonzero(linked)[to_mc]] = sim_particle[sims[to_mc]]
        momentum[linked] = sim_momentum[sims]
        flag[linked] = (sim_quality[sims] & (1 << 30)) != 0
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite IDEA {digi} hit features")
        rows.append(x)
        producers.append(producer)
        momenta.append(momentum)
        unsaved.append(flag)
    return np.concatenate(rows), np.concatenate(producers), np.concatenate(momenta), np.concatenate(unsaved)


def build_idea_full(
    prop_data, iev, collection_ids, gen, n_track, n_cluster, direct_tracks, direct_clusters, cluster_energy, source_energy, jetdef, jet_ptcut
):
    """Full-view hit inputs, targets and labels for one IDEA event."""
    from mlpf.data.key4hep.postprocessing import add_daughters_to_status1, full_truth_links, mc_coll

    n_gp = len(gen["PDG"])
    x_tracker, tracker_producer, tracker_momentum, tracker_unsaved = tracker_hit_features(prop_data, iev, int(collection_ids[mc_coll]))
    x_calo, calo_links = hit_inputs_and_links(prop_data, iev, collection_ids, n_gp)
    n_tracker = len(x_tracker)
    linked = (tracker_producer >= 0) & (tracker_producer < n_gp)
    tracker_links = (tracker_producer[linked], np.flatnonzero(linked), x_tracker[linked, 5].astype(np.float64))
    mc_to_hit = tuple(np.concatenate([a, b]) for a, b in zip(tracker_links, (calo_links[0], calo_links[1] + n_tracker, calo_links[2])))

    # Visible status-1 targets: the track/calo rule of the previous IDEA view. Calorimeter
    # responses of descendants count for their status-1 ancestor; only direct status-1 track
    # links establish track support (truth-seeded descendant tracks do not).
    inclusive_calo, _ = add_daughters_to_status1(gen, calo_links, direct_tracks)
    calo_response = np.asarray(
        coo_matrix((inclusive_calo[2], (inclusive_calo[0], inclusive_calo[1])), shape=(n_gp, max(len(x_calo), 1))).sum(axis=1)
    ).ravel()
    track_weight = np.zeros(n_gp)
    if len(direct_tracks[0]):
        np.maximum.at(track_weight, np.asarray(direct_tracks[0], dtype=np.int64), np.asarray(direct_tracks[2], dtype=np.float64))
    energy = np.asarray(gen["energy"], dtype=np.float64)
    charge = np.asarray(gen["charge"], dtype=np.float64)
    status1 = (np.asarray(gen["generatorStatus"]) == 1) & ~np.isin(np.abs(np.asarray(gen["PDG"])), [12, 14, 16])
    visible = status1 & visibility_mask(calo_response, energy, charge != 0, track_weight >= 0.2)
    targets = np.flatnonzero(visible)

    hit_features = {
        "type": np.zeros(n_tracker + len(x_calo)),
        "subdetector": np.concatenate([x_tracker[:, 10], x_calo[:, 10]]),
        **{f"position.{axis}": np.concatenate([x_tracker[:, 6 + i], x_calo[:, 6 + i]]) for i, axis in enumerate("xyz")},
    }
    sim_info = {
        "momentum": np.concatenate([tracker_momentum, np.zeros((len(x_calo), 3), dtype=np.float32)]),
        "unsaved_secondary": np.concatenate([tracker_unsaved, np.zeros(len(x_calo), dtype=bool)]),
    }
    truth = full_truth_links(prop_data, iev, gen, targets, mc_to_hit, direct_tracks, ([], [], []), hit_features, sim_info)
    truth.is_tracker_hit = np.arange(n_tracker + len(x_calo)) < n_tracker
    cluster_mc, cluster_index, fraction = (np.asarray(x) for x in direct_clusters)
    truth.mc_to_cluster = (
        cluster_mc,
        cluster_index,
        fraction.astype(np.float64) * np.asarray(cluster_energy, dtype=np.float64)[cluster_index.astype(np.int64)],
    )

    target_features = {
        name: np.asarray(gen[name])[targets]
        for name in ("PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "ispu", "generatorStatus", "simulatorStatus")
    }
    target_features["gp_to_track"] = track_weight[targets]
    target_features["gp_to_cluster"] = calo_response[targets]
    full = build_full_targets(truth, target_features, n_track, n_cluster, source_energy=source_energy, jetdef=jetdef, jet_ptcut=jet_ptcut)
    return {"X_hit_tracker": x_tracker, "X_hit_calo": x_calo, **full}
