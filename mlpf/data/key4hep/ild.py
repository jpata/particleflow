"""ILD collection and PF representation contracts shared by conversion and TFDS."""

import numpy as np

PROVENANCE = {
    "metadata_detector": "ILD_FCCee_v02",
    "metadata_track_source": "clupatra_kaltest",
    "metadata_track_collection": "MarlinTrkTracks",
    "metadata_target_policy": "ild-direct-hit-majority-v1",
    "metadata_track_truth_min_purity": 0.75,
    "metadata_candidate_source": "pandora",
    "metadata_candidate_assignment": "one-pfo-one-element-v1",
}


def track_truth(prop, links, event, collection_ids, tracker_relations, gen_features):
    """Direct charged-primary labels from distinct measured hits, at 75% purity.

    Every distinct measurement supplies at most one vote; duplicate relations
    cannot inflate purity. Shared hits split their vote between distinct MC
    contributors. Unlinked hits remain in the denominator. Secondary owners
    compete for the majority but never acquire primary ancestor momenta.
    """
    measurement_owners = {}
    mc_id = collection_ids["MCParticles"]
    for hit, (relation, sim) in tracker_relations.items():

        def values(name):
            return np.asarray(links[name][event])

        from_index = values(f"_{relation}_from/_{relation}_from.index")
        from_id = values(f"_{relation}_from/_{relation}_from.collectionID")
        to_index = values(f"_{relation}_to/_{relation}_to.index")
        to_id = values(f"_{relation}_to/_{relation}_to.collectionID")
        if np.any(from_id != collection_ids[hit]) or np.any(to_id != collection_ids[sim]):
            raise ValueError(f"ILD tracker relation {relation} has unexpected collection IDs")
        particles = values(f"_{sim}_particle/_{sim}_particle.index")
        particle_ids = values(f"_{sim}_particle/_{sim}_particle.collectionID")
        for measurement, simulated in zip(from_index, to_index):
            if simulated < 0 or simulated >= len(particles) or particle_ids[simulated] != mc_id:
                raise ValueError(f"ILD tracker relation {relation} has invalid simulation truth")
            gp = int(particles[simulated])
            if not 0 <= gp < len(gen_features["PDG"]):
                raise ValueError(f"ILD tracker relation {relation} has invalid MC index")
            measurement_owners.setdefault((collection_ids[hit], int(measurement)), set()).add(gp)

    prefix = "_MarlinTrkTracks_trackerHits/_MarlinTrkTracks_trackerHits"
    hit_indices = np.asarray(prop[prefix + ".index"][event])
    hit_ids = np.asarray(prop[prefix + ".collectionID"][event])
    tracks = prop["MarlinTrkTracks"][event]
    gp_indices, track_indices, purities = [], [], []
    for track, (begin, end) in enumerate(zip(tracks["MarlinTrkTracks.trackerHits_begin"], tracks["MarlinTrkTracks.trackerHits_end"])):
        measurements = set(zip(hit_ids[begin:end].tolist(), hit_indices[begin:end].tolist()))
        if not measurements:
            continue
        votes = {}
        for measurement in sorted(measurements):
            owners = measurement_owners.get(measurement, ())
            for gp in sorted(owners):
                votes[gp] = votes.get(gp, 0.0) + 1.0 / len(owners)
        if not votes:
            continue
        gp = min(votes, key=lambda index: (-votes[index], index))
        purity = votes[gp] / len(measurements)
        if gen_features["generatorStatus"][gp] == 1 and gen_features["charge"][gp] != 0 and purity >= PROVENANCE["metadata_track_truth_min_purity"]:
            gp_indices.append(gp)
            track_indices.append(track)
            purities.append(purity)
    return np.asarray(gp_indices, dtype=np.int32), np.asarray(track_indices, dtype=np.int32), np.asarray(purities, dtype=np.float32)


def pfo_representatives(reco, track_indices, cluster_indices, track_momenta, cluster_energies):
    """Preserve each complete Pandora PFO once, including multi-track V0s.

    Charged PFOs prefer their highest-momentum track; neutrals prefer their
    highest-energy cluster. If that input kind is absent, use the other kind.
    Equal scores prefer the lowest native index. No truth information is used.
    """
    tracks, clusters = {}, {}
    for particle in range(len(reco["charge"])):
        tr = np.asarray(track_indices[int(reco["tracks_begin"][particle]) : int(reco["tracks_end"][particle])], dtype=int)
        cl = np.asarray(cluster_indices[int(reco["clusters_begin"][particle]) : int(reco["clusters_end"][particle])], dtype=int)
        prefer_track = len(tr) and (reco["charge"][particle] != 0 or not len(cl))
        if prefer_track:
            index = min(tr, key=lambda i: (-track_momenta[i], i))
            mapping = tracks
        elif len(cl):
            index = min(cl, key=lambda i: (-cluster_energies[i], i))
            mapping = clusters
        else:
            raise ValueError(f"ILD Pandora PFO {particle} has neither a track nor a cluster")
        if index in mapping:
            raise ValueError(f"ILD Pandora PFOs share representative input {index}")
        mapping[int(index)] = particle
    return tracks, clusters
