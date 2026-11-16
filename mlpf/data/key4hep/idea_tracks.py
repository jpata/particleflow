"""IDEA fitted-track selection and training-only hit truth matching.

Input selection uses reconstructed quantities only. MC information is used
afterwards to build labels, never to replace or correct track measurements.
"""

from collections import defaultdict

import awkward as ak
import numpy as np

TRACK_SOURCE = "ggtf_genfit"
PROXY_TRACK_SOURCE = "truth_seeded"
TRACK_COLLECTION = "FittedTracks"
TRACK_QUALITY_POLICY = "finite_at_ip_chi2_ge_0_ndf_gt_0_with_hits"
TRACK_TRUTH_POLICY = "direct_charged_status1_hit_majority"
TRACK_TRUTH_MIN_PURITY = 0.75

# FittedTracks retains original hit references. The filtered-hit output instead
# refers to Fitted_hits clones without the original sim-hit associations.
TRACK_HIT_COLLECTIONS = ("VTXBDigis", "VTXDDigis", "SiWrBDigis", "SiWrDDigis", "DCHDigis")
TRACK_HIT_LINKS = {
    "VTXBSimDigiLinks": "VertexBarrelCollection",
    "VTXDSimDigiLinks": "VertexEndcapCollection",
    "SiWrBSimDigiLinks": "SiWrBCollection",
    "SiWrDSimDigiLinks": "SiWrDCollection",
    "DCHDigisSimAssociationCollection": "DCHCollection",
}


def validate_track_provenance(data, expected_track_source=TRACK_SOURCE):
    """Reject silent proxy/fitted mixing, including unmarked legacy input."""
    if expected_track_source not in {TRACK_SOURCE, PROXY_TRACK_SOURCE}:
        raise ValueError(f"Unknown IDEA track provenance: {expected_track_source}")
    if "metadata_track_source" not in data.fields:
        if expected_track_source == PROXY_TRACK_SOURCE:
            return PROXY_TRACK_SOURCE  # explicit legacy diagnostic/builder only
        raise ValueError("IDEA parquet lacks fitted-track provenance; regenerate it. Legacy proxy files require explicit truth_seeded mode.")
    num_events = len(data["X_track"])
    if len(data["metadata_track_source"]) != num_events:
        raise ValueError("IDEA track provenance length does not match event count")
    sources = set(ak.to_list(data["metadata_track_source"]))
    if sources != {expected_track_source}:
        raise ValueError(f"IDEA track provenance {sources} does not match expected {expected_track_source}")
    if expected_track_source == TRACK_SOURCE:
        expected = {
            "metadata_track_collection": TRACK_COLLECTION,
            "metadata_track_quality_policy": TRACK_QUALITY_POLICY,
            "metadata_track_truth_policy": TRACK_TRUTH_POLICY,
            "metadata_track_truth_min_purity": TRACK_TRUTH_MIN_PURITY,
        }
        for field, value in expected.items():
            if field not in data.fields or len(data[field]) != num_events or any(item != value for item in ak.to_list(data[field])):
                raise ValueError(f"IDEA fitted-track provenance requires {field}={value}")
    return expected_track_source


def fitted_track_selection(prop_data, event, b_field):
    """Return original track indices and unique AtIP state indices after quality cuts."""
    tracks = prop_data[TRACK_COLLECTION][event]
    states = prop_data[f"_{TRACK_COLLECTION}_trackStates"][event]
    track_arrays = {
        field: ak.to_numpy(tracks[f"{TRACK_COLLECTION}.{field}"])
        for field in ["trackStates_begin", "trackStates_end", "trackerHits_begin", "trackerHits_end", "chi2", "ndf"]
    }
    state_arrays = {
        field: ak.to_numpy(states[f"_{TRACK_COLLECTION}_trackStates.{field}"])
        for field in ["location", "tanLambda", "D0", "phi", "omega", "Z0", "time", "referencePoint.x", "referencePoint.y", "referencePoint.z"]
    }

    def t(field):
        return track_arrays[field]

    def s(field):
        return state_arrays[field]

    locations = s("location")
    selected, selected_states = [], []
    for index, (begin, end) in enumerate(zip(t("trackStates_begin"), t("trackStates_end"))):
        if begin < 0 or end < begin or end > len(locations):
            raise ValueError("IDEA fitted-track state range is invalid")
        at_ip = np.flatnonzero(locations[begin:end] == 1) + begin  # edm4hep::TrackState::AtIP
        if len(at_ip) != 1:
            continue
        state = int(at_ip[0])
        values = [
            s(field)[state] for field in ("tanLambda", "D0", "phi", "omega", "Z0", "time", "referencePoint.x", "referencePoint.y", "referencePoint.z")
        ]
        chi2, ndf = t("chi2")[index], t("ndf")[index]
        if not np.isfinite(values).all() or not np.isfinite(chi2) or chi2 < 0 or ndf <= 0:
            continue
        omega, tanlambda = s("omega")[state], s("tanLambda")[state]
        if omega == 0 or t("trackerHits_end")[index] <= t("trackerHits_begin")[index]:
            continue
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            momentum = 0.000299792458 * abs(b_field / omega) * np.hypot(1.0, tanlambda)
        if not np.isfinite(momentum) or momentum <= 0 or momentum > np.finfo(np.float32).max:
            continue
        selected.append(index)
        selected_states.append(state)
    return np.asarray(selected, dtype=np.int64), np.asarray(selected_states, dtype=np.int64)


def fitted_track_truth(prop_data, event, collection_ids, track_indices, min_purity=TRACK_TRUTH_MIN_PURITY):
    """Match accepted tracks to direct charged status-1 truth by weighted hit votes.

    Each distinct measurement contributes at most one vote. Unlinked hits and
    null particle links remain in the purity denominator. Low-purity tracks,
    secondaries, and fakes stay in X but receive no track representative label.
    """
    if not 0.5 < min_purity <= 1.0:
        raise ValueError("IDEA track truth purity must be greater than 0.5 and at most 1")

    def a(name):
        return ak.to_numpy(prop_data[name][event])

    n_mc = len(a("MCParticles.PDG"))
    hit_truth = defaultdict(list)
    for links, simulation in TRACK_HIT_LINKS.items():
        from_prefix = f"_{links}_from/_{links}_from."
        to_prefix = f"_{links}_to/_{links}_to."
        sim_particles = a(f"_{simulation}_particle/_{simulation}_particle.index")
        sim_particle_ids = a(f"_{simulation}_particle/_{simulation}_particle.collectionID")
        for fi, fc, ti, tc, weight in zip(
            a(from_prefix + "index"), a(from_prefix + "collectionID"), a(to_prefix + "index"), a(to_prefix + "collectionID"), a(links + ".weight")
        ):
            if int(tc) != collection_ids[simulation] or ti < 0 or ti >= len(sim_particles):
                raise ValueError(f"IDEA {links} has an invalid simulation-hit reference")
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"IDEA {links} has an invalid association weight")
            mc = int(sim_particles[ti])
            if mc >= n_mc or (mc >= 0 and int(sim_particle_ids[ti]) != collection_ids["MCParticles"]):
                raise ValueError(f"IDEA {simulation} has an invalid MCParticle reference")
            hit_truth[int(fc), int(fi)].append((mc, float(weight)))

    positions = {collection_ids[name]: np.column_stack([a(f"{name}.position.{axis}") for axis in "xyz"]) for name in TRACK_HIT_COLLECTIONS}
    tracks = prop_data[TRACK_COLLECTION][event]
    begin = ak.to_numpy(tracks[f"{TRACK_COLLECTION}.trackerHits_begin"])
    end = ak.to_numpy(tracks[f"{TRACK_COLLECTION}.trackerHits_end"])
    hit_indices = a(f"_{TRACK_COLLECTION}_trackerHits/_{TRACK_COLLECTION}_trackerHits.index")
    hit_collections = a(f"_{TRACK_COLLECTION}_trackerHits/_{TRACK_COLLECTION}_trackerHits.collectionID")
    status, charge = a("MCParticles.generatorStatus"), a("MCParticles.charge")
    gp, trk, purity = [], [], []
    radii, hit_counts = [], []
    for compressed_index, original_index in enumerate(track_indices):
        lo, hi = begin[original_index], end[original_index]
        if lo < 0 or hi < lo or hi > len(hit_indices):
            raise ValueError("IDEA fitted-track hit range is invalid")
        hits = sorted(set(zip(hit_collections[lo:hi].tolist(), hit_indices[lo:hi].tolist())))
        votes = defaultdict(float)
        hit_radii = []
        for collection, index in hits:
            if collection not in positions or index < 0 or index >= len(positions[collection]):
                raise ValueError("IDEA fitted track refers to an unknown or invalid original hit")
            pos = positions[collection][index]
            if not np.isfinite(pos).all():
                raise ValueError("IDEA fitted track contains a hit with non-finite position")
            hit_radii.append(np.hypot(pos[0], pos[1]))
            contributions = hit_truth.get((collection, index), [])
            total = sum(weight for _, weight in contributions)
            if total > 0:
                for mc, weight in contributions:
                    if mc >= 0:
                        votes[mc] += weight / total
        radii.append(min(hit_radii) if hit_radii else 0.0)
        hit_counts.append(len(hits))
        if votes:
            winner = max(sorted(votes), key=votes.get)
            fraction = votes[winner] / len(hits)
            if fraction >= min_purity and status[winner] == 1 and charge[winner] != 0:
                gp.append(winner)
                trk.append(compressed_index)
                purity.append(fraction)
    relation = (np.asarray(gp, dtype=np.int64), np.asarray(trk, dtype=np.int64), np.asarray(purity, dtype=np.float64))
    return relation, np.asarray(radii, dtype=np.float32), np.asarray(hit_counts, dtype=np.int64)


def preferred_track_representatives(relation, tracks, hit_counts):
    """Choose a duplicate's representative without changing input order or measurements."""
    preferred, scores = {}, {}
    chi2, ndf = np.asarray(tracks["chi2"]), np.asarray(tracks["ndf"])
    for particle, track, purity in zip(*relation):
        score = (purity, hit_counts[track], -chi2[track] / ndf[track], -track)
        if particle not in scores or score > scores[particle]:
            preferred[int(particle)] = int(track)
            scores[particle] = score
    return preferred
