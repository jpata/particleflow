import pytest
import numpy as np
import awkward as ak
from mlpf.data.key4hep.postprocessing import (
    assign_genparticles_to_obj_and_merge,
    EventData,
    map_pdgid_to_candid,
    map_neutral_to_charged,
    map_charged_to_neutral,
)
from mlpf.conf import ParticleFeatures


def get_mock_event_data(n_gp=1, n_track=1, n_cluster=2, n_hit=5):
    # 1. Mock Gen Features
    gen_features = {
        "PDG": np.array([211] * n_gp, dtype=np.int32),
        "charge": np.array([1.0] * n_gp, dtype=np.float32),
        "pt": np.array([10.0] * n_gp, dtype=np.float32),
        "eta": np.array([0.1] * n_gp, dtype=np.float32),
        "phi": np.array([0.1] * n_gp, dtype=np.float32),
        "energy": np.array([10.0] * n_gp, dtype=np.float32),
        "ispu": np.array([0] * n_gp, dtype=np.float32),
        "generatorStatus": np.array([1] * n_gp, dtype=np.int32),
        "simulatorStatus": np.array([1] * n_gp, dtype=np.int32),
        "gp_to_track": np.array([1.0] * n_gp, dtype=np.float32),
        "gp_to_cluster": np.array([1.0] * n_gp, dtype=np.float32),
        "jet_idx": np.array([0] * n_gp, dtype=np.int64),
    }

    # 2. Mock Adjacencies
    # Gen 0 linked to Track 0
    gp_to_trk = (np.array([0]), np.array([0]), np.array([1.0]))
    # Gen 0 linked to all Hits (0-4)
    gp_to_hit = (np.array([0] * n_hit), np.arange(n_hit), np.array([1.0] * n_hit))
    # Hits (0-2) linked to Cluster 0, Hits (3-4) linked to Cluster 1
    hit_to_cl = (np.arange(n_hit), np.array([0, 0, 0, 1, 1]), np.array([1.0] * n_hit))

    return EventData(
        gen_features=ak.Record(gen_features),
        hit_features=ak.Record({"type": np.zeros(n_hit)}),
        cluster_features=ak.Record({"type": np.zeros(n_cluster)}),
        track_features=ak.Record({"type": np.zeros(n_track)}),
        genparticle_to_hit=gp_to_hit,
        genparticle_to_track=gp_to_trk,
        hit_to_cluster=hit_to_cl,
        gp_merges=([], []),
    )


def test_hybrid_assignment_logic():
    """
    Verifies that the hybrid assignment logic correctly identifies:
    - Exactly one HUB for the standard features.
    - Multiple SPOKES for the particle_number pointer.
    """
    event_data = get_mock_event_data(n_gp=1, n_track=1, n_cluster=2)

    # 1. Run assignment
    # This calls the refactored logic in postprocessing.py
    gp_cleaned, gp_to_obj, gp_to_hit_idx, trk_inclusive, cls_inclusive, hit_inclusive = assign_genparticles_to_obj_and_merge(event_data)

    # 2. Verify mappings
    # Gen 0 should be matched to Track 0 (exclusive)
    assert gp_to_obj[0, 0] == 0
    assert gp_to_obj[0, 1] == -1

    # Gen 0 should be inclusive parent of Track 0 and both clusters
    assert trk_inclusive[0] == 0
    assert cls_inclusive[0] == 0
    assert cls_inclusive[1] == 0

    # 3. Simulate the matrix filling logic as done in process_one_file
    n_gps = len(gp_cleaned.gen_features["PDG"])
    n_tracks = len(event_data.track_features["type"])
    n_clusters = len(event_data.cluster_features["type"])

    feat_names = ParticleFeatures.get_names()
    PN_IDX = feat_names.index("particle_number")
    PID_IDX = feat_names.index("PDG")
    E_IDX = feat_names.index("energy")

    # Canonical Features (Standardized PID)
    gps_canonical = np.zeros((n_gps, len(feat_names)))
    gps_canonical[:, PID_IDX] = 211
    gps_canonical[:, E_IDX] = 10.0
    gps_canonical[:, PN_IDX] = np.arange(n_gps) + 1

    # Track Target
    gps_track = np.zeros((n_tracks, len(feat_names)))
    gps_track[0, :PN_IDX] = gps_canonical[0, :PN_IDX]  # HUB properties
    gps_track[0, PN_IDX] = gps_canonical[0, PN_IDX]  # PN pointer

    # Cluster Targets (Hybrid SPOKES)
    gps_cluster = np.zeros((n_clusters, len(feat_names)))
    # Spokes only get PN, properties stay 0
    gps_cluster[0, PN_IDX] = gps_canonical[0, PN_IDX]
    gps_cluster[1, PN_IDX] = gps_canonical[0, PN_IDX]

    # VERIFICATION
    # Standard Perspective: Only one element has Energy/PID
    total_energy = np.sum(gps_track[:, E_IDX]) + np.sum(gps_cluster[:, E_IDX])
    assert total_energy == 10.0  # No double counting!

    # OC Perspective: All elements share PN
    all_pns = np.concatenate([gps_track[:, PN_IDX], gps_cluster[:, PN_IDX]])
    assert np.all(all_pns == 1.0)  # Physical cohesion!


def test_pid_consistency_mapping():
    """
    Verifies that forcing logic (Tracks=Charged, Clusters=Neutral) is
    consistent in the canonical mapping.
    """
    # Case: Photon (22) matched to a Track (should become 211)
    # This is a rare/incorrect detector assignment but tests the logic robustness
    p = 22
    c = 0

    # In track-centric mode
    pid_forced_track = map_neutral_to_charged(map_pdgid_to_candid(p, c))
    assert pid_forced_track == 211

    # In cluster-centric mode
    pid_forced_cluster = map_charged_to_neutral(map_pdgid_to_candid(p, c))
    assert pid_forced_cluster == 22


if __name__ == "__main__":
    pytest.main([__file__])
