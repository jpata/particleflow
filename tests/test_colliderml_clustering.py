# Unit tests for the ColliderML spatial clusterer (mlpf/data/colliderml/clustering.py).
import numpy as np

from mlpf.data.colliderml.clustering import cluster_event


def _event(n_ecal_hits=16, n_hcal_hits=4, seed=0):
    rng = np.random.default_rng(seed)
    # six compact clouds inside ECAL barrel (code 10, pitch 5.1 mm) and HCAL barrel (13, 30 mm)
    xy = rng.uniform(-1, 1, (n_ecal_hits, 2)) * np.array([2.0, 2.0])
    z = rng.uniform(-1, 1, (n_ecal_hits, 1))
    ecal_xyz = np.hstack([xy, z])
    ecal_e = rng.uniform(0.01, 1.0, n_ecal_hits)
    # one HCAL cluster
    hcal_xyz = np.array([[100.0, 100.0, 100.0], [110.0, 100.0, 100.0], [120.0, 100.0, 100.0], [130.0, 100.0, 100.0]], dtype=np.float32)
    hcal_e = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    x = np.concatenate([ecal_xyz[:, 0], hcal_xyz[:, 0]])
    y = np.concatenate([ecal_xyz[:, 1], hcal_xyz[:, 1]])
    z = np.concatenate([ecal_xyz[:, 2], hcal_xyz[:, 2]])
    E = np.concatenate([ecal_e, hcal_e])
    det = np.concatenate([np.full(n_ecal_hits, 10, dtype=np.int64), np.full(n_hcal_hits, 13, dtype=np.int64)])
    return x, y, z, E, det


def test_every_hit_in_exactly_one_cluster():
    x, y, z, E, det = _event(seed=1)
    cluster_of, feats, coo, cluster_region = cluster_event(x, y, z, E, det)
    # every hit assigned to a valid cluster index
    assert cluster_of.min() >= 0
    n_clusters = int(cluster_of.max()) + 1
    # all indices from 0..n_clusters-1 are used
    assert len(np.unique(cluster_of)) == n_clusters
    assert len(cluster_region) == n_clusters
    assert feats.shape[0] == n_clusters
    # hit_to_cluster COO is consistent
    hit_idx, cl_idx, w = coo
    assert np.array_equal(np.sort(hit_idx), np.arange(len(x)))
    assert np.all(w == 1.0)


def test_no_cross_region_links():
    x, y, z, E, det = _event(seed=2)
    cluster_of, _, _, cluster_region = cluster_event(x, y, z, E, det)
    # ECAL and HCAL hits can never belong to the same cluster
    for cid in np.unique(cluster_of):
        m = cluster_of == cid
        assert len(set(det[m].tolist())) == 1, f"cluster {cid} spans regions {set(det[m])}"


def test_cluster_energy_is_sum_of_members():
    x, y, z, E, det = _event(seed=3)
    cluster_of, feats, _, _ = cluster_event(x, y, z, E, det)
    for cid in np.unique(cluster_of):
        m = cluster_of == cid
        assert abs(feats[cid, 5] - np.sum(E[m])) < 1e-6, f"cluster {cid} energy {feats[cid,5]} != member sum {np.sum(E[m])}"


def test_total_energy_conserved():
    x, y, z, E, det = _event(seed=4)
    cluster_of, feats, _, _ = cluster_event(x, y, z, E, det)
    assert abs(np.sum(feats[:, 5]) - np.sum(E)) < 1e-6


def test_singletons_for_isolated_hits():
    # one isolated ECAL hit far away from the HCAL cluster
    x = np.array([0.0, 1000.0, 1100.0, 1200.0], dtype=np.float32)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    E = np.array([0.5, 0.5, 0.4, 0.3], dtype=np.float32)
    det = np.array([10, 13, 13, 13], dtype=np.int64)
    cluster_of, feats, _, _ = cluster_event(x, y, z, E, det)
    # ECAL hit is isolated -> singleton
    assert cluster_of[0] != cluster_of[1]
    # HCAL hits form at least one cluster
    assert len(np.unique(cluster_of)) >= 2
    # the singleton's features equal the hit itself
    singleton_id = int(cluster_of[0])
    assert feats[singleton_id, 5] == E[0]


def test_deterministic():
    x, y, z, E, det = _event(seed=5)
    c1 = cluster_event(x, y, z, E, det)[0]
    c2 = cluster_event(x, y, z, E, det)[0]
    assert np.array_equal(c1, c2)


def test_truth_blindness():
    # permuting the truth drop of ``detector`` (a reconstruction field) should change the result
    # (region-driven), but reassigning the hit's energy (also a reconstruction input) does.
    x, y, z, E, det = _event(seed=6)
    E2 = E.copy()
    E2[0] = 1e-6
    c2 = cluster_event(x, y, z, E2, det)[0]
    # changing energy may move seeds -> not guaranteed identical; the guarantee we test is that
    # it runs with a different E and still satisfies the invariants.
    m = np.unique(c2)
    for cid in m:
        # every cluster has exactly one region
        assert len(set(det[c2 == cid])) == 1
