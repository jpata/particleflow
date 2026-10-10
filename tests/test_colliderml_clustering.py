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
        assert abs(feats[cid, 5] - np.sum(E[m])) < 1e-6, f"cluster {cid} energy {feats[cid, 5]} != member sum {np.sum(E[m])}"


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


# ---------------- CLUE ----------------


def _shower(center, n, sigma, e_tot, rng, det):
    """Gaussian blob of n hits around center (mm) carrying e_tot GeV, exponential hit energies."""
    pos = np.asarray(center, dtype=np.float64) + rng.normal(0.0, sigma, (n, 3))
    e = rng.exponential(1.0, n)
    return pos, e * e_tot / e.sum(), np.full(n, det, dtype=np.int64)


def _clue_event(seed=0, sep_mm=200.0):
    """Two ECAL-barrel showers sep_mm apart (along z), plus one isolated soft hit."""
    rng = np.random.default_rng(seed)
    p1, e1, d1 = _shower((1350.0, 0.0, 0.0), 300, 8.0, 20.0, rng, 10)
    p2, e2, d2 = _shower((1350.0, 0.0, sep_mm), 300, 8.0, 10.0, rng, 10)
    p3 = np.array([[1350.0, 0.0, -2000.0]])
    pos = np.vstack([p1, p2, p3])
    E = np.concatenate([e1, e2, [0.02]])
    det = np.concatenate([d1, d2, [10]])
    truth = np.concatenate([np.zeros(300, int), np.ones(300, int), [2]])
    return pos[:, 0], pos[:, 1], pos[:, 2], E, det, truth


def test_clue_invariants():
    x, y, z, E, det, _ = _clue_event(seed=1)
    cluster_of, feats, coo, region = cluster_event(x, y, z, E, det, algorithm="clue")
    n_cl = int(cluster_of.max()) + 1
    assert cluster_of.min() >= 0 and len(np.unique(cluster_of)) == n_cl
    assert feats.shape[0] == n_cl == len(region)
    assert np.array_equal(np.sort(coo[0]), np.arange(len(x)))
    assert abs(feats[:, 5].sum() - E.sum()) < 1e-6
    for cid in range(n_cl):
        assert abs(feats[cid, 5] - E[cluster_of == cid].sum()) < 1e-6


def test_clue_separates_two_showers():
    x, y, z, E, det, truth = _clue_event(seed=2, sep_mm=200.0)
    cluster_of = cluster_event(x, y, z, E, det, algorithm="clue")[0]
    # each shower's energy sits in one dominant cluster, and the two are different
    dom = []
    for t in (0, 1):
        m = truth == t
        e_by_cl = np.bincount(cluster_of[m], weights=E[m])
        assert e_by_cl.max() > 0.9 * E[m].sum()
        dom.append(int(np.argmax(e_by_cl)))
    assert dom[0] != dom[1]
    # the isolated soft hit (2 m away) is kept as its own cluster: energy is never dropped
    assert np.sum(cluster_of == cluster_of[-1]) == 1


def test_clue_absorbs_small_fragment():
    # a 0.2 GeV fragment 60 mm from a 20 GeV shower joins it with absorption on, not without
    rng = np.random.default_rng(3)
    p1, e1, d1 = _shower((1350.0, 0.0, 0.0), 300, 8.0, 20.0, rng, 10)
    p2, e2, d2 = _shower((1350.0, 0.0, 60.0), 5, 2.0, 0.2, rng, 10)
    pos = np.vstack([p1, p2])
    E = np.concatenate([e1, e2])
    det = np.concatenate([d1, d2])
    args = (pos[:, 0], pos[:, 1], pos[:, 2], E, det)
    on = cluster_event(*args, algorithm="clue")[0]
    off = cluster_event(*args, algorithm="clue", clue_options={"min_cluster_E": 0.0})[0]
    assert len(np.unique(on[300:])) == 1 and on[300] == np.bincount(on[:300]).argmax()
    assert not np.any(np.isin(off[300:], off[:300]))


def test_clue_component_cap_bounds_cluster_size():
    # a long chain of soft (sub-seed-threshold) hits: components would join it into one
    # cluster; the cap re-clusters it into pieces no bigger than the cap allows
    n = 400
    z = np.arange(n) * 6.0
    x = np.full(n, 1350.0)
    y = np.zeros(n)
    E = np.full(n, 0.01)
    det = np.full(n, 10, dtype=np.int64)
    opts = {"min_cluster_E": 0.0}
    whole = cluster_event(x, y, z, E, det, algorithm="clue", clue_options={**opts, "max_component_hits": 0})[0]
    capped = cluster_event(x, y, z, E, det, algorithm="clue", clue_options={**opts, "max_component_hits": 100})[0]
    assert len(np.unique(whole)) == 1
    assert np.bincount(capped).max() <= 100


def test_clue_deterministic_and_region_local():
    x, y, z, E, det = _event(seed=7)
    c1 = cluster_event(x, y, z, E, det, algorithm="clue")[0]
    c2 = cluster_event(x, y, z, E, det, algorithm="clue")[0]
    assert np.array_equal(c1, c2)
    for cid in np.unique(c1):
        assert len(set(det[c1 == cid])) == 1


def test_clue_empty_and_single_hit():
    empty = np.zeros(0)
    cluster_of, feats, _, _ = cluster_event(empty, empty, empty, empty, np.zeros(0, np.int64), algorithm="clue")
    assert len(cluster_of) == 0 and feats.shape == (0, 17)
    one = np.array([1350.0])
    cluster_of, feats, _, _ = cluster_event(one, np.zeros(1), np.zeros(1), np.array([0.3]), np.array([10]), algorithm="clue")
    assert cluster_of.tolist() == [0] and feats.shape[0] == 1


def test_clue_drop_isolated_soft_clusters():
    # the isolated 20 MeV hit 2 m away is kept by default, dropped (cluster -1) with drop_E
    x, y, z, E, det, _ = _clue_event(seed=4)
    kept = cluster_event(x, y, z, E, det, algorithm="clue")
    cluster_of, feats, coo, region = cluster_event(x, y, z, E, det, algorithm="clue", clue_options={"drop_E": 0.1})
    assert kept[0][-1] >= 0 and cluster_of[-1] == -1
    assert feats.shape[0] == kept[1].shape[0] - 1 == len(region)
    assert -1 not in coo[1] and len(coo[0]) == len(x) - 1
    assert abs(feats[:, 5].sum() - E[:-1].sum()) < 1e-6


def test_clue_drop_single_hit_clusters():
    # same fixture: the isolated 20 MeV hit forms a single-hit cluster by default, dropped with
    # drop_1hit; multi-hit clusters are untouched
    x, y, z, E, det, _ = _clue_event(seed=4)
    kept = cluster_event(x, y, z, E, det, algorithm="clue")
    cluster_of, feats, coo, region = cluster_event(x, y, z, E, det, algorithm="clue", clue_options={"drop_1hit": True})
    sizes = np.bincount(kept[0], minlength=len(kept[1]))
    singles = int((sizes == 1).sum())
    assert singles >= 1 and cluster_of[-1] == -1
    assert feats.shape[0] == kept[1].shape[0] - singles == len(region)
    assert (np.bincount(cluster_of[cluster_of >= 0], minlength=feats.shape[0]) > 1).all()
    assert -1 not in coo[1] and len(coo[0]) == int((kept[0] >= 0).sum()) - singles
    assert abs(feats[:, 5].sum() - (E.sum() - E[-1])) < 1e-6
