# Unit tests for the clusterer-agnostic metrics (mlpf/data/colliderml/cluster_metrics.py).
import contextlib
import io

import numpy as np

from mlpf.data.colliderml.cluster_metrics import aggregate, cluster_metrics, greedy_assign, pp_metrics, _candidates, _sum_pairs
from mlpf.data.target_building import EventData, assign_genparticles_to_obj_and_merge


def _truth(hit, leaf, w, n_rows, energy=None, ispu=None):
    return dict(
        hit=np.asarray(hit),
        leaf=np.asarray(leaf),
        w=np.asarray(w, dtype=np.float64),
        n_rows=n_rows,
        energy=np.arange(n_rows, 0, -1, dtype=np.float64) * 10 if energy is None else np.asarray(energy, np.float64),
        ispu=np.zeros(n_rows, bool) if ispu is None else np.asarray(ispu),
    )


# 3 particles, 6 hits of 2 GeV: particle 0 owns hits 0-2 (6 GeV), particle 1 hits 3-4 (4 GeV),
# particle 2 hit 5 (2 GeV); true energies 30 / 20 / 10 GeV (the allocator's order)
HIT_E = np.full(6, 2.0)
TRUTH = _truth([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 2], HIT_E, 3)


def test_perfect_clustering():
    m = aggregate([cluster_metrics(np.array([0, 0, 0, 1, 1, 2]), HIT_E, TRUTH)])
    for rule in ("dm", "alloc"):
        assert m[f"eff_{rule}"] == 1.0 and m[f"fake_{rule}"] == 0.0
        for k in ("purity", "recall"):
            assert m[f"{k}_{rule}_E"] == 1.0 and m[f"{k}_{rule}_hits"] == 1.0
    assert m["purity_E"] == 1.0 and m["completeness_E"] == 1.0 and m["dropped_E_frac"] == 0.0
    assert m["n_clusters"] == 3 and m["n_clusters_1hit"] == 1


def test_everything_merged():
    # one cluster: particle 0 makes up exactly 50% -> double-majority match (>= 50%); the
    # allocator gives the cluster to the most energetic particle, 0; 1 and 2 are unmatched
    m = aggregate([cluster_metrics(np.zeros(6, np.int64), HIT_E, TRUTH)])
    assert np.isclose(m["purity_E"], 0.5) and m["completeness_E"] == 1.0
    for rule in ("dm", "alloc"):
        assert np.isclose(m[f"eff_{rule}"], 1 / 3) and m[f"fake_{rule}"] == 0.0
        assert np.isclose(m[f"purity_{rule}_E"], 0.5) and m[f"recall_{rule}_E"] == 1.0
        assert np.isclose(m[f"purity_{rule}_hits"], 0.5) and m[f"recall_{rule}_hits"] == 1.0


def test_fragment_and_mixture():
    # particle 0 split 4 + 2 GeV (cluster 1 = its satellite); cluster 2 is a 50/50 mixture of
    # particle 1 (hit 3) and particle 2 (hit 5); hit 4 (particle 1) is unclustered
    co = np.array([0, 0, 1, 2, -1, 2])
    m = aggregate([cluster_metrics(co, HIT_E, TRUTH)])
    assert np.isclose(m["dropped_E_frac"], 2 / 12)
    # double majority: 0 matches cluster 0 (4 of 6 GeV); 1 makes up 50% of cluster 2 but the
    # cluster holds only 2 of its 4 GeV (exactly 50%) -> match; 2 is not its cluster's dominant
    # particle (tie broken by input order: particle 1) -> unmatched. Clusters > 1 GeV: 0, 1, 2;
    # cluster 1 (satellite) unmatched
    assert np.isclose(m["eff_dm"], 2 / 3) and np.isclose(m["fake_dm"], 1 / 3)
    # allocator (true energy 30 / 20 / 10): 0 takes cluster 0, 1 takes cluster 2, 2 finds
    # cluster 2 taken -> none; cluster 1 unmatched
    assert np.isclose(m["eff_alloc"], 2 / 3) and np.isclose(m["fake_alloc"], 1 / 3)
    assert np.isclose(m["recall_alloc_E"], (4 + 2) / (6 + 4)) and np.isclose(m["purity_alloc_E"], (4 + 2) / (4 + 4))


def test_allocator_order_differs_from_dominance():
    # one cluster where the low-energy particle dominates: the allocator still gives it to the
    # most energetic particle (true energy), the double majority to the dominant one
    hit_e = np.array([2.0, 6.0])
    truth = _truth([0, 1], [0, 1], hit_e, 2, energy=[100.0, 5.0])
    m = aggregate([cluster_metrics(np.array([0, 0]), hit_e, truth)])
    assert np.isclose(m["purity_alloc_E"], 0.25) and np.isclose(m["purity_dm_E"], 0.75)


def test_per_visible_counts():
    m = aggregate([cluster_metrics(np.array([0, 0, 0, 1, 1, 2]), HIT_E, TRUTH, n_tracks=2, n_visible=4)])
    assert np.isclose(m["clusters_per_visible"], 3 / 4) and np.isclose(m["elements_per_visible"], 5 / 4)
    assert "elements_per_visible" not in aggregate([cluster_metrics(np.array([0, 0, 0, 1, 1, 2]), HIT_E, TRUTH)])


def _allocator_event(rng, n_gp=40, n_hit=300, n_trk=15, n_cl=25):
    energy = rng.uniform(0.5, 50.0, n_gp).astype(np.float32)
    gh = (rng.integers(0, n_gp, 900), rng.integers(0, n_hit, 900), rng.uniform(0.01, 2.0, 900).astype(np.float32))
    gt = (rng.integers(0, n_gp, 30), rng.integers(0, n_trk, 30), rng.uniform(0.2, 1.0, 30).astype(np.float32))
    cluster_of = rng.integers(0, n_cl, n_hit)
    return energy, gh, gt, cluster_of


def test_greedy_assign_matches_the_allocator():
    rng = np.random.default_rng(1)
    for _ in range(5):
        energy, gh, gt, cluster_of = _allocator_event(rng)
        n_gp, n_hit, n_trk, n_cl = len(energy), len(cluster_of), 15, 25
        gen = {k: np.zeros(n_gp, np.float32) for k in ("PDG", "charge", "pt", "eta", "phi", "ispu", "generatorStatus", "simulatorStatus")}
        gen.update(energy=energy, gp_to_track=np.zeros(n_gp, np.float32), gp_to_cluster=np.zeros(n_gp, np.float32), jet_idx=np.zeros(n_gp))
        h2c = (np.arange(n_hit), cluster_of, np.ones(n_hit, np.float32))
        gd = EventData(gen, {"type": np.zeros(n_hit)}, {"type": np.zeros(n_cl)}, {"type": np.zeros(n_trk)}, gh, gt, h2c, (np.array([]), np.array([])))
        with contextlib.redirect_stdout(io.StringIO()):
            _, gp_to_obj, *_ = assign_genparticles_to_obj_and_merge(gd)
        Mp, Mc, Mw = _sum_pairs(gh[0], cluster_of[gh[1]], gh[2], n_cl)
        tp, tt, tw = _sum_pairs(*gt, n_trk)
        own_t, own_c = greedy_assign(energy, [_candidates(tp, tt, tw, n_gp), _candidates(Mp, Mc, Mw, n_gp)], [n_trk, n_cl])
        has = (own_t >= 0) | (own_c >= 0)
        assert np.array_equal(np.stack([own_t[has], own_c[has]], 1), gp_to_obj)


def test_pp_metrics_tracks_and_clusters():
    # particle 0 (charged, 30 GeV) owns track 0 (all 4 of its hits) and deposits in cluster 0;
    # particle 1 (neutral, 20 GeV) deposits in cluster 0 too; cluster 1 belongs to nobody
    gen_energy = np.array([30.0, 20.0])
    gp_trk = (np.array([0]), np.array([0]), np.array([1.0]))
    gp_calo = (np.array([0, 1, 1]), np.array([0, 1, 2]), np.array([2.0, 3.0, 3.0]))
    cluster_of = np.array([0, 0, 0, 1])
    hit_e = np.array([2.0, 3.0, 3.0, 2.0])
    m = aggregate([pp_metrics(gen_energy, gp_trk, gp_calo, np.array([4, 0]), cluster_of, hit_e, np.array([4]), np.array([5.0]))])
    # 0 takes the track, 1 takes cluster 0; cluster 1 (2 GeV) has no target
    assert m["eff_pp"] == 1.0 and np.isclose(m["fake_pp"], 1 / 3)
    assert np.isclose(m["purity_pp_clu_E"], 6 / 8) and m["recall_pp_clu_E"] == 1.0
    assert np.isclose(m["purity_pp_clu_hits"], 2 / 3) and m["recall_pp_clu_hits"] == 1.0
    assert m["purity_pp_trk_hits"] == 1.0 and m["recall_pp_trk_hits"] == 1.0
