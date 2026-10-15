# Unit tests for the ColliderML surrogate ancestors (truth.use_surrogate_ancestors).
import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from mlpf.data.colliderml import truth
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables, find_surrogate_ancestors

REPO = Path(__file__).resolve().parent.parent
RELEASE1 = Path("/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1")

# One tracked pi+ (id 1) plus a generator-decayed K0S (id 10 -> pi+ 11, pi- 12) that Geant4 let
# interact instead: its secondaries (neutron 20, its pi+ 21) carry the deposits, the record
# daughters are silent.
PARTICLES = [
    # id, pdg, charge, px, py, pz, E, primary, parent
    (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, True, -1),
    (10, 310, 0.0, 0.0, 12.0, 9.0, 15.0, True, -1),
    (11, 211, 1.0, 0.0, 7.0, 5.0, 8.6, True, 10),
    (12, -211, -1.0, 0.0, 5.0, 4.0, 6.4, True, 10),
    (20, 2112, 0.0, 0.0, 6.0, 4.0, 7.9, False, 10),
    (21, 211, 1.0, 0.0, 2.0, 1.0, 2.2, False, 20),
]


def _jagged(lists, dtype):
    """Typed jagged array (ak.from_iter types an empty list as unknown, which ak.num rejects)."""
    return ak.unflatten(np.asarray([v for vs in lists for v in vs], dtype=dtype), np.asarray([len(vs) for vs in lists], dtype=np.int64))


def _particles(rows):
    cols = ["particle_id", "pdg_id", "charge", "px", "py", "pz", "energy", "primary", "parent_id"]
    ev = {k: [r[i] for r in rows] for i, k in enumerate(cols)}
    ev["vertex_primary"] = [1] * len(rows)
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _calo(contribs):
    """One ECAL-barrel hit per contribution list [(particle id, raw energy), ...]."""
    n = len(contribs)
    ev = {
        "detector": np.full(n, 10, dtype=np.int64),
        "total_energy": np.asarray([sum(e for _, e in c) for c in contribs], dtype=np.float64),
        "x": np.arange(n, dtype=np.float64),
        "y": np.zeros(n),
        "z": np.zeros(n),
        "contrib_particle_ids": _jagged([[p for p, _ in c] for c in contribs], np.int64),
        "contrib_energies": _jagged([[e for _, e in c] for c in contribs], np.float64),
        "contrib_times": _jagged([[0.0] * len(c) for c in contribs], np.float64),
    }
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _tracks(hit_ids):
    n = len(hit_ids)
    ev = {k: np.full(n, v) for k, v in [("d0", 0.0), ("z0", 0.0), ("phi", 0.0), ("theta", 1.0), ("qop", 0.1), ("majority_particle_id", -1)]}
    ev["hit_ids"] = _jagged(hit_ids, np.int64)
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _tracker(owners):
    return ak.Record({"particle_id": ak.Array(np.asarray(owners, dtype=np.int64))})


def _gen_tables(particles, calo, tracks, tracker, enabled=True, monkeypatch=None):
    monkeypatch.setattr(truth, "use_surrogate_ancestors", enabled)
    return compute_gen_tables(particles, calo, tracks, DEFAULT_CALIBRATION, tracker_ev=tracker, return_rows=True)


def _event(extra_calo=(), extra_tracker=()):
    """K0S event: the pi+ owns track 0 (hits 0-2); the K0S secondaries deposit 0.2 + 0.1 raw."""
    calo = _calo([[(20, 0.2)], [(21, 0.1)]] + [list(c) for c in extra_calo])
    tracker = [1, 1, 1] + list(extra_tracker)
    return _particles(PARTICLES), calo, _tracks([[0, 1, 2]]), _tracker(tracker)


def test_k0s_interaction_promotes_surrogate(monkeypatch):
    particles, calo, tracks, tracker = _event()
    gen, gp_to_hit, _, genref, rows = _gen_tables(particles, calo, tracks, tracker, monkeypatch=monkeypatch)
    assert rows.tolist() == [0, 1]  # the pi+ and the K0S; its silent record daughters stay out
    assert gen["PDG"].tolist() == [211.0, 310.0]
    assert gen["generatorStatus"].tolist() == [1.0, 2.0]
    assert gen["simulatorStatus"].tolist() == [float(0x01000000)] * 2
    np.testing.assert_allclose(gen["energy"][1], 15.0)
    np.testing.assert_allclose(gen["gp_to_cluster"][1], 37.5 * 0.3, rtol=1e-6)  # both secondaries' deposits
    calo_links = np.asarray(gp_to_hit[2]) > 0
    assert sorted(np.asarray(gp_to_hit[1])[calo_links].tolist()) == [0, 1]
    assert set(np.asarray(gp_to_hit[0])[calo_links].tolist()) == {1}
    # the gen reference gains the same particle (both-sides guarantee)
    assert genref["PDG"].tolist() == [211.0, 310.0]
    assert genref["generatorStatus"].tolist() == [1.0, 2.0]


def test_switch_off_drops_the_deposits(monkeypatch):
    particles, calo, tracks, tracker = _event()
    gen, gp_to_hit, _, genref, rows = _gen_tables(particles, calo, tracks, tracker, enabled=False, monkeypatch=monkeypatch)
    assert rows.tolist() == [0]
    assert genref["PDG"].tolist() == [211.0]
    assert not np.any(np.asarray(gp_to_hit[2]) > 0)


@pytest.mark.parametrize("via", ["calo", "tracker"])
def test_partial_veto(monkeypatch, via):
    # one record daughter registered: promoting the K0S as well would double count it
    if via == "calo":
        particles, calo, tracks, tracker = _event(extra_calo=[[(11, 0.1)]])
    else:
        particles, calo, tracks, tracker = _event(extra_tracker=[11])
    gen, _, _, _, rows = _gen_tables(particles, calo, tracks, tracker, monkeypatch=monkeypatch)
    assert 1 not in rows.tolist()
    assert np.all(gen["generatorStatus"] == 1.0)
    pid = np.asarray(particles["particle_id"])
    hit_ids = np.concatenate([np.asarray(ak.flatten(calo["contrib_particle_ids"])), np.asarray(tracker["particle_id"])])
    accepted, n_partial, n_nested = find_surrogate_ancestors(
        hit_ids, pid, np.asarray(particles["parent_id"]), np.asarray(particles["primary"]), np.asarray(particles["primary"]) & (pid != 10)
    )
    assert (accepted.tolist(), n_partial, n_nested) == ([], 1, 0)


def test_nested_keeps_deepest(monkeypatch):
    # Xi- (30) -> Lambda (31) + pi- (32), Lambda -> p (33) + pi- (34), all generator-decayed and
    # silent. The Lambda interacted (secondary 41); the Xi left a delta ray (40) before decaying.
    rows = [
        (30, 3312, -1.0, 0.0, 9.0, 0.0, 9.2, True, -1),
        (31, 3122, 0.0, 0.0, 7.0, 0.0, 7.1, True, 30),
        (32, -211, -1.0, 0.0, 2.0, 0.0, 2.0, True, 30),
        (33, 2212, 1.0, 0.0, 5.0, 0.0, 5.1, True, 31),
        (34, -211, -1.0, 0.0, 2.0, 0.0, 2.0, True, 31),
        (40, 11, -1.0, 0.0, 0.1, 0.0, 0.1, False, 30),
        (41, 2112, 0.0, 0.0, 3.0, 0.0, 3.1, False, 31),
    ]
    particles = _particles(rows)
    calo = _calo([[(40, 0.01)], [(41, 0.1)]])
    gen, _, _, genref, rows_out = _gen_tables(particles, calo, _tracks([]), _tracker([]), monkeypatch=monkeypatch)
    assert rows_out.tolist() == [1]  # the Lambda only
    assert gen["PDG"].tolist() == [3122.0]
    assert gen["generatorStatus"].tolist() == [2.0]
    np.testing.assert_allclose(gen["gp_to_cluster"], [37.5 * 0.1], rtol=1e-6)  # the Xi's delta ray is not its
    primary = np.asarray(particles["primary"])
    pid = np.asarray(particles["particle_id"])
    is_leaf = primary & np.isin(pid, [32, 33, 34])
    accepted, n_partial, n_nested = find_surrogate_ancestors(np.array([40, 41]), pid, np.asarray(particles["parent_id"]), primary, is_leaf)
    assert (accepted.tolist(), n_partial, n_nested) == ([1], 0, 1)


def test_tracker_only_surrogate(monkeypatch):
    # the K0S secondary 21 leaves a whole track (hits 3-5) and no calo deposit: the K0S owns it
    particles, _, _, _ = _event()
    tracks = _tracks([[0, 1, 2], [3, 4, 5]])
    tracker = _tracker([1, 1, 1, 21, 21, 21])
    gen, _, gp_to_track, genref, rows = _gen_tables(particles, _calo([]), tracks, tracker, monkeypatch=monkeypatch)
    assert rows.tolist() == [0, 1]
    assert gen["generatorStatus"].tolist() == [1.0, 2.0]
    np.testing.assert_allclose(gen["gp_to_track"], [1.0, 1.0])
    assert sorted(zip(np.asarray(gp_to_track[0]).tolist(), np.asarray(gp_to_track[1]).tolist())) == [(0, 0), (1, 1)]
    assert genref["PDG"].tolist() == [211.0, 310.0]


def _study():
    spec = importlib.util.spec_from_file_location("colliderml_surrogate_impact", REPO / "scripts" / "colliderml" / "colliderml_surrogate_impact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not RELEASE1.is_dir(), reason="needs the ColliderML Release-1 source shards")
def test_native_equals_study_simulation_on_real_events(monkeypatch):
    """compute_gen_tables with the native surrogates == the study's independent simulation of the
    fix (its candidate rows forced into the leaf set with the native rule switched off)."""
    from mlpf.data.colliderml import reader

    study = _study()
    paths = [reader.shard_paths(RELEASE1, "ttbar_pu0", obj)[0] for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]]
    n_surrogates = 0
    for iev, ev in enumerate(reader.iter_event_records(*paths, batch_size=5)):
        if iev >= 10:
            break
        particles, calo, tracks, tracker = ev["particles"], ev["calo_hits"], ev["tracks"], ev["tracker_hits"]
        pid, _, parent, is_leaf, redirect = truth._build_parent_maps(particles)
        primary = np.asarray(ak.to_numpy(particles["primary"])).astype(bool)
        cid = truth._redirect_ids(np.asarray(ak.to_numpy(ak.flatten(calo["contrib_particle_ids"], axis=None))).ravel(), redirect)
        nctr = np.asarray(ak.to_numpy(ak.num(calo["contrib_particle_ids"])))
        w = np.asarray(ak.to_numpy(ak.flatten(calo["contrib_energies"], axis=None))) * truth.calibration_factors(
            np.repeat(np.asarray(ak.to_numpy(calo["detector"])), nctr), DEFAULT_CALIBRATION
        )
        th_pid = truth._redirect_ids(np.asarray(ak.to_numpy(tracker["particle_id"])).astype(np.int64), redirect)
        leaf_row = truth.walk_to_leaf_many(cid, pid, is_leaf, parent)
        th_leaf = truth.walk_to_leaf_many(th_pid, pid, is_leaf, parent)
        clean, _, n_partial, n_nested = study.simulated_surrogates(pid, parent, primary, cid, w, leaf_row, th_pid, th_leaf)

        # the native helper finds the same rows and exclusions
        accepted, n_partial_native, n_nested_native = find_surrogate_ancestors(
            np.concatenate([cid.astype(np.int64), th_pid]), pid, parent, primary, is_leaf
        )
        assert (accepted.tolist(), n_partial_native, n_nested_native) == (sorted(clean), n_partial, n_nested)

        native = _gen_tables(particles, calo, tracks, tracker, enabled=True, monkeypatch=monkeypatch)
        with study.surrogates_as_leaves(clean):
            simulated = _gen_tables(particles, calo, tracks, tracker, enabled=False, monkeypatch=monkeypatch)
        gen_n, hit_n, trk_n, ref_n, rows_n = native
        gen_s, hit_s, trk_s, ref_s, rows_s = simulated
        np.testing.assert_array_equal(rows_n, rows_s)
        for coo_n, coo_s in [(hit_n, hit_s), (trk_n, trk_s)]:
            for a, b in zip(coo_n, coo_s):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for table_n, table_s in [(gen_n, gen_s), (ref_n, ref_s)]:
            assert set(table_n) == set(table_s)
            for k in table_n:
                if k != "generatorStatus":
                    np.testing.assert_array_equal(table_n[k], table_s[k], err_msg=k)
        # generatorStatus 2 exactly on the surrogates among the targets (the simulation leaves 1)
        np.testing.assert_array_equal(gen_n["generatorStatus"], np.where(np.isin(rows_n, clean), 2.0, 1.0))
        assert np.all(gen_s["generatorStatus"] == 1.0)
        n_surrogates += int(np.isin(rows_n, clean).sum())
    assert n_surrogates > 0, "no surrogate target in the scanned events: the equivalence check is vacuous"
