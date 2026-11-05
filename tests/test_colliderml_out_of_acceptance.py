# Unit tests for the ColliderML out-of-acceptance drop (truth.drop_out_of_acceptance).
import awkward as ak
import numpy as np

from mlpf.data.colliderml import truth
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables, out_of_acceptance


def _p(pid, pdg, charge, pt, eta, primary, parent, vertex=(0.0, 0.0, 0.0), mass=0.14):
    pz = pt * np.sinh(eta)
    return (pid, pdg, charge, pt, 0.0, pz, float(np.sqrt(pt**2 + pz**2 + mass**2)), primary, parent, *vertex)


def _particles(rows):
    cols = ["particle_id", "pdg_id", "charge", "px", "py", "pz", "energy", "primary", "parent_id", "vx", "vy", "vz"]
    ev = {k: [r[i] for r in rows] for i, k in enumerate(cols)}
    ev["vertex_primary"] = [1] * len(rows)
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _jagged(lists, dtype):
    return ak.unflatten(np.asarray([v for vs in lists for v in vs], dtype=dtype), np.asarray([len(vs) for vs in lists], dtype=np.int64))


def _calo(contribs):
    """One ECAL-endcap hit per contribution list [(particle id, raw energy), ...]."""
    n = len(contribs)
    ev = {
        "detector": np.full(n, 11, dtype=np.int64),
        "total_energy": np.asarray([sum(e for _, e in c) for c in contribs], dtype=np.float64),
        "x": np.full(n, 400.0),
        "y": np.zeros(n),
        "z": np.full(n, 3300.0),
        "contrib_particle_ids": _jagged([[p for p, _ in c] for c in contribs], np.int64),
        "contrib_energies": _jagged([[e for _, e in c] for c in contribs], np.float64),
        "contrib_times": _jagged([[0.0] * len(c) for c in contribs], np.float64),
    }
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _no_tracks():
    ev = {k: np.zeros(0) for k in ("d0", "z0", "phi", "theta", "qop", "majority_particle_id")}
    ev["hit_ids"] = _jagged([], np.int64)
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _targets(particles, calo, enabled, monkeypatch):
    monkeypatch.setattr(truth, "drop_out_of_acceptance", enabled)
    tracker = ak.Record({"particle_id": ak.Array(np.zeros(0, dtype=np.int64))})
    gen, gp_to_hit, _, _, rows = compute_gen_tables(particles, calo, _no_tracks(), DEFAULT_CALIBRATION, tracker_ev=tracker, return_rows=True)
    pid = np.asarray(ak.to_numpy(particles["particle_id"]))
    return sorted(pid[rows].tolist()), float(np.sum(gp_to_hit[2]))


# A 500 GeV pi+ at eta 7 (id 1) interacts in the beam pipe at z = 2.9 m; its children pi+ (20) and
# n (21) spray into the endcap and carry all of the deposits. A central pi+ (id 2, eta 1) deposits itself.
BEAM_PIPE = (24.0, 0.0, 2900.0)
FORWARD_EVENT = [
    _p(1, 211, 1.0, 0.9, 7.0, True, -1),
    _p(2, 211, 1.0, 5.0, 1.0, True, -1),
    _p(20, 211, 1.0, 1.5, 2.9, False, 1, BEAM_PIPE),
    _p(21, 2112, 0.0, 1.0, 3.0, False, 1, BEAM_PIPE, mass=0.94),
]
FORWARD_CALO = [[(20, 0.05)], [(21, 0.03)], [(2, 0.1)]]


def test_out_of_acceptance_mask():
    # id 1 heads beyond 3.5; the central pi+ (2) and the beam-pipe secondaries (20, 21) point into the acceptance
    np.testing.assert_array_equal(out_of_acceptance(_particles(FORWARD_EVENT)), [True, False, False, False])


def test_forward_particle_dropped(monkeypatch):
    particles, calo = _particles(FORWARD_EVENT), _calo(FORWARD_CALO)
    before, dep_before = _targets(particles, calo, False, monkeypatch)
    after, dep_after = _targets(particles, calo, True, monkeypatch)
    assert before == [1, 2]
    assert after == [2]
    # only the central pi+'s own deposit (raw 0.1 in the ECAL endcap) stays attributed
    np.testing.assert_allclose(dep_after, 0.1 * DEFAULT_CALIBRATION[11], rtol=1e-5)
    assert dep_after < dep_before


def test_central_particle_unchanged(monkeypatch):
    rows = [_p(2, 211, 1.0, 5.0, 3.3, True, -1), _p(20, 211, 1.0, 1.0, 3.0, False, 2, (300.0, 0.0, 3000.0))]
    particles, calo = _particles(rows), _calo([[(2, 0.1)], [(20, 0.05)]])
    assert _targets(particles, calo, True, monkeypatch)[0] == _targets(particles, calo, False, monkeypatch)[0] == [2]


def test_dropped_particle_and_ancestor_are_not_surrogates(monkeypatch):
    # a generator-decayed rho (5) at eta 6 with a pi+ daughter (1) that interacts in the beam pipe: neither the pi+
    # (through its own or its secondary's deposits) nor the rho comes back as a surrogate ancestor
    rows = [
        _p(5, 113, 0.0, 1.2, 6.0, True, -1, mass=0.775),
        _p(1, 211, 1.0, 1.0, 6.0, True, 5),
        _p(20, 211, 1.0, 1.0, 3.0, False, 1, BEAM_PIPE),
    ]
    particles, calo = _particles(rows), _calo([[(1, 0.05)], [(20, 0.1)]])
    assert _targets(particles, calo, False, monkeypatch)[0] == [1]
    assert _targets(particles, calo, True, monkeypatch)[0] == []


def test_forward_surrogate_is_dropped(monkeypatch):
    # a generator-decayed K0S (10) at eta 4.5 interacts in the beam pipe: its generator daughters (11, 12) are
    # silent and its Geant4 secondary (20) deposits, so without the cut the K0S is a surrogate target
    rows = [
        _p(10, 310, 0.0, 0.5, 4.5, True, -1, mass=0.498),
        _p(11, 211, 1.0, 0.3, 4.5, True, 10),
        _p(12, -211, -1.0, 0.2, 4.5, True, 10),
        _p(20, 2112, 0.0, 1.0, 3.0, False, 10, BEAM_PIPE, mass=0.94),
    ]
    particles, calo = _particles(rows), _calo([[(20, 0.1)]])
    assert _targets(particles, calo, False, monkeypatch)[0] == [10]
    assert _targets(particles, calo, True, monkeypatch)[0] == []
