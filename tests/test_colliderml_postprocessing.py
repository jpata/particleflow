# Unit tests for the ColliderML converter's per-event record (hits view).
import awkward as ak

from mlpf.data.colliderml.postprocessing import _event_record_one, particle_feature_order

PN = particle_feature_order.index("particle_number")


def _record(fields):
    return ak.Record({k: ak.from_iter(v) for k, v in fields.items()})


def _shared_hits_event():
    # Two photons A (pid 1) and B (pid 2) share four well-separated ECAL cells (one cluster
    # each). A leads on cells 0 and 2 and B on cells 1 and 3, with the minor contributor listed
    # last on cells 0 and 2 and first on 1 and 3, so neither a hit-major nor a particle-major
    # COO order lets a "last entry wins" write agree with the largest contributor on all four.
    # The exclusive rows land on cells 0 (A) and 1 (B); cells 2 and 3 carry only inclusive
    # marks. A tracked pi+ (pid 3, no calo deposit) keeps the track/tracker tables non-empty.
    particles = _record(
        {
            "particle_id": [1, 2, 3],
            "pdg_id": [22, 22, 211],
            "charge": [0.0, 0.0, 1.0],
            "px": [200.0, 0.0, 1.0],
            "py": [0.0, 150.0, 0.0],
            "pz": [0.0, 0.0, 10.0],
            "energy": [200.0, 150.0, 10.1],
            "vertex_primary": [1, 1, 1],
            "primary": [True, True, True],
            "parent_id": [-1, -1, -1],
        }
    )
    calo = _record(
        {
            "detector": [10, 10, 10, 10],
            "total_energy": [3.1, 2.1, 1.05, 0.85],
            "x": [1500.0, -1500.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1500.0, -1500.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "contrib_particle_ids": [[1, 2], [1, 2], [1, 2], [1, 2]],
            "contrib_energies": [[3.0, 0.1], [0.1, 2.0], [1.0, 0.05], [0.05, 0.8]],
            "contrib_times": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        }
    )
    tracks = _record({"d0": [0.01], "z0": [0.02], "phi": [0.3], "theta": [0.1], "qop": [0.1], "majority_particle_id": [3], "hit_ids": [[0, 1, 2]]})
    tracker = _record({"particle_id": [3, 3, 3], "x": [30.0, 60.0, 90.0], "y": [0.0, 0.0, 0.0], "z": [300.0, 600.0, 900.0], "time": [0.0, 0.0, 0.0]})
    return particles, tracks, calo, tracker


def test_calo_hit_pn_is_largest_contributor():
    particles, tracks, calo, tracker = _shared_hits_event()
    rec = _event_record_one(0, particles, tracks, calo, tracker)

    y = rec["ytarget_hit_calo"]
    # both photons are kept calo-hosted targets; particle_number follows leaf order (A=1, B=2)
    assert sorted(y[y[:, 0] != 0, PN].tolist()) == [1.0, 2.0]
    assert y[:, PN].tolist() == [1.0, 2.0, 1.0, 2.0]

