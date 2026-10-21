# Unit tests for the ColliderML converter's per-event record (hits view).
import awkward as ak
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlpf.data.colliderml.make_test_fixture import make_fixture
from mlpf.data.colliderml.postprocessing import _event_record_one, particle_feature_order, process_one_file
from mlpf.data.colliderml.reader import shard_paths

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


def test_track_x_carries_perigee_track_state_parameters():
    # col 10 = tanLambda = 1/tan(theta), col 11 = omega = q/pT [1/mm], col 12 =
    # radiusOfInnermostHit = transverse radius of the innermost owned tracker hit (30 mm here).
    particles, tracks, calo, tracker = _shared_hits_event()
    rec = _event_record_one(0, particles, tracks, calo, tracker)

    x = rec["X_track"]
    assert x.shape == (1, 17)
    assert np.isclose(x[0, 10], 1.0 / np.tan(0.1), rtol=1e-5)
    # qop = 0.1, theta = 0.1 -> pt = sin(0.1)/0.1; omega = 3e-4 * 3 T / pt
    pt = np.sin(0.1) / 0.1
    assert np.isclose(x[0, 11], 9e-4 / pt, rtol=1e-5)
    assert np.isclose(x[0, 12], 30.0, atol=1e-6)


def test_calo_hit_pn_is_largest_contributor():
    particles, tracks, calo, tracker = _shared_hits_event()
    rec = _event_record_one(0, particles, tracks, calo, tracker)

    y = rec["ytarget_hit_calo"]
    # both photons are kept calo-hosted targets; particle_number follows leaf order (A=1, B=2)
    assert sorted(y[y[:, 0] != 0, PN].tolist()) == [1.0, 2.0]
    assert y[:, PN].tolist() == [1.0, 2.0, 1.0, 2.0]


def test_target_without_calo_claim_falls_back_to_innermost_tracker_hit():
    # The pi+ has no calo deposit, so its full target row lands on its innermost owned tracker
    # hit (x=30 mm) while every owned tracker hit carries its particle_number. Tracker-hit
    # kinematic inputs stay zero: the release has no tracker-hit energy.
    particles, tracks, calo, tracker = _shared_hits_event()
    rec = _event_record_one(0, particles, tracks, calo, tracker)

    x, y = rec["X_hit_tracker"], rec["ytarget_hit_tracker"]
    assert (y[:, 0] != 0).tolist() == [True, False, False]
    assert y[:, PN].tolist() == [3.0, 3.0, 3.0]
    assert np.all(x[:, 1:6] == 0.0)


def test_resume_reconverts_partial_output_and_skips_complete_one(tmp_path):
    make_fixture(tmp_path / "src", n_events=6)
    paths = [shard_paths(tmp_path / "src", "ttbar_pu0", obj)[0] for obj in ("particles", "tracks", "calo_hits", "tracker_hits")]
    ofn = tmp_path / "out" / "shard.parquet"

    def rows():
        return pq.ParquetFile(ofn).metadata.num_rows

    process_one_file(*paths, ofn, num_events=2)  # debug cap
    assert rows() == 2
    process_one_file(*paths, ofn)  # a full run must not take the capped file for done
    assert rows() == 3
    mtime = ofn.stat().st_mtime_ns
    process_one_file(*paths, ofn)  # complete: skipped
    assert ofn.stat().st_mtime_ns == mtime


def test_event_missing_from_one_table_is_dropped(tmp_path):
    # release-1 ttbar_pu0 train-00991: the tracks table lacks one event (999 vs 1000 rows)
    make_fixture(tmp_path / "src", n_events=8)
    paths = [shard_paths(tmp_path / "src", "ttbar_pu0", obj)[0] for obj in ("particles", "tracks", "calo_hits", "tracker_hits")]
    tracks = pq.read_table(paths[1])
    missing = tracks["event_id"][1].as_py()
    pq.write_table(tracks.filter(pc.not_equal(tracks["event_id"], missing)), paths[1])
    ofn = tmp_path / "out" / "shard.parquet"

    process_one_file(*paths, ofn)
    ids = pq.read_table(ofn)["event_id"].to_pylist()
    assert len(ids) == 3 and missing not in ids
    mtime = ofn.stat().st_mtime_ns
    process_one_file(*paths, ofn)  # the 3-event output counts as complete: skipped
    assert ofn.stat().st_mtime_ns == mtime


def test_unsplit_pi0_target_is_a_photon():
    # a pi0 without recorded decay products stays a leaf in truth.py; the converter labels its
    # target a photon (like the split ones), not the shared mapping's neutral hadron
    particles, tracks, calo, tracker = _shared_hits_event()
    pdg = ak.to_list(particles["pdg_id"])
    pdg[1] = 111
    particles = ak.Record({k: (ak.Array(pdg) if k == "pdg_id" else particles[k]) for k in particles.fields})
    rec = _event_record_one(0, particles, tracks, calo, tracker)

    y = rec["ytarget_cluster"]
    classes = sorted(y[y[:, 0] != 0, 0].tolist())
    assert classes == [22.0, 22.0]
