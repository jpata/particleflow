"""Unit tests for the genparticle merge accounting in the Key4hep postprocessing."""

import contextlib
import io

import awkward as ak
import numpy as np
import pytest

from mlpf.data.key4hep import postprocessing as pp


def make_eventdata(energies, cluster_weights):
    """Build an EventData where each genparticle may share weight on a single cluster.

    genparticles 0..(n-2) share cluster 0 (so only the highest-energy one becomes its
    exclusive owner), and the last genparticle has no cluster weight at all.
    """
    n_gp = len(energies)
    gen_features = ak.Array(
        {
            "PDG": [211] * n_gp,
            "charge": [1] * n_gp,
            "pt": energies,
            "eta": [0.1] * n_gp,
            "phi": [0.0] * n_gp,
            "energy": energies,
            "ispu": [0] * n_gp,
            "generatorStatus": [1] * n_gp,
            "simulatorStatus": [0x05000000] * n_gp,
            "gp_to_track": [0] * n_gp,
            "gp_to_cluster": [0] * n_gp,
            "jet_idx": [0] * n_gp,
            "particle_number": list(range(1, n_gp + 1)),
        }
    )
    gp_to_hit = (
        np.arange(n_gp, dtype=np.int32),
        np.zeros(n_gp, dtype=np.int32),
        np.array(cluster_weights, dtype=float),
    )
    hit_to_cluster = (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), np.array([1.0]))
    return pp.EventData(
        gen_features=gen_features,
        hit_features=ak.Array({"type": [0]}),
        cluster_features=ak.Array({"type": [0]}),
        track_features=ak.Array({"type": []}),
        genparticle_to_hit=gp_to_hit,
        genparticle_to_track=(np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=float)),
        hit_to_cluster=hit_to_cluster,
        gp_merges=(np.array([], dtype=np.int32), np.array([], dtype=np.int32)),
    )


def test_multiple_merges_into_same_host_accumulate():
    # host (100) absorbs three unmatched particles (10, 20, 30) sharing its cluster.
    energies = [100.0, 10.0, 20.0, 30.0]
    gpdata = make_eventdata(energies, cluster_weights=[10.0, 1.0, 1.0, 1.0])

    cleaned, *_ = pp.assign_genparticles_to_obj_and_merge(gpdata)

    after_e = np.asarray(ak.to_numpy(cleaned.gen_features["energy"])).astype(float)
    hosts = np.asarray(cleaned.gp_merges[0]).astype(int)
    merged = np.asarray(cleaned.gp_merges[1]).astype(int)

    assert len(after_e) == 1
    assert after_e[0] == pytest.approx(160.0, abs=1e-6)
    assert list(zip(hosts, merged)) == [(0, 1), (0, 2), (0, 3)]


def test_particle_without_cluster_host_is_dropped_but_accounted():
    # The last particle has no cluster weight: no track/cluster host exists, so it must be
    # removed from the target, but its energy has to be accounted for (the two-sided
    # conservation assert inside the function would otherwise fire).
    energies = [100.0, 10.0, 20.0, 30.0, 5.0]
    gpdata = make_eventdata(energies, cluster_weights=[10.0, 1.0, 1.0, 1.0, 0.0])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cleaned, *_ = pp.assign_genparticles_to_obj_and_merge(gpdata)

    after_e = np.asarray(ak.to_numpy(cleaned.gen_features["energy"])).astype(float)
    assert after_e[0] == pytest.approx(160.0, abs=1e-6)
    assert "Dropped 1 unmatched genparticles" in buf.getvalue()
    assert after_e.sum() + 5.0 == pytest.approx(sum(energies), abs=1e-6)
