"""Unit tests for the genparticle merge accounting in the Key4hep postprocessing."""

import contextlib
import dataclasses
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


def test_merge_host_tie_breaks_on_lowest_cluster_index():
    # gp0/gp1/gp2 each own one cluster; gp3 deposits equally into all three, loses every
    # claim and is merged. The host must be the owner of the lowest-index tied cluster
    # (gp0), as with the dense argmax — scipy's matmul stores that row as [2, 1, 0].
    energies = [100.0, 90.0, 80.0, 5.0]
    gpdata = dataclasses.replace(
        make_eventdata(energies, cluster_weights=[1.0] * 4),
        hit_features=ak.Array({"type": [0, 0, 0]}),
        cluster_features=ak.Array({"type": [0, 0, 0]}),
        genparticle_to_hit=(
            np.array([0, 1, 2, 3, 3, 3], dtype=np.int32),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
            np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        ),
        hit_to_cluster=(np.arange(3, dtype=np.int32), np.arange(3, dtype=np.int32), np.ones(3)),
    )

    cleaned, *_ = pp.assign_genparticles_to_obj_and_merge(gpdata)

    assert list(zip(np.asarray(cleaned.gp_merges[0]).astype(int), np.asarray(cleaned.gp_merges[1]).astype(int))) == [(0, 3)]


def _two_particle_event(gp_to_hit, n_hits, hit_clusters):
    return dataclasses.replace(
        make_eventdata([100.0, 5.0], cluster_weights=[1.0, 1.0]),
        hit_features=ak.Array({"type": [0] * n_hits}),
        cluster_features=ak.Array({"type": [0] * (max(hit_clusters) + 1)}),
        genparticle_to_hit=tuple(np.asarray(a) for a in gp_to_hit),
        hit_to_cluster=(np.arange(n_hits, dtype=np.int32), np.asarray(hit_clusters, dtype=np.int32), np.ones(n_hits)),
    )


def test_negative_weight_hit_is_never_claimed():
    # gp1's only link is a negative weight on hit 1. It may still own that cluster (the dense
    # code sorted all nonzero cluster weights), but no hit representative: only weights > 0
    # can be claimed.
    gpdata = _two_particle_event(([0, 1], [0, 1], [10.0, -0.5]), n_hits=2, hit_clusters=[0, 1])

    _, gp_to_obj, gp_to_hit_idx, *_ = pp.assign_genparticles_to_obj_and_merge(gpdata)

    assert gp_to_obj[:, 1].tolist() == [0, 1]
    assert gp_to_hit_idx.tolist() == [0, -1]


def test_unmatched_particle_with_only_negative_cluster_weight_is_dropped():
    # gp1 loses cluster 0 to gp0 and has no positive cluster weight, so there is no merge
    # host: it is dropped (and accounted), not merged into gp0.
    gpdata = _two_particle_event(([0, 1], [0, 0], [10.0, -1.0]), n_hits=1, hit_clusters=[0])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cleaned, *_ = pp.assign_genparticles_to_obj_and_merge(gpdata)

    assert len(cleaned.gp_merges[0]) == 0
    assert "Dropped 1 unmatched genparticles" in buf.getvalue()
    assert np.asarray(ak.to_numpy(cleaned.gen_features["energy"])).tolist() == [100.0]
