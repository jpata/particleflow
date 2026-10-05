# Unit tests for the ColliderML postprocessing batch writer
# (mlpf/data/colliderml/postprocessing.py::_batch_arrow_arrays).
import numpy as np

from mlpf.data.colliderml.postprocessing import _batch_arrow_arrays


def _event(n_hits, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "event_id": np.int64(seed),
        "genmet": np.float32(1.5),
        "hit_to_cluster": rng.integers(0, 3, n_hits).astype(np.int64),
        "X_hit_calo": rng.random((n_hits, 12)).astype(np.float32),
        "genjet": rng.random((2, 4)).astype(np.float32),
    }


def test_scalar_and_ragged_routing_by_field_name():
    # degenerate first event: exactly 1 calo hit. The old shape-inferred detection misrouted
    # hit_to_cluster (1-D of length 1) into the scalar branch, truncating every event to its
    # first element; routing must be by field name instead.
    evs = [_event(1, seed=0), _event(7, seed=1), _event(3, seed=2)]
    arrays = dict(_batch_arrow_arrays(evs))
    import pyarrow as pa

    # event_id / genmet: one scalar per event
    assert arrays["event_id"].to_pylist() == [0, 1, 2]
    assert arrays["genmet"].to_pylist() == [1.5, 1.5, 1.5]
    # hit_to_cluster: ragged, full contents per event, regardless of first-event length
    assert pa.types.is_list(arrays["hit_to_cluster"].type) or pa.types.is_large_list(arrays["hit_to_cluster"].type)
    assert [len(v) for v in arrays["hit_to_cluster"].to_pylist()] == [1, 7, 3]
    assert arrays["hit_to_cluster"].to_pylist()[0] == [int(_event(1, 0)["hit_to_cluster"][0])]
    # 2-D fields keep their fixed inner width
    X = arrays["X_hit_calo"].to_pylist()
    assert [len(v) for v in X] == [1, 7, 3]
    assert np.asarray(X[0]).shape == (1, 12) and np.asarray(X[1]).shape == (7, 12)
    gj = arrays["genjet"].to_pylist()
    assert np.asarray(gj[0]).shape == (2, 4)


def test_consistent_schema_across_batches_regardless_of_first_event():
    # the same fields routed through a batch whose first event is degenerate and one that is
    # not must produce the same arrow schema (a mismatched row group crashes ParquetWriter)
    evs_ok = [_event(5, seed=3), _event(7, seed=4)]
    evs_degenerate = [_event(1, seed=0), _event(7, seed=1)]
    sch_ok = dict((k, a.type) for k, a in _batch_arrow_arrays(evs_ok))
    sch_deg = dict((k, a.type) for k, a in _batch_arrow_arrays(evs_degenerate))
    assert sch_ok == sch_deg
