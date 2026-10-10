#!/usr/bin/env python3
"""Check real PF event splits, tensors, track slots and persisted TFDS fidelity."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import tensorflow_datasets as tfds

from scripts.compare_detector_features import verify_persisted_colliderml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-events", type=int, default=1000)
    args = parser.parse_args()
    manual = args.run_dir / "manual"
    tfds_path = args.run_dir / "tensorflow_datasets/colliderml_ttbar_nopu_pf/10/1.1.0"
    shards = [ak.from_parquet(path, columns=["event_id"]) for path in sorted(manual.glob("*.parquet"))]
    ids = [set(ak.to_list(shard.event_id)) for shard in shards]
    assert len(ids) == 2 and not ids[0].intersection(ids[1]), "Event overlap between train and held-out splits"
    assert sum(len(shard) for shard in shards) == args.num_events
    assert sum(map(len, ids)) == args.num_events, "Duplicate event IDs"
    report = {
        "events": args.num_events,
        "split_events": {},
        "input_objects": 0,
        "active_targets": 0,
        "max_tanLambda_residual": 0.0,
        "max_omega_residual": 0.0,
    }
    builder = tfds.builder_from_directory(str(tfds_path))
    for split in ("train", "test"):
        source = builder.as_data_source(split=split)
        report["split_events"][split] = len(source)
        for index in range(len(source)):
            event = source[index]
            assert all(np.isfinite(value).all() for value in event.values()), (split, index, "nonfinite")
            x, y = event["X"], event["ytarget"]
            assert x.ndim == 2 and x.shape[1] == 17
            assert y.shape == (len(x), 14) and event["ycand"].shape == y.shape
            assert np.isin(x[:, 0], [1, 2]).all()
            assert np.isin(y[:, 0], np.arange(6)).all()
            report["input_objects"] += len(x)
            report["active_targets"] += int(np.count_nonzero(y[:, 0]))
            tracks = x[x[:, 0] == 1]
            if len(tracks):
                np.testing.assert_allclose(tracks[:, 10], np.sinh(tracks[:, 2]), rtol=2e-5, atol=2e-5)
                expected_omega = np.sign(np.where(tracks[:, 9] == 0, 0.001, tracks[:, 9])) * 0.0009 / tracks[:, 1]
                np.testing.assert_allclose(tracks[:, 11], expected_omega, rtol=2e-5, atol=1e-7)
                assert (tracks[:, 12] >= 0).all()
                report["max_tanLambda_residual"] = max(report["max_tanLambda_residual"], float(np.max(np.abs(tracks[:, 10] - np.sinh(tracks[:, 2])))))
                report["max_omega_residual"] = max(report["max_omega_residual"], float(np.max(np.abs(tracks[:, 11] - expected_omega))))
    assert report["split_events"]["train"] == len(shards[0])
    assert report["split_events"]["test"] == len(shards[1])
    report["persisted_roundtrip"] = verify_persisted_colliderml(manual, tfds_path)
    assert report["persisted_roundtrip"]["exact_match"]
    report["passed"] = True
    (args.run_dir / "data_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
