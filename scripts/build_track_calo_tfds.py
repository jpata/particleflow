"""Build an event-disjoint TFDS smoke sample from one track/calo Parquet file."""

import argparse
import importlib
from pathlib import Path

import awkward as ak
import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.track_calo_utils.utils import required_fields

BUILDERS = {
    "idea": "IdeaTrackCaloTtbar",
    "cld": "CldTrackCaloTtbar",
    "clic": "ClicTrackCaloTtbar",
    "maia": "MaiaTrackCaloTtbar",
    "colliderml": "CollidermlTrackCaloTtbar",
}


def build(input_path, detector, output_dir):
    required = required_fields(detector)
    table = ak.from_parquet(input_path)
    missing = required - set(table.fields)
    if missing:
        raise ValueError(f"Missing {sorted(missing)}; regenerate input with --track-calo-targets")
    # Write only the new view, preserving event boundaries and original source index ranges.
    table = ak.Array({name: table[name] for name in sorted(required)})
    if len(table) < 2:
        raise ValueError("At least two events are needed for disjoint train/test samples")
    boundary = max(1, min(len(table) - 1, int(0.9 * len(table))))
    manual = output_dir / "manual" / detector
    for split, selection in (("train", table[:boundary]), ("test", table[boundary:])):
        directory = manual / split
        directory.mkdir(parents=True, exist_ok=True)
        ak.to_parquet(selection, directory / f"{input_path.stem}.parquet")
    module = importlib.import_module(f"mlpf.heptfds.{detector}_track_calo.ttbar")
    builder = getattr(module, BUILDERS[detector])(config="10", data_dir=output_dir / "tensorflow_datasets")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual.resolve(), try_download_gcs=False))
    from mlpf.heptfds.track_calo_utils.utils import generate_examples

    for split in ("train", "test"):
        expected = [event for _, event in generate_examples(sorted((manual / split).glob("*.parquet")), detector)]
        actual = builder.as_data_source(split=split)
        assert len(actual) == len(expected)

        # TFDS shuffles examples. Match invariant target/feature content by exact array bytes.
        def fingerprint(event):
            return tuple(np.asarray(event[key], dtype=np.float32).tobytes() for key in sorted(event))

        assert sorted(map(fingerprint, expected)) == sorted(fingerprint(actual[i]) for i in range(len(actual)))
    print(f"Validated {detector}: {boundary} train + {len(table) - boundary} test events at {builder.data_dir}")
    return str(builder.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--detector", choices=BUILDERS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a fresh directory when changing the source sample")
    args = parser.parse_args()
    build(args.input, args.detector, args.output_dir)
