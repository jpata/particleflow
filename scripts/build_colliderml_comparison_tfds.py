#!/usr/bin/env python3
"""Build a small ColliderML or MAIA TFDS sample through its native builder."""
import argparse
from pathlib import Path

import awkward as ak
import tensorflow_datasets as tfds

from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf


def write_smoke_shards(input_path, manual_dir, train_fraction=0.5):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    # The PF builder does not consume the much larger raw-hit columns.
    fields = ak.metadata_from_parquet(input_path)["columns"]
    required = (
        "event_id",
        "X_track",
        "X_cluster",
        "ytarget_track",
        "ytarget_cluster",
        "ycand_track",
        "ycand_cluster",
        "genmet",
        "genjet",
        "targetjet",
    )
    table = ak.from_parquet(input_path, columns=[field for field in required if any(c == field or c.startswith(field + ".") for c in fields)])
    # Key4hep writes one outer Record of event lists; ColliderML writes rows.
    if isinstance(table, ak.Record):
        table = ak.Array({field: table[field] for field in table.fields})
    if len(table) < 2:
        raise ValueError("At least two events are required")
    manual_dir.mkdir(parents=True, exist_ok=True)
    middle = max(1, min(len(table) - 1, int(len(table) * train_fraction)))
    for index, (start, count) in enumerate(((0, middle), (middle, len(table) - middle))):
        ak.to_parquet(table[start : start + count], manual_dir / f"train-{index:05d}-of-00002.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--detector", choices=("colliderml", "maia"), default="colliderml")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    args = parser.parse_args()
    write_smoke_shards(args.input, args.manual_dir, args.train_fraction)
    if args.detector == "maia":
        from mlpf.heptfds.maia_pf_edm4hep.ttbar import MaiaEdmTtbarPf

        builder = MaiaEdmTtbarPf(config="10", data_dir=str(args.data_dir))
    else:
        builder = CollidermlTtbarNopuPf(config="10", data_dir=str(args.data_dir))
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=str(args.manual_dir.resolve())))
    print(builder.data_path)


if __name__ == "__main__":
    main()
