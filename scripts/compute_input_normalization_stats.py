"""Compute the input normalization statistics of one dataset version.

The mean and std of each track/cluster input column, after its transform, over the whole train
split of every <dataset>_*_pf sample and config of that version in the data directory, written
with the name and transform of each column to mlpf/model/input_normalization_stats/<dataset>_<version>.yaml.
Saved values are then used in the normalization module mlpf/model/input_normalization.py.

In a single process:
    python scripts/compute_input_normalization_stats.py map idea 0.1.0 --data-dir DATA_DIR --work-dir WORK_DIR
    python scripts/compute_input_normalization_stats.py reduce idea 0.1.0 --work-dir WORK_DIR

CMS 2.8.0 has 28 million train events. To inspect a reproducible spread across every config:
    python scripts/compute_input_normalization_stats.py map cms 2.8.0 --data-dir DATA_DIR --work-dir WORK_DIR --max-events-per-build 100
    python scripts/compute_input_normalization_stats.py reduce cms 2.8.0 --work-dir WORK_DIR
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import yaml

from mlpf.conf import ELEMENT_TYPES, X_FEATURES, column_names, column_transform, stats_path

# Events read and summed at once.
CHUNK_EVENTS = 256
# Arguments of a run without any, e.g. from a debugger: about 270 CLIC events, written outside the repository.
DEBUG_ARGS = [
    "map",
    "clic",
    "3.3.0",
    "--data-dir",
    "/mnt/ceph/users/jpata/mlpf/clic/v1.2.5_key4hep_2025-05-29/tfds",
    "--work-dir",
    "/tmp/input_normalization_debug",
    "--shard",
    "0",
    "--num-shards",
    "10000",
]


def open_builds(dataset, version, data_dir, samples=None):
    """TFDS builders of every config of the version of the track/cluster samples in the data directory, all by default."""
    paths = []
    # CMS names its samples cms_pf_<process>; the other PF datasets use <detector>_<process>_pf.
    if dataset == "cms":
        sample_names = [f"cms_pf_{sample}" for sample in samples] if samples else ["cms_pf_*"]
    else:
        sample_names = [f"{dataset}_{sample}_pf" for sample in samples] if samples else [f"{dataset}_*_pf"]
    for sample in sample_names:
        # TFDS layout: <data dir>/<sample>/<config>/<version>.
        matches = sorted(data_dir.glob(f"{sample}/*/{version}"))
        if not matches:
            raise SystemExit(f"No {sample} samples of version {version} in {data_dir}")
        paths += matches
    return [tfds.builder(f"{path.parent.parent.name}/{path.parent.name}:{version}", data_dir=data_dir) for path in paths]


def shard_path(work_dir, dataset, version, shard):
    return work_dir / f"{dataset}_{version}_shard{shard}.npz"


def map_shard(args):
    """Count, sum and sum of squares of the transformed columns of each element type over one shard."""
    builders = open_builds(args.dataset, args.version, args.data_dir, args.samples)
    full_sizes = [builder.info.splits["train"].num_examples for builder in builders]
    max_events = getattr(args, "max_events_per_build", None)
    sizes = [min(size, max_events) if max_events else size for size in full_sizes]
    # Shards are contiguous ranges of the train events of all builds, taken one after the other.
    start, stop = (sum(sizes) * shard // args.num_shards for shard in (args.shard, args.shard + 1))

    num_columns = builders[0].info.features["X"].shape[-1]
    # log1p(x / scale) where scale is not NaN.
    scale = np.full((3, num_columns), np.nan)
    low, high = np.full((3, num_columns), -np.inf), np.full((3, num_columns), np.inf)
    for elemtype in ELEMENT_TYPES:
        for column, name in enumerate(column_names(args.dataset, elemtype)[:num_columns]):
            transform = column_transform(name)
            if transform["log1p_scale"] is not None:
                scale[elemtype, column] = transform["log1p_scale"]
            if transform["clip"] is not None:
                low[elemtype, column], high[elemtype, column] = transform["clip"]

    count, total, total_sq = np.zeros(3), np.zeros((3, num_columns)), np.zeros((3, num_columns))
    # Raw range, clipped and non-finite values of each column, to diagnose the statistics.
    minimum, maximum = np.full((3, num_columns), np.inf), np.full((3, num_columns), -np.inf)
    clipped, nonfinite = np.zeros((3, num_columns)), np.zeros((3, num_columns))
    offset = 0
    for builder, size, full_size in zip(builders, sizes, full_sizes):
        first, last = max(start - offset, 0), min(stop - offset, size)
        offset += size
        if first >= last:
            continue
        source = builder.as_data_source(split="train")
        indices = np.linspace(0, full_size - 1, size, dtype=np.int64) if max_events else None
        for chunk in range(first, last, CHUNK_EVENTS):
            selected = indices[chunk : min(chunk + CHUNK_EVENTS, last)] if indices is not None else range(chunk, min(chunk + CHUNK_EVENTS, last))
            X = np.concatenate([source[int(i)]["X"] for i in selected]).astype(np.float64)
            for t in ELEMENT_TYPES:
                # CMS has several calorimeter element IDs (2..11); inference maps all of them
                # to the shared cluster normalization table.
                rows = X[X[:, 0] >= 2] if args.dataset == "cms" and t == 2 else X[X[:, 0] == t]
                if len(rows) == 0:
                    continue
                finite = np.isfinite(rows)
                nonfinite[t] += (~finite).sum(axis=0)
                minimum[t] = np.minimum(minimum[t], np.where(finite, rows, np.inf).min(axis=0))
                maximum[t] = np.maximum(maximum[t], np.where(finite, rows, -np.inf).max(axis=0))
                clipped[t] += ((rows < low[t]) | (rows > high[t])).sum(axis=0)
                rows = np.clip(rows, low[t], high[t])
                log = ~np.isnan(scale[t])
                rows[:, log] = np.log1p(rows[:, log] / scale[t, log])
                count[t] += len(rows)
                total[t] += rows.sum(axis=0)
                total_sq[t] += np.square(rows).sum(axis=0)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_path(args.work_dir, args.dataset, args.version, args.shard),
        shard=args.shard,
        num_shards=args.num_shards,
        events=stop - start,
        population_events=sum(full_sizes),
        max_events_per_build=max_events or 0,
        builds=[builder.info.full_name for builder in builders],
        count=count,
        total=total,
        total_sq=total_sq,
        minimum=minimum,
        maximum=maximum,
        clipped=clipped,
        nonfinite=nonfinite,
    )
    print(f"Shard {args.shard} of {args.num_shards}: train events {start} to {stop} of {sum(sizes)}")


def reduce_shards(args):
    """Mean and std of each column from the sums of all shards."""
    shards = [dict(np.load(path)) for path in sorted(args.work_dir.glob(f"{args.dataset}_{args.version}_shard*.npz"))]
    num_shards = int(shards[0]["num_shards"]) if shards else 1
    if sorted((int(shard["shard"]), int(shard["num_shards"])) for shard in shards) != [(shard, num_shards) for shard in range(num_shards)]:
        raise SystemExit(f"Missing or mixed shards of {args.dataset} {args.version} in {args.work_dir}")

    count, total, total_sq, clipped, nonfinite = (
        sum(shard[key] for shard in shards) for key in ("count", "total", "total_sq", "clipped", "nonfinite")
    )
    minimum = np.min([shard["minimum"] for shard in shards], axis=0)
    maximum = np.max([shard["maximum"] for shard in shards], axis=0)
    stats = {"samples": shards[0]["builds"].tolist(), "events": int(sum(shard["events"] for shard in shards))}
    max_events = int(shards[0].get("max_events_per_build", 0))
    if max_events:
        stats["sampling"] = {
            "method": "evenly_spaced_train_indices",
            "max_events_per_build": max_events,
            "population_events": int(shards[0]["population_events"]),
        }
    for elemtype, name in ELEMENT_TYPES.items():
        mean = total[elemtype] / count[elemtype]
        # Rounding can make the variance of a constant column slightly negative.
        std = np.sqrt(np.maximum(total_sq[elemtype] / count[elemtype] - np.square(mean), 0.0))
        names = column_names(args.dataset, elemtype)
        stats[name] = [
            {
                "column": c,
                "name": names[c],
                **column_transform(names[c]),
                "mean": float(mean[c]),
                "std": float(std[c]),
                "min": float(minimum[elemtype, c]),
                "max": float(maximum[elemtype, c]),
                "clipped": int(clipped[elemtype, c]),
                "nonfinite": int(nonfinite[elemtype, c]),
            }
            for c in range(1, len(mean))
        ]

    invalid = [
        f"{name} {column}" for name in ELEMENT_TYPES.values() for column in stats[name] if not np.isfinite([column["mean"], column["std"]]).all()
    ]
    if invalid:
        # A clipped value <= -scale makes log1p invalid; non-finite raw values propagate.
        raise SystemExit(f"Statistics of {args.dataset} {args.version} that are not finite, not written:\n" + "\n".join(invalid))

    path = stats_path(args.dataset, args.version)
    path.parent.mkdir(exist_ok=True)
    header = "# Written by scripts/compute_input_normalization_stats.py: mean and std of each column after its transform.\n"
    path.write_text(header + yaml.safe_dump(stats, sort_keys=False, default_flow_style=None, width=150))
    print(f"Wrote {path}: {stats['events']} train events of {len(stats['samples'])} sample configs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(required=True)
    map_parser = steps.add_parser("map", help="sum the transformed columns over one shard of the train events")
    map_parser.set_defaults(run=map_shard)
    map_parser.add_argument("--data-dir", type=Path, required=True)
    map_parser.add_argument("--samples", nargs="+", help="process names, e.g. ttbar_nopu (default: all)")
    map_parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)), help="default: the srun task index")
    map_parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)), help="default: the number of srun tasks")
    map_parser.add_argument("--max-events-per-build", type=int, help="sample this many evenly spaced train events per config (default: all)")
    reduce_parser = steps.add_parser("reduce", help="combine the sums of all shards into the statistics")
    reduce_parser.set_defaults(run=reduce_shards)
    for step_parser in (map_parser, reduce_parser):
        step_parser.add_argument("dataset", choices=sorted(X_FEATURES))
        step_parser.add_argument("version")
        step_parser.add_argument("--work-dir", type=Path, required=True, help="directory of the per-shard sums")
    args = parser.parse_args(sys.argv[1:] or DEBUG_ARGS)
    if getattr(args, "max_events_per_build", None) is not None and args.max_events_per_build < 1:
        parser.error("--max-events-per-build must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
