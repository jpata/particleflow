#!/usr/bin/env python3
"""Render ColliderML release events using the same renderer as Key4hep.

Supply --parquet to display the exact converted cluster positions; otherwise
clusters are built with the native default clustering and calibration. Track
helices use ACTS q/p and theta with the nominal 3 T axial field. Particle guides
show primary leaves, not the final merged/allocated MLPF targets.
"""
import argparse
from itertools import islice
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from mlpf.data.colliderml.clustering import cluster_event
from mlpf.data.colliderml.reader import iter_event_records, shard_paths
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, _build_parent_maps, calibration_factors
from scripts.visualize_key4hep import DETECTORS, render_event


class EventBranch:
    def __init__(self, values, event):
        self.values, self.event = values, event

    def array(self, entry_start, entry_stop):
        if entry_start != self.event or entry_stop != self.event + 1:
            raise IndexError("Only the selected event was loaded")
        return ak.Array([self.values])


class EventTree(dict):
    """Minimal single-event adapter for the shared ROOT display renderer."""
    def __init__(self, branches, event, num_entries):
        super().__init__({key: EventBranch(value, event) for key, value in branches.items()})
        self.num_entries = num_entries


def adapt_event(record, event, num_entries, clusters=None):
    branches = {}

    def collection(name, values):
        branches[name] = []
        branches.update({f"{name}/{name}.{key}": np.asarray(value) for key, value in values.items()})

    tracks = record["tracks"]
    theta = np.clip(np.asarray(tracks["theta"]), 1e-9, np.pi - 1e-9)
    collection("ActsTracks", {"trackStates_begin": np.arange(len(theta))})
    collection("_ActsTracks_trackStates", {
        "phi": tracks["phi"], "D0": tracks["d0"], "Z0": tracks["z0"],
        "omega": 2.99792458e-4 * DETECTORS["colliderml"].magnetic_field_tesla * np.asarray(tracks["qop"]) / np.sin(theta),
        "tanLambda": np.cos(theta) / np.sin(theta),
    })
    calo = record["calo_hits"]
    if clusters is None:
        region = np.asarray(calo["detector"])
        energy = np.asarray(calo["total_energy"]) * calibration_factors(region, DEFAULT_CALIBRATION)
        clusters = cluster_event(*(np.asarray(calo[axis]) for axis in "xyz"), energy, region)[1]
    clusters = np.asarray(clusters).reshape(-1, 17)
    collection("PandoraClusters", {"energy": clusters[:, 5], **{f"position.{axis}": clusters[:, 6 + i] for i, axis in enumerate("xyz")}})
    for obj, prefix, regions in (("tracker_hits", "TrackerRegion", range(9)), ("calo_hits", "CaloRegion", range(9, 15))):
        hits = record[obj]
        for region in regions:
            selected = np.asarray(hits["detector"]) == region
            collection(f"{prefix}{region}", {f"position.{axis}": np.asarray(hits[axis])[selected] for axis in "xyz"})
    particles = record["particles"]
    _, _, _, leaf, _ = _build_parent_maps(particles)
    collection("MCParticles", {
        "generatorStatus": leaf.astype(int), "PDG": particles["pdg_id"],
        "charge": particles["charge"], "mass": particles["mass"],
        **{f"momentum.{axis}": particles[f"p{axis}"] for axis in "xyz"},
    })
    return EventTree(branches, event, num_entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="aligned release shard root directory")
    parser.add_argument("--sample", default="ttbar_pu0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--events", type=int, nargs="+", default=[0], help="row indices within the source shard")
    parser.add_argument("--parquet", type=Path, help="converted MLPF parquet containing the selected event IDs")
    parser.add_argument("--output-dir", type=Path, default=Path("event_displays"))
    parser.add_argument("--plot-limit", type=float, default=6500.0, help="transverse half-width in mm")
    parser.add_argument("--max-hits", type=int, default=800, help="display cap per detector region")
    parser.add_argument("--input-view", choices=("combined", "pf", "hits"), default="combined")
    parser.add_argument("--no-particles", action="store_true")
    parser.add_argument("--target-only", action="store_true", help="primary-leaf proxy, not final allocated targets")
    args = parser.parse_args()
    paths = [shard_paths(args.input, args.sample, obj)[args.shard] for obj in ("particles", "tracks", "calo_hits", "tracker_hits")]
    count = pq.ParquetFile(paths[0]).metadata.num_rows
    if not args.events or min(args.events) < 0 or max(args.events) >= count:
        parser.error(f"event indices must lie in [0, {count})")
    if args.max_hits < 1 or args.plot_limit <= 0:
        parser.error("--max-hits and --plot-limit must be positive")
    if args.target_only and args.input_view != "combined":
        parser.error("--input-view cannot be combined with --target-only")
    clusters_by_id = None
    if args.parquet:
        data = ak.from_parquet(args.parquet, columns=["event_id", "X_cluster"])
        clusters_by_id = {int(event_id): ak.to_numpy(clusters) for event_id, clusters in zip(data["event_id"], data["X_cluster"])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.events)
    for index, record in enumerate(islice(iter_event_records(*paths, batch_size=1), max(requested) + 1)):
        if index not in requested:
            continue
        clusters = clusters_by_id[record["event_id"]] if clusters_by_id is not None else None
        tree = adapt_event(record, index, count, clusters)
        suffix = "_targets" if args.target_only else ""
        output = args.output_dir / f"colliderml_event_{index}{suffix}.png"
        render_event(args.input, index, output, args.max_hits, "colliderml", args.plot_limit,
                     show_particles=not args.no_particles, target_only=args.target_only, input_view=args.input_view, tree=tree)
        print(f"{output} (source event_id={record['event_id']})")


if __name__ == "__main__":
    main()
