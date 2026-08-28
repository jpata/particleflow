#!/usr/bin/env python3
"""Add explicit pipeline-validation candidates and metadata to IDEA parquet."""

import argparse
from pathlib import Path

import awkward as ak

from mlpf.heptfds.edm4hep_utils.utils_idea import (
    CANDIDATE_SOURCE,
    TRACK_SOURCE,
    repair_jet_indices,
)


def prepare_pipeline_file(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    data = ak.from_parquet(input_path)
    fields = {field: data[field] for field in data.fields if field not in {"ycand_track", "ycand_cluster"}}
    target_tracks = []
    target_clusters = []
    repaired = 0

    for event_index in range(len(data["X_track"])):
        num_jets = len(data["targetjet"][event_index])
        tracks, n_track = repair_jet_indices(data["ytarget_track"][event_index], num_jets)
        clusters, n_cluster = repair_jet_indices(data["ytarget_cluster"][event_index], num_jets)
        target_tracks.append(tracks)
        target_clusters.append(clusters)
        repaired += n_track + n_cluster

    fields["ytarget_track"] = ak.Array(target_tracks)
    fields["ytarget_cluster"] = ak.Array(target_clusters)
    # This is deliberately an oracle reference for evaluator plumbing.  The
    # actual MLPF training target remains ytarget; no baseline reconstruction
    # is available in the IDEA steering yet.
    fields["ycand_track"] = ak.Array(target_tracks)
    fields["ycand_cluster"] = ak.Array(target_clusters)
    num_events = len(target_tracks)
    fields["metadata_track_source"] = ak.Array([TRACK_SOURCE] * num_events)
    fields["metadata_candidate_source"] = ak.Array([CANDIDATE_SOURCE] * num_events)
    fields["metadata_suitable_for_physics"] = ak.Array([False] * num_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(ak.Record(fields), output_path)
    return {"events": num_events, "repaired_jet_indices": repaired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input IDEA parquet")
    parser.add_argument("output", help="Output pipeline-validation parquet")
    args = parser.parse_args()
    summary = prepare_pipeline_file(args.input, args.output)
    print(f"wrote {summary['events']} events to {args.output}; repaired jet_idx={summary['repaired_jet_indices']}")


if __name__ == "__main__":
    main()
