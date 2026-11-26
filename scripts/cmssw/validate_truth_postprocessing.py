#!/usr/bin/env python3
"""Validate new targets directly against fiducial visible GEN status-1 particles."""

import argparse
import collections
import json
import os
import pickle
import shutil
from pathlib import Path
import subprocess
import sys

from mlpf.data.cms.postprocessing_truth import process_event, read_events

ENGINE = "fiducial_gen_response_v2_full_hgcal"


def validate_sample(label, run, output, cached_events=None):
    directory = output / label
    directory.mkdir(parents=True, exist_ok=True)
    if cached_events:
        results = pickle.loads(cached_events.read_bytes())
    else:
        results = [process_event(event) for event in read_events(run / "step3_inNANOAODSIM.root", run / "truth_postprocessing_weights.json")]
    events = directory / "events.pkl"
    if not cached_events or cached_events.resolve() != events.resolve():
        with events.open("wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    response = directory / "gen_target_response"
    command = [sys.executable, str(Path(__file__).with_name("plot_gen_target_response.py")), str(events), "--output", str(response), "--label", label]
    subprocess.run(command, check=True, env=dict(os.environ, MPLCONFIGDIR="/tmp/particleflow-matplotlib"))
    for name in ("targets_vs_gen.png", "targets_vs_gen_fiducial.png"):
        shutil.copy2(response / "particle_distributions.png", directory / name)
    shutil.copy2(response / "particle_content.png", directory / "particle_content.png")
    physics = json.loads((response / "response.json").read_text())["summary"]
    summary = dict(
        validation_engine=ENGINE,
        events=len(results),
        inputs=sum(len(r["elements"]) for r in results),
        original_truth=sum(len(r["truth"]) for r in results),
        targets=sum(int((r["target_id"] > 0).sum()) for r in results),
        splits=sum(len(r["split_truth"]) for r in results),
        merges=sum(len(r["merged_rows"]) for r in results),
        unmatched_truth=sum(len(r["unmatched_truth"]) for r in results),
        assigned_collections=dict(collections.Counter(r["elements"][i].collection for r in results for i in r["target_id"].nonzero()[0])),
        response=physics,
        response_directory=str(response.resolve()),
    )
    (directory / "validation.json").write_text(
        json.dumps(
            dict(
                summary=summary,
                reference="Fiducial visible GEN status 1 only",
                source_nano=str((run / "step3_inNANOAODSIM.root").resolve()),
                target_source=str(events.resolve()),
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs=2, action="append", metavar=("LABEL", "RUN_DIRECTORY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-events", type=Path, help="Reuse the existing assigned targets for one sample; no assignment changes")
    args = parser.parse_args()
    if args.cached_events and len(args.sample) != 1:
        parser.error("--cached-events requires exactly one sample")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {label: validate_sample(label, Path(run), args.output, args.cached_events) for label, run in args.sample}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
