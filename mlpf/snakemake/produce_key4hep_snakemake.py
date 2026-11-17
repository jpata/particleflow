"""Generate one namespaced dataset DAG for a multi-detector Key4hep campaign."""

import argparse
from pathlib import Path

from mlpf.snakemake import produce_snakemake
from mlpf.utils import load_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="key4hep_50k")
    parser.add_argument("--productions", nargs="+", help="Optional subset of the campaign's detectors")
    args = parser.parse_args(argv)
    spec = load_spec(produce_snakemake.SPEC_FILE)
    campaign = spec["campaigns"][args.campaign]
    detectors = args.productions or campaign["productions"]
    if not detectors or len(set(detectors)) != len(detectors) or not set(detectors).issubset(campaign["productions"]):
        raise ValueError("Select distinct productions from this campaign")
    jobs = Path(campaign["jobs_dir"])
    stages = {stage: [] for stage in ("gen", "post", "tfds", "tfds_hit")}
    modules = []
    for detector in detectors:
        production = spec["productions"][detector]
        if production.get("allow_partial", True) or not production.get("validate_events", False):
            raise ValueError(f"{detector}: campaign requires complete, event-validated datasets")
        for sample, settings in production["samples"].items():
            begin, end = settings["seed_range"]
            if (end - begin) * settings["events_per_job"] != campaign["expected_events_per_sample"]:
                raise ValueError(f"{detector}/{sample}: wrong event count for campaign")
            if sample not in production["tfds_mapping"]:
                raise ValueError(f"{detector}/{sample}: missing track/cluster TFDS mapping")
        steps = ["gen", "post", "tfds"]
        if production.get("tfds_hit_mapping"):
            steps.append("tfds_hit")
        produce_snakemake.main(["--production", detector, "--jobs-root", str(jobs), "--steps", *steps])
        for step in steps:
            stages[step].append(str(jobs / detector / step / "all.done"))
        modules.append(f'\nmodule {detector}:\n    snakefile: "{detector}/Snakefile"\n\n' f"use rule * from {detector} as {detector}_*\n")
    targets = stages["tfds"] + stages["tfds_hit"]
    text = "# Generated dataset campaign: simulation through TFDS; no training.\n"
    text += f"rule all:\n    input: {targets!r}\n\nrule datasets:\n    input: {targets!r}\n"
    for stage, outputs in stages.items():
        if outputs:
            text += f"\nrule {stage}:\n    input: {outputs!r}\n"
    text += "".join(modules)
    (jobs / "Snakefile").write_text(text)
    print(f"Combined dataset workflow: {jobs / 'Snakefile'}")


if __name__ == "__main__":
    main()
