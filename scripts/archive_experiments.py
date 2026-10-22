#!/usr/bin/env python3
"""Archive experiments, retaining per-run outputs and the latest scenario-wide step.

Run from the repository root:
    python3 scripts/archive_experiments.py --dry-run
    python3 scripts/archive_experiments.py --output-dir experiment_archives
    python3 scripts/archive_experiments.py --live --overwrite

One uncompressed tar per immediate experiment directory; paths inside the tar
start with experiments/<experiment>/. All files are kept except older numbered
checkpoints, preds_step_N and plots_step_N directories. Latest means highest
numeric step, selected independently for each output type in each run. Predictions
include ALL dataset/rank shards. No source files are modified or deleted.
Additionally, retain checkpoints and predictions at the highest numeric step
where BOTH exist for EVERY run in the scenario (immediate experiment directory).
This shared-step selection uses presence on disk, including in --live mode; it
does not prove prediction shards are complete if that step is still being written.

On your laptop (GNU parallel, replacing USER@HOST and the remote directory):
    scp USER@HOST:/absolute/path/experiment_archives/archives.txt .
    mkdir -p results
    parallel -j 4 rsync -a --partial --protect-args USER@HOST:{} ./results/ :::: archives.txt

archives.txt contains absolute remote archive paths, one per line. selection.json
records exactly which periodic outputs were selected. Dry runs write neither
archives nor manifests. Prefer archiving after training stops: outputs still being
written may be incomplete. --live selects the latest numeric step for runs whose
rank-0 log ends with a training completion marker, and the second-latest step for
other runs. It omits unfinished periodic groups with fewer than two steps and accepts tar exit
code 1 (files changed while reading) with a warning. Logs may end with a partial
line. This is best effort, not a consistent snapshot. Other tar errors leave only
a .partial file, never a final tar.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


CHECKPOINT = re.compile(r"checkpoint-(\d+)\.pth\Z")
PERIODIC = re.compile(r"(preds|plots)_step_(\d+)\Z")


def training_completed(run):
    """Use the final rank-0 log, including early stopping, not checkpoint presence.

    The completion message is emitted after final validation/prediction/plotting.
    Only inspect the tail so large training logs need not be read in full.
    """
    log = run / "train.log.0"
    if not log.is_file():
        log = run / "train.log"
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return False
    lines = [line for line in tail.splitlines() if line.strip()]
    return bool(lines and "Training completed. Total time on device 0:" in lines[-1])


def select_entries(experiment, live=False):
    """Walk without following symlinks; prune older periodic artifacts per parent."""
    entries = []
    selections = []
    runs = [
        p
        for p in sorted(experiment.iterdir())
        if p.is_dir()
        and not p.is_symlink()
        and (
            (p / "checkpoints").is_dir()
            or (p / "train-config.yaml").is_file()
            or (p / "hyperparameters.json").is_file()
            or any(p.glob("preds_step_*"))
        )
    ]
    common_steps = None
    for run in runs:
        checkpoints = {
            int(match[1]) for p in (run / "checkpoints").glob("checkpoint-*.pth") for match in [CHECKPOINT.fullmatch(p.name)] if match and p.is_file()
        }
        predictions = {int(match[2]) for p in run.glob("preds_step_*") for match in [PERIODIC.fullmatch(p.name)] if match and p.is_dir()}
        paired_steps = checkpoints & predictions
        common_steps = paired_steps if common_steps is None else common_steps & paired_steps
    common_step = max(common_steps) if common_steps else None
    if runs and common_step is None:
        print(f"WARNING: {experiment.name}: no step has checkpoints and predictions for every run", file=sys.stderr)

    def visit(directory, use_previous):
        if directory.parent == experiment:
            use_previous = live and not training_completed(directory)
        children = sorted(directory.iterdir())
        groups = {}
        for child in children:
            match = PERIODIC.fullmatch(child.name)
            if match and child.is_dir():
                groups.setdefault(match[1], []).append((int(match[2]), child))
            match = CHECKPOINT.fullmatch(child.name)
            if directory.name == "checkpoints" and match and child.is_file():
                groups.setdefault("checkpoint", []).append((int(match[1]), child))
        excluded = set()
        for kind, candidates in groups.items():
            ordered = sorted(candidates, key=lambda item: (item[0], item[1].name))
            kept = {}
            if common_step is not None and ((kind == "checkpoint" and directory.parent in runs) or (kind == "preds" and directory in runs)):
                for step, path in candidates:
                    if step == common_step:
                        kept[path] = ["scenario-common"]
            if use_previous and len(ordered) < 2:
                print(f"WARNING: {kind} in {directory}: no previous step available; retaining only any shared-step output", file=sys.stderr)
            else:
                latest = ordered[-2 if use_previous else -1][1]
                kept.setdefault(latest, []).append("previous" if use_previous else "latest")
            excluded.update(path for _, path in candidates if path not in kept)
            for path, reasons in sorted(kept.items()):
                selections.append({"kind": kind, "path": str(path.relative_to(experiment.parent.parent)), "reasons": reasons})
        entries.append(directory)
        for child in children:
            if child in excluded:
                continue
            if child.is_dir() and not child.is_symlink():
                visit(child, use_previous)
            else:
                entries.append(child)

    visit(experiment, live)
    return entries, selections


def check_tar_result(result, live):
    if live and result.returncode == 1:
        print("WARNING: tar reports files changed while reading; accepting best-effort live archive.", file=sys.stderr)
    else:
        result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_archives"))
    parser.add_argument("--experiment", action="append", help="Immediate experiment folder name; repeat to select several")
    parser.add_argument("--dry-run", action="store_true", help="Print selections and sizes without writing anything")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing archives after a successful tar")
    parser.add_argument(
        "--live", action="store_true", help="Use latest outputs for completed runs, previous outputs otherwise; accept tar file-change exit code 1"
    )
    args = parser.parse_args()
    root = args.experiments_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        parser.error("output directory must be outside the experiments directory")
    experiments = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    if args.experiment:
        unknown = set(args.experiment) - {p.name for p in experiments}
        if unknown:
            parser.error(f"unknown experiments: {sorted(unknown)}")
        experiments = [p for p in experiments if p.name in args.experiment]
    if not experiments:
        parser.error("no experiment directories found")
    archives = [output / (p.name + ".tar") for p in experiments]
    if not args.overwrite and not args.dry_run:
        for archive in archives:
            if archive.exists():
                parser.error(f"archive already exists: {archive}; use --overwrite to replace")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        # Do not leave a previous successful transfer list after a failed rebuild.
        try:
            (output / "archives.txt").unlink()
        except FileNotFoundError:
            pass
    manifest = {}
    for experiment, archive in zip(experiments, archives):
        entries, selections = select_entries(experiment, live=args.live)
        size = sum(p.lstat().st_size for p in entries if p.is_file() and not p.is_symlink())
        print(f"{experiment.name}: {len(entries)} entries, {size / 1024**3:.2f} GiB -> {archive}", flush=True)
        for selection in selections:
            print(f"  {selection['kind']} ({', '.join(selection['reasons'])}): {selection['path']}", flush=True)
        manifest[experiment.name] = selections
        if args.dry_run:
            continue
        # Explicit members plus --no-recursion prevent tar from reintroducing pruned files.
        with tempfile.NamedTemporaryFile(dir=output, suffix=".members") as members:
            for entry in entries:
                members.write(os.fsencode(entry.relative_to(root.parent)) + b"\0")
            members.flush()
            partial = archive.with_suffix(".tar.partial")
            result = subprocess.run(
                [
                    "tar",
                    "--create",
                    "--file",
                    str(partial),
                    "--directory",
                    str(root.parent),
                    "--no-recursion",
                    "--null",
                    "--verbatim-files-from",
                    "--files-from",
                    members.name,
                ]
            )
            check_tar_result(result, args.live)
            partial.replace(archive)
    if not args.dry_run:
        (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
        pending = output / "archives.txt.partial"
        pending.write_text("".join(str(p) + "\n" for p in archives))
        pending.replace(output / "archives.txt")
        print(f"Transfer list: {output / 'archives.txt'}")


if __name__ == "__main__":
    main()
