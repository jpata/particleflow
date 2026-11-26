#!/usr/bin/env python3
"""Run four single-species guns in bounded-size, resumable chunks."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

from run_gun_validation import REPO, prepare, run_sample

SPECIES = {"electron": 11, "charged_pion": 211, "muon": 13, "photon": 22}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked(command, log):
    with log.open("w") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10000, help="Events per species")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--chunk-events", type=int, default=500)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.events, args.threads, args.streams, args.chunk_events) < 1 or args.streams > args.threads:
        parser.error("Counts must be positive and streams must not exceed threads")
    if args.resume and args.run_dir:
        parser.error("Choose --resume or --run-dir")
    run = (args.resume or args.run_dir or REPO / "logs/cmssw" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-single-guns-10000ev-8threads"
    )).resolve()
    os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["MPLCONFIGDIR"] = "/tmp/particleflow-matplotlib"
    if not args.resume:
        run.mkdir(parents=True, exist_ok=False)
        manifest = dict(events_per_species=args.events, threads=args.threads, streams=args.streams,
                        chunk_events=args.chunk_events, pt=[0.7, 10.0], eta=[-3.0, 3.0], antiparticles=True,
                        pileup=0, geometry="ExtendedRun4D121", samples={})
        for species_index, (name, pid) in enumerate(SPECIES.items()):
            chunks = []
            for chunk_index, first in enumerate(range(1, args.events + 1, args.chunk_events)):
                count = min(args.chunk_events, args.events - first + 1)
                directory = run / name / f"chunk_{chunk_index:03d}"
                settings = dict(ids=[pid], pt=manifest["pt"], eta=manifest["eta"],
                                seed=285001 + species_index * 1000000 + chunk_index * 10000)
                prepared = prepare(directory, count, count, samples={name: settings},
                                   threads=args.threads, streams=args.streams, first_event=first)
                chunks.append(prepared[name])
            manifest["samples"][name] = dict(pdg_id=pid, chunks=chunks)
        save(run / "manifest.json", manifest)
    manifest = json.loads((run / "manifest.json").read_text())
    output = REPO / "reports/cmssw_truth_postprocessing_guns" / run.name
    output.mkdir(parents=True, exist_ok=True)
    state_path = run / "suite_status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(samples={})
    state["state"] = "running"
    save(state_path, state)
    print(f"Suite: {run}\nReports: {output}", flush=True)
    summary = {}
    for name, sample in manifest["samples"].items():
        for index, settings in enumerate(sample["chunks"]):
            chunk = Path(settings["directory"])
            complete = chunk / "postprocessing_complete.json"
            if complete.exists():
                continue
            state["active"] = dict(sample=name, chunk=index, directory=str(chunk))
            save(state_path, state)
            print(f"{name}: chunk {index + 1}/{len(sample['chunks'])}", flush=True)
            _, ok = run_sample(name, settings)
            if not ok:
                raise RuntimeError(f"Simulation/association stage failed: {chunk}")
            checked([sys.executable, "-m", "mlpf.data.cms.postprocessing_truth",
                     str(chunk / "step3_inNANOAODSIM.root"), "--associations",
                     str(chunk / "truth_postprocessing_weights.json"), "--output", str(chunk / "events.pkl")],
                    chunk / "postprocessing_run.log")
            results = pickle.loads((chunk / "events.pkl").read_bytes())
            ids = [tuple(event["event_id"]) for event in results]
            expected = set(range(settings["first_event"], settings["first_event"] + settings["events"]))
            if len(results) != settings["events"] or len(set(ids)) != len(ids) or {event[2] for event in ids} != expected:
                raise RuntimeError(f"Unexpected event count or event identities: {chunk}")
            meta = json.loads((chunk / "truth_postprocessing_weights.json").read_text())["events"]
            if len(meta) != len(results) or any(event["uncovered_detected_particles"] for event in meta):
                raise RuntimeError(f"Incomplete truth coverage: {chunk}")
            # These files were generated by this chunk and are reproducible from
            # its retained configs/seeds. Keep NanoAOD, weights, targets and logs.
            removed = []
            for filename in ("step1.root", "step2.root", "step3.root", "step3_inDQM.root", "step3_inMINIAODSIM.root"):
                path = chunk / filename
                if path.exists():
                    path.unlink()
                    removed.append(filename)
            save(complete, dict(events=len(results), targets=sum(int((r["target_id"] > 0).sum()) for r in results),
                                removed_intermediates=removed, finished=datetime.now(timezone.utc).isoformat()))
            state["samples"][name] = dict(completed_events=settings["first_event"] + settings["events"] - 1)
            save(state_path, state)
            del results, meta
        assigned = []
        for settings in sample["chunks"]:
            assigned.extend(pickle.loads((Path(settings["directory"]) / "events.pkl").read_bytes()))
        if len(assigned) != manifest["events_per_species"] or len({tuple(r["event_id"]) for r in assigned}) != len(assigned):
            raise RuntimeError(f"Wrong total or repeated event IDs: {name}")
        target_directory = output / name
        target_directory.mkdir(exist_ok=True)
        cached = target_directory / "events.pkl"
        with cached.open("wb") as handle:
            pickle.dump(assigned, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del assigned
        checked([sys.executable, str(REPO / "scripts/cmssw/validate_truth_postprocessing.py"), "--sample", name,
                 sample["chunks"][0]["directory"], "--output", str(output), "--cached-events", str(cached)],
                run / f"{name}_validation_run.log")
        validation_path = target_directory / "validation.json"
        validation = json.loads(validation_path.read_text())
        validation.pop("source_nano", None)
        validation["source_nanos"] = [str(Path(c["directory"]) / "step3_inNANOAODSIM.root") for c in sample["chunks"]]
        save(validation_path, validation)
        summary[name] = validation["summary"]
        state["samples"][name]["validation_complete"] = True
        save(state_path, state)
    save(output / "summary.json", summary)
    checked([sys.executable, str(REPO / "scripts/cmssw/summarize_gun_validation.py"), str(output)], run / "summary_run.log")
    state.update(state="complete", finished=datetime.now(timezone.utc).isoformat())
    state.pop("active", None)
    save(state_path, state)
    print("All four gun samples and GEN-only validation completed", flush=True)


if __name__ == "__main__":
    main()
