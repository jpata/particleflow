#!/usr/bin/env python3
"""Produce independent gun samples, export graph weights, and validate assignments.

Run with the repository Python environment. CMSSW commands use env.sh.
The manifest and per-sample status.json make a completed stage resumable.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "CMSSW_20_1_0_pre5/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal"
SAMPLES = {
    "mixed": dict(ids=[11, 13, 22, 130, 211], pt=[0.7, 10.0], eta=[-3.0, 3.0], seed=41001),
    "electron_forward": dict(ids=[11], pt=[5.0, 50.0], eta=[1.6, 2.8], seed=52001),
    "neutral_barrel": dict(ids=[22, 130, 2112], pt=[0.7, 10.0], eta=[-1.3, 1.3], seed=63001),
    "charged_lowpt": dict(ids=[13, 211, 321, 2212], pt=[0.3, 3.0], eta=[-3.0, 3.0], seed=74001),
}


def prepare(directory, events, specialized_events, samples=None, threads=2, streams=1, first_event=1):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, settings in (SAMPLES if samples is None else samples).items():
        run = directory / name
        run.mkdir(exist_ok=True)
        count = events if name == "mixed" else specialized_events
        manifest[name] = dict(**settings, events=count, antiparticles=True, threads=threads, streams=streams, first_event=first_event, directory=str(run))
        sources = [SOURCE / "SinglePiFlatPt0p7To10_cfi_GEN_SIM.py", *sorted(SOURCE.glob("step2_*.py")), *sorted(SOURCE.glob("step3_*.py"))]
        for stage, source in enumerate(sources, start=1):
            text = source.read_text()
            text += f"\n# Independent gun-validation overrides: {name}.\nprocess.maxEvents.input = {count}\n"
            text += f"process.options.numberOfThreads = {threads}\nprocess.options.numberOfStreams = {streams}\n"
            if stage == 1:
                text += f"process.source.firstEvent = cms.untracked.uint32({first_event})\n"
                text += f"process.generator.PGunParameters.ParticleID = cms.vint32({', '.join(map(str, settings['ids']))})\n"
                for field, value in zip(("MinPt", "MaxPt", "MinEta", "MaxEta"), settings["pt"] + settings["eta"]):
                    text += f"process.generator.PGunParameters.{field} = cms.double({value})\n"
                text += f"process.generator.psethack = cms.string('{name} validation gun')\n"
            # Change every configured RNG seed, including digitization/reconstruction.
            text += f"""if hasattr(process, 'RandomNumberGeneratorService'):
    for _engine_index, _engine_name in enumerate(sorted(process.RandomNumberGeneratorService.parameterNames_())):
        _engine = getattr(process.RandomNumberGeneratorService, _engine_name)
        if hasattr(_engine, 'initialSeed'):
            _engine.initialSeed = cms.untracked.uint32({settings["seed"]} + {stage} * 1000 + _engine_index)
"""
            (run / f"step{stage}_cfg.py").write_text(text)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_sample(name, settings):
    run = Path(settings["directory"])
    status_path = run / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    env = dict(os.environ, APPTAINER_NV="true", SITECONFIG_PATH="/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia", CMSSW_RUN_DIR=str(run))
    stages = [
        (f"step{i}", [str(REPO / "env.sh"), "cmsRun", "-j", str(run / f"step{i}_FrameworkJobReport.xml"), str(run / f"step{i}_cfg.py")])
        for i in (1, 2, 3)
    ]
    stages.append(
        (
            "associations",
            [
                str(REPO / "env.sh"),
                "python3",
                str(REPO / "scripts/cmssw/extract_association_metrics.py"),
                str(run / "step3.root"),
                "--skip-legacy",
                "--sparse",
                "--output",
                str(run / "truth_postprocessing_weights.json"),
            ],
        )
    )
    for stage, command in stages:
        if status.get(stage, {}).get("exit_code") == 0:
            continue
        print(f"{name}: starting {stage}", flush=True)
        status[stage] = dict(started=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        with (run / f"{stage}_run.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[stage].update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"{name}: {stage} exit {result.returncode}", flush=True)
        if result.returncode:
            return name, False
    return name, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--specialized-events", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.events, args.specialized_events, args.jobs) < 1:
        parser.error("event counts and jobs must be positive")
    directory = (
        args.resume.resolve() if args.resume else REPO / "logs/cmssw" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-gun-validation")
    )
    manifest = json.loads((directory / "manifest.json").read_text()) if args.resume else prepare(directory, args.events, args.specialized_events)
    print(f"Suite: {directory}", flush=True)
    (REPO / "logs/cmssw/latest_gun_validation.txt").write_text(str(directory) + "\n")
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_sample, name, settings) for name, settings in manifest.items()]
        for future in as_completed(futures):
            name, ok = future.result()
            if not ok:
                failures.append(name)
    if failures:
        print(f"Failed samples: {failures}. Inspect logs; resume using --resume {directory}", flush=True)
        return 1
    output = REPO / "reports/cmssw_truth_postprocessing_extended" / directory.name
    command = [sys.executable, str(REPO / "scripts/cmssw/validate_truth_postprocessing.py"), "--output", str(output)]
    for name, settings in manifest.items():
        command.extend(["--sample", name, settings["directory"]])
    print(f"Validating all samples into {output}", flush=True)
    with (directory / "validation_run.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    print(f"Validation exit {result.returncode}; report: {output}", flush=True)
    if result.returncode == 0:
        summary = json.loads((output / "summary.json").read_text())
        for name, settings in manifest.items():
            if summary[name]["events"] != settings["events"]:
                raise RuntimeError(f"{name}: processed {summary[name]['events']} events, expected {settings['events']}")
        with (directory / "summary_run.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/cmssw/summarize_gun_validation.py"), str(output)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
            )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
