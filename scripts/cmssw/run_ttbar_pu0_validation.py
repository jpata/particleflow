#!/usr/bin/env python3
"""Generate PU0 ttbar in Run4D121, export sparse overlaps, and validate PF targets."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "CMSSW_20_1_0_pre5/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal"
FRAGMENT = "Configuration.Generator.TTbar_14TeV_TuneCP5_cfi"


def prepare(run, events, threads=2, streams=1, seed_base=85001):
    run.mkdir(parents=True)
    sources = [SOURCE / "SinglePiFlatPt0p7To10_cfi_GEN_SIM.py", *sorted(SOURCE.glob("step2_*.py")), *sorted(SOURCE.glob("step3_*.py"))]
    for stage, source in enumerate(sources, start=1):
        text = source.read_text()
        if stage == 1:
            start = text.index("process.generator = cms.EDFilter(")
            end = text.index("# Path and EndPath definitions", start)
            text = text[:start] + f"process.load('{FRAGMENT}')\n\n" + text[end:]
        text += f"""\n# Independent PU0 ttbar validation overrides.
process.maxEvents.input = {events}
process.options.numberOfThreads = {threads}
process.options.numberOfStreams = {streams}
process.configurationMetadata.annotation = cms.untracked.string('ttbar 14 TeV CP5 PU0 inclusive PF validation')
if hasattr(process, 'RandomNumberGeneratorService'):
    for _engine_index, _engine_name in enumerate(sorted(process.RandomNumberGeneratorService.parameterNames_())):
        _engine = getattr(process.RandomNumberGeneratorService, _engine_name)
        if hasattr(_engine, 'initialSeed'):
            _engine.initialSeed = cms.untracked.uint32({seed_base} + {stage} * 1000 + _engine_index)
"""
        if stage == 2:
            text += """# Seed on both incoming beam protons, so PFReconstructable walks the entire
# collision (hard scatter, ISR/FSR, beam remnants and underlying event).
# An unseeded collision has no Signal legs; top-only seeds omit the underlying event.
process.truthLogicalGraphProducer.postProcessing.seedPdgIds = cms.vint32(2212, -2212)
process.truthLogicalGraphProducer.postProcessing.seedParentDepth = cms.uint32(0)
process.truthLogicalGraphProducer.postProcessing.keepStableSpectators = cms.bool(True)
process.truthLogicalGraphProducer.postProcessing.reconstructablePdgIds = cms.vint32()
"""
        # Fail before running if the baseline unexpectedly changes to a pileup configuration.
        text += "\nassert not hasattr(process.mix, 'input'), 'PU0 sample must not have pileup input'\n"
        (run / f"step{stage}_cfg.py").write_text(text)
    manifest = dict(
        events=events,
        generator=FRAGMENT,
        com_energy_GeV=14000,
        generator_top_mass_GeV=175,
        geometry="ExtendedRun4D121",
        pileup=0,
        truth_seeds=[2212, -2212],
        reconstructable_pdg_ids=[],
        truth_scope="inclusive beam-seeded collision; no top-only selection",
        threads=threads,
        streams=streams,
        seed_base=seed_base,
        directory=str(run),
    )
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--streams", type=int, default=1, help="Concurrent events; use 8 with 8 threads for parallel simulation")
    parser.add_argument("--seed-base", type=int, default=85001)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.events, args.threads, args.streams, args.seed_base) < 1:
        parser.error("event, thread, stream and seed values must be positive")
    if args.streams > args.threads:
        parser.error("--streams must not exceed --threads")
    if args.run_dir and args.resume:
        parser.error("--run-dir and --resume are mutually exclusive")
    run = (
        args.resume.resolve()
        if args.resume
        else (
            args.run_dir.resolve() if args.run_dir else REPO / "logs/cmssw" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-ttbar-pu0")
        )
    )
    manifest = (
        json.loads((run / "manifest.json").read_text()) if args.resume else prepare(run, args.events, args.threads, args.streams, args.seed_base)
    )
    (REPO / "logs/cmssw/latest_ttbar_pu0_validation.txt").write_text(str(run) + "\n")
    print(f"Run: {run}", flush=True)
    env = dict(os.environ, APPTAINER_NV="true", SITECONFIG_PATH="/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia", CMSSW_RUN_DIR=str(run))
    status_file = run / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
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
    for name, command in stages:
        if status.get(name, {}).get("exit_code") == 0:
            continue
        print(f"Starting {name}", flush=True)
        status[name] = dict(started=datetime.now(timezone.utc).isoformat())
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        with (run / f"{name}_run.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[name].update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            return result.returncode
    output = REPO / "reports/cmssw_truth_postprocessing_ttbar" / run.name
    metadata = json.loads((run / "truth_postprocessing_weights.json").read_text())["events"]
    assert len(metadata) == manifest["events"]
    for event in metadata:
        assert sorted(p["pdg_id"] for p in event["gen_last_copy_tops"]) == [-6, 6], event["event"]
        assert len(event["roots"]) > 0, "Empty inclusive PF truth selection"
        assert len(event["signal_seeds"]) >= 2 and all(p["status"] == 4 for p in event["signal_seeds"]), event["signal_seeds"]
        assert all(bx == 0 for bx in event["root_bunch_crossing"])
        assert all(index == 0 for index in event["root_sim_event"])
        assert not event["uncovered_detected_particles"], (event["event"], event["uncovered_detected_particles"])
    (run / "sample_checks.json").write_text(
        json.dumps(dict(events=len(metadata), ttbar_in_every_event=True, beam_seeded=True, nonempty_pf_truth=True, all_roots_pu0=True), indent=2)
        + "\n"
    )
    command = [
        sys.executable,
        str(REPO / "scripts/cmssw/validate_truth_postprocessing.py"),
        "--sample",
        "ttbar_pu0",
        str(run),
        "--output",
        str(output),
    ]
    with (run / "validation_run.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"Validation exit {result.returncode}; output {output}", flush=True)
    if result.returncode:
        return result.returncode
    summary = json.loads((output / "summary.json").read_text())["ttbar_pu0"]
    assert summary["events"] == manifest["events"]
    assert summary["original_truth"] > 0, "Empty PF truth selection for collision events"
    with (run / "summary_run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/cmssw/summarize_gun_validation.py"), str(output)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
