#!/usr/bin/env python3
"""Run resumable postprocessing validation and plots on a completed ttbar sample."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
ENGINE = "fiducial_gen_response_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-targets", action="store_true", help="Validate the existing events.pkl without changing assignment")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else REPO / "reports/cmssw_truth_postprocessing_ttbar" / run.name
    script = REPO / "scripts/cmssw"
    events = output / "ttbar_pu0/events.pkl"
    commands = [
        (
            "validation",
            [sys.executable, str(script / "validate_truth_postprocessing.py"), "--sample", "ttbar_pu0", str(run), "--output", str(output)],
        ),
        ("summary", [sys.executable, str(script / "summarize_gun_validation.py"), str(output)]),
        ("assignments", [sys.executable, str(script / "plot_target_assignments.py"), str(events), "--output", str(output / "assignments")]),
        ("transverse", [sys.executable, str(script / "validate_transverse_activity.py"), str(events), "--output", str(output / "transverse")]),
    ]
    if args.reuse_targets:
        if not events.exists():
            parser.error("--reuse-targets requires an existing events.pkl")
        commands[0][1].extend(["--cached-events", str(events)])
    path = run / "postprocessing_status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    env = dict(os.environ, MPLCONFIGDIR="/tmp/particleflow-matplotlib", PYTHONUNBUFFERED="1")
    for name, command in commands:
        if status.get(name, {}).get("exit_code") == 0 and status[name].get("engine") == ENGINE:
            continue
        status[name] = dict(started=datetime.now(timezone.utc).isoformat(), command=command, engine=ENGINE)
        path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"Starting {name}", flush=True)
        with (run / f"{name}_run.log").open("w") as handle:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        status[name].update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            return result.returncode
    summary = json.loads((output / "summary.json").read_text())["ttbar_pu0"]
    expected = json.loads((run / "manifest.json").read_text())["events"]
    assert summary["events"] == expected
    (run / "postprocessing_complete.json").write_text(
        json.dumps(dict(events=expected, output=str(output), finished=datetime.now(timezone.utc).isoformat()), indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
