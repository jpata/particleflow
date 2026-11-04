"""Locate MET tails and particle decision migrations in retained paired events."""

import argparse
import json
from pathlib import Path
import numpy as np
import awkward as ak


def met(particles):
    active = particles["cls_id"] != 0
    pt, phi = particles["pt"][active], particles["phi"][active]
    return float(np.hypot(np.sum(pt * np.cos(phi)), np.sum(pt * np.sin(phi))))


def inspect(root, analysis):
    manifest = json.loads((root / "input-manifest.json").read_text())["cases"]
    events = ak.from_parquet(analysis / "paired-events.parquet")
    assert len(events) == len(manifest)
    rows = []
    for case, event in zip(manifest, events, strict=True):
        pr = {k: ak.to_numpy(event.particles_reference[k]) for k in ["cls_id", "pt", "eta", "phi", "energy"]}
        pc = {k: ak.to_numpy(event.particles_candidate[k]) for k in pr}
        changed = np.flatnonzero(pr["cls_id"] != pc["cls_id"])
        decisions = []
        for index in sorted(changed, key=lambda i: max(pr["pt"][i], pc["pt"][i]), reverse=True)[:20]:
            decisions.append(
                dict(element=int(index), reference={k: float(v[index]) for k, v in pr.items()}, candidate={k: float(v[index]) for k, v in pc.items()})
            )
        rows.append(
            dict(
                case=case["file"],
                dataset_index=case["index"],
                met_reference=met(pr),
                met_candidate=met(pc),
                delta_met=met(pc) - met(pr),
                particle_decisions_changed=len(changed),
                largest_decision_migrations=decisions,
            )
        )
    rows = sorted(rows, key=lambda r: abs(r["delta_met"]), reverse=True)[:10]
    (analysis / "met-outliers.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(analysis.name, rows[0]["case"], rows[0]["delta_met"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--analysis", type=Path)
    a = p.parse_args()
    for folder in [a.analysis] if a.analysis else sorted((a.root / "analysis").iterdir()):
        if (folder / "paired-events.parquet").exists():
            inspect(a.root, folder)
