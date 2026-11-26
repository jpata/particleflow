#!/usr/bin/env python3
"""Audit top ancestry in inclusive events; shared strings prevent an exclusive hard-process interpretation."""

import argparse
import json
from pathlib import Path

import ROOT
from DataFormats.FWLite import Events, Handle

ROOT.gInterpreter.Declare('#include "DataFormats/HepMCCandidate/interface/GenParticle.h"')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("--associations", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
payload = json.loads(args.associations.read_text())
metadata = {int(e["event"]): e for e in payload["events"]}
handle = Handle("std::vector<reco::GenParticle>")
checks = []
for event in Events(str(args.input.resolve())):
    event.getByLabel("genParticles", handle)
    particles = handle.product()
    count = len(particles)
    children = [[] for _ in range(count)]
    for i, p in enumerate(particles):
        for j in range(p.numberOfMothers()):
            ref = p.motherRef(j)
            if ref.isNonnull():
                key = int(ref.key())
                if not 0 <= key < count:
                    raise ValueError("GEN mother reference outside source collection")
                children[key].append(i)
    tops = [i for i, p in enumerate(particles) if abs(p.pdgId()) == 6]
    last_tops = [i for i in tops if particles[i].statusFlags().isLastCopy()]
    assert sorted(int(particles[i].pdgId()) for i in last_tops) == [-6, 6]
    descendants = set()
    stack = list(tops)
    while stack:
        node = stack.pop()
        if node in descendants:
            continue
        descendants.add(node)
        stack.extend(children[node])
    number = int(event.eventAuxiliary().event())
    meta = metadata[number]
    selected = [i for i, p in enumerate(particles) if p.status() == 1 and i in descendants]
    stable = [i for i, p in enumerate(particles) if p.status() == 1]
    assert {int(p["index"]) for p in meta["gen_status1"]} == set(stable)
    for row in meta["gen_status1"]:
        row["has_top_ancestor"] = int(row["index"]) in descendants

    def energy(indices, visible=False):
        return sum(float(particles[i].energy()) for i in indices if not visible or abs(particles[i].pdgId()) not in (12, 14, 16))

    check = dict(
        event=number,
        all_status1_energy=energy(stable),
        visible_status1_energy=energy(stable, True),
        ttbar_status1=len(selected),
        ttbar_status1_energy=energy(selected),
        ttbar_visible_energy=energy(selected, True),
        last_top_pair_energy=energy(last_tops),
        from_hard_process_status1=sum(p.status() == 1 and p.statusFlags().fromHardProcess() for p in particles),
    )
    meta["naive_top_ancestry"] = dict(
        definition="status-1 descendants of any GEN top or antitop copy, via GEN mother links",
        gen_indices=selected,
        top_indices=tops,
        last_top_indices=last_tops,
        energy_check=check,
    )
    checks.append(check)
    print(check, flush=True)
payload["ancestry_caveat"] = "Any-top ancestry includes shared string hadrons and is not an exclusive hard-process selection"
args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
args.output.with_suffix(".scope_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
