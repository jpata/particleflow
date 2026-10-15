#!/usr/bin/env python3
"""Dump a few real surrogate-ancestor examples with full provenance.

Scans the first events of a source shard with the converter's own rule
(truth.find_surrogate_ancestors), selects the N accepted surrogates with the largest recovered
calo deposits (default 3 per sample), and for
each prints: what was deposited (and where), the surrogate's identity and its *silent*
record daughters, the walk of a representative contribution from hit to surrogate, and the
one-line "why it was dropped / why the surrogate rule adds it".

Usage (from repo root, .venv):
  python3 scripts/colliderml/colliderml_surrogate_examples.py --sample pu0 --num-events 30
  python3 scripts/colliderml/colliderml_surrogate_examples.py --sample pu200 --num-events 5
  # --top N to change the number of examples
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # import the study script's helpers
sys.path.insert(0, str(Path(__file__).parents[2]))  # and the repo

from colliderml_surrogate_impact import _row_lookup, walk_to_primary_many, PDG_NAMES  # noqa: E402
from mlpf.data.colliderml import reader  # noqa: E402
from mlpf.data.colliderml.truth import (  # noqa: E402
    DEFAULT_CALIBRATION,
    _build_parent_maps,
    _redirect_ids,
    calibration_factors,
    find_surrogate_ancestors,
    walk_to_leaf_many,
)

ECAL, HCAL = {9, 10, 11}, {12, 13, 14}


def _name(pdg_id, charge=None):
    n = PDG_NAMES.get(abs(int(pdg_id)), str(abs(int(pdg_id))))
    chargeless = abs(int(pdg_id)) in (310, 3122, 3312)  # neutral hyperons carry 0 charge in these releases
    if abs(int(pdg_id)) >= 3000 and abs(int(pdg_id)) < 4000 and not chargeless:  # sign on baryons
        pass  # keep the generic name; sign is usually obvious from pdg sign
    return n


def event_surrogates(ev):
    """Accepted surrogates of one event (truth.find_surrogate_ancestors) with calo deposits, plus metadata."""
    particles, calo = ev["particles"], ev["calo_hits"]
    pid, pdg, parent, is_leaf, redirect = _build_parent_maps(particles)
    primary = np.asarray(ak.to_numpy(particles["primary"])).astype(bool)

    cid = _redirect_ids(np.asarray(ak.to_numpy(ak.flatten(calo["contrib_particle_ids"], axis=None))).ravel(), redirect)
    ce = np.asarray(ak.to_numpy(ak.flatten(calo["contrib_energies"], axis=None))).ravel()
    nctr = np.asarray(ak.to_numpy(ak.num(calo["contrib_particle_ids"]))).ravel()
    det = np.repeat(np.asarray(ak.to_numpy(calo["detector"])), nctr)
    w = (ce.astype(np.float32) * calibration_factors(det, DEFAULT_CALIBRATION)).astype(np.float64)

    leaf_row = walk_to_leaf_many(cid, pid, is_leaf, parent)
    th_pid = _redirect_ids(np.asarray(ak.to_numpy(ev["tracker_hits"]["particle_id"])).astype(np.int64), redirect)

    out = []
    dropped = leaf_row == -1
    clean, _, _ = find_surrogate_ancestors(np.concatenate([cid.astype(np.int64), th_pid]), pid, parent, primary, is_leaf)
    if dropped.any():
        uniq, inv = np.unique(cid[dropped], return_inverse=True)
        row_per_contrib = walk_to_primary_many(uniq, pid, primary, parent)[inv]
        for r in clean:
            # lift the (dropped-space == r) mask back to full-contribution space
            m = np.zeros(len(cid), bool)
            m[np.nonzero(dropped)[0]] = row_per_contrib == r
            if not m.any():
                continue  # tracker-only surrogate: no calo deposit to show
            out.append(
                dict(
                    row=int(r),
                    dep=float(w[m].sum()),
                    dep_ecal=float(w[m & np.isin(det, list(ECAL))].sum()),
                    dep_hcal=float(w[m & np.isin(det, list(HCAL))].sum()),
                    n_contrib=int(m.sum()),
                    contrib_rows=np.nonzero(m)[0],
                    cid=cid,
                    dep_per_leaf=(
                        np.bincount(leaf_row[leaf_row >= 0], weights=w[leaf_row >= 0], minlength=len(pid))
                        if (leaf_row >= 0).any()
                        else np.zeros(len(pid))
                    ),
                )
            )
    return dict(pid=pid, pdg=pdg, parent=parent, is_leaf=is_leaf, primary=primary, cid=cid, w=w, dropped=dropped, leaf_row=leaf_row, surr=out)


def describe(particles, ctx, s):
    pid, pdg, parent, is_leaf = ctx["pid"], ctx["pdg"], ctx["parent"], ctx["is_leaf"]
    row_of = _row_lookup(pid)
    E = np.asarray(ak.to_numpy(particles["energy"]))
    px = np.asarray(ak.to_numpy(particles["px"]))
    py = np.asarray(ak.to_numpy(particles["py"]))
    pz = np.asarray(ak.to_numpy(particles["pz"]))
    pt = np.hypot(px, py)
    eta = np.arcsinh(np.where(pt == 0, 0, pz / np.maximum(pt, 1e-9)))
    vprim = np.asarray(ak.to_numpy(particles["vertex_primary"]))
    dep_leaf = s["dep_per_leaf"]

    r = s["row"]
    lines = []
    hs = "hard scatter" if int(vprim[r]) == 1 else "pileup"
    lines.append(f"{_name(pdg[r])} (pdg {int(pdg[r])}, id {int(pid[r])}), {hs}: E={E[r]:.1f} GeV, pT={pt[r]:.1f} GeV, η={eta[r]:.2f}")
    lines.append(
        f"  recovered deposit: {s['dep']:.1f} GeV calibrated (ECAL {s['dep_ecal']:.1f} / HCAL {s['dep_hcal']:.1f}) "
        f"on {s['n_contrib']} contribution(s); deposit/E = {s['dep'] / max(E[r], 1e-9):.2f}"
    )

    # record daughters of the surrogate: are they generator-final? do they deposit anything?
    kids = [c for c in range(len(pid)) if int(parent[c]) == int(pid[r])]
    leaf_kids = [c for c in kids if is_leaf[c]]
    g4_kids = sorted((c for c in kids if not is_leaf[c]), key=lambda c: -E[c])
    if leaf_kids:
        kid_lines = [
            f"{_name(pdg[c])}(id {int(pid[c])}, E={E[c]:.1f}, deposit {dep_leaf[c] if c < len(dep_leaf) else 0.0:.2f} GeV)" for c in leaf_kids
        ]
        lines.append(f"  generator-assigned decay products (final at record level): {'; '.join(kid_lines)}")
    else:
        lines.append("  generator-assigned decay products: none in the release record")
    if g4_kids:
        top = "; ".join(f"{_name(pdg[c])}(E={E[c]:.1f})" for c in g4_kids[:6])
        lines.append(f"  Geant4's actual children ({len(g4_kids)}, ΣE={E[g4_kids].sum():.1f} GeV, top: {top})")

    # walk of the largest contribution that walks to this surrogate
    cid, w = ctx["cid"], ctx["w"]
    i_max = s["contrib_rows"][np.argmax(w[s["contrib_rows"]])]
    chain, cur, seen = [], int(cid[i_max]), set()
    while cur != -1 and cur not in seen:
        seen.add(cur)
        row = row_of([cur])[0]
        if row < 0:
            break
        chain.append(row)
        if row == r:
            break
        cur = int(parent[row])
    bits = []
    for row in chain:
        lbl = "PRIMARY" if row == r else ("leaf" if is_leaf[row] else "G4 secondary")
        bits.append(f"{_name(pdg[row])}(id {int(pid[row])}, E={E[row]:.2f}, {lbl})")
    lines.append(f"  largest deposit ({w[i_max]:.1f} GeV) walks: " + "  <-  ".join(bits))

    kids_silent = all((dep_leaf[c] if c < len(dep_leaf) else 0.0) == 0 for c in kids)
    why = f"  why: {_name(pdg[r])} is a non-leaf generator primary; the release pre-assigned its decay"
    if leaf_kids and kids_silent:
        why += ", whose recorded daughters above deposit nothing (Geant4 re-ran the fate — the particle interacted/"
        why += "showered in the calorimeter first), so the shower energy sat on G4 secondaries that walk up to a "
        why += "non-leaf and was dropped. Promoting the surrogate recovers it on the object the detector actually saw."
    else:
        why += "; some record daughters did deposit — check the partial-guard handling."
    lines.append(why)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sample", choices=["pu0", "pu200"], default="pu0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-events", type=int, default=30)
    p.add_argument("--top", type=int, default=3, help="examples to show (largest recovered deposit)")
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1"))
    args = p.parse_args()

    sample = f"ttbar_{args.sample}"
    paths = [reader.shard_paths(args.data_dir, sample, obj)[args.shard] for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]]
    print(f"sample={sample} shard={args.shard} scanning first {args.num_events} events")

    found = []  # (dep, ev_index, event_id, ctx, surrogate)
    for iev, ev in enumerate(reader.iter_event_records(*paths, batch_size=5)):
        if iev >= args.num_events:
            break
        ctx = event_surrogates(ev)
        for s in ctx["surr"]:
            found.append((s["dep"], iev, int(ev["event_id"]), ev, ctx, s))
        print(f"\r  event {iev} ({len(found)} surrogates so far)", end="", flush=True)
    print()
    found.sort(key=lambda x: -x[0])
    for rank, (dep, iev, eid, ev, ctx, s) in enumerate(found[: args.top], 1):
        print("=" * 100)
        print(f"EXAMPLE {rank}: {sample} shard {args.shard} event {eid}")
        print(describe(ev["particles"], ctx, s))
    print("=" * 100)


if __name__ == "__main__":
    main()
