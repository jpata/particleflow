#!/usr/bin/env python3
"""Before/after impact of adding surrogate ancestors to the ColliderML truth targets.

Context (mlpf/data/colliderml/truth.py, compute_gen_tables): calo contributions whose parent
chain never reaches a leaf primary are dropped. The chain that fails is a generator-decayed
K0S/Lambda/Sigma/Xi (a non-leaf primary) that Geant4 let interact before its preassigned decay,
so its generator daughters are silent — key4hep's `find_surrogate_ancestors` case.

"After" simulates the fix as key4hep defines it: the deepest primary on a dropped chain becomes a
leaf (a target candidate), unless one of its leaf descendants registered anything (adding it would
double count; "partial") or it is the ancestor of another candidate ("nested"). compute_gen_tables
is then rerun unchanged, so visibility, track links and the gen reference follow the converter's
own rules. Per sample the report shows what we gain (attributed energy, gen-MET closure against
the neutrino MET) and what it costs (which targets are added, how measurable they are).

Usage (from repo root, .venv; pu200 only via sbatch — scripts/flatiron/colliderml_surrogate_impact.sh):
  python3 scripts/colliderml_surrogate_impact.py --sample pu0 --workers 10 --out /tmp/surr/pu0.json
  # defaults: pu0 = shards 0-9 x 30 events, pu200 = shards 0-9 x 5 events
"""

import argparse
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import awkward as ak
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from mlpf.data.colliderml import reader, truth
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, NEUTRINO_PDGS, _redirect_ids, calibration_factors, compute_gen_tables, walk_to_leaf_many
from mlpf.data.target_building import map_pdgid_to_candid

DEFAULTS = {"pu0": dict(num_shards=10, events_per_shard=30), "pu200": dict(num_shards=10, events_per_shard=5)}

PDG_NAMES = {310: "K0S", 3122: "Λ", 3222: "Σ⁺", 3112: "Σ⁻", 3312: "Ξ⁻", 3322: "Ξ⁰", 3334: "Ω⁻"}
CLASS_NAMES = {130: "neutral had", 211: "charged had", 22: "γ", 11: "e", 13: "μ"}


def _row_lookup(pid):
    """Vectorized particle id -> row index (-1 if absent)."""
    order = np.argsort(pid, kind="stable")
    pid_s = pid[order]

    def row_of(v):
        v = np.asarray(v)
        if len(pid_s) == 0 or v.size == 0:
            return np.full(v.shape, -1, np.int64)
        pos = np.clip(np.searchsorted(pid_s, v), 0, len(pid_s) - 1)
        return np.where(pid_s[pos] == v, order[pos], -1)

    return row_of


def walk_to_primary_many(particle_ids, pid, primary, parent, max_depth=256):
    """Like walk_to_leaf_many, but stops at the first row with primary=True (leaf or not).

    Returns the surrogate row per id (the ancestor the detector actually saw), -1 if the
    chain dies before reaching any primary (genuinely orphaned contribution)."""
    ids = np.asarray(particle_ids)
    if len(ids) == 0:
        return np.empty(0, np.int64)
    row_of = _row_lookup(pid)
    cur = row_of(ids)
    res = np.full(len(ids), -1, np.int64)
    alive = cur >= 0
    for _ in range(max_depth):
        if not alive.any():
            break
        idx_alive = np.nonzero(alive)[0]
        cur_alive = cur[alive]
        prim_now = primary[cur_alive]
        res[idx_alive[prim_now]] = cur_alive[prim_now]
        cont = idx_alive[~prim_now]
        if len(cont) == 0:
            break
        cur[cont] = row_of(parent[cur_alive[~prim_now]])
        alive[cont[cur[cont] < 0]] = False
    return res


def _ancestors(rows, parent, row_of, max_depth=256):
    """Set of all ancestor rows of `rows` (excluding the rows themselves)."""
    anc = set()
    cur = row_of(parent[np.asarray(list(rows), np.int64)]) if len(rows) else np.empty(0, np.int64)
    for _ in range(max_depth):
        cur = np.unique(cur[cur >= 0])
        cur = cur[~np.isin(cur, list(anc))] if anc else cur
        if len(cur) == 0:
            break
        anc.update(cur.tolist())
        cur = row_of(parent[cur])
    return anc


@contextmanager
def surrogates_as_leaves(rows):
    """Make compute_gen_tables treat `rows` as leaf primaries (the simulated fix)."""
    orig = truth._build_parent_maps

    def patched(particles_ev):
        pid, pdg, parent, is_leaf, redirect = orig(particles_ev)
        is_leaf[rows] = True
        return pid, pdg, parent, is_leaf, redirect

    truth._build_parent_maps = patched
    try:
        yield
    finally:
        truth._build_parent_maps = orig


def _added_rows(after, before):
    """Indices of `after` target rows absent from `before` (multiset diff), and #before rows lost."""

    def keys(t):
        return list(zip(t["PDG"].tolist(), t["energy"].tolist(), t["pt"].tolist(), t["eta"].tolist()))

    remaining = Counter(keys(before))
    idx = []
    for i, k in enumerate(keys(after)):
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            idx.append(i)
    return np.asarray(idx, np.int64), sum(remaining.values())


def _met_closure(genref, nu_px, nu_py):
    """|MET(genref) - MET(neutrinos)| = |sum genref pT + sum neutrino pT| [GeV]."""
    sx = np.sum(genref["pt"] * np.cos(genref["phi"])) + nu_px
    sy = np.sum(genref["pt"] * np.sin(genref["phi"])) + nu_py
    return float(np.hypot(sx, sy))


def _missing_pt(ref, rows, pdg, E, pt, px, py):
    """|sum pT| of the final-state `rows` absent from the gen reference (rows matched on PDG, E, pT at float32,
    as compute_gen_tables stores them). Equals the closure up to the record's own pT imbalance (~0.1 GeV)."""
    remaining = Counter(zip(ref["PDG"].tolist(), ref["energy"].tolist(), ref["pt"].tolist()))
    sx = sy = 0.0
    for i in rows:
        k = (float(np.float32(pdg[i])), float(np.float32(E[i])), float(np.float32(pt[i])))
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            sx += px[i]
            sy += py[i]
    return float(np.hypot(sx, sy))


def analyze_event(ev):
    particles, calo, tracks, tracker = ev["particles"], ev["calo_hits"], ev["tracks"], ev["tracker_hits"]
    pid, pdg, parent, is_leaf, redirect = truth._build_parent_maps(particles)
    row_of = _row_lookup(pid)
    primary = np.asarray(ak.to_numpy(particles["primary"])).astype(bool)
    px = np.asarray(ak.to_numpy(particles["px"]))
    py = np.asarray(ak.to_numpy(particles["py"]))
    pz = np.asarray(ak.to_numpy(particles["pz"]))
    E = np.asarray(ak.to_numpy(particles["energy"]))
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-9))
    vprim = np.asarray(ak.to_numpy(particles["vertex_primary"]))

    # calibrated per-contribution energies, as in compute_gen_tables
    cid = _redirect_ids(np.asarray(ak.to_numpy(ak.flatten(calo["contrib_particle_ids"], axis=None))).ravel(), redirect)
    ce = np.asarray(ak.to_numpy(ak.flatten(calo["contrib_energies"], axis=None))).ravel()
    nctr = np.asarray(ak.to_numpy(ak.num(calo["contrib_particle_ids"]))).ravel()
    det = np.repeat(np.asarray(ak.to_numpy(calo["detector"])), nctr)
    w = (ce.astype(np.float32) * calibration_factors(det, DEFAULT_CALIBRATION)).astype(np.float64)
    e_total = float(w.sum())

    leaf_row = walk_to_leaf_many(cid, pid, is_leaf, parent)
    dropped = leaf_row == -1
    th_pid = _redirect_ids(np.asarray(ak.to_numpy(tracker["particle_id"])).astype(np.int64), redirect)
    th_leaf = walk_to_leaf_many(th_pid, pid, is_leaf, parent)

    # surrogate candidates: deepest primary on each dropped chain, with its dropped deposit
    clean, dep_of, n_partial, n_nested = [], {}, 0, 0
    if dropped.any():
        uniq, inv = np.unique(cid[dropped], return_inverse=True)
        row_per_contrib = walk_to_primary_many(uniq, pid, primary, parent)[inv]
        ok = row_per_contrib >= 0
        cand, cinv = np.unique(row_per_contrib[ok], return_inverse=True)
        cand_dep = np.bincount(cinv, weights=w[dropped][ok], minlength=len(cand))
        # key4hep guards: no registered leaf descendant (else double counting), deepest of nested
        measured = np.unique(np.concatenate([leaf_row[leaf_row >= 0], th_leaf[th_leaf >= 0]]))
        partial = np.isin(cand, list(_ancestors(measured, parent, row_of)))
        rest = cand[~partial]
        nested = np.isin(rest, list(_ancestors(rest, parent, row_of)))
        clean = rest[~nested].tolist()
        n_partial, n_nested = int(partial.sum()), int(nested.sum())
        dep_of = dict(zip(cand.tolist(), cand_dep.tolist()))

    # neutrino MET: the truth the gen reference should close against
    nu = is_leaf & np.isin(np.abs(pdg), list(NEUTRINO_PDGS))
    nu_px, nu_py = float(px[nu].sum()), float(py[nu].sum())

    gen_b, _, _, ref_b = compute_gen_tables(particles, calo, tracks, DEFAULT_CALIBRATION, tracker)
    if clean:
        with surrogates_as_leaves(clean):
            gen_a, _, _, ref_a = compute_gen_tables(particles, calo, tracks, DEFAULT_CALIBRATION, tracker)
    else:
        gen_a, ref_a = gen_b, ref_b
    added, lost = _added_rows(gen_a, gen_b)

    # final-state set without / with surrogates; a surrogate replaces its (silent) generator descendants
    final_b = np.nonzero(is_leaf & ~nu)[0]
    in_surr = np.zeros(len(pid), bool)
    if clean:
        cur = row_of(parent[final_b])
        for _ in range(256):
            hit = cur >= 0
            if not hit.any():
                break
            in_surr[final_b[hit & np.isin(cur, clean)]] = True
            cur = np.where(hit, row_of(parent[np.where(hit, cur, 0)]), -1)
    final_a = np.concatenate([final_b[~in_surr[final_b]], np.asarray(clean, np.int64)])

    e_dropped = float(w[dropped].sum())
    e_recovered = float(sum(dep_of[r] for r in clean))
    return dict(
        affected=bool(dropped.any()),
        e_total=e_total,
        e_unattr_before=e_dropped,
        e_unattr_after=e_dropped - e_recovered,
        e_targets_before=float(np.sum(gen_b["gp_to_cluster"])),
        e_targets_after=float(np.sum(gen_a["gp_to_cluster"])),
        n_targets_before=len(gen_b["pt"]),
        n_targets_after=len(gen_a["pt"]),
        n_partial=n_partial,
        n_nested=n_nested,
        n_lost=int(lost),
        closure_before=_met_closure(ref_b, nu_px, nu_py),
        closure_after=_met_closure(ref_a, nu_px, nu_py),
        met_nu=float(np.hypot(nu_px, nu_py)),
        missing_before=_missing_pt(ref_b, final_b, pdg, E, pt, px, py),
        missing_after=_missing_pt(ref_a, final_a, pdg, E, pt, px, py),
        th_orphan=float(np.mean(th_leaf == -1)) if len(th_leaf) else 0.0,
        surr=[dict(pdg=abs(int(pdg[r])), dep=dep_of[r], pt=float(pt[r]), abs_eta=float(abs(eta[r])), ispu=int(vprim[r] != 1)) for r in clean],
        added=[
            dict(
                cls=int(map_pdgid_to_candid(int(abs(gen_a["PDG"][i])), float(gen_a["charge"][i]))),
                no_track=bool(gen_a["charge"][i] != 0 and gen_a["gp_to_track"][i] == 0),
                dep_frac=float(gen_a["gp_to_cluster"][i] / max(gen_a["energy"][i], 1e-9)),
            )
            for i in added
        ],
    )


def run_shard(job):
    pu_label, shard, num_events, data_dir = job
    sample = f"ttbar_{pu_label}"
    paths = [reader.shard_paths(data_dir, sample, obj)[shard] for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]]
    rows = []
    for iev, ev in enumerate(reader.iter_event_records(*paths, batch_size=5)):
        if iev >= num_events:
            break
        rows.append(dict(shard=shard, event_id=int(ev["event_id"]), **analyze_event(ev)))
    print(f"{sample} shard {shard}: {len(rows)} events", flush=True)
    return rows


def percentiles(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "mean": float(np.mean(x))}


def summarize(rows):
    aff = [r for r in rows if r["affected"]]
    surr = [s for r in rows for s in r["surr"]]
    added = [a for r in rows for a in r["added"]]
    species = defaultdict(float)
    for s in surr:
        species[s["pdg"]] += s["dep"]
    tot_e = sum(r["e_total"] for r in rows)
    return {
        "events": len(rows),
        "events_affected": len(aff),
        "surrogates": len(surr),
        "excluded_partial": sum(r["n_partial"] for r in rows),
        "excluded_nested": sum(r["n_nested"] for r in rows),
        "targets_lost_sanity": sum(r["n_lost"] for r in rows),
        "unattr_rel_before": percentiles([r["e_unattr_before"] / r["e_total"] for r in rows if r["e_total"]]),
        "unattr_rel_after": percentiles([r["e_unattr_after"] / r["e_total"] for r in rows if r["e_total"]]),
        "targets_E_share_before": sum(r["e_targets_before"] for r in rows) / tot_e if tot_e else 0.0,
        "targets_E_share_after": sum(r["e_targets_after"] for r in rows) / tot_e if tot_e else 0.0,
        "targets_per_event_before": float(np.mean([r["n_targets_before"] for r in rows])) if rows else 0.0,
        "targets_per_event_after": float(np.mean([r["n_targets_after"] for r in rows])) if rows else 0.0,
        "added_targets": len(added),
        "added_by_class": dict(Counter(CLASS_NAMES.get(a["cls"], str(a["cls"])) for a in added).most_common()),
        "added_charged_no_track": sum(a["no_track"] for a in added),
        "added_dep_frac": percentiles([a["dep_frac"] for a in added]),
        "closure_affected_before_GeV": percentiles([r["closure_before"] for r in aff]),
        "closure_affected_after_GeV": percentiles([r["closure_after"] for r in aff]),
        "met_nu_affected_GeV": percentiles([r["met_nu"] for r in aff]),
        "missing_pt_affected_before_GeV": percentiles([r["missing_before"] for r in aff]),
        "missing_pt_affected_after_GeV": percentiles([r["missing_after"] for r in aff]),
        "surrogate_ispu_frac": float(np.mean([s["ispu"] for s in surr])) if surr else 0.0,
        "surrogate_pt_GeV": percentiles([s["pt"] for s in surr]),
        "surrogate_abs_eta": percentiles([s["abs_eta"] for s in surr]),
        "tracker_hit_orphan_frac": percentiles([r["th_orphan"] for r in rows]),
        "species_recovered_GeV": dict(sorted(species.items(), key=lambda kv: -kv[1])),
    }


# Row labels shared by the results table and the glossary (same order).
LABELS = [
    "events affected",
    "surrogates added as candidates",
    "unattributed calo energy / event [med / p90]",
    "calo energy on targets",
    "targets / event",
    "added targets by class",
    "added targets: E_deposit / E_truth [med / p90]",
    "gen-MET closure, affected events [GeV, med / p90]",
    "surrogate origin",
    "recovered energy by species [GeV]",
]

GLOSSARY = {
    "events affected": "events with any calibrated calo deposit whose chain reaches no leaf (dropped today)",
    "surrogates added as candidates": "deepest primaries on dropped chains that pass the key4hep guards; excluded = a leaf descendant registered "
    "(partial: would double count) or ancestor of another candidate (nested)",
    "unattributed calo energy / event [med / p90]": "share of the event's calibrated deposit attributed to no truth particle, today → with surrogates"
    " (the remainder after is genuinely orphaned)",
    "calo energy on targets": "share of all calibrated deposit carried by the (visible) target set, summed over events",
    "targets / event": "mean target count today → with surrogates",
    "added targets by class": "the new targets in MLPF classes; 'no track' = charged with zero track-hit share, so they can never get a track "
    "and are relabelled neutral",
    "added targets: E_deposit / E_truth [med / p90]": "per added target, E_deposit = calibrated calo energy attributed to it (its own and "
    "its secondaries' deposits), E_truth = its true energy; ~1 = contained and well measured, small = mostly leaked (hard to regress), "
    "can exceed 1 (one calibration constant per region). The visibility cut requires E_deposit / E_truth > 0.10 for neutrals; "
    "surrogates failing it enter only the gen reference (count shown)",
    "gen-MET closure, affected events [GeV, med / p90]": "|MET(gen reference) − MET(neutrinos)|, today → with surrogates; the neutrino MET "
    "median is shown for scale",
    "surrogate origin": "pileup share, median pT and |η| of all accepted surrogates — where the change lands",
    "recovered energy by species [GeV]": "confirms the mechanism — expect K0S/Λ/Σ/Ξ",
}


def _pct(x):
    return f"{100 * x:.2f}%"


def _arrow(a, b):
    return f"{a} → {b}"


def table_rows(s):
    """Label -> value-string rows for one sample summary, in LABELS order."""

    def mp(p, f):
        return f"{f(p['median'])} / {f(p['p90'])}" if p else "–"

    def num(x):
        return f"{x:.1f}"

    species = " > ".join(f"{PDG_NAMES.get(int(k), k)} {v:.0f}" for k, v in s["species_recovered_GeV"].items())
    classes = ", ".join(f"{k} {v}" for k, v in s["added_by_class"].items()) or "none"
    n_meas_only = s["surrogates"] - s["added_targets"]
    vals = {
        "events affected": f"{s['events_affected']}/{s['events']}",
        "surrogates added as candidates": f"{s['surrogates']} (excluded: {s['excluded_partial']} partial, {s['excluded_nested']} nested)",
        "unattributed calo energy / event [med / p90]": _arrow(mp(s["unattr_rel_before"], _pct), mp(s["unattr_rel_after"], _pct)),
        "calo energy on targets": _arrow(_pct(s["targets_E_share_before"]), _pct(s["targets_E_share_after"])),
        "targets / event": _arrow(num(s["targets_per_event_before"]), num(s["targets_per_event_after"])),
        "added targets by class": f"{classes} ({s['added_charged_no_track']} charged w/o track)",
        "added targets: E_deposit / E_truth [med / p90]": f"{mp(s['added_dep_frac'], lambda x: f'{x:.2f}')}; {n_meas_only} surrogates gen-ref only",
        "gen-MET closure, affected events [GeV, med / p90]": _arrow(
            mp(s["closure_affected_before_GeV"], num), mp(s["closure_affected_after_GeV"], num)
        )
        + f" (MET_ν med {num(s['met_nu_affected_GeV'].get('median', 0.0))})",
        "surrogate origin": f"PU {_pct(s['surrogate_ispu_frac'])}, pT med {num(s['surrogate_pt_GeV'].get('median', 0.0))} GeV, "
        f"|η| med {num(s['surrogate_abs_eta'].get('median', 0.0))}",
        "recovered energy by species [GeV]": species,
    }
    return [(label, vals[label]) for label in LABELS]


def _cell(x):
    """Escape markdown-table-breaking pipes inside a cell (e.g. the |η| notation)."""
    return str(x).replace("|", "\\|")


def format_report(summaries, runs_header):
    """Markdown report: glossary, results table (one column per sample), sanity footnote."""
    columns = list(summaries.keys())
    rows_by_sample = {name: dict(table_rows(s)) for name, s in summaries.items()}
    g_lines = ["## Metrics", "", "| metric | meaning |", "|---|---|"] + [f"| {_cell(label)} | {_cell(GLOSSARY[label])} |" for label in LABELS]
    r_lines = ["## Results (today → with surrogates)", "", "| metric | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    r_lines += [f"| {_cell(label)} | " + " | ".join(_cell(rows_by_sample[name][label]) for name in columns) + " |" for label in LABELS]
    notes = [
        f"- {name}: tracker-hit orphan share (count-level) med {_pct(s['tracker_hit_orphan_frac'].get('median', 0.0))}; "
        f"targets lost by the fix (sanity, must be 0): {s['targets_lost_sanity']}"
        for name, s in summaries.items()
    ]
    return "\n".join([f"# surrogate-ancestor impact: {runs_header}", ""] + g_lines + [""] + r_lines + [""] + notes)


# Plot styling: light chart surface, recessive grid/axes, two validated categorical slots.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]
TRACK_ETA_MAX = 3.0  # Release-1 track acceptance


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _before_after_scatter(per_events, path, plt, key, better, title):
    """Per affected event: r[key_before] (x) vs r[key_after] (y), one panel per sample."""
    samples = list(per_events)
    aff = {name: [r for r in rows if r["affected"]] for name, rows in per_events.items()}
    if any(f"{key}_before" not in r for rows in aff.values() for r in rows):
        return False  # results written before this quantity existed
    vals = np.array([r[k] for rows in aff.values() for r in rows for k in (f"{key}_before", f"{key}_after")])
    if not len(vals):
        return False
    lo, hi = max(vals[vals > 0].min() if (vals > 0).any() else 0.1, 0.1) / 1.5, vals.max() * 1.5
    fig, axes = plt.subplots(1, len(samples), figsize=(max(5 * len(samples), 6.5), 5), squeeze=False, facecolor=SURFACE)
    for ax, name in zip(axes[0], samples):
        b = np.array([r[f"{key}_before"] for r in aff[name]])
        a = np.array([r[f"{key}_after"] for r in aff[name]])
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, ls="--", zorder=1)
        ax.scatter(np.clip(b, lo, None), np.clip(a, lo, None), s=36, color=SERIES[0], edgecolor=SURFACE, lw=0.8, zorder=2)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="without surrogates [GeV]")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{name}: {len(b)} affected events", fontsize=11)
        if len(b):
            summary = f"median {np.median(b):.1f} → {np.median(a):.1f} GeV\n{100 * np.mean(a < b):.0f}% of events {better}"
            ax.text(0.04, 0.96, summary, transform=ax.transAxes, va="top", fontsize=9, color=INK2)
        _style(ax)
    axes[0][0].set_ylabel("with surrogates [GeV]")
    fig.suptitle(f"{title} per event\nbelow the diagonal: surrogates help", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_closure(per_events, path, plt):
    return _before_after_scatter(per_events, path, plt, "closure", "closer to MET(ν)", "Gen-MET closure |MET(gen ref) − MET(ν)|")


def plot_missing(per_events, path, plt):
    title = "|Σ pT| of particles missing from the gen ref"
    return _before_after_scatter(per_events, path, plt, "missing", "with less missing pT", title)


def plot_unattributed(per_events, path, plt):
    """Histogram of today's per-event unattributed calo energy fraction, samples overlaid."""
    frac = {name: 100 * np.array([r["e_unattr_before"] / r["e_total"] for r in rows if r["e_total"]]) for name, rows in per_events.items()}
    pos = np.concatenate([f[f > 0] for f in frac.values()])
    if not len(pos):
        return False
    bins = np.logspace(np.log10(pos.min() * 0.9), np.log10(100), 31)
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    for color, (name, f) in zip(SERIES, frac.items()):
        n0 = int((f == 0).sum())
        label = f"{name}: {len(f)} events, median {np.median(f):.2f}%" + (f" ({n0} at 0%, not shown)" if n0 else "")
        ax.hist(f[f > 0], bins=bins, weights=np.full((f > 0).sum(), 1 / len(f)), histtype="step", lw=2, color=color, label=label)
    ax.set(xscale="log", yscale="log", xlabel="calibrated calo energy attributed to no truth particle [% of event]", ylabel="fraction of events")
    ax.set_title("Unattributed calo energy per event today (0% in every event with surrogates)", fontsize=11)
    _style(ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)  # headroom for the legend
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_surrogates(per_events, path, plt):
    """Accepted surrogates in |eta| vs pT, marker area ~ sqrt(recovered deposit), hard scatter vs pileup."""
    samples = list(per_events)
    surr = {name: [s for r in rows for s in r["surr"]] for name, rows in per_events.items()}
    deps = [s["dep"] for ss in surr.values() for s in ss]
    if not deps:
        return False
    dmax = max(deps)

    def size(dep):  # sqrt keeps 1 / 10 / 100 GeV distinguishable over a ~1000x range
        return 20 + 30 * np.sqrt(np.asarray(dep))

    fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 4.6), squeeze=False, sharey=True, sharex=True, facecolor=SURFACE)
    for ax, name in zip(axes[0], samples):
        ss = sorted(surr[name], key=lambda s: -s["dep"])  # big markers first, small ones stay visible
        for color, ispu, label in [(SERIES[0], 0, "hard scatter"), (SERIES[1], 1, "pileup")]:
            sel = [s for s in ss if s["ispu"] == ispu]
            if sel:
                ax.scatter(
                    [s["abs_eta"] for s in sel],
                    [s["pt"] for s in sel],
                    s=size([s["dep"] for s in sel]),
                    color=color,
                    alpha=0.75,
                    edgecolor=SURFACE,
                    lw=0.8,
                    label=f"{label} ({len(sel)})",
                )
        ax.axvline(TRACK_ETA_MAX, color=AXIS, lw=1, ls="--")
        ax.text(TRACK_ETA_MAX, 1.0, " track acceptance", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=MUTED)
        ax.set(yscale="log", xlabel="|η| of surrogate")
        ax.set_title(f"{name}: {len(ss)} surrogates", fontsize=11)
        _style(ax)
        leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
        for h in leg.legend_handles:
            h.set_sizes([36])
    axes[0][0].set_ylabel("surrogate pT [GeV]")
    ref = [d for d in (1, 10, 100) if d <= dmax] or [dmax]
    handles = [plt.scatter([], [], s=size(d), color=MUTED, edgecolor=SURFACE) for d in ref]
    fig.legend(handles, [f"{d:g} GeV recovered" for d in ref], loc="upper right", frameon=False, fontsize=8, labelcolor=INK2, ncol=len(ref))
    fig.suptitle("Where the surrogates are (marker area ∝ √ recovered calo deposit)", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def write_plots(per_events, out):
    """Write the three plots next to `out` (the JSON path); returns the PNG file names written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for suffix, fn in [("closure", plot_closure), ("missing", plot_missing), ("unattributed", plot_unattributed), ("surrogates", plot_surrogates)]:
        path = out.with_name(f"{out.stem}_{suffix}.png")
        if fn(per_events, path, plt):
            names.append(path.name)
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sample", nargs="+", choices=["pu0", "pu200"], default=["pu0"], help="one or both, e.g. --sample pu0 pu200")
    p.add_argument("--num-shards", type=int, default=None, help="use source shards 0..N-1 (default: 10)")
    p.add_argument("--events-per-shard", type=int, default=None, help="first N events of each shard (default: 30 pu0, 5 pu200)")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (one shard each)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1"),
        help="ColliderML Release-1 download root",
    )
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path (a .md table is written next to it)")
    args = p.parse_args()

    samples = list(dict.fromkeys(args.sample))  # dedupe, preserve order
    summaries, per_events, headers = {}, {}, []
    for pu in samples:
        nshard = args.num_shards if args.num_shards is not None else DEFAULTS[pu]["num_shards"]
        nev = args.events_per_shard if args.events_per_shard is not None else DEFAULTS[pu]["events_per_shard"]
        jobs = [(pu, shard, nev, args.data_dir) for shard in range(nshard)]
        with Pool(max(1, min(args.workers, len(jobs)))) as pool:
            rows = [r for shard_rows in pool.map(run_shard, jobs, chunksize=1) for r in shard_rows]
        summaries[pu], per_events[pu] = summarize(rows), rows
        headers.append(f"{pu}: shards 0-{nshard - 1} x {nev} ev = {len(rows)} ev")

    report = format_report(summaries, " | ".join(headers))
    print()
    print(report)
    print()
    print("Full JSON summary:")
    print(json.dumps(summaries, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "samples": {pu: {"summary": summaries[pu], "per_event": per_events[pu]} for pu in samples},
                },
                indent=2,
            )
        )
        pngs = write_plots(per_events, args.out)
        table_path = args.out.with_suffix(".md")
        table_path.write_text(report + "\n" + "".join(f"\n![{name}]({name})\n" for name in pngs))
        print(f"wrote {args.out} (json), {table_path} (table) and {len(pngs)} plots")


if __name__ == "__main__":
    main()
