#!/usr/bin/env python3
"""Inspect which detector inputs carry the assigned PF particle targets."""

import argparse
import collections
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
import numpy as np

from mlpf.data.cms.postprocessing_truth import delta_r, kinematics

KINDS = ("GSF", "Track", "ECAL", "HCAL", "HGCal")
KIND_LABELS = ("GSF tracks", "Tracks", "ECAL clusters", "HCAL clusters", "HGCal tracksters")
COLORS = ("#8e5ea2", "#2878b5", "#f5a623", "#bc5346", "#319c89")
SPECIES = (
    (11, r"$e^\pm$"),
    (13, r"$\mu^\pm$"),
    (22, r"$\gamma$"),
    (211, r"$\pi^\pm$"),
    (321, r"$K^\pm$"),
    (2212, r"$p/\bar p$"),
    (130, r"$K_L^0$"),
    (2112, r"$n/\bar n$"),
    (310, r"$K_S^0$"),
    (0, "Other hadrons"),
)


def assignment_rows(events):
    rows = []
    for event in events:
        gen = {p.truth_id: p for p in event["gen"]}
        contributions = collections.defaultdict(list)
        for parent in event["provenance"]:
            contributions[int(parent["row"])].append(parent)
        for row in np.flatnonzero(event["target_id"]):
            parents = contributions[int(row)]
            dominant = max(parents, key=lambda c: (float(c["fraction"]) * event["truth"][int(c["truth"])].p4[3], -int(c["truth"])))
            truth = int(dominant["truth"])
            particle = event["truth"][truth]
            element = event["elements"][row]
            pt, eta, phi, energy = kinematics(event["target_p4"][row])
            ancestors = event["gen_ancestors"][truth]
            ancestor_pid = gen[ancestors[0]].pdg_id if len(ancestors) == 1 and ancestors[0] in gen else None
            rows.append(
                dict(
                    event_id=list(event["event_id"]),
                    row=int(row),
                    collection=element.collection,
                    input_index=element.index,
                    pid=int(event["ytarget"]["pid"][row]),
                    charge=float(event["ytarget"]["charge"][row]),
                    pt=pt,
                    eta=eta,
                    phi=phi,
                    energy=energy,
                    input_charge=element.charge,
                    input_energy=float(element.p4[3]),
                    input_pt=kinematics(element.p4)[0],
                    dr=delta_r(element.p4, event["target_p4"][row]),
                    merged=len(parents) > 1,
                    mixed_label=bool(event["mixed"][row]),
                    n_parents=len(parents),
                    dominant_truth_id=particle.truth_id,
                    dominant_gen_pid=ancestor_pid,
                    split=any(int(p["truth"]) in event["split_truth"] for p in parents),
                )
            )
    return rows


def species_id(pid):
    return pid if pid in {pid for pid, _ in SPECIES[:-1]} else 0


def matrix(rows):
    counts = np.zeros((len(SPECIES), len(KINDS)), dtype=int)
    for row in rows:
        counts[[pid for pid, _ in SPECIES].index(species_id(row["pid"])), KINDS.index(row["collection"])] += 1
    return counts


def heatmaps(rows, output, title, name):
    counts = matrix(rows)
    totals = counts.sum(axis=1)
    active = totals > 0
    counts, totals = counts[active], totals[active]
    labels = [f"{label}  (N={total:,})" for (_, label), total in zip([s for s, keep in zip(SPECIES, active) if keep], totals)]
    fractions = counts / np.maximum(totals[:, None], 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    images = [
        axes[0].imshow(np.ma.masked_where(counts == 0, counts), cmap="Blues", norm=LogNorm(vmin=1, vmax=max(2, counts.max())), aspect="auto"),
        axes[1].imshow(fractions, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto"),
    ]
    for index, ax in enumerate(axes):
        for y in range(len(totals)):
            for x in range(len(KINDS)):
                text = f"{counts[y, x]:,}" if index == 0 else f"{100 * fractions[y, x]:.1f}%"
                dark = counts[y, x] > np.sqrt(counts.max()) if index == 0 else fractions[y, x] > 0.5
                ax.text(x, y, text, ha="center", va="center", color="white" if dark else "#222222", fontsize=10)
        ax.set(
            xticks=np.arange(len(KINDS)),
            xticklabels=KIND_LABELS,
            yticks=np.arange(len(totals)),
            yticklabels=labels,
            title="Assigned target counts" if index == 0 else "Destination fractions within each species",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.set_xticks(np.arange(-0.5, len(KINDS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(totals), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(images[0], ax=axes[0], label="Targets (log color scale)", shrink=0.8)
    fig.colorbar(images[1], ax=axes[1], label="Fraction", format=PercentFormatter(1), shrink=0.8)
    fig.suptitle(title + f"\n{len(rows):,} output targets; merged rows use the largest-energy contributor’s PID")
    fig.savefig(output / name, dpi=170)
    plt.close(fig)


def pt_plots(rows, output, title):
    groups = [
        ("Electrons", lambda r: r["pid"] == 11),
        ("Muons", lambda r: r["pid"] == 13),
        ("Photons", lambda r: r["pid"] == 22),
        ("Charged hadrons", lambda r: r["pid"] not in (11, 13, 22) and r["charge"] != 0),
        ("Neutral hadrons", lambda r: r["pid"] not in (11, 13, 22) and r["charge"] == 0),
        ("All targets", lambda r: True),
    ]
    bins = np.array([0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100, 300])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (name, select) in zip(axes.flat, groups):
        selected = [r for r in rows if select(r) and bins[0] <= r["pt"] < bins[-1]]
        counts = np.array([np.histogram([r["pt"] for r in selected if r["collection"] == kind], bins=bins)[0] for kind in KINDS])
        fractions = counts / np.maximum(counts.sum(axis=0), 1)
        bottom = np.zeros(len(bins) - 1)
        for color, label, values in zip(COLORS, KIND_LABELS, fractions):
            ax.bar(bins[:-1], values, width=np.diff(bins), bottom=bottom, align="edge", color=color, label=label, edgecolor="white", linewidth=0.4)
            bottom += values
        ax.set(
            xscale="log",
            xlim=(bins[0], bins[-1]),
            ylim=(0, 1),
            xlabel="Assigned target pT [GeV]",
            ylabel="Fraction in pT bin",
            title=f"{name}: N={len(selected):,}",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5)
    fig.suptitle(title + "\nAssignment destinations versus target pT; 0.1 ≤ pT < 300 GeV, empty bins left blank")
    fig.savefig(output / "assignment_vs_pt.png", dpi=170)
    plt.close(fig)


def quality_plots(rows, output, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(KINDS))
    totals = np.array([sum(r["collection"] == kind for r in rows) for kind in KINDS])
    bottom = np.zeros(len(KINDS))
    for values, label, color in [
        (lambda r: not r["merged"], "One original truth root", "#2878b5"),
        (lambda r: r["merged"] and not r["mixed_label"], "Merged, same labels", "#f5a623"),
        (lambda r: r["mixed_label"], "Merged, mixed labels", "#bc5346"),
    ]:
        counts = np.array([sum(values(r) for r in rows if r["collection"] == kind) for kind in KINDS])
        fractions = counts / np.maximum(totals, 1)
        axes[0].bar(x, fractions, bottom=bottom, label=label, color=color)
        bottom += fractions
    axes[0].set(ylabel="Fraction of assigned targets", title="Merge composition", ylim=(0, 1.12))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for i, total in enumerate(totals):
        axes[0].text(i, 1.02, f"N={total:,}", ha="center", fontsize=9)
    axes[0].legend(fontsize=8)
    tails = [sum(r["collection"] == kind and r["dr"] > 0.1 for r in rows) for kind in KINDS]
    axes[1].bar(x, np.array(tails) / np.maximum(totals, 1), color=COLORS)
    for i, (tail, total) in enumerate(zip(tails, totals)):
        axes[1].text(i, tail / max(total, 1) + 0.025, f"{tail}/{total}", ha="center", fontsize=9)
    axes[1].set(ylim=(0, 1), ylabel="Fraction of assigned targets", title="Angular separation dR > 0.1")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    electrons = [r for r in rows if r["pid"] == 11]
    bottom = np.zeros(len(KINDS))
    for select, label, color in [
        (lambda r: abs(r["dominant_gen_pid"] or 0) == 11, "GEN electron ancestor", "#2878b5"),
        (lambda r: r["dominant_gen_pid"] == 22, "GEN photon ancestor", "#f5a623"),
        (lambda r: abs(r["dominant_gen_pid"] or 0) not in (11, 22), "GEN hadron / other ancestor", "#319c89"),
    ]:
        counts = np.array([sum(select(r) and r["collection"] == kind for r in electrons) for kind in KINDS])
        axes[2].bar(x, counts, bottom=bottom, label=label, color=color)
        bottom += counts
    axes[2].set(ylabel="Electron-labelled targets", title="Electron origins (dominant parent)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, KINDS)
    fig.suptitle(title + "\nProvenance and assignment diagnostics; full-event targets")
    fig.savefig(output / "assignment_diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="PU0 ttbar")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pickle.loads(args.events.read_bytes())
    rows = assignment_rows(events)
    selected = [r for r in rows if r["pt"] > 0.5 and abs(r["eta"]) < 3]
    title = f"{args.label}: {len(events)} events, inclusive truth graph"
    heatmaps(rows, args.output, title, "target_to_element.png")
    heatmaps(selected, args.output, title + "; target pT > 0.5 GeV, |eta| < 3", "target_to_element_fiducial.png")
    pt_plots(rows, args.output, title)
    quality_plots(rows, args.output, title)
    counts = matrix(rows)
    species = {str(pid): dict(zip(KINDS, map(int, count))) for (pid, _), count in zip(SPECIES, counts)}
    neutral_tracks = [r for r in rows if r["collection"] in ("Track", "GSF") and r["charge"] == 0]
    electron_origins = {
        kind: dict(collections.Counter(str(r["dominant_gen_pid"]) for r in rows if r["pid"] == 11 and r["collection"] == kind)) for kind in KINDS
    }
    summary = dict(
        events=len(events),
        targets=len(rows),
        target_species_to_elements=species,
        elements=dict(collections.Counter(r["collection"] for r in rows)),
        fiducial_targets=len(selected),
        merged=dict(collections.Counter(r["collection"] for r in rows if r["merged"])),
        mixed_label_merged=dict(collections.Counter(r["collection"] for r in rows if r["mixed_label"])),
        electron_origins=electron_origins,
        neutral_targets_on_tracks=neutral_tracks,
    )
    (args.output / "assignments.json").write_text(json.dumps(dict(summary=summary, assignments=rows), indent=2, allow_nan=False) + "\n")
    lines = [
        f"{args.label}: detector destinations of {len(rows)} assigned targets in {len(events)} events",
        "",
        "Counts refer to derived output rows, after splitting/merging; row fractions exclude unmatched truth roots.",
        "Merged rows use the highest-energy contributor's PID and charge. Electron ancestry refers to that contributor.",
        "This is the inclusive event sample, including underlying event.",
        "",
        "Species | GSF | Track | ECAL | HCAL | HGCal | total",
    ]
    for (pid, _), count in zip(SPECIES, counts):
        lines.append(f"{pid if pid else 'other hadrons'} | " + " | ".join(map(str, count)) + f" | {int(count.sum())}")
    photon_electrons = sum(r["pid"] == 11 and r["dominant_gen_pid"] == 22 for r in rows)
    electrons = sum(r["pid"] == 11 for r in rows)
    lines.extend(
        [
            "",
            f"Electron-labelled targets with a GEN photon ancestor: {photon_electrons}/{electrons}",
            f"Neutral targets on charged tracks: {len(neutral_tracks)}; PID counts {dict(collections.Counter(r['pid'] for r in neutral_tracks))}",
            f"Among these, merged rows: {sum(r['merged'] for r in neutral_tracks)}",
            "The track association uses branch hit purity and has no requirement that its target be charged.",
            f"Merged output rows by detector: {summary['merged']}",
            f"Mixed-label merged rows by detector: {summary['mixed_label_merged']}",
            f"Fiducial targets (pT > 0.5 GeV, |eta| < 3): {len(selected)}",
            "The pT panels include only 0.1 <= target pT < 300 GeV; each panel is normalized within each pT bin.",
            "",
            "The JSON contains each event/row, detector collection/index, target PID/charge, kinematics, dR, dominant truth ID and ancestry.",
        ]
    )
    (args.output / "report.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
