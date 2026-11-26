#!/usr/bin/env python3
"""GEN-only fiducial particle, jet and MET response plots from assigned targets."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlpf.data.cms.gen_target_response import ResponseConfig, analyze_event, binned_stats, bootstrap_stats, kinematics

JET_PT_BINS = [20, 30, 50, 80, 120, 200, 350, 600, 1000, 2000]
JET_ETA_BINS = [0, 0.5, 1, 1.5, 2, 2.6, 2.8, 3.0, 3.2]
MET_BINS = [20, 30, 50, 80, 120, 200, 400, 800]
MET_COMPONENT_BINS = [0, 10, 20, 30, 50, 80, 120, 200, 400, 800]


def curve(ax, bins, field, label="New targets", color="#1764ab", log=False, percent=False, counts=False):
    selected = [b for b in bins if b["plot"] and b[field] is not None]
    if not selected:
        return
    x = np.array([np.sqrt(b["low"] * b["high"]) if log else (b["low"] + b["high"]) / 2 for b in selected])
    factor = 100 if percent else 1
    y = np.array([b[field] for b in selected]) * factor
    intervals = [b["intervals"].get(field, [b[field], b[field]]) for b in selected]
    error = np.maximum(0, np.array([[b[field] - ci[0], ci[1] - b[field]] for b, ci in zip(selected, intervals)]).T) * factor
    xerr = np.array([[v - b["low"], b["high"] - v] for v, b in zip(x, selected)]).T
    ax.errorbar(x, y, xerr=xerr, yerr=error, fmt="o-", markersize=4, capsize=3, label=label, color=color)
    if log:
        ax.set_xscale("log")
    if counts:
        for xx, yy, b in zip(x, y, selected):
            ax.annotate(f"N={b['n']}", (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)


def jet_plots(events, pairs, bins_pt, bins_eta, output, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, bins, log, xlabel in [(axes[0, 0], bins_pt, True, "GEN jet pT [GeV]"), (axes[0, 1], bins_eta, False, "|GEN jet eta|")]:
        curve(ax, bins, "median", log=log, counts=True)
        ax.axhline(1, color="0.6", linestyle=":")
        ax.set(xlabel=xlabel, ylabel="Median target / GEN jet pT")
    for ax, bins, log, xlabel in [(axes[1, 0], bins_pt, True, "GEN jet pT [GeV]"), (axes[1, 1], bins_eta, False, "|GEN jet eta|")]:
        curve(ax, bins, "relative_sigma68", log=log, percent=True)
        ax.set(xlabel=xlabel, ylabel="Relative resolution: sigma68 / |median| [%]")
    fig.suptitle(
        title + "\nMatched anti-kT R=0.4 jets; GEN pT ≥ 20 GeV, target pT ≥ 5 GeV, |eta| ≤ 3.2, dR < 0.2"
        "\nResolution: sigma68=(Q84−Q16)/2; error bars: event-bootstrap 68% intervals; bins require N≥20"
    )
    fig.savefig(output / "jet_response.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    gen = [j for e in events for j in e["gen_jets"]]
    target = [j for e in events for j in e["target_jets"] if j["pt"] >= 20]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    maximum = max([j["pt"] for j in gen + target], default=40)
    for collection, name, color in [(gen, "Fiducial GEN status 1", "black"), (target, "New targets", "#1764ab")]:
        axes[0, 0].hist([j["pt"] for j in collection], bins=np.geomspace(20, maximum * 1.01, 35), histtype="step", label=name, color=color)
        axes[0, 1].hist([j["eta"] for j in collection], bins=np.linspace(-3.2, 3.2, 38), histtype="step", label=name, color=color)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Jet pT [GeV]", ylabel="Jets (both spectra pT ≥ 20 GeV)")
    axes[0, 0].legend()
    axes[0, 1].set(xlabel="Jet eta", ylabel="Jets (both spectra pT ≥ 20 GeV)")
    ratios = [p["response"] for p in pairs]
    axes[1, 0].hist(ratios, bins=np.linspace(0, max(1.6, max(ratios, default=1) * 1.01), 65), histtype="step")
    axes[1, 0].axvline(1, color="0.6", linestyle=":")
    axes[1, 0].set(yscale="log", xlabel="Matched target / GEN jet pT", ylabel="Matched pairs")
    axes[1, 1].scatter([p["gen_pt"] for p in pairs], [p["target_pt"] for p in pairs], s=5, alpha=0.25)
    axes[1, 1].plot([5, maximum], [5, maximum], color="0.6", linestyle=":")
    axes[1, 1].set(xscale="log", yscale="log", xlabel="Matched GEN jet pT [GeV]", ylabel="Matched target jet pT [GeV]")
    fig.suptitle(title + "\nJets clustered independently from visible particles with pT > 0.5 GeV, |eta| ≤ 3.2 (including electrons/muons)")
    fig.savefig(output / "jet_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    efficiency = []
    for ax, collection, miss, label in [(axes[0], gen, False, "GEN matching efficiency"), (axes[1], target, True, "Unmatched target fraction")]:
        for low, high in zip(JET_PT_BINS[:-1], JET_PT_BINS[1:]):
            sample = [j for j in collection if low <= j["pt"] < high]
            n = len(sample)
            k = sum(j["matched"] != miss for j in sample)
            efficiency.append(dict(quantity=label, low=low, high=high, n=n, selected=k))
            if n:
                value = k / n
                center = (value + 0.5 / n) / (1 + 1 / n)
                half = np.sqrt(value * (1 - value) / n + 0.25 / n**2) / (1 + 1 / n)
                ax.errorbar(
                    np.sqrt(low * high),
                    value,
                    yerr=[[max(0, value - center + half)], [max(0, center + half - value)]],
                    fmt="o",
                    color="#1764ab",
                    capsize=3,
                )
        ax.set(xscale="log", ylim=(0, 1.05), xlabel=("GEN" if not miss else "Target") + " jet pT [GeV]", ylabel=label)
    fig.suptitle(title + "\nUnique angular matching only; no response cut. Target unmatched fraction uses target jets pT ≥ 20 GeV.")
    fig.savefig(output / "jet_matching.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return efficiency


def met_plots(rows, tables, output, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for field, label, color in [("magnitude_response", "Magnitude response", "#1764ab"), ("projected_response", "Projected response", "#dd6b20")]:
        curve(axes[0, 0], tables[field], "median", label=label, color=color)
        curve(axes[0, 1], tables[field], "relative_sigma68", label=label, color=color, percent=True)
    axes[0, 0].axhline(1, color="0.6", linestyle=":")
    axes[0, 0].set(xlabel="GEN fiducial MET [GeV]", ylabel="Median response")
    axes[0, 0].legend()
    axes[0, 1].set(xlabel="GEN fiducial MET [GeV]", ylabel="Relative resolution: sigma68 / |median| [%]")
    axes[0, 1].legend()
    for field, label, color in [("delta_parallel", "Parallel residual", "#1764ab"), ("delta_perp", "Perpendicular residual", "#dd6b20")]:
        curve(axes[1, 0], tables[field], "median", label=label, color=color)
        curve(axes[1, 1], tables[field], "sigma68", label=label, color=color)
    axes[1, 0].axhline(0, color="0.6", linestyle=":")
    axes[1, 0].set(xlabel="GEN fiducial MET [GeV]", ylabel="Median target − GEN residual [GeV]")
    axes[1, 0].legend()
    axes[1, 1].set(xlabel="GEN fiducial MET [GeV]", ylabel="Residual sigma68 [GeV]")
    axes[1, 1].legend()
    fig.suptitle(
        title + "\nFiducial MET = −sum of visible particle transverse vectors (same cuts on GEN/targets)"
        "\nMagnitude/projected response: GEN MET ≥ 20 GeV; sigma68=(Q84−Q16)/2; 68% event-bootstrap intervals"
    )
    fig.savefig(output / "met_response.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    maximum = max(max(r["gen_met"], r["target_met"]) for r in rows)
    for field, label, color in [("gen_met", "Fiducial GEN status 1", "black"), ("target_met", "New targets", "#1764ab")]:
        axes[0, 0].hist([r[field] for r in rows], bins=np.linspace(0, maximum * 1.01, 45), histtype="step", label=label, color=color)
    axes[0, 0].set(xlabel="Fiducial MET magnitude [GeV]", ylabel="Events")
    axes[0, 0].legend()
    selected = [r for r in rows if r["gen_met"] >= 20]
    response_values = [r[field] for r in selected for field in ("magnitude_response", "projected_response")]
    response_bins = np.linspace(min(-0.5, min(response_values, default=0)), max(2, max(response_values, default=1) * 1.01), 60)
    for field, label, color in [("magnitude_response", "Magnitude ratio", "#1764ab"), ("projected_response", "Projected response", "#dd6b20")]:
        values = [r[field] for r in selected]
        axes[0, 1].hist(values, bins=response_bins, histtype="step", label=label, color=color)
    axes[0, 1].axvline(1, color="0.6", linestyle=":")
    axes[0, 1].set(xlabel="Target / GEN response (GEN MET ≥ 20 GeV)", ylabel="Events")
    axes[0, 1].legend()
    component_max = max(1.0, max(abs(r[field]) for r in rows for field in ("delta_x", "delta_y")))
    for field, label, color in [("delta_x", "x residual", "black"), ("delta_y", "y residual", "#1764ab")]:
        values = [r[field] for r in rows]
        axes[1, 0].hist(
            values,
            bins=np.linspace(-component_max * 1.01, component_max * 1.01, 60),
            histtype="step",
            label=label,
            color=color,
        )
    axes[1, 0].set(xlabel="Target − GEN MET component [GeV]", ylabel="Events", yscale="log")
    axes[1, 0].legend()
    axes[1, 1].scatter([r["gen_met"] for r in rows], [r["target_met"] for r in rows], s=8, alpha=0.4)
    axes[1, 1].plot([0, maximum], [0, maximum], color="0.6", linestyle=":")
    axes[1, 1].set(xlabel="GEN fiducial MET [GeV]", ylabel="Target fiducial MET [GeV]")
    fig.suptitle(title + "\nFiducial imbalance from visible particles with pT > 0.5 GeV, |eta| ≤ 3.2; not full-acceptance neutrino MET")
    fig.savefig(output / "met_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def particle_plots(events, output, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    collections = [
        (np.concatenate([np.asarray(e[field]).reshape(-1, 4) for e in events]), label, color)
        for field, label, color in [("gen_particle_p4", "Fiducial GEN status 1", "black"), ("target_particle_p4", "New targets", "#1764ab")]
    ]
    pt_bins = np.geomspace(0.5, max(2, max(np.max(kinematics(v)[0], initial=0) for v, _, _ in collections)) * 1.01, 45)
    energy_bins = np.geomspace(0.5, max(2, max(np.max(v[:, 3], initial=0) for v, _, _ in collections)) * 1.01, 45)
    for vectors, label, color in collections:
        pt, eta, _ = kinematics(vectors)
        axes[0].hist(pt, bins=pt_bins, histtype="step", label=label, color=color)
        axes[1].hist(eta, bins=np.linspace(-3.2, 3.2, 49), histtype="step", label=label, color=color)
        axes[2].hist(vectors[:, 3], bins=energy_bins, histtype="step", label=label, color=color)
    for ax, xlabel in zip(axes, ["Particle pT [GeV]", "Particle eta", "Particle energy [GeV]"]):
        ax.set(xlabel=xlabel, ylabel="Particles / targets")
    for ax in (axes[0], axes[2]):
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle(title + "\nVisible fiducial particles only: pT > 0.5 GeV, |eta| ≤ 3.2; no association or ancestor matching imposed")
    fig.savefig(output / "particle_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    import collections

    counts = [collections.Counter(pid for e in events for pid in e[field]) for field in ("gen_pid", "target_pid")]
    species = sorted(set(counts[0]) | set(counts[1]))
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(species))
    for i, (count, label, color) in enumerate(zip(counts, ("Fiducial GEN status 1", "New targets"), ("black", "#1764ab"))):
        ax.bar(x + (i - 0.5) * 0.4, [count[pid] for pid in species], width=0.4, label=label, color=color)
    ax.set(xticks=x, xticklabels=species, xlabel="Absolute PDG ID", ylabel="Particles / targets", yscale="log")
    ax.legend()
    fig.suptitle(title + "\nVisible particles with pT > 0.5 GeV, |eta| ≤ 3.2; assigned targets after splitting/merging")
    fig.savefig(output / "particle_content.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--label", default="PU0 ttbar")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = ResponseConfig()
    results = pickle.loads(args.events.read_bytes())
    events = [analyze_event(result, i, config) for i, result in enumerate(results)]
    pairs = [p for e in events for p in e["jet_matches"]]
    met = [e["met"] for e in events]
    jet_pt = binned_stats(pairs, "gen_pt", "response", JET_PT_BINS, args.bootstrap)
    jet_eta = binned_stats(pairs, "gen_eta", "response", JET_ETA_BINS, args.bootstrap, absolute_x=True)
    met_tables = {
        field: binned_stats(met, "gen_met", field, MET_BINS if "response" in field else MET_COMPONENT_BINS, args.bootstrap)
        for field in ("magnitude_response", "projected_response", "delta_parallel", "delta_perp")
    }
    selected = [r for r in met if r["gen_met"] >= config.met_response_min]
    gen_count = sum(len(e["gen_jets"]) for e in events)
    targets = [j for e in events for j in e["target_jets"] if j["pt"] >= config.genjet_pt_min]
    summary = dict(
        events=len(events),
        config=config.metadata(),
        gen_fiducial_particles=sum(e["gen_particles"] for e in events),
        target_fiducial_particles=sum(e["target_particles"] for e in events),
        gen_jets=gen_count,
        target_jets_above20=len(targets),
        matched_jets=len(pairs),
        gen_matching_efficiency=len(pairs) / gen_count if gen_count else None,
        target_unmatched_fraction=sum(not j["matched"] for j in targets) / len(targets) if targets else None,
        jet_response=bootstrap_stats([p["response"] for p in pairs], [p["event_index"] for p in pairs], args.bootstrap),
        met_response_events=len(selected),
        met_magnitude_response=bootstrap_stats([r["magnitude_response"] for r in selected], [r["event_index"] for r in selected], args.bootstrap),
        met_projected_response=bootstrap_stats([r["projected_response"] for r in selected], [r["event_index"] for r in selected], args.bootstrap),
        met_delta_parallel=bootstrap_stats([r["delta_parallel"] for r in selected], [r["event_index"] for r in selected], args.bootstrap),
        met_delta_perp=bootstrap_stats([r["delta_perp"] for r in selected], [r["event_index"] for r in selected], args.bootstrap),
        met_delta_x=bootstrap_stats([r["delta_x"] for r in met], list(range(len(met))), args.bootstrap),
        met_delta_y=bootstrap_stats([r["delta_y"] for r in met], list(range(len(met))), args.bootstrap),
    )
    tables = dict(jet_pt=jet_pt, jet_abs_eta=jet_eta, met=met_tables)
    title = f"{args.label}: {len(events):,} events, fiducial GEN status 1 versus new targets"
    matching = jet_plots(events, pairs, jet_pt, jet_eta, args.output, title)
    met_plots(met, met_tables, args.output, title)
    particle_plots(events, args.output, title)
    (args.output / "response.json").write_text(
        json.dumps(dict(summary=summary, bins=tables, jet_matching=matching, events=events), indent=2, allow_nan=False) + "\n"
    )
    lines = [
        title,
        "",
        json.dumps(config.metadata(), indent=2),
        "",
        f"GEN jet matching efficiency: {summary['gen_matching_efficiency']}",
        f"Target jet unmatched fraction (pT >= 20 GeV): {summary['target_unmatched_fraction']}",
    ]
    for name in ("jet_response", "met_magnitude_response", "met_projected_response", "met_delta_parallel", "met_delta_perp"):
        s = summary[name]
        lines.append(
            f"{name}: N={s['n']}, median={s['median']}, sigma68={s['sigma68']}"
            + (f", relative sigma68={s['relative_sigma68']}" if "response" in name else " GeV")
        )
    lines.extend(
        [
            "",
            "Response/resolution curves require at least 20 entries per bin. Jet uncertainty resampling preserves per-event correlations.",
            "Jets include visible charged leptons. MET uses the same fiducial particle definition on both sides.",
            "This evaluates target closure and assignment effects; these targets are not independent reconstructed PF candidates.",
        ]
    )
    (args.output / "report.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
