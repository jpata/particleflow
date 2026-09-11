#!/usr/bin/env python3
"""Make the final physics and computing plots for the keyed-hit lookup fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from uproot.source.file import MemmapSource


CASE = "idea_qq"
VARIANTS = {"pre_fix": "pre-fix (linear scan)", "main": "current (keyed lookup)"}
COLORS = {"pre_fix": "#dc2626", "main": "#059669"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=10.0)
    return parser.parse_args()


def run_path(root: Path, variant: str, name: str) -> Path:
    return root / "runs" / variant / CASE / "rep-1" / name


def load_metrics(root: Path, variant: str) -> dict:
    path = run_path(root, variant, "metrics.json")
    data = json.loads(path.read_text())
    required = {
        "user_s",
        "system_s",
        "wall_s",
        "max_rss_kb",
        "startup_s",
        "processing_s",
        "processing_s_per_event",
    }
    missing = required - set(data)
    if missing or any(data.get(key) is None for key in required):
        raise ValueError(f"missing metrics in {path}: {sorted(missing)}")
    data["cpu_s"] = data["user_s"] + data["system_s"]
    return data


def payload_leaves(tree: uproot.TTree) -> dict[str, uproot.TBranch]:
    leaves = {}
    for name in tree.keys(recursive=True):
        branch = tree[name]
        if len(branch.branches) == 0 and "EventHeader.timeStamp" not in name:
            leaves[name] = branch
    return leaves


def branch_by_suffix(tree: uproot.TTree, suffix: str) -> uproot.TBranch:
    matches = [name for name in tree.keys(recursive=True) if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one branch ending in {suffix!r}, found {matches}")
    return tree[matches[0]]


def load_and_validate_physics(root: Path) -> dict:
    paths = {variant: run_path(root, variant, "events.root") for variant in VARIANTS}
    with (
        uproot.open(paths["pre_fix"], handler=MemmapSource) as pre_file,
        uproot.open(paths["main"], handler=MemmapSource) as current_file,
    ):
        pre_tree = pre_file["events"]
        current_tree = current_file["events"]
        if pre_tree.num_entries != current_tree.num_entries:
            raise ValueError(f"event-count mismatch: {pre_tree.num_entries} != {current_tree.num_entries}")
        pre_leaves = payload_leaves(pre_tree)
        current_leaves = payload_leaves(current_tree)
        if set(pre_leaves) != set(current_leaves):
            missing = sorted(set(pre_leaves) - set(current_leaves))
            extra = sorted(set(current_leaves) - set(pre_leaves))
            raise ValueError(f"leaf mismatch: missing={missing[:5]} extra={extra[:5]}")
        mismatches = []
        for name in sorted(pre_leaves):
            before = pre_leaves[name].array(library="ak")
            after = current_leaves[name].array(library="ak")
            if not ak.almost_equal(before, after, rtol=0.0, atol=0.0, dtype_exact=True):
                mismatches.append(name)
        if mismatches:
            raise ValueError(f"payload mismatches: {mismatches[:10]}")

        suffix = "DRcaloSiPMreadout_scint.energy"
        energies = {
            "pre_fix": branch_by_suffix(pre_tree, suffix).array(library="ak"),
            "main": branch_by_suffix(current_tree, suffix).array(library="ak"),
        }
        event_count = pre_tree.num_entries
        exact_leaf_count = len(pre_leaves)
    return {
        "events": event_count,
        "exact_leaves": exact_leaf_count,
        "energies": energies,
    }


def plot_physics(data: dict, output: Path) -> dict:
    energies = data["energies"]
    flattened = {variant: ak.to_numpy(ak.flatten(values)) for variant, values in energies.items()}
    positive = {variant: values[values > 0] for variant, values in flattened.items()}
    if any(len(values) == 0 for values in positive.values()):
        raise ValueError("the sensitive scintillation-energy distribution is empty")
    low = min(values.min() for values in positive.values())
    high = max(values.max() for values in positive.values())
    bins = np.geomspace(low, high * (1.0 + 1e-9), 55)
    counts = {variant: np.histogram(values, bins=bins)[0] for variant, values in positive.items()}
    event_energy = {variant: np.asarray([float(ak.sum(event)) for event in values]) for variant, values in energies.items()}
    event_cells = {variant: np.asarray([len(event) for event in values]) for variant, values in energies.items()}

    fig = plt.figure(figsize=(13.5, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.28)
    histogram = fig.add_subplot(grid[0, 0])
    ratio_axis = fig.add_subplot(grid[1, 0], sharex=histogram)
    energy_closure = fig.add_subplot(grid[0, 1])
    cell_closure = fig.add_subplot(grid[1, 1])

    styles = {"pre_fix": "--", "main": "-"}
    for variant in ("pre_fix", "main"):
        histogram.stairs(
            counts[variant],
            bins,
            color=COLORS[variant],
            linestyle=styles[variant],
            linewidth=2.2,
            label=f"{VARIANTS[variant]} ({len(positive[variant]):,} positive cells)",
        )
    histogram.set_xscale("log")
    histogram.set_yscale("log")
    histogram.set_ylabel("cells / bin")
    histogram.set_title("Sensitive observable: scintillation energy per DRC cell")
    histogram.legend(frameon=False, fontsize=9)
    histogram.grid(alpha=0.2, which="both")
    histogram.tick_params(labelbottom=False)

    valid = counts["pre_fix"] > 0
    ratio = np.divide(
        counts["main"],
        counts["pre_fix"],
        out=np.ones_like(counts["main"], dtype=float),
        where=valid,
    )
    centers = np.sqrt(bins[:-1] * bins[1:])
    ratio_axis.plot(centers[valid], ratio[valid], "o", color="#111827", markersize=3)
    ratio_axis.axhline(1.0, color="#059669", linewidth=1.5)
    ratio_axis.set_xscale("log")
    ratio_axis.set_ylim(0.95, 1.05)
    ratio_axis.set_ylabel("current / pre-fix")
    ratio_axis.set_xlabel("positive scintillation hit energy [GeV]")
    ratio_axis.grid(alpha=0.2, which="both")

    before_energy = event_energy["pre_fix"]
    after_energy = event_energy["main"]
    energy_low = min(before_energy.min(), after_energy.min())
    energy_high = max(before_energy.max(), after_energy.max())
    energy_closure.plot([energy_low, energy_high], [energy_low, energy_high], color="#111827")
    energy_closure.scatter(before_energy, after_energy, color="#059669", s=55, zorder=3)
    for event, (before, after) in enumerate(zip(before_energy, after_energy)):
        energy_closure.annotate(str(event), (before, after), xytext=(4, 4), textcoords="offset points")
    energy_closure.set_xlabel("pre-fix event scintillation energy [GeV]")
    energy_closure.set_ylabel("current event scintillation energy [GeV]")
    energy_closure.set_title("Event-level energy equality")
    energy_closure.grid(alpha=0.2)

    before_cells = event_cells["pre_fix"]
    after_cells = event_cells["main"]
    cell_low = min(before_cells.min(), after_cells.min())
    cell_high = max(before_cells.max(), after_cells.max())
    cell_closure.plot([cell_low, cell_high], [cell_low, cell_high], color="#111827")
    cell_closure.scatter(before_cells, after_cells, color="#2563eb", s=45, zorder=3)
    cell_closure.set_xlabel("pre-fix cells / event")
    cell_closure.set_ylabel("current cells / event")
    cell_closure.grid(alpha=0.2)

    fig.suptitle(
        "PR #620 physics validation: keyed lookup preserves IDEA output exactly\n"
        f"{data['events']} fixed-seed 365 GeV qq events; "
        f"{data['exact_leaves']} typed EDM4hep leaves equal",
        fontsize=15,
    )
    fig.savefig(output / "keyed_lookup_physics.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "keyed_lookup_physics.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "events": data["events"],
        "exact_leaves": data["exact_leaves"],
        "positive_scintillation_cells": len(positive["main"]),
        "max_event_energy_difference": float(np.max(np.abs(after_energy - before_energy))),
        "max_event_cell_difference": int(np.max(np.abs(after_cells - before_cells))),
    }


def plot_computing(metrics: dict[str, dict], output: Path, minimum_speedup: float) -> dict:
    before = metrics["pre_fix"]
    after = metrics["main"]
    processing_speedup = before["processing_s_per_event"] / after["processing_s_per_event"]
    cpu_speedup = before["cpu_s"] / after["cpu_s"]
    wall_speedup = before["wall_s"] / after["wall_s"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    categories = ["processing / event", "total CPU / run", "wall / run"]
    before_times = [before["processing_s_per_event"], before["cpu_s"], before["wall_s"]]
    after_times = [after["processing_s_per_event"], after["cpu_s"], after["wall_s"]]
    x = np.arange(len(categories))
    width = 0.36
    axes[0].bar(x - width / 2, before_times, width, color=COLORS["pre_fix"], label=VARIANTS["pre_fix"])
    axes[0].bar(x + width / 2, after_times, width, color=COLORS["main"], label=VARIANTS["main"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("seconds (log scale)")
    axes[0].set_xticks(x, categories)
    axes[0].set_title("Runtime reduction")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2, which="both")
    for position, factor in enumerate((processing_speedup, cpu_speedup, wall_speedup)):
        axes[0].text(
            position,
            max(before_times[position], after_times[position]) * 1.12,
            f"{factor:.1f}×",
            ha="center",
            fontweight="bold",
        )

    rss = [before["max_rss_kb"] / 1024**2, after["max_rss_kb"] / 1024**2]
    axes[1].bar([0, 1], rss, color=[COLORS["pre_fix"], COLORS["main"]], width=0.65)
    axes[1].set_xticks([0, 1], ["pre-fix", "current"])
    axes[1].set_ylabel("peak RSS [GiB]")
    axes[1].set_title("Memory usage")
    axes[1].grid(axis="y", alpha=0.2)
    for position, value in enumerate(rss):
        axes[1].text(position, value, f"{value:.2f} GiB", ha="center", va="bottom")

    status = "PASS" if processing_speedup >= minimum_speedup else "FAIL"
    fig.suptitle(
        f"PR #620 computing validation: {processing_speedup:.1f}× faster IDEA event processing [{status}]\n"
        f"required speedup ≥ {minimum_speedup:.1f}×; startup excluded only from the per-event metric",
        color="#059669" if status == "PASS" else "#dc2626",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output / "keyed_lookup_computing.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "keyed_lookup_computing.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "status": status,
        "minimum_speedup": minimum_speedup,
        "processing_speedup": processing_speedup,
        "cpu_speedup": cpu_speedup,
        "wall_speedup": wall_speedup,
        "pre_fix": before,
        "current": after,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    physics = plot_physics(load_and_validate_physics(args.workdir), args.output)
    metrics = {variant: load_metrics(args.workdir, variant) for variant in VARIANTS}
    computing = plot_computing(metrics, args.output, args.minimum_speedup)
    summary = {"physics": physics, "computing": computing}
    (args.output / "keyed_lookup_summary.json").write_text(json.dumps(summary, indent=2))
    if computing["status"] != "PASS":
        raise SystemExit(f"processing speedup {computing['processing_speedup']:.2f}x is below " f"{args.minimum_speedup:.2f}x")
    print(f"wrote final keyed-lookup validation plots to {args.output}")


if __name__ == "__main__":
    main()
