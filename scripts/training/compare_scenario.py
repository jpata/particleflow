"""Compare the variants of one training scenario from their history/step_<n>.json files.

Reads every run under a scenario's experiment directory (<experiments_dir>/<scenario name>/, as
written by scripts/training/run_scenario.py) and writes <out>.md (per run: final and best
validation loss per validation group, test-jet response median/IQR per test sample at the last
validation) and <out>.png (validation loss per group vs step, one panel per group, one line per
variant; further seeds dashed in their variant's colour). Per-group losses need a model whose
validation_datasets hold more than one group.

  PYTHONPATH=. python scripts/training/compare_scenario.py <experiments_dir>/<scenario name>
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.plot_experiment_comparison import load_history  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed per variant
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
JET = re.compile(r"step/(.+)/jet_ratio/jet_ratio_target_to_pred_pt/(med|iqr)$")


def load_runs(scenario_dir):
    """Runs with a history in scenario order (variants as listed in the scenario, then by seed), with the variants.
    Each variant's lowest seed with a history is its primary run, drawn solid and labelled."""
    runs, variants = [], []
    for manifest_path in sorted(Path(scenario_dir).glob("*/scenario-manifest.json")):
        manifest = json.load(open(manifest_path))
        for name in manifest["scenario"]["variants"]:
            if name not in variants:
                variants.append(name)
        try:
            steps = load_history(manifest_path.parent / "history")
        except FileNotFoundError:
            continue  # no validation yet
        job = manifest["job"]
        runs.append({"dir": manifest_path.parent.name, "seed": job["seed"], "arm": job["variant_name"], "steps": steps})
    runs.sort(key=lambda r: (variants.index(r["arm"]), r["seed"]))
    for i, r in enumerate(runs):
        r["primary"] = i == 0 or runs[i - 1]["arm"] != r["arm"]
    return runs, variants


def group_losses(record):
    # The unweighted group totals, so runs with different group step weights stay comparable;
    # fall back to the weighted Total for runs that predate per-group weighting.
    losses = {
        k.split("/")[1]: v
        for k, v in record["valid"].items()
        if k.startswith("group/") and k.endswith("/Total_unweighted")
    }
    return losses or {k.split("/")[1]: v for k, v in record["valid"].items() if k.startswith("group/") and k.endswith("/Total")}


def jets(record):
    out = {}
    for k, v in record.items():
        m = JET.match(k)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = v
    return out


def table(runs):
    groups = sorted({g for r in runs for g in group_losses(r["steps"][-1][1])})
    samples = sorted({s for r in runs for s in jets(r["steps"][-1][1])})
    head = ["variant", "seed", "last step"] + [f"{g} val (best)" for g in groups] + [f"{s} jet med / iqr" for s in samples]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        last_step, last = r["steps"][-1]
        best = {g: min(group_losses(rec).get(g, float("inf")) for _, rec in r["steps"]) for g in groups}
        cells = [r["arm"], str(r["seed"]), str(last_step)]
        cells += [f"{group_losses(last).get(g, float('nan')):.4f} ({best[g]:.4f})" for g in groups]
        cells += [f"{jets(last).get(s, {}).get('med', float('nan')):.3f} / {jets(last).get(s, {}).get('iqr', float('nan')):.3f}" for s in samples]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), groups


def _label_line_ends(ax, ends, min_gap_pt=10):
    """Direct labels at the line ends, pushed apart vertically so they never overlap."""
    ax.margins(x=0.15)
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    placed = []
    for (x, y), label in sorted(ends, key=lambda e: e[0][1]):
        y_pt = ax.transData.transform((x, y))[1] * to_pt
        target = max(y_pt, placed[-1] + min_gap_pt) if placed else y_pt
        placed.append(target)
        ax.annotate(label, (x, y), xytext=(4, target - y_pt), textcoords="offset points", va="center", fontsize=8, color=INK2)


def plot(runs, groups, variants, title, path):
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), sharex=True, facecolor=SURFACE)
    axes = [axes] if len(groups) == 1 else list(axes)
    for ax, g in zip(axes, groups):
        ax.set_facecolor(SURFACE)
        ends = []
        for r in runs:
            pts = [(s, group_losses(rec)[g]) for s, rec in r["steps"] if g in group_losses(rec)]
            if not pts:
                continue
            color = COLORS[variants.index(r["arm"]) % len(COLORS)]
            dashed = not r["primary"]
            ax.plot(*zip(*pts), color=color, lw=1.6, ls="--" if dashed else "-", label=None if dashed else r["arm"])
            if not dashed:
                ends.append((pts[-1], r["arm"]))
        _label_line_ends(ax, ends)
        ax.set_title(g, fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.set_xlabel("training step", fontsize=8, color=INK2)
    axes[0].set_ylabel("validation loss (Total, unweighted)", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"{title}: validation loss per group by variant (dashed: other seeds)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", help="<experiments_dir>/<scenario name>")
    parser.add_argument("--out", help="output path prefix for .md and .png (default: <scenario_dir>/comparison)")
    args = parser.parse_args()

    runs, variants = load_runs(args.scenario_dir)
    if not runs:
        raise SystemExit(f"no runs with history under {args.scenario_dir}")
    out = args.out or str(Path(args.scenario_dir) / "comparison")
    md, groups = table(runs)
    Path(f"{out}.md").write_text(md + "\n")
    if groups:
        plot(runs, groups, variants, Path(args.scenario_dir).name, f"{out}.png")
    print(md)
    print(f"wrote {out}.md" + (f" and {out}.png" if groups else " (no per-group validation losses to plot)"))


if __name__ == "__main__":
    main()
