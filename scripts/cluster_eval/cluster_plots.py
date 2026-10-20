#!/usr/bin/env python3
"""Plots for judging calorimeter clusterings, from a YAML spec (columns as in cluster_table.yaml).

    python scripts/cluster_eval/cluster_plots.py --workers 12 --out-dir cluster_plots

Each column (a dataset + a clustering) is evaluated and stored as for the table (columns.py,
same results_dir: a column defined identically in both specs is computed once), here together
with its per-event values and per-cluster / per-particle distributions. Every figure has one
panel per dataset, overlaying that dataset's clusterings; a plot entry's optional `columns`
(a list of labels) restricts which columns it shows. A column's `slot` (1-8) fixes its colour
and line style, so the same clusterer looks the same in every panel and figure.

Figures (<out-dir>/<name>.<format>; name = the entry's `name`, else its kind):
  hits_per_cluster    distribution of the number of hits per cluster, per event (log-log): the
                      single-hit fragments at one end, giant merged clusters at the other
  clusters_per_event  histogram of the number of clusters per event (legend: the mean)
  efficiency          the table's efficiency vs the particle's calorimeter deposit, and its fake
                      rate vs the cluster energy, binned (Wilson 68% intervals); the table's
                      numbers are these above 1 GeV (dotted line). `rule: dm` (default, double
                      majority) or `alloc` (the allocator, clusters only)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import yaml  # noqa: E402

from columns import column_result, dataset_id, dataset_title  # noqa: E402

HERE = Path(__file__).resolve().parent

# categorical slots, fixed order (the first three stay distinguishable for every colour-vision
# type), each with its own line style as a second cue
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.handlelength": 2.6,
    "lines.linewidth": 1.6,
}


def _style(col, slot_of):
    i = slot_of[col["label"]]
    return dict(color=SLOTS[i], ls=LINESTYLES[i], label=col["label"])


def _panels(columns, results, entry):
    """[(title, [(col, result), ...])], one per dataset in spec order, restricted to the plot
    entry's `columns` labels if given."""
    only = entry.get("columns")
    groups = {}
    for c, r in zip(columns, results):
        if only is None or c["label"] in only:
            groups.setdefault(dataset_id(c), []).append((c, r))
    return [(dataset_title(items[0][0]), items) for items in groups.values()]


def _having(panels, key):
    """The panels' columns whose stored distributions include `key` (empty panels dropped)."""
    panels = [(t, [(c, r) for c, r in items if key in r["events"]]) for t, items in panels]
    return [(t, items) for t, items in panels if items]


def _grid(n_panels, n_rows):
    return plt.subplots(n_rows, n_panels, figsize=(3.7 * n_panels, 2.9 * n_rows), squeeze=False, layout="constrained")


def _plain(axis):
    """Log axis labelled with plain numbers (0.1, 1, 10, 100) at the decades."""
    axis.set_major_locator(LogLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _wilson(k, n, z=1.0):
    """Wilson score interval (z = 1: 68%) of k successes out of n; nan where n == 0."""
    n = np.asarray(n, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z**2 / n
        mid = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, mid - half, mid + half


def hits_per_cluster(panels, slot_of, entry):
    panels = _having(panels, "clu_nhit")
    fig, axes = _grid(len(panels), 1)
    for ax, (title, items) in zip(axes[0], panels):
        nmax = max(int(r["events"]["clu_nhit"].max(initial=1)) for _, r in items)
        # log-spaced bins on integer boundaries (1, 2, 3, ..., then wider), the last ending at nmax + 0.5
        edges = np.unique(np.round(np.logspace(0, np.log10(nmax + 1), 40)).astype(np.int64)) - 0.5
        for c, r in items:
            cnt, _ = np.histogram(r["events"]["clu_nhit"], bins=edges)
            dens = cnt / (r["n_events"] * np.diff(edges))
            ax.stairs(np.where(cnt > 0, dens, np.nan), edges, **_style(c, slot_of))
        ax.set(xscale="log", yscale="log", title=title, xlabel="hits per cluster")
        _plain(ax.xaxis)
        ax.legend()
    axes[0, 0].set_ylabel(r"$dN_\mathrm{clusters}/dn_\mathrm{hits}$ per event")
    return fig


def clusters_per_event(panels, slot_of, entry):
    fig, axes = _grid(len(panels), 1)
    for ax, (title, items) in zip(axes[0], panels):
        vals = [r["events"]["ev_n_clusters"] for _, r in items]
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        pad = max(0.05 * (hi - lo), 1.0)
        n_bins = int(np.clip(2.5 * np.sqrt(max(len(v) for v in vals)), 8, 40))  # ~18 bins for 50 events
        edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
        top = 0
        for (c, r), v in zip(items, vals):
            st = dict(_style(c, slot_of), label=f"{c['label']} (mean {v.mean():.0f})")
            cnt, _, _ = ax.hist(v, bins=edges, histtype="step", linewidth=1.6, **st)
            top = max(top, cnt.max())
        ax.set(title=title, xlabel="clusters per event", ylim=(0, 1.45 * top))  # headroom for the legend
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(loc="upper center")
    axes[0, 0].set_ylabel("events")
    return fig


RULES = {"dm": "double majority", "alloc": "allocator, clusters only"}


def efficiency(panels, slot_of, entry):
    rule = entry.get("rule", "dm")
    panels = _having(panels, f"par_{rule}")
    fig, axes = _grid(len(panels), 2)
    for j, (title, items) in enumerate(panels):
        rows = [
            (axes[0, j], "par_T", lambda ev: ev[f"par_{rule}"]),
            (axes[1, j], "clu_E", lambda ev: ~ev[f"clu_{rule}"].astype(bool)),
        ]
        for ax, xkey, success in rows:
            xmax = max(float(r["events"][xkey].max(initial=1.0)) for _, r in items)
            edges = np.logspace(-1, np.log10(max(xmax, 1.0)) + 1e-6, 17)
            centers = np.sqrt(edges[1:] * edges[:-1])
            for i, (c, r) in enumerate(items):
                ev = r["events"]
                n, _ = np.histogram(ev[xkey], bins=edges)
                k, _ = np.histogram(ev[xkey][success(ev)], bins=edges)
                p, lo, hi = _wilson(k, n)
                ok = n > 0
                dodge = 10 ** (0.025 * (i - (len(items) - 1) / 2))  # side by side, not on top
                st = _style(c, slot_of)
                yerr = np.clip([p[ok] - lo[ok], hi[ok] - p[ok]], 0, None)  # rounding at p = 0 or 1
                ax.errorbar(centers[ok] * dodge, p[ok], yerr=yerr, marker="o", markersize=3.5, capsize=0, elinewidth=1.0, **st)
            ax.axvline(1.0, color=MUTED, ls=":", lw=1.0)
            ax.set(xscale="log", ylim=(-0.02, 1.02))
            _plain(ax.xaxis)
        axes[0, j].set(title=title, xlabel="particle calorimeter deposit (GeV)")
        axes[1, j].set(xlabel="cluster energy (GeV)")
        axes[0, j].legend()
    axes[0, 0].set_ylabel(f"efficiency ({RULES[rule]})")
    axes[1, 0].set_ylabel(f"fake rate ({RULES[rule]})")
    return fig


FIGURES = {f.__name__: f for f in (hits_per_cluster, clusters_per_event, efficiency)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(HERE / "cluster_plots.yaml"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true", help="ignore stored column results")
    ap.add_argument("--out-dir", default="cluster_plots")
    ap.add_argument("--formats", nargs="+", default=["png"], help="e.g. png pdf")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text())
    columns = spec["columns"]
    results = [column_result(c, spec["defaults"], args.workers, args.recompute, per_event=True) for c in columns]
    slot_of = {}  # label -> slot index: the column's `slot`, else the next free one
    for c in columns:
        if "slot" in c:
            slot_of[c["label"]] = int(c["slot"]) - 1
    for c in columns:
        if c["label"] not in slot_of:
            slot_of[c["label"]] = min(set(range(len(SLOTS))) - set(slot_of.values()))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for entry in spec["plots"]:
            fig = FIGURES[entry["kind"]](_panels(columns, results, entry), slot_of, entry)
            for fmt in args.formats:
                path = out / f"{entry.get('name', entry['kind'])}.{fmt}"
                fig.savefig(path, dpi=150)
                print(f"wrote {path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
