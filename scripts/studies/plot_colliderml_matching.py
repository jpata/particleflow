"""Summarize the paired event comparison and plot target direction distributions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "legacy": "Current greedy",
    "majority_greedy": "Majority tracks + greedy clusters",
    "ownership_greedy": "Greedy on ownership scores",
    "solver": "Joint solver",
    "majority_energy": "Bipartite matching, energy score",
}


def confidence_intervals(events, output):
    rng = np.random.default_rng(12345)
    result = []
    metrics = [
        ("track_failure_delta_pp", "track_all_bad", "track_all_count"),
        ("cluster_failure_delta_pp", "cluster_all_bad", "cluster_all_count"),
        ("cluster_pt5_failure_delta_pp", "cluster_pt5_bad", "cluster_pt5_count"),
        ("jet_recall_delta_pp", "matched_jets", "reference_jets"),
        ("jet_response_recall_delta_pp", "response_qualified_jets", "reference_jets"),
        ("exclusive_hs_delta_pp", "exclusive_hs", "selected_hs"),
    ]
    for sample in ["pu0", "pu200"]:
        x = events[events["sample"] == sample]
        solver = x[x.rule == "solver"].sort_values("event_id")
        ix = rng.integers(0, len(solver), (5000, len(solver)))
        for baseline in ["legacy", "ownership_greedy", "majority_energy"]:
            other = x[x.rule == baseline].sort_values("event_id")
            assert np.array_equal(solver.event_id.values, other.event_id.values)
            for name, numerator, denominator in metrics:
                sn, sd = solver[numerator].values, solver[denominator].values
                gn, gd = other[numerator].values, other[denominator].values
                values = 100 * (sn[ix].sum(axis=1) / sd[ix].sum(axis=1) - gn[ix].sum(axis=1) / gd[ix].sum(axis=1))
                point = 100 * (sn.sum() / sd.sum() - gn.sum() / gd.sum())
                lo, hi = np.quantile(values, [0.025, 0.975])
                result.append(dict(sample=sample, baseline=baseline, metric=name, estimate=point, low95=lo, high95=hi))
            s, g = solver.ownership_objective.values, other.ownership_objective.values
            gain = 100 * (s[ix].sum(axis=1) / g[ix].sum(axis=1) - 1)
            lo, hi = np.quantile(gain, [0.025, 0.975])
            result.append(
                dict(
                    sample=sample,
                    baseline=baseline,
                    metric="ownership_objective_gain_percent",
                    estimate=100 * (s.sum() / g.sum() - 1),
                    low95=lo,
                    high95=hi,
                )
            )
    frame = pd.DataFrame(result)
    frame.to_csv(output / "paired_bootstrap.csv", index=False)
    (output / "paired_bootstrap.json").write_text(json.dumps(result, indent=2))


def plot(output):
    events = pd.read_csv(output / "events.csv")
    counts = events.groupby(["sample", "rule"]).size().unstack("rule")
    assert counts.notna().all().all() and counts.nunique(axis=1).eq(1).all()
    confidence_intervals(events, output)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig_high, axes_high = plt.subplots(1, 2, figsize=(12, 4.5))
    quantiles = []
    for row, sample in enumerate(["pu0", "pu200"]):
        files = sorted((output / sample / "events").glob("event_*.parquet"))
        objects = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        for column, kind in enumerate(["track", "cluster"]):
            ax = axes[row, column]
            for rule, label in LABELS.items():
                subset = objects[(objects.kind == kind) & (objects.rule == rule)]
                dr = np.sort(subset.delta_r.values)
                ax.plot(dr, np.arange(1, len(dr) + 1) / len(dr), label=label)
                quantiles.append(dict(sample=sample, rule=rule, kind=kind, median=float(np.median(dr)), p90=float(np.quantile(dr, 0.9))))
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 15)
            ax.set_ylim(0, 1)
            ax.axvline(0.1 if kind == "track" else 0.3, color="black", ls="--", lw=1)
            ax.set_title(f"{sample.upper()}: {kind} targets, {int(counts.loc[sample].iloc[0])} events")
            ax.set_xlabel("Delta-R between input and target particle")
            ax.set_ylabel("Fraction of targets")
            ax.grid(alpha=0.2)
        ax = axes_high[row]
        for rule, label in LABELS.items():
            dr = np.sort(objects[(objects.kind == "cluster") & (objects.rule == rule) & (objects.pt >= 5)].delta_r.values)
            ax.plot(dr, np.arange(1, len(dr) + 1) / len(dr), label=label)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 10)
        ax.set_ylim(0, 1)
        ax.axvline(0.3, color="black", ls="--", lw=1)
        ax.set_title(f"{sample.upper()}: cluster targets with pT >= 5 GeV")
        ax.set_xlabel("Delta-R between cluster and target")
        ax.set_ylabel("Fraction of targets")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes_high[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "direction_cdf.png", dpi=160)
    plt.close(fig)
    fig_high.tight_layout()
    fig_high.savefig(output / "high_pt_cluster_direction.png", dpi=160)
    plt.close(fig_high)
    pd.DataFrame(quantiles).to_csv(output / "direction_quantiles.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    plot(parser.parse_args().input)
