"""Summarize context-ablation events and individual-hit prediction curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_mean(values):
    values = np.asarray(values)
    rng = np.random.default_rng(128743)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), event_bootstrap_95pct=np.quantile(means, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    context = json.loads((args.output / "context_events.json").read_text())
    individual = json.loads((args.output / "individual_hit_results.json").read_text())
    conditions = ["full_repeat", "keep_60pct", "foreign_context", "queries_only", "isolated_hits"]
    groups = ["1", "2", "all"]
    summary = {"context": [], "individual_hit": [], "type_marginal_baseline": individual["type_marginal_baseline"]}
    for network in ["initial_encoder", "teacher", "student"]:
        for condition in conditions:
            rows = [r for r in context if r["network"] == network and r["condition"] == condition]
            for group in groups:
                item = dict(network=network, condition=condition, group=group, events=len(rows))
                for section in ["groups", "against_own_full", "against_full_teacher"]:
                    if section not in rows[0]:
                        continue
                    item[section] = {}
                    for metric in rows[0][section]["1"]:
                        if metric == "hits":
                            continue
                        values = [np.mean([g[metric] for k, g in r[section].items() if group == "all" or k == group]) for r in rows]
                        item[section][metric] = bootstrap_mean(values)
                summary["context"].append(item)
    for features in ["geometry", "all_raw"]:
        trials = [t for t in individual["trials"] if t["features"] == features]
        for group in groups:
            for metric in ["agreement", "kl", "cross_entropy", "js"]:
                values = [t["history"][-1]["valid"][group][metric] for t in trials]
                summary["individual_hit"].append(dict(features=features, group=group, metric=metric,
                    values=values, mean=float(np.mean(values)), seed_std=float(np.std(values, ddof=1))))
    summary["uncertainty"] = "Context intervals bootstrap events; predictor std spans three initialization/training seeds on a fixed event split. Neither includes SSL training-seed uncertainty."
    summary["max_reused_donor_rows"] = max(r["reused_donor_rows"] for r in context)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = ["Full repeat", "60% retained", "Foreign context", "Queries only", "Isolated"]
    for col, group in enumerate(["1", "2"]):
        for network in ["teacher", "student"]:
            rows = [next(r for r in summary["context"] if r["network"] == network and r["condition"] == c and r["group"] == group) for c in conditions]
            values = [r["against_full_teacher"]["agreement"]["mean"] for r in rows]
            cis = np.array([r["against_full_teacher"]["agreement"]["event_bootstrap_95pct"] for r in rows])
            axes[0, col].plot(labels, values, "o-", label=network)
            axes[0, col].fill_between(labels, cis[:, 0], cis[:, 1], alpha=.15)
        for network in ["initial_encoder", "teacher", "student"]:
            values = [next(r for r in summary["context"] if r["network"] == network and r["condition"] == c and r["group"] == group)["groups"]["relative_centered_rms"]["mean"] for c in conditions]
            axes[1, col].plot(labels, values, "o-", label=network)
        axes[0, col].set(title=("Tracker" if group == "1" else "Calorimeter") + ": prototype agreement with full teacher", ylim=(0, 1.02))
        axes[1, col].set(title="Embedding change / within-query reference spread", ylabel="Relative RMS change")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed query measurements; manipulated context | shading: 95% event-bootstrap interval")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"context_dependence.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for col, group in enumerate(["1", "2", "all"]):
        for features in ["geometry", "all_raw"]:
            trials = [t for t in individual["trials"] if t["features"] == features]
            steps = [r["updates"] for r in trials[0]["history"]]
            for row, metric in enumerate(["agreement", "kl"]):
                values = np.array([[r["valid"][group][metric] for r in t["history"]] for t in trials])
                mean, std = values.mean(0), values.std(0, ddof=1)
                axes[row, col].plot(steps, mean, "o-", label=features)
                axes[row, col].fill_between(steps, mean-std, mean+std, alpha=.15)
                axes[row, col].axhline(individual["type_marginal_baseline"][group][metric], ls=":", color="gray")
                axes[row, col].set(xlabel="Predictor updates", ylabel="Prototype agreement" if row == 0 else "KL(full teacher || predictor)")
                axes[row, col].grid(alpha=.2)
        axes[0, col].set_title({"1": "Tracker", "2": "Calorimeter", "all": "Equal-type average"}[group])
        axes[0, col].legend()
    fig.suptitle("Individual-hit predictors: held-out events | mean ± one seed std | dotted: detector-type marginal baseline")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"individual_hit_predictors.{ext}", dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
