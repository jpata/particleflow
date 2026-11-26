"""Fixed-budget downstream benefit versus region SSL updates, paired by seed."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def stats(x):
    return dict(values=x.tolist(), mean=float(x.mean()), seed_std=float(x.std(ddof=1)))


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    seeds = manifest["seeds"]
    sources = {step: "regions" for step in manifest["regions_steps"]}
    pending = []
    extension_path = ROOT / "early_manifest.json"
    if extension_path.exists():
        extension = json.loads(extension_path.read_text())
        assert extension["seeds"] == seeds
        assert extension["budgets"] == manifest["budgets"]
        for step in extension["regions_steps"]:
            if all(
                (
                    ROOT / "regions_early" / f"step-{step}-seed-{seed}" / "result.json"
                ).exists()
                for seed in seeds
            ):
                sources[step] = "regions_early"
            else:
                pending.append(step)
        if any(source == "regions_early" for source in sources.values()):
            old = json.loads((ROOT / "regions/config.json").read_text())
            new = json.loads((ROOT / "regions_early/config.json").read_text())
            old.pop("ssl_steps")
            new.pop("ssl_steps")
            assert old == new, (
                "Early probes do not match the original probe configuration"
            )
    steps = sorted(sources)
    trials = {
        (step, seed): json.loads(
            (
                ROOT / sources[step] / f"step-{step}-seed-{seed}" / "result.json"
            ).read_text()
        )
        for step in steps
        for seed in seeds
    }
    for seed in seeds:
        assert (
            len({trials[step, seed]["initial_readout_sha256"] for step in steps}) == 1
        )

    def values(step, budget, key, split="valid"):
        rows = []
        for seed in seeds:
            value = next(
                row for row in trials[step, seed]["history"] if row["updates"] == budget
            )[split]
            for part in key.split("."):
                value = value[part]
            rows.append(value)
        return np.array(rows, dtype=float)

    summaries, increments = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for budget, color in [(1000, "C0"), (2000, "C1"), (5000, "C2")]:
        baseline = values(0, budget, "unweighted_task_loss")
        points = []
        for step in steps:
            loss = values(step, budget, "unweighted_task_loss")
            benefit = baseline - loss
            f1 = values(step, budget, "particle_f1")
            row = dict(
                ssl_steps=step,
                probe_updates=budget,
                loss=stats(loss),
                loss_reduction=stats(benefit),
                loss_reduction_percent_ratio_of_means=float(
                    100 * benefit.mean() / baseline.mean()
                ),
                f1=stats(f1),
            )
            summaries.append(row)
            points.append(row)
        for ax, key in [
            (axes[0, 0], "loss"),
            (axes[0, 1], "loss_reduction"),
            (axes[1, 0], "f1"),
        ]:
            ax.errorbar(
                np.array(steps) / 1000,
                [p[key]["mean"] for p in points],
                yerr=[p[key]["seed_std"] for p in points],
                marker="o",
                capsize=3,
                color=color,
                label=f"{budget:,} probe updates",
            )
    for left, right in zip(steps[:-1], steps[1:]):
        delta = values(left, 5000, "unweighted_task_loss") - values(
            right, 5000, "unweighted_task_loss"
        )
        increments.append(
            dict(
                from_step=left,
                to_step=right,
                reduction=stats(delta),
                reduction_per_10k=stats(delta * 10000 / (right - left)),
                improving_seeds=int((delta > 0).sum()),
            )
        )
    ax = axes[1, 1]
    ax.bar(
        range(len(increments)),
        [r["reduction_per_10k"]["mean"] for r in increments],
        yerr=[r["reduction_per_10k"]["seed_std"] for r in increments],
        capsize=5,
        color="C2",
        alpha=0.8,
    )
    ax.set(
        xticks=range(len(increments)),
        xticklabels=[
            f"{r['from_step'] // 1000}–{r['to_step'] // 1000}k" for r in increments
        ],
        xlabel="SSL interval (5,000-update probes)",
        ylabel="PF loss reduction per 10k SSL steps",
    )
    ax.axhline(0, color="gray", ls=":")
    axes[0, 0].set(ylabel="Validation unweighted PF task loss ↓")
    axes[0, 1].set(ylabel="Paired PF loss reduction from SSL step 0 ↑")
    axes[1, 0].set(ylabel="Validation matched particle F1 ↑")
    axes[0, 0].legend(fontsize=9)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set(xlabel="SSL updates (thousands)", xticks=np.array(steps) / 1000)
    for ax in axes.flat:
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(
        "Region SSL benefit at fixed PF probe budgets | five paired seeds, mean ± seed SD\n"
        "Frozen encoder; same 1,024 train / 256 validation events. Lines connect measured checkpoints only."
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            ROOT / f"ssl_step_benefit.{extension}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)

    components = []
    keys = [
        "Classification_binary",
        "Classification",
        "Regression_pt",
        "Regression_eta",
        "Regression_sin_phi",
        "Regression_cos_phi",
        "Regression_energy",
    ]
    for step in steps:
        # Confirm that reported PF loss is the fixed base-task sum, not calibrated Total.
        reconstructed = sum(values(step, 5000, "losses." + key) for key in keys)
        assert np.allclose(reconstructed, values(step, 5000, "unweighted_task_loss"))
        for key in keys + ["Cardinality", "HitGrouping", "Auxiliary", "Total"]:
            loss = values(step, 5000, "losses." + key)
            reduction = values(0, 5000, "losses." + key) - loss
            components.append(
                dict(
                    ssl_steps=step,
                    loss_term=key,
                    included_in_pf_base_sum=key in keys,
                    loss=stats(loss),
                    reduction=stats(reduction),
                )
            )
    result = dict(
        seeds=seeds,
        step_sources=sources,
        pending_steps=pending,
        points=summaries,
        increments=increments,
        components_at_5000=components,
        interpretation="Positive reduction means improvement versus a matched-seed trained readout on the initialized encoder. "
        "Relative reductions divide the mean paired reduction by the baseline mean. Error bars are probe-seed SD, not confidence intervals. "
        f"Completed matched multiseed SSL checkpoints: {steps}; pending: {pending}. No scaling law or convergence limit is fitted. "
        "Unweighted PF loss excludes Cardinality, HitGrouping and Auxiliary; calibrated Total is reported separately.",
    )
    (ROOT / "ssl_step_benefit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                points=[r for r in summaries if r["probe_updates"] == 5000],
                increments=increments,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
