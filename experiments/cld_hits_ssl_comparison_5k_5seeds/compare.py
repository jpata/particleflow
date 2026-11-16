"""Compare paired PF probes across SSL methods at matched event exposure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
metrics = [
    ("particle_f1", "Particle F1"),
    ("matched_pid_accuracy", "Matched PID accuracy"),
    ("unweighted_task_loss", "PF task loss"),
    ("particle_metrics.event/energy_relative_abs_error", "Event energy relative error"),
    ("particle_metrics.count/mae", "Particle count MAE"),
    ("particle_metrics.matching/efficiency", "Particle recall"),
]


def value(row, key):
    result = row["valid"]
    for part in key.split("."):
        result = result[part]
    return result


def statistics(values):
    valid = values[np.isfinite(values)]
    return dict(
        values=[float(v) if np.isfinite(v) else None for v in values],
        valid_seeds=len(valid),
        mean=float(valid.mean()) if len(valid) else None,
        seed_std=float(valid.std(ddof=1)) if len(valid) > 1 else None,
    )


def main():
    manifest = json.loads((root / "manifest.json").read_text())
    data = {}
    for method, factor in [("regions", 1), ("thinning", 2)]:
        data[method] = {}
        for p in (root / method).glob("step-*-seed-*/result.json"):
            row = json.loads(p.read_text())
            exposure = int(p.parent.name.split("-")[1]) * factor
            data[method][exposure, row["seed"]] = row
    expected = {
        (exposure, seed)
        for exposure in manifest["event_exposures"]
        for seed in manifest["seeds"]
    }
    if any(set(data[method]) != expected for method in data):
        raise ValueError("Comparison requires every planned trial to finish")
    for seed in manifest["seeds"]:
        heads = {
            d["initial_readout_sha256"]
            for rows in data.values()
            for (exposure, s), d in rows.items()
            if s == seed
        }
        assert len(heads) == 1, f"Unpaired initial readouts for seed {seed}"
    summaries = []
    for exposure in manifest["event_exposures"]:
        for budget in manifest["budgets"]:
            for metric, _ in metrics:
                values = {}
                for method in data:
                    values[method] = np.array(
                        [
                            value(
                                next(
                                    r
                                    for r in data[method][exposure, seed]["history"]
                                    if r["updates"] == budget
                                ),
                                metric,
                            )
                            for seed in manifest["seeds"]
                        ],
                        dtype=float,
                    )
                delta = values["regions"] - values["thinning"]
                summaries.append(
                    dict(
                        event_exposure=exposure,
                        updates=budget,
                        metric=metric,
                        **{method: statistics(v) for method, v in values.items()},
                        region_minus_thinning=statistics(delta),
                    )
                )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(
                seeds=manifest["seeds"],
                rows=summaries,
                note="Paired seed differences; standard deviations measure seed variation, not event confidence intervals. Undefined metrics remain null; aggregate counts record available seeds.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, metrics):
        rows = [r for r in summaries if r["metric"] == metric and r["updates"] == 5000]
        for method in data:
            ax.errorbar(
                [r["event_exposure"] for r in rows],
                [r[method]["mean"] for r in rows],
                yerr=[r[method]["seed_std"] for r in rows],
                marker="o",
                capsize=3,
                label=method,
            )
        ax.set(xlabel="Sampled SSL events", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(
        "Matched PF readouts: 5,000 updates, five paired seeds\nMean ± seed standard deviation; 1,024 training / 256 validation events"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(root / f"comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, (metric, label) in zip(axes, metrics[:3]):
        for exposure, style in [(0, "--"), (100000, "-")]:
            rows = [
                r
                for r in summaries
                if r["metric"] == metric
                and r["event_exposure"] == exposure
                and r["updates"] >= 500
            ]
            for method, color in [("regions", "C0"), ("thinning", "C1")]:
                x = np.array([r["updates"] for r in rows])
                mean = np.array([r[method]["mean"] for r in rows])
                sd = np.array([r[method]["seed_std"] for r in rows])
                ax.plot(
                    x,
                    mean,
                    marker="o",
                    ls=style,
                    color=color,
                    label=f"{method}: "
                    + ("initialized" if exposure == 0 else "100k event exposure"),
                )
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.08)
        ax.set(xlabel="Probe updates", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Matched readout learning curves | mean ± standard deviation across five seeds"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            root / f"final_checkpoint_learning_curves.{extension}",
            dpi=160,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("Completed cross-method comparison:", root / "comparison.json")


if __name__ == "__main__":
    main()
