"""Reproduce the completed-run comparison using saved artifacts only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path("experiments/cld_hits_ssl_regions")
out = run / "analysis"
records = [json.loads(line) for line in (run / "metrics.jsonl").open()]
valid = [r for r in records if "valid" in r]
steps = np.array([r["step"] for r in valid])
probes = {}
for name in ["cld_hits_ssl_regions", "cld_hits_ssl_large"]:
    probes[name] = [
        json.loads(p.read_text())
        for p in sorted(
            (Path("experiments") / name / "probes").glob("step-*.json"),
            key=lambda p: int(p.stem.split("-")[1]),
        )
    ]


def save(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
for ax, key, label, nested in zip(
    axes.flat,
    [
        "particle_f1",
        "matched_pid_accuracy",
        "unweighted_task_loss",
        "event/energy_relative_abs_error",
        "matching/purity",
        "count/mae",
    ],
    [
        "Matched particle F1",
        "Matched PID accuracy",
        "Unweighted PF task loss",
        "Event energy relative absolute error",
        "Matched particle purity",
        "Particle count MAE",
    ],
    [False, False, False, True, True, True],
):
    for name, factor, legend in [
        ("cld_hits_ssl_regions", 1, "Region masking"),
        ("cld_hits_ssl_large", 2, "Hit thinning"),
    ]:
        data = probes[name]
        ax.plot(
            [r["ssl_step"] * factor for r in data],
            [
                (r["valid"]["particle_metrics"] if nested else r["valid"])[key]
                for r in data
            ],
            "o-",
            label=legend,
        )
    ax.set(xlabel="Sampled SSL events (step × batch size)", ylabel=label)
    ax.grid(alpha=0.3)
axes[0, 0].legend()
fig.suptitle(
    "Frozen PF probes: 1,000 updates, 1,024 train / 256 validation events\nOne seed per run; exposure matched, optimizer steps and compute differ"
)
save(fig, "comparison")

fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
for key, weight in [
    ("occupancy_loss", 1),
    ("count_loss", 0.1),
    ("energy_loss", 1),
    ("latent_loss", 0.1),
]:
    axes[0, 0].plot(
        steps, [r["valid"][key] * weight for r in valid], label=f"{weight:g} × {key}"
    )
axes[0, 0].set(ylabel="Weighted validation objective term")
axes[0, 0].legend(fontsize=8)
for key in ["occupancy_precision", "occupancy_recall", "occupancy_f1"]:
    axes[0, 1].plot(
        steps, [r["valid"][key] for r in valid], label=key.removeprefix("occupancy_")
    )
axes[0, 1].legend()
axes[0, 1].set(ylabel="Occupancy metrics at threshold 0.5")
p = valid[-1]["valid"]["occupied_fraction"]
axes[0, 2].plot(
    steps, [r["valid"]["occupancy_accuracy"] for r in valid], label="Predictor"
)
axes[0, 2].axhline(1 - p, color="gray", ls="--", label="Predict all empty")
axes[0, 2].legend()
axes[0, 2].set(ylabel="Occupancy accuracy")
for ax, key in zip(axes[1], ["energy_loss", "teacher_entropy", "effective_prototypes"]):
    ax.plot(steps, [r["valid"][key] for r in valid])
    ax.set(ylabel=key)
for ax in axes.flat:
    ax.set(xlabel="SSL updates")
    ax.grid(alpha=0.3)
fig.suptitle(
    "Region prediction on 128 fixed validation events / masks\nPrototype count is effective within-event diversity, not particle count"
)
save(fig, "region_objective")

spectra = []
for path in sorted(
    (run / "embeddings").glob("step-*.npz"), key=lambda p: int(p.stem.split("-")[1])
):
    with np.load(path) as d:
        for kind, name in [(1, "tracker"), (2, "calorimeter")]:
            x = d["student"][d["X"][:, 0] == kind].astype(float)
            x -= x.mean(0)
            eig = np.maximum(np.linalg.eigvalsh(x.T @ x), 0)
            w = eig / eig.sum()
            spectra.append(
                dict(
                    step=int(d["step"]),
                    detector=name,
                    rows=len(x),
                    effective_rank=float(np.exp(-(w * np.log(w.clip(1e-30))).sum())),
                    participation_ratio=float(1 / (w * w).sum()),
                    pc1_fraction=float(w[-1]),
                    feature_std=float(x.std(0).mean()),
                )
            )
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
for name in ["tracker", "calorimeter"]:
    rows = [r for r in spectra if r["detector"] == name]
    for ax, key in zip(axes, ["effective_rank", "pc1_fraction", "feature_std"]):
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=name)
        ax.set(xlabel="SSL updates", ylabel=key)
        ax.grid(alpha=0.3)
axes[0].legend()
fig.suptitle(
    "Centered embedding covariance: fixed four-event cohort, 8,192 sampled hits"
)
save(fig, "embedding_spectrum")

rng = np.random.default_rng(27182)
pair_deltas = {}
first, final = probes["cld_hits_ssl_regions"][0], probes["cld_hits_ssl_regions"][-1]
for detector in ["tracker", "calorimeter"]:
    pair_deltas[detector] = {}
    for metric in ["average_precision", "roc_auc"]:
        for name, reference in [
            ("vs_initial", first["hit_pairs"]["detectors"][detector]["embedding"]),
            ("vs_raw", final["hit_pairs"]["detectors"][detector]["raw"]),
        ]:
            lookup = {
                r["event_index"]: r[metric] for r in reference["events"] if metric in r
            }
            differences = np.array(
                [
                    r[metric] - lookup[r["event_index"]]
                    for r in final["hit_pairs"]["detectors"][detector]["embedding"][
                        "events"
                    ]
                    if metric in r and r["event_index"] in lookup
                ]
            )
            boot = differences[
                rng.integers(len(differences), size=(5000, len(differences)))
            ].mean(1)
            pair_deltas[detector][f"{metric}_{name}"] = dict(
                mean=float(differences.mean()),
                event_bootstrap_95ci=np.quantile(boot, [0.025, 0.975]).tolist(),
                events=len(differences),
            )
train = [r for r in records if "train" in r]
performance = dict(
    training_seconds=sum(r["performance"]["training_seconds"] for r in train),
    logged_period_seconds=train[-1]["performance"]["start_unix_time"]
    + train[-1]["performance"]["step_seconds"]
    - train[0]["performance"]["start_unix_time"],
    max_recorded_step_allocated_gib=max(
        r["performance"]["step_peak_allocated_bytes"] for r in train
    )
    / 2**30,
)
summary = dict(
    validation=[dict(step=r["step"], **r["valid"]) for r in valid],
    spectra=spectra,
    probes={
        name: [dict(step=r["ssl_step"], valid=r["valid"]) for r in data]
        for name, data in probes.items()
    },
    pair_deltas=pair_deltas,
    performance=performance,
    final_train_2000_mean={
        k: float(np.mean([r["train"][k] for r in train[-2000:]]))
        for k in train[-1]["train"]
    },
    uncertainty="Pair bootstrap samples events, not pairs; conditional on fitted probes and one SSL seed. PF aggregate artifacts do not support full-cohort paired confidence intervals.",
)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(dict(pair_deltas=pair_deltas, performance=performance), indent=2))
