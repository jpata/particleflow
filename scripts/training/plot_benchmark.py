"""Render portable sweep CSVs. Requires matplotlib, numpy and pandas; no training stack."""

import argparse
from pathlib import Path


def render(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    directory = Path(directory)
    cases = pd.read_csv(directory / "cases.csv")
    if (cases.world_size != 1).any() or cases.duplicated(["platform", "variant", "batch"]).any():
        raise ValueError("Plot one single-device session per platform/model/batch; select repeats explicitly")
    completed = cases[cases.status == "completed"].copy()
    if completed.empty:
        raise ValueError("No completed cases to plot")
    completed["allocated_gib"] = completed.allocated_bytes / 2**30
    completed["khits_per_second"] = completed.hits_per_second / 1000
    completed["p99_over_p50"] = completed.step_p99_seconds / completed.step_p50_seconds
    platforms = list(cases.platform.drop_duplicates())
    variants = list(cases.variant.drop_duplicates())
    colors = dict(zip(platforms, plt.get_cmap("tab10").colors * (1 + len(platforms) // 10)))
    figdir = directory / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(figdir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def compare(data, metric, ylabel, name):
        fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 4), squeeze=False, layout="constrained", sharey=True)
        for ax, variant in zip(axes.flat, variants):
            for platform in platforms:
                rows = data[(data.platform == platform) & (data.variant == variant)].sort_values("batch")
                ax.plot(rows.batch, rows[metric], "-o", label=platform, color=colors[platform])
            ax.set(title=variant, xlabel="Training batch size", ylabel=ylabel, xscale="log")
            batches = sorted(data.batch.unique())
            ax.set_xticks(batches, labels=batches)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(alpha=0.2)
        axes.flat[-1].legend(fontsize=10)
        save(fig, name)

    for metric, ylabel in [
        ("events_per_second", "Steady training events/s"),
        ("khits_per_second", "Steady training kHits/s"),
        ("allocated_gib", "Process peak allocated GiB"),
        ("p99_over_p50", "Host step p99 / p50"),
    ]:
        compare(completed, metric, ylabel, metric)
    steps = pd.read_csv(directory / "steps.csv")
    keys = completed[["platform", "variant", "batch", "seed", "run_id", "session_id", "rank", "warmup_steps"]]
    steps = steps.merge(keys, on=list(keys.columns[:-1]), validate="many_to_one")
    for index, variant in enumerate(variants):
        selected = steps[steps.variant == variant]
        batches = sorted(selected.batch.unique())
        ncols = min(2, len(batches))
        nrows = (len(batches) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False, layout="constrained", sharey=True)
        for ax, batch in zip(axes.flat, batches):
            for i, platform in enumerate(platforms):
                rows = selected[(selected.platform == platform) & (selected.batch == batch)].sort_values("step")
                ax.plot(rows.step, rows.total_unweighted, ["-", "--", ":", "-."][i % 4], label=platform, color=colors[platform])
            ax.set(title=f"{variant}, batch {batch}", xlabel="Training step", ylabel="Total unweighted loss")
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(batches) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=10)
        save(fig, f"training_loss_{index + 1}")
    validation = pd.read_csv(directory / "validation.csv")
    if not validation.empty:
        endpoints = validation.merge(
            completed[["platform", "variant", "batch", "seed", "run_id", "session_id", "rank", "steps"]],
            on=["platform", "variant", "batch", "seed", "run_id", "session_id", "rank"],
            validate="many_to_one",
        )
        endpoints = endpoints[endpoints.step == endpoints.steps]
        if endpoints.step.nunique() != 1:
            raise ValueError("Validation endpoints have different training lengths; compare explicitly")
        compare(endpoints, "valid_total_unweighted", f"Validation loss at step {endpoints.step.iloc[0]}", "validation_loss")
    print(f"Wrote figures to {figdir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    render(args.directory)


if __name__ == "__main__":
    main()
