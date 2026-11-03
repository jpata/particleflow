"""Aggregate detector matching validation, paired intervals and direction plots."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.studies.validate_detector_matching import summarize

DETECTORS = ["cld", "clic", "idea", "cms"]
LABELS = {"native": "Native assignment", "majority_energy": "Majority tracks + energy matching"}


def report(output):
    summaries = []
    events = []
    centroids = []
    for d in DETECTORS:
        summarize(SimpleNamespace(detector=d, output=output / d))
        summaries.extend(json.loads((output / d / "summary.json").read_text()))
        events.append(pd.read_csv(output / d / "events.csv"))
        if d in ["cld", "clic"]:
            arrays = ak.from_parquet(output / d / "native.parquet", columns=["X_cluster"])
            for i in range(len(arrays["X_cluster"])):
                x = np.asarray(ak.to_numpy(arrays["X_cluster"][i]))
                eta = np.arcsinh(x[:, 8] / np.maximum(np.hypot(x[:, 6], x[:, 7]), 1e-9))
                phi = np.arctan2(x[:, 7], x[:, 6])
                data = pd.read_parquet(output / d / "events" / f"{i:05d}.parquet")
                data = data[data.kind == 1]
                for rule, z in data.groupby("rule"):
                    ids = z.object.to_numpy(dtype=int)
                    dp = z.phi.to_numpy() - phi[ids]
                    dr = np.hypot(z.eta.to_numpy() - eta[ids], np.arctan2(np.sin(dp), np.cos(dp)))
                    centroids.append(
                        dict(
                            detector=d,
                            event_id=i,
                            rule=rule,
                            count=len(z),
                            bad=int((dr > 0.3).sum()),
                            pt5_count=int((z.pt >= 5).sum()),
                            pt5_bad=int(((dr > 0.3) & (z.pt.to_numpy() >= 5)).sum()),
                        )
                    )
    s = pd.DataFrame(summaries)
    s.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    e = pd.concat(events, ignore_index=True)
    assert e.groupby(["detector", "rule"]).size().eq(50).all()
    rng = np.random.default_rng(12345)
    intervals = []
    for d in DETECTORS:
        candidate = e[(e.detector == d) & (e.rule == "majority_energy")].sort_values("event_id")
        idx = rng.integers(0, len(candidate), (5000, len(candidate)))
        for baseline in ["native"]:
            native = e[(e.detector == d) & (e.rule == baseline)].sort_values("event_id")
            assert np.array_equal(candidate.event_id, native.event_id)
            for name, num, den in [
                ("jet_recall_delta_pp", "matched_jets", "reference_jets"),
                ("jet_response_recall_delta_pp", "response_jets", "reference_jets"),
                ("track_failure_delta_pp", "track_bad", "track_targets"),
                ("cluster_failure_delta_pp", "cluster_bad", "cluster_targets"),
            ]:
                sn, sd = candidate[num].values, candidate[den].values
                bn, bd = native[num].values, native[den].values
                boot = 100 * (sn[idx].sum(axis=1) / sd[idx].sum(axis=1) - bn[idx].sum(axis=1) / bd[idx].sum(axis=1))
                lo, hi = np.quantile(boot, [0.025, 0.975])
                intervals.append(
                    dict(detector=d, baseline=baseline, metric=name, estimate=100 * (sn.sum() / sd.sum() - bn.sum() / bd.sum()), low95=lo, high95=hi)
                )
    pd.DataFrame(intervals).to_csv(output / "paired_intervals.csv", index=False)
    pd.DataFrame(centroids).to_csv(output / "centroid_events.csv", index=False)
    cs = []
    for (d, r), z in pd.DataFrame(centroids).groupby(["detector", "rule"]):
        cs.append(
            dict(detector=d, rule=r, bad_fraction=float(z.bad.sum() / z["count"].sum()), pt5_bad_fraction=float(z.pt5_bad.sum() / z.pt5_count.sum()))
        )
    pd.DataFrame(cs).to_csv(output / "centroid_summary.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for row, d in enumerate(DETECTORS):
        objects = pd.concat([pd.read_parquet(path) for path in sorted((output / d / "events").glob("*.parquet"))], ignore_index=True)
        for col, kind in enumerate([0, 1]):
            ax = axes[row, col]
            for rule, label in LABELS.items():
                dr = np.sort(objects[(objects.rule == rule) & (objects.kind == kind)].delta_r.values)
                ax.plot(dr, np.arange(1, len(dr) + 1) / len(dr), label=label)
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 10)
            ax.set_ylim(0, 1)
            ax.axvline(0.1 if kind == 0 else 0.3, color="black", ls="--", lw=1)
            ax.set_title(f'{d.upper()}: {"track" if kind == 0 else "cluster"} targets')
            ax.set_xlabel("Delta-R between input and target")
            ax.set_ylabel("Fraction of targets")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "direction_cdf.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    report(parser.parse_args().input)
