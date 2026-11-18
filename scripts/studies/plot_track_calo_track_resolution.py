"""Check track/target eta and phi alignment in the track + calo Parquet view.

Full target rows are representative tracks. Rows carrying only particle_number
are membership links; IDEA descendant tracks make this distinction essential.
"""

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlpf.conf import ParticleFeatures
from scripts.studies.validate_track_calo_targets import validate


COL = {name: i for i, name in enumerate(ParticleFeatures.get_names())}
DEFAULT_DETECTORS = ("cld", "clic", "idea", "colliderml")
ALL_DETECTORS = (*DEFAULT_DETECTORS, "maia")
PT_BINS = np.array([0, 0.1, 0.3, 1, 3, 10, 30, np.inf])
COLORS = {"representative": "#2563a6", "membership_only": "#df8530"}


def matrix(value, width):
    return np.asarray(ak.to_numpy(value)).reshape(-1, width)


def angular_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def load_detector(paths, detector):
    values = {kind: {key: [] for key in ("deta", "dphi", "target_pt", "track_pt", "event", "track", "pdg")} for kind in COLORS}
    legacy_comparison = {"new_delta_r_gt_0p3": 0, "legacy_full_on_same_track": 0, "legacy_delta_r_gt_0p3": 0, "same_target_kinematics": 0}
    events = 0
    for path in paths:
        validate(path)
        data = ak.from_parquet(path)
        if isinstance(data, ak.Record):
            data = ak.Array({name: data[name] for name in data.fields})
        for event in data:
            track = matrix(event.X_track, 17 if detector == "colliderml" else 16)
            labels = matrix(event.ytarget_track_calo_track, len(COL))
            legacy = matrix(event.ytarget_track, len(COL))
            targets = matrix(event.track_calo_target_particles, len(COL))
            ids = labels[:, COL["particle_number"]].astype(np.int64)
            assigned = ids > 0
            if assigned.any():
                norm = np.hypot(track[assigned, 3], track[assigned, 4])
                if not np.isfinite(track[assigned, :5]).all() or np.any(norm == 0):
                    raise ValueError(f"Nonfinite or undefined assigned track direction in {path}, event {events}")
            for kind, mask in (
                ("representative", assigned & (labels[:, COL["PDG"]] != 0)),
                ("membership_only", assigned & (labels[:, COL["PDG"]] == 0)),
            ):
                if not mask.any():
                    continue
                selected = targets[ids[mask] - 1]
                track_phi = np.arctan2(track[mask, 3], track[mask, 4])
                target_phi = np.arctan2(selected[:, COL["sin_phi"]], selected[:, COL["cos_phi"]])
                batch = {
                    "deta": track[mask, 2] - selected[:, COL["eta"]],
                    "dphi": angular_difference(track_phi, target_phi),
                    "target_pt": selected[:, COL["pt"]],
                    "track_pt": track[mask, 1],
                    "event": np.full(mask.sum(), events, dtype=np.int64),
                    "track": np.flatnonzero(mask),
                    "pdg": selected[:, COL["PDG"]].astype(np.int64),
                }
                if kind == "representative":
                    delta_r = np.hypot(batch["deta"], batch["dphi"])
                    outlier_rows = np.flatnonzero(mask)[delta_r > 0.3]
                    legacy_comparison["new_delta_r_gt_0p3"] += len(outlier_rows)
                    old_rows = outlier_rows[legacy[outlier_rows, COL["PDG"]] != 0]
                    legacy_comparison["legacy_full_on_same_track"] += len(old_rows)
                    if len(old_rows):
                        old_eta = track[old_rows, 2] - legacy[old_rows, COL["eta"]]
                        old_phi = angular_difference(
                            np.arctan2(track[old_rows, 3], track[old_rows, 4]),
                            np.arctan2(legacy[old_rows, COL["sin_phi"]], legacy[old_rows, COL["cos_phi"]]),
                        )
                        legacy_comparison["legacy_delta_r_gt_0p3"] += int(np.count_nonzero(np.hypot(old_eta, old_phi) > 0.3))
                        new_rows = labels[old_rows, 2:7]
                        legacy_comparison["same_target_kinematics"] += int(
                            np.count_nonzero(np.all(np.isclose(new_rows, legacy[old_rows, 2:7], atol=1e-5), axis=1))
                        )
                for key, value in batch.items():
                    values[kind][key].append(value)
            events += 1
    for kind in values:
        for key, chunks in values[kind].items():
            values[kind][key] = np.concatenate(chunks) if chunks else np.array([])
    return values, events, legacy_comparison


def metrics(values):
    deta, dphi = values["deta"], values["dphi"]
    dr = np.hypot(deta, dphi)
    result = {"tracks": len(dr)}
    if not len(dr):
        return result
    for name, residual in (("eta", deta), ("phi", dphi), ("delta_r", dr)):
        absolute = np.abs(residual)
        result[name] = {
            "median_signed": float(np.median(residual)),
            "q68_abs": float(np.quantile(absolute, 0.68)),
            "q95_abs": float(np.quantile(absolute, 0.95)),
        }
    result["delta_r_lt_0p1_fraction"] = float(np.mean(dr < 0.1))
    result["delta_r_gt_0p3_count"] = int(np.count_nonzero(dr > 0.3))
    result["abs_dphi_gt_3_count"] = int(np.count_nonzero(np.abs(dphi) > 3))
    result["pt_bins"] = []
    for lower, upper in zip(PT_BINS[:-1], PT_BINS[1:]):
        mask = (values["target_pt"] >= lower) & (values["target_pt"] < upper)
        result["pt_bins"].append(
            {
                "lower": float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "tracks": int(mask.sum()),
                "eta_q68_abs": float(np.quantile(np.abs(deta[mask]), 0.68)) if mask.any() else None,
                "eta_q95_abs": float(np.quantile(np.abs(deta[mask]), 0.95)) if mask.any() else None,
                "phi_q68_abs": float(np.quantile(np.abs(dphi[mask]), 0.68)) if mask.any() else None,
                "phi_q95_abs": float(np.quantile(np.abs(dphi[mask]), 0.95)) if mask.any() else None,
            }
        )
    outliers = np.argsort(dr)[-10:][::-1]
    result["largest_delta_r"] = [
        {
            "event": int(values["event"][i]),
            "track": int(values["track"][i]),
            "pdg": int(values["pdg"][i]),
            "track_pt": float(values["track_pt"][i]),
            "target_pt": float(values["target_pt"][i]),
            "delta_eta": float(deta[i]),
            "delta_phi": float(dphi[i]),
            "delta_r": float(dr[i]),
        }
        for i in outliers
    ]
    return result


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(fig)


def plot_cdf(data, directory, detectors):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (key, label) in enumerate((("deta", "|Δη|"), ("dphi", "|Δφ| [rad]"))):
            ax = axes[row, column]
            for kind, description in (("representative", "Representative"), ("membership_only", "Membership only")):
                residual = np.abs(data[detector][kind][key])
                if len(residual):
                    ordered = np.sort(np.maximum(residual, 1e-9))
                    ax.step(
                        ordered,
                        np.arange(1, len(ordered) + 1) / len(ordered),
                        where="post",
                        color=COLORS[kind],
                        linestyle="-" if kind == "representative" else "--",
                        label=f"{description} (n={len(ordered)})",
                    )
            ax.axhline(0.68, color="0.7", lw=0.8)
            ax.axhline(0.95, color="0.7", lw=0.8)
            ax.set(xscale="log", xlim=(1e-9, 4), ylim=(0, 1.01), xlabel=label, ylabel="Fraction within residual", title=detector.upper())
            if row == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Track direction against assigned target direction\nFull representatives and membership links are separate; zero residuals shown at 10⁻⁹",
        fontsize=14,
    )
    save(fig, directory, "residual_cdf")


def plot_signed(data, directory, detectors, *, window=0.1, name="signed_residuals"):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    bins = np.linspace(-window, window, 81)
    for row, detector in enumerate(detectors):
        for column, (key, label) in enumerate((("deta", "Δη"), ("dphi", "Δφ [rad]"))):
            ax = axes[row, column]
            for kind, description in (("representative", "Representative"), ("membership_only", "Membership only")):
                residual = data[detector][kind][key]
                if len(residual):
                    outside = np.count_nonzero(np.abs(residual) > window)
                    ax.hist(
                        residual,
                        bins=bins,
                        weights=np.full(len(residual), 1 / len(residual)),
                        histtype="step",
                        lw=1.4,
                        color=COLORS[kind],
                        linestyle="-" if kind == "representative" else "--",
                        label=f"{description}: {outside}/{len(residual)} outside",
                    )
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set(xlabel=f"Track − target {label}", ylabel="Fraction of group / bin", title=detector.upper(), xlim=(-window, window))
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Signed track/target residuals in the ±{window:g} core\nLegend gives counts outside the displayed range", fontsize=14)
    save(fig, directory, name)


def plot_signed_wide(data, directory, detectors):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (key, label, limit) in enumerate((("deta", "Δη", 4.0), ("dphi", "Δφ [rad]", np.pi))):
            ax = axes[row, column]
            bins = np.linspace(-limit, limit, 161)
            for kind, description in (("representative", "Representative"), ("membership_only", "Membership only")):
                residual = data[detector][kind][key]
                if len(residual):
                    outside = np.count_nonzero(np.abs(residual) > limit)
                    ax.hist(
                        residual,
                        bins=bins,
                        weights=np.full(len(residual), 1 / len(residual)),
                        histtype="step",
                        lw=1.5,
                        color=COLORS[kind],
                        linestyle="-" if kind == "representative" else "--",
                        label=f"{description}: {outside}/{len(residual)} outside",
                    )
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set(xlabel=f"Track − target {label}", ylabel="Fraction of group / bin", title=detector.upper(), xlim=(-limit, limit), yscale="log")
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Signed track/target residuals across the full angular range\nLog fraction axis exposes low-population tails", fontsize=14)
    save(fig, directory, "signed_residuals_wide")


def plot_signed_vs_pt(data, directory, detectors):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (key, label, limit) in enumerate((("deta", "Δη", 4.0), ("dphi", "Δφ [rad]", np.pi))):
            ax = axes[row, column]
            for kind, description in (("membership_only", "Membership only"), ("representative", "Representative")):
                sample = data[detector][kind]
                if len(sample[key]):
                    ax.scatter(
                        sample["target_pt"],
                        sample[key],
                        s=9,
                        alpha=0.4,
                        color=COLORS[kind],
                        marker="x" if kind == "membership_only" else "o",
                        linewidths=0.7,
                        label=f"{description} (n={len(sample[key])})",
                        rasterized=True,
                    )
            ax.axhline(0, color="0.5", lw=0.8)
            ax.axhline(0.1, color="0.75", lw=0.8, linestyle=":")
            ax.axhline(-0.1, color="0.75", lw=0.8, linestyle=":")
            ax.set(xlabel="Target pT [GeV]", ylabel=f"Track − target {label}", title=detector.upper(), xscale="log", ylim=(-limit, limit))
            ax.set_yscale("symlog", linthresh=1e-4)
            if row == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Individual signed track/target residuals versus target pT\nSymmetric-log residual axis resolves the core and large tails", fontsize=14
    )
    save(fig, directory, "signed_residuals_vs_pt")


def plot_vs_pt(data, directory, detectors):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        sample = data[detector]["representative"]
        for column, (key, label) in enumerate((("deta", "|Δη|"), ("dphi", "|Δφ| [rad]"))):
            ax = axes[row, column]
            centers, q68, q95 = [], [], []
            for lower, upper in zip(PT_BINS[:-1], PT_BINS[1:]):
                mask = (sample["target_pt"] >= lower) & (sample["target_pt"] < upper)
                if mask.sum() < 10:
                    continue
                centers.append(float(np.median(sample["target_pt"][mask])))
                q68.append(float(np.quantile(np.abs(sample[key][mask]), 0.68)))
                q95.append(float(np.quantile(np.abs(sample[key][mask]), 0.95)))
            ax.plot(centers, np.maximum(q68, 1e-9), "o-", color=COLORS["representative"], label="68% |residual|")
            ax.plot(centers, np.maximum(q95, 1e-9), "s--", color="#b35434", label="95% |residual|")
            ax.axhline(0.1, color="0.6", lw=0.8, linestyle=":")
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Target pT [GeV]",
                ylabel=label,
                title=f"{detector.upper()} · {len(sample['deta'])} representative tracks",
            )
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Representative-track angular resolution versus target pT\nMinimum 10 tracks per bin; zero residuals displayed at 10⁻⁹", fontsize=14)
    save(fig, directory, "resolution_vs_pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("local_test_data/track_calo_targets"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/track_calo_targets/track_resolution"))
    parser.add_argument("--detectors", nargs=4, choices=ALL_DETECTORS, default=DEFAULT_DETECTORS)
    args = parser.parse_args()
    if len(set(args.detectors)) != 4:
        parser.error("--detectors requires four distinct detector names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    data, report = {}, {}
    for detector in args.detectors:
        paths = sorted((args.input_dir / detector).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No {detector} Parquet files in {args.input_dir / detector}")
        data[detector], events, legacy_comparison = load_detector(paths, detector)
        report[detector] = {
            "paths": [str(p) for p in paths],
            "events": events,
            "outlier_legacy_comparison": legacy_comparison,
            **{kind: metrics(data[detector][kind]) for kind in COLORS},
        }
    plot_cdf(data, args.output_dir, args.detectors)
    plot_signed(data, args.output_dir, args.detectors)
    plot_signed(data, args.output_dir, args.detectors, window=0.01, name="signed_residuals_zoom")
    plot_signed_wide(data, args.output_dir, args.detectors)
    plot_signed_vs_pt(data, args.output_dir, args.detectors)
    plot_vs_pt(data, args.output_dir, args.detectors)
    (args.output_dir / "resolution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote six figures (PNG + SVG) and resolution.json to {args.output_dir}")


if __name__ == "__main__":
    main()
