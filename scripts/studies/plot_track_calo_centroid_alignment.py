"""Compare tracks and particle-number grouped calorimeter hits to targets.

The hit direction comes from the energy-weighted XYZ centroid, measured from the
origin. Track |p| is an energy proxy; grouped-hit energy is the sum of hit weights.
Paired track and hit residuals use exactly the same target particles.
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
from scripts.studies.plot_track_calo_track_resolution import angular_difference, matrix
from scripts.studies.validate_track_calo_targets import validate


COL = {name: i for i, name in enumerate(ParticleFeatures.get_names())}
DEFAULT_DETECTORS = ("cld", "clic", "idea", "colliderml")
ALL_DETECTORS = (*DEFAULT_DETECTORS, "maia")
PT_BINS = np.array([0, 0.1, 0.3, 1, 3, 10, 30, np.inf])
ENERGY_BINS = np.array([0, 0.1, 0.3, 1, 3, 10, 30, 100, np.inf])
GROUPS = {
    "paired_track": ("Track / target (paired)", "#2563a6"),
    "paired_centroid": ("Hit centroid / target (paired)", "#df8530"),
    "calo_only_centroid": ("Hit centroid / target (calo representative)", "#2c8a65"),
}
ENERGY_LABELS = {
    "paired_track": "Track |p| / target (paired)",
    "paired_centroid": "Σ hit E / target (paired)",
    "calo_only_centroid": "Σ hit E / target (calo representative)",
}
SIGNED_COLUMNS = (("deta", "Δη", 4.0), ("dphi", "Δφ [rad]", np.pi), ("log_energy_ratio", "log₁₀(input / target E)", 3.0))
FIELDS = ("deta", "dphi", "target_pt", "target_energy", "input_energy", "log_energy_ratio", "hit_count", "event", "target_id")


def energy_description(key, detector):
    if detector == "idea" and key != "paired_track":
        return ENERGY_LABELS[key].replace("Σ hit E", "Σ raw hit signal")
    return ENERGY_LABELS[key]


def empty_group():
    return {key: [] for key in FIELDS}


def append_group(group, *, deta, dphi, target_pt, target_energy, input_energy, hit_count, event, target_id):
    target_energy = np.asarray(target_energy, dtype=np.float64)
    input_energy = np.asarray(input_energy, dtype=np.float64)
    if not np.isfinite(target_energy).all() or not np.isfinite(input_energy).all() or np.any(target_energy <= 0) or np.any(input_energy <= 0):
        raise ValueError("Target energy and input energy proxy must be finite and positive")
    batch = dict(
        deta=deta,
        dphi=dphi,
        target_pt=target_pt,
        target_energy=target_energy,
        input_energy=input_energy,
        log_energy_ratio=np.log10(input_energy / target_energy),
        hit_count=hit_count,
        event=np.full(len(target_id), event, dtype=np.int64),
        target_id=target_id,
    )
    for key in FIELDS:
        group[key].append(np.asarray(batch[key]))


def load_detector(paths, detector):
    groups = {name: empty_group() for name in GROUPS}
    coverage = {
        "events": 0,
        "targets": 0,
        "track_representatives": 0,
        "paired_targets": 0,
        "track_targets_without_labelled_hits": 0,
        "calo_representatives": 0,
        "calo_targets_without_labelled_hits": 0,
        "labelled_hits": 0,
        "background_hits": 0,
        "zero_radius_centroids": 0,
    }
    for path in paths:
        validate(path)
        data = ak.from_parquet(path)
        if isinstance(data, ak.Record):
            data = ak.Array({name: data[name] for name in data.fields})
        for event in data:
            xtrack = matrix(event.X_track, 17 if detector == "colliderml" else 16)
            ytrack = matrix(event.ytarget_track_calo_track, len(COL))
            hits = matrix(event.X_hit_calo, 15)
            yhit = matrix(event.ytarget_track_calo_hit, len(COL))
            target = matrix(event.track_calo_target_particles, len(COL))
            n_target = len(target)
            hit_ids = yhit[:, COL["particle_number"]].astype(np.int64)
            if not np.array_equal(hit_ids, yhit[:, COL["particle_number"]]):
                raise ValueError(f"Noninteger hit particle_number in {path}, event {coverage['events']}")
            weights = hits[:, 5].astype(np.float64)
            if not np.isfinite(hits[:, 5:9]).all() or np.any(weights <= 0):
                raise ValueError(f"Nonpositive hit energy or nonfinite hit position in {path}, event {coverage['events']}")
            sum_weight = np.bincount(hit_ids, weights=weights, minlength=n_target + 1)
            hit_count = np.bincount(hit_ids, minlength=n_target + 1)
            weighted_xyz = np.column_stack([np.bincount(hit_ids, weights=weights * hits[:, axis], minlength=n_target + 1) for axis in (6, 7, 8)])
            xyz = np.divide(weighted_xyz, sum_weight[:, None], out=np.zeros_like(weighted_xyz), where=sum_weight[:, None] > 0)
            radius = np.hypot(xyz[:, 0], xyz[:, 1])
            valid_centroid = (sum_weight > 0) & (radius > 0)
            valid_centroid[0] = False
            coverage["zero_radius_centroids"] += int(np.count_nonzero((sum_weight[1:] > 0) & (radius[1:] == 0)))

            rep_rows = np.flatnonzero(ytrack[:, COL["PDG"]] != 0)
            rep_ids = ytrack[rep_rows, COL["particle_number"]].astype(np.int64)
            rep_has_hits = valid_centroid[rep_ids]
            paired_ids, paired_rows = rep_ids[rep_has_hits], rep_rows[rep_has_hits]
            all_ids = np.arange(1, n_target + 1)
            calo_ids = np.setdiff1d(all_ids, rep_ids, assume_unique=True)
            calo_has_hits = valid_centroid[calo_ids]
            coverage["events"] += 1
            coverage["targets"] += n_target
            coverage["track_representatives"] += len(rep_ids)
            coverage["paired_targets"] += len(paired_ids)
            coverage["track_targets_without_labelled_hits"] += int(np.count_nonzero(~rep_has_hits))
            coverage["calo_representatives"] += len(calo_ids)
            coverage["calo_targets_without_labelled_hits"] += int(np.count_nonzero(~calo_has_hits))
            coverage["labelled_hits"] += int(np.count_nonzero(hit_ids))
            coverage["background_hits"] += int(np.count_nonzero(hit_ids == 0))
            if not calo_has_hits.all():
                raise ValueError(f"Calo-represented target lacks a valid hit centroid in {path}, event {coverage['events'] - 1}")

            def target_direction(ids):
                selected = target[ids - 1]
                return (
                    selected[:, COL["eta"]],
                    np.arctan2(selected[:, COL["sin_phi"]], selected[:, COL["cos_phi"]]),
                    selected[:, COL["pt"]],
                    selected[:, COL["energy"]],
                )

            def centroid_residual(ids):
                eta, phi, pt, energy = target_direction(ids)
                centroid_eta = np.arcsinh(xyz[ids, 2] / radius[ids])
                centroid_phi = np.arctan2(xyz[ids, 1], xyz[ids, 0])
                return centroid_eta - eta, angular_difference(centroid_phi, phi), pt, energy

            if len(paired_ids):
                if np.any(np.hypot(xtrack[paired_rows, 3], xtrack[paired_rows, 4]) == 0):
                    raise ValueError(f"Undefined paired track direction in {path}, event {coverage['events'] - 1}")
                target_eta, target_phi, pt, energy = target_direction(paired_ids)
                append_group(
                    groups["paired_track"],
                    deta=xtrack[paired_rows, 2] - target_eta,
                    dphi=angular_difference(np.arctan2(xtrack[paired_rows, 3], xtrack[paired_rows, 4]), target_phi),
                    target_pt=pt,
                    target_energy=energy,
                    input_energy=xtrack[paired_rows, 5],
                    hit_count=hit_count[paired_ids],
                    event=coverage["events"] - 1,
                    target_id=paired_ids,
                )
                deta, dphi, pt, energy = centroid_residual(paired_ids)
                append_group(
                    groups["paired_centroid"],
                    deta=deta,
                    dphi=dphi,
                    target_pt=pt,
                    target_energy=energy,
                    input_energy=sum_weight[paired_ids],
                    hit_count=hit_count[paired_ids],
                    event=coverage["events"] - 1,
                    target_id=paired_ids,
                )
            if len(calo_ids):
                deta, dphi, pt, energy = centroid_residual(calo_ids)
                append_group(
                    groups["calo_only_centroid"],
                    deta=deta,
                    dphi=dphi,
                    target_pt=pt,
                    target_energy=energy,
                    input_energy=sum_weight[calo_ids],
                    hit_count=hit_count[calo_ids],
                    event=coverage["events"] - 1,
                    target_id=calo_ids,
                )
    for group in groups.values():
        for key in FIELDS:
            group[key] = np.concatenate(group[key]) if group[key] else np.array([])
    if not np.array_equal(groups["paired_track"]["target_id"], groups["paired_centroid"]["target_id"]):
        raise AssertionError("Paired track and centroid targets are not aligned")
    if not np.array_equal(groups["paired_track"]["target_energy"], groups["paired_centroid"]["target_energy"]):
        raise AssertionError("Paired track and centroid target energies are not aligned")
    return groups, coverage


def metrics(group):
    deta, dphi = group["deta"], group["dphi"]
    dr = np.hypot(deta, dphi)
    result = {"targets": len(dr)}
    if not len(dr):
        return result
    for name, values in (("eta", deta), ("phi", dphi), ("delta_r", dr)):
        result[name] = {
            "median_signed": float(np.median(values)),
            "q68_abs": float(np.quantile(np.abs(values), 0.68)),
            "q95_abs": float(np.quantile(np.abs(values), 0.95)),
        }
    log_response = group["log_energy_ratio"]
    result["energy_response"] = {
        "median_input_over_target": float(np.median(group["input_energy"] / group["target_energy"])),
        "median_log10_input_over_target": float(np.median(log_response)),
        "q68_abs_log10_input_over_target": float(np.quantile(np.abs(log_response), 0.68)),
        "q95_abs_log10_input_over_target": float(np.quantile(np.abs(log_response), 0.95)),
    }
    result["delta_r_lt_0p1_fraction"] = float(np.mean(dr < 0.1))
    result["delta_r_lt_0p3_fraction"] = float(np.mean(dr < 0.3))
    result["hit_count_median"] = float(np.median(group["hit_count"]))
    result["pt_bins"] = []
    for lower, upper in zip(PT_BINS[:-1], PT_BINS[1:]):
        mask = (group["target_pt"] >= lower) & (group["target_pt"] < upper)
        result["pt_bins"].append(
            {
                "lower": float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "targets": int(mask.sum()),
                **{
                    f"{name}_q68_abs": float(np.quantile(np.abs(values[mask]), 0.68)) if mask.any() else None
                    for name, values in (("eta", deta), ("phi", dphi))
                },
            }
        )
    result["energy_bins"] = []
    for lower, upper in zip(ENERGY_BINS[:-1], ENERGY_BINS[1:]):
        mask = (group["target_energy"] >= lower) & (group["target_energy"] < upper)
        result["energy_bins"].append(
            {
                "lower": float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "targets": int(mask.sum()),
                "q68_abs_log10_input_over_target": float(np.quantile(np.abs(log_response[mask]), 0.68)) if mask.any() else None,
                "q95_abs_log10_input_over_target": float(np.quantile(np.abs(log_response[mask]), 0.95)) if mask.any() else None,
            }
        )
    return result


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(fig)


def signed_histograms(data, directory, detectors, *, window, name):
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (field, label, limit) in enumerate(SIGNED_COLUMNS):
            ax = axes[row, column]
            width = limit if window is None else window
            bins = np.linspace(-width, width, 81 if window is not None else 161)
            for key, (description, color) in GROUPS.items():
                if field == "log_energy_ratio":
                    description = energy_description(key, detector)
                residual = data[detector][key][field]
                if len(residual):
                    outside = np.count_nonzero(np.abs(residual) > width)
                    ax.hist(
                        residual,
                        bins=bins,
                        weights=np.full(len(residual), 1 / len(residual)),
                        histtype="step",
                        lw=1.4,
                        color=color,
                        linestyle="--" if key == "calo_only_centroid" else "-",
                        label=f"{description}: {outside}/{len(residual)} outside",
                    )
            ax.axvline(0, color="0.6", lw=0.8)
            title = detector.upper() + (" · raw hit signals" if field == "log_energy_ratio" and detector == "idea" else "")
            ax.set(xlabel=label, ylabel="Fraction of group / bin", title=title, xlim=(-width, width))
            if window is None:
                ax.set_yscale("log")
            if row == 0:
                ax.legend(fontsize=7, loc="upper left")
    scale = "full angular range" if window is None else f"±{window:g}"
    fig.suptitle(
        f"Track and hit-group signed residuals: {scale}\nEnergy column: track |p| or Σ hit E (IDEA: raw signal); paired groups share targets",
        fontsize=14,
    )
    save(fig, directory, name)


def cdf(data, directory, detectors):
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (field, label) in enumerate((("deta", "|Δη|"), ("dphi", "|Δφ| [rad]"), ("log_energy_ratio", "|log₁₀(input / target E)|"))):
            ax = axes[row, column]
            for key, (description, color) in GROUPS.items():
                if field == "log_energy_ratio":
                    description = energy_description(key, detector)
                residual = np.abs(data[detector][key][field])
                if len(residual):
                    ordered = np.sort(np.maximum(residual, 1e-9))
                    ax.step(
                        ordered,
                        np.arange(1, len(ordered) + 1) / len(ordered),
                        where="post",
                        color=color,
                        linestyle="--" if key == "calo_only_centroid" else "-",
                        label=f"{description} (n={len(ordered)})",
                    )
            ax.axhline(0.68, color="0.75", lw=0.8)
            ax.axhline(0.95, color="0.75", lw=0.8)
            title = detector.upper() + (" · raw hit signals" if field == "log_energy_ratio" and detector == "idea" else "")
            ax.set(xscale="log", xlim=(1e-9, 10), ylim=(0, 1.01), xlabel=label, ylabel="Fraction within residual", title=title)
            if row == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Absolute angular and log-energy residuals to target\nZero residuals shown at 10⁻⁹; paired groups share target IDs", fontsize=14)
    save(fig, directory, "residual_cdf")


def signed_vs_pt(data, directory, detectors):
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (field, label, limit) in enumerate(SIGNED_COLUMNS):
            ax = axes[row, column]
            if field == "log_energy_ratio":
                energies = []
                for key in ("calo_only_centroid", "paired_centroid", "paired_track"):
                    _, color = GROUPS[key]
                    sample = data[detector][key]
                    if len(sample[field]):
                        ax.scatter(
                            sample["target_energy"],
                            sample["input_energy"],
                            s=8,
                            alpha=0.35,
                            color=color,
                            marker="x" if key == "calo_only_centroid" else "o",
                            linewidths=0.6,
                            label=f"{energy_description(key, detector)} (n={len(sample[field])})",
                            rasterized=True,
                        )
                        energies.extend((sample["target_energy"], sample["input_energy"]))
                if energies:
                    minimum = min(float(np.min(values)) for values in energies) * 0.8
                    maximum = max(float(np.max(values)) for values in energies) * 1.2
                    equal_label = "Track |p| = target E" if detector == "idea" else "Input = target E"
                    ax.plot([minimum, maximum], [minimum, maximum], color="0.4", lw=0.9, linestyle=":", label=equal_label)
                    ax.set(xlim=(minimum, maximum), ylim=(minimum, maximum))
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlabel="Target energy [GeV]",
                    ylabel="Track |p| [GeV] or Σ hit signal [a.u.]" if detector == "idea" else "Track |p| or Σ hit E [GeV]",
                    title=detector.upper() + (" · uncalibrated hit signals" if detector == "idea" else ""),
                )
                if row == 0:
                    ax.legend(fontsize=7, loc="upper left")
                continue
            for key in ("calo_only_centroid", "paired_centroid", "paired_track"):
                description, color = GROUPS[key]
                sample = data[detector][key]
                if len(sample[field]):
                    ax.scatter(
                        sample["target_pt"],
                        sample[field],
                        s=8,
                        alpha=0.35,
                        color=color,
                        marker="x" if key == "calo_only_centroid" else "o",
                        linewidths=0.6,
                        label=f"{description} (n={len(sample[field])})",
                        rasterized=True,
                    )
            ax.axhline(0, color="0.5", lw=0.8)
            ax.axhline(0.1, color="0.75", lw=0.8, linestyle=":")
            ax.axhline(-0.1, color="0.75", lw=0.8, linestyle=":")
            ax.set(xlabel="Target pT [GeV]", ylabel=label, title=detector.upper(), xscale="log", ylim=(-limit, limit))
            ax.set_yscale("symlog", linthresh=1e-4)
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Signed angular residuals versus pT and input versus target energy\nTrack |p| is an energy proxy; IDEA hit sums are raw signals", fontsize=14
    )
    save(fig, directory, "signed_residuals_vs_pt")


def resolution_vs_pt(data, directory, detectors):
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout="constrained")
    for row, detector in enumerate(detectors):
        for column, (field, label) in enumerate((("deta", "|Δη|"), ("dphi", "|Δφ| [rad]"), ("log_energy_ratio", "|log₁₀(input / target E)|"))):
            ax = axes[row, column]
            for key, (description, color) in GROUPS.items():
                if field == "log_energy_ratio":
                    description = energy_description(key, detector)
                sample = data[detector][key]
                centers, q68, q95 = [], [], []
                bin_field = "target_energy" if field == "log_energy_ratio" else "target_pt"
                bins = ENERGY_BINS if field == "log_energy_ratio" else PT_BINS
                for lower, upper in zip(bins[:-1], bins[1:]):
                    mask = (sample[bin_field] >= lower) & (sample[bin_field] < upper)
                    if mask.sum() < 10:
                        continue
                    centers.append(float(np.median(sample[bin_field][mask])))
                    q68.append(float(np.quantile(np.abs(sample[field][mask]), 0.68)))
                    q95.append(float(np.quantile(np.abs(sample[field][mask]), 0.95)))
                if centers:
                    ax.plot(centers, np.maximum(q68, 1e-9), "o-", ms=4, lw=1.4, color=color, label=description)
                    ax.plot(centers, np.maximum(q95, 1e-9), "--", lw=0.9, color=color)
            ax.axhline(0.1, color="0.7", lw=0.8, linestyle=":")
            title = detector.upper() + (" · raw hit signals" if field == "log_energy_ratio" and detector == "idea" else "")
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Target energy [GeV]" if field == "log_energy_ratio" else "Target pT [GeV]",
                ylabel=label,
                title=title,
            )
            if row == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        "Angular alignment and input-to-target spread versus target scale\nSolid: 68% |residual|; dashed: 95%; IDEA hit sums are uncalibrated",
        fontsize=14,
    )
    save(fig, directory, "resolution_vs_pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("local_test_data/track_calo_targets"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/track_calo_targets/centroid_alignment"))
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
        data[detector], coverage = load_detector(paths, detector)
        report[detector] = {
            "paths": [str(path) for path in paths],
            "coverage": coverage,
            "energy_units": {
                "target": "GeV",
                "track_proxy": "|p| in GeV",
                "hit_sum": "raw digitized signal (uncalibrated)" if detector == "idea" else "GeV",
            },
            **{key: metrics(group) for key, group in data[detector].items()},
        }
    signed_histograms(data, args.output_dir, args.detectors, window=0.01, name="signed_residuals_zoom")
    signed_histograms(data, args.output_dir, args.detectors, window=0.1, name="signed_residuals")
    signed_histograms(data, args.output_dir, args.detectors, window=None, name="signed_residuals_wide")
    cdf(data, args.output_dir, args.detectors)
    signed_vs_pt(data, args.output_dir, args.detectors)
    resolution_vs_pt(data, args.output_dir, args.detectors)
    (args.output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote six figures (PNG + SVG) and alignment.json to {args.output_dir}")


if __name__ == "__main__":
    main()
