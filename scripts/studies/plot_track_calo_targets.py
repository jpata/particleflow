"""Plot validated track/calo Parquet targets; these are truth-label diagnostics, not model predictions."""

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
TRACK, CALO, LEGACY = "#2563a6", "#df8530", "#525866"
DETECTORS = ("cld", "clic", "maia", "colliderml")


def matrix(value, width=14):
    return ak.to_numpy(value).reshape(-1, width)


def read_sample(paths):
    events, reports = [], []
    for path in paths:
        reports.append(validate(path))
        data = ak.from_parquet(path)
        if isinstance(data, ak.Record):
            data = ak.Array({key: data[key] for key in data.fields})
        for event in data:
            target = matrix(event.track_calo_target_particles)
            track, calo = matrix(event.ytarget_track_calo_track), matrix(event.ytarget_track_calo_hit)
            legacy = np.concatenate((matrix(event.ytarget_track), matrix(event.ytarget_cluster)))
            legacy = legacy[legacy[:, 0] != 0]
            tracked = np.zeros(len(target), dtype=bool)
            tracked[track[track[:, 0] != 0, COL["particle_number"]].astype(int) - 1] = True
            membership = np.bincount(calo[:, COL["particle_number"]].astype(int), minlength=len(target) + 1)[1:]
            events.append(
                dict(
                    target=target,
                    tracked=tracked,
                    legacy=legacy,
                    membership=membership,
                    accounting=ak.to_numpy(event.track_calo_energy_accounting),
                    raw=event,
                )
            )
    return events, reports


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    plt.close(fig)


def overview(samples, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = [f"{d.upper()}\n{len(samples[d])} events" for d in DETECTORS]
    x = np.arange(4)
    legacy = np.array([sum(len(e["legacy"]) for e in samples[d]) for d in DETECTORS])
    tracks = np.array([sum(e["tracked"].sum() for e in samples[d]) for d in DETECTORS])
    calo = np.array([sum((~e["tracked"]).sum() for e in samples[d]) for d in DETECTORS])
    ax = axes[0, 0]
    ax.bar(x - 0.2, legacy, 0.36, color=LEGACY, label="Legacy tracks + clusters")
    ax.bar(x + 0.2, tracks, 0.36, color=TRACK, label="New: track representative")
    ax.bar(x + 0.2, calo, 0.36, bottom=tracks, color=CALO, label="New: calo-hit representative")
    for i in x:
        ax.text(i - 0.2, legacy[i] + 9, str(legacy[i]), ha="center", fontsize=9)
        ax.text(i + 0.2, tracks[i] + calo[i] + 9, str(tracks[i] + calo[i]), ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Total target particles", title="Target population")
    ax.margins(y=0.2)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    for i, d in enumerate(DETECTORS):
        old = [len(e["legacy"]) for e in samples[d]]
        new = [len(e["target"]) for e in samples[d]]
        ax.scatter(old, new, s=42, label=d.upper(), color=plt.get_cmap("tab10")(i))
    upper = max(max(len(e["target"]), len(e["legacy"])) for events in samples.values() for e in events) * 1.12
    ax.plot([0, upper], [0, upper], "--", color="0.6", lw=1)
    ax.set(xlabel="Legacy particles / event", ylabel="New particles / event", title="Paired events", xlim=(0, upper), ylim=(0, upper))
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for i, d in enumerate(DETECTORS):
        values = [
            100 * (e["target"][:, 6].astype(float).sum() - e["legacy"][:, 6].astype(float).sum()) / max(e["accounting"][0], 1e-12) for e in samples[d]
        ]
        offset = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(i + offset, values, s=38, color=plt.get_cmap("tab10")(i))
    ax.axhline(0, color="0.6", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=labels, ylabel="(New − legacy) target energy / input [%]", title="Energy retained relative to legacy targets")
    ax = axes[1, 1]
    for i, d in enumerate(DETECTORS):
        values = [abs(e["accounting"][0] - e["accounting"][1:].sum()) / max(e["accounting"][0], 1e-12) for e in samples[d]]
        ax.scatter(i + np.linspace(-0.13, 0.13, len(values)), values, s=38, color=plt.get_cmap("tab10")(i))
    ax.set(xticks=x, xticklabels=labels, ylabel="Absolute fractional closure residual", title="Input = target + invisible + dropped")
    ax.set_ylim(-5e-14, 1e-12)
    residuals = [abs(e["accounting"][0] - e["accounting"][1:].sum()) / max(e["accounting"][0], 1e-12) for events in samples.values() for e in events]
    ax.set_ylim(-5e-14, max(1e-12, max(residuals) * 1.3))
    ax.text(0.5, 0.8, f"Maximum residual: {max(residuals):.2g} ({len(residuals)} events)", transform=ax.transAxes, ha="center", fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Track + calorimeter-hit target validation\nSmall input samples; truth-label checks, not reconstruction performance", fontsize=14)
    save(fig, output, "overview")


def kinematics(samples, output):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for row, d in enumerate(DETECTORS):
        target = np.concatenate([e["target"] for e in samples[d]])
        tracked = np.concatenate([e["tracked"] for e in samples[d]])
        legacy = np.concatenate([e["legacy"] for e in samples[d]])
        for column, (field, label) in enumerate((("pt", "Target transverse momentum [GeV]"), ("eta", "Target pseudorapidity"))):
            ax = axes[row, column]
            values = np.concatenate((target[:, COL[field]], legacy[:, COL[field]]))
            if field == "pt":
                positive = values[values > 0]
                bins = np.geomspace(max(float(positive.min()) * 0.8, 1e-7), float(positive.max()) * 1.2, 30)
                ax.set_xscale("log")
            else:
                limit = max(5, np.ceil(np.max(np.abs(values))))
                bins = np.linspace(-limit, limit, 36)
            ax.hist(
                [target[tracked, COL[field]], target[~tracked, COL[field]]],
                bins=bins,
                stacked=True,
                color=[TRACK, CALO],
                label=["New: track", "New: calo hit"],
                alpha=0.8,
            )
            ax.hist(legacy[:, COL[field]], bins=bins, histtype="step", color=LEGACY, lw=1.6, label="Legacy")
            ax.set(xlabel=label, ylabel="Particles / bin", title=f"{d.upper()} · {len(samples[d])} events")
            if row == 0 and column == 0:
                ax.legend(fontsize=9)
    fig.suptitle("Target kinematics by representative type\nCounts are unnormalized; detector samples have different collision energies", fontsize=14)
    save(fig, output, "kinematics")


def membership(samples, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, d in zip(axes.flat, DETECTORS):
        counts = np.concatenate([e["membership"] for e in samples[d]])
        tracked = np.concatenate([e["tracked"] for e in samples[d]])
        bins = np.unique(np.concatenate(([-0.5, 0.5], np.geomspace(1.5, max(2, counts.max() + 1), 25))))
        ax.hist(counts[tracked], bins=bins, histtype="step", color=TRACK, lw=1.7, label="Track-represented targets")
        ax.hist(counts[~tracked], bins=bins, histtype="step", color=CALO, lw=1.7, label="Calo-represented targets")
        ax.set_xscale("symlog", linthresh=1)
        ax.margins(y=0.5)
        ax.set(xlabel="Calorimeter hits carrying target particle_number", ylabel="Particles / bin", title=f"{d.upper()} · {len(samples[d])} events")
        raw_count = sum(len(e["raw"].X_hit_calo) for e in samples[d])
        assigned = int(counts.sum())
        ax.text(
            0.97,
            0.95,
            f"Labelled calo hits: {assigned:,} / {raw_count:,}\nTrack targets with zero calo hits: {np.sum(tracked & (counts == 0))}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.82), fontsize=8)
    fig.suptitle(
        "Calorimeter-hit membership labels\nRepresentative labels override dominant-contributor labels; these are not purity measurements",
        fontsize=14,
    )
    save(fig, output, "membership")


def event_display(samples, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    colors = plt.get_cmap("tab10")
    rng = np.random.default_rng(42)
    for ax, d in zip(axes.flat, DETECTORS):
        event = samples[d][0]
        raw, target = event["raw"], event["target"]
        hits = matrix(raw.X_hit_calo, 15)
        labels = matrix(raw.ytarget_track_calo_hit)[:, COL["particle_number"]].astype(int)
        xyz = hits[:, 6:9]
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        eta = np.arcsinh(xyz[:, 2] / np.maximum(radius, 1e-9))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        top = np.argsort(target[:, COL["energy"]])[-8:][::-1]
        background = np.flatnonzero(~np.isin(labels, top + 1))
        if len(background) > 10000:
            background = rng.choice(background, 10000, replace=False)
        ax.scatter(eta[background], phi[background], c="0.8", s=2, alpha=0.3, rasterized=True)
        tracks = ak.to_numpy(raw.X_track)
        track_labels = matrix(raw.ytarget_track_calo_track)[:, COL["particle_number"]].astype(int)
        for rank, index in enumerate(top):
            color = colors(rank if rank < 7 else 9)
            selected = labels == index + 1
            ax.scatter(eta[selected], phi[selected], s=4, color=color, alpha=0.65, rasterized=True)
            ax.scatter(
                target[index, COL["eta"]],
                np.arctan2(target[index, COL["sin_phi"]], target[index, COL["cos_phi"]]),
                marker="*",
                s=130,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=f"{index + 1}: {target[index, COL['energy']]:.1f} GeV",
            )
            selected_tracks = track_labels == index + 1
            if selected_tracks.any():
                ax.scatter(
                    tracks[selected_tracks, 2],
                    np.arctan2(tracks[selected_tracks, 3], tracks[selected_tracks, 4]),
                    color=color,
                    marker="x",
                    s=40,
                    linewidth=1.3,
                )
        ax.set(
            xlabel="η (hit position / track or target direction)",
            ylabel="φ [rad]",
            ylim=(-np.pi, np.pi),
            title=f"{d.upper()} · event 0 · {len(target)} targets",
        )
        ax.legend(title="Target ID: energy", fontsize=7, title_fontsize=8, loc="upper right", ncol=2)
    fig.suptitle(
        "First-event label displays: eight highest-energy targets\nDots: measured calo-hit positions · ×: reconstructed track direction · ★: truth target direction",
        fontsize=13,
    )
    save(fig, output, "event_display")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("local_test_data/track_calo_targets"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/track_calo_targets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    samples, reports = {}, {}
    for d in DETECTORS:
        paths = sorted((args.input_dir / d).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No converted {d} samples under {args.input_dir}")
        samples[d], reports[d] = read_sample(paths)
    for plot in (overview, kinematics, membership, event_display):
        plot(samples, args.output_dir)
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Wrote four figures (PNG + SVG) and validation.json to {args.output_dir}")


if __name__ == "__main__":
    main()
