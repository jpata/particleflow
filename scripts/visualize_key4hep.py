#!/usr/bin/env python3
"""Render perspective Key4HEP event displays from EDM4hep ROOT files.

The display overlays reconstructed tracks, calorimeter clusters, detector hits,
and stable generator-level particles. Detector type is inferred from the EDM4hep
collections unless explicitly selected. Adapted from erwulff/particlemind's
``notebooks/cld-visualize.ipynb``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


PARTICLE_STYLES = {
    11: ("electron", "#17becf"),
    13: ("muon", "#9467bd"),
    22: ("photon", "#e6b800"),
    130: ("neutral hadron", "#ff7f0e"),
    211: ("charged hadron", "#e377c2"),
}


@dataclass(frozen=True)
class DetectorConfig:
    key: str
    title: str
    track_collection: str
    track_label: str
    cluster_collection: str
    hit_collections: tuple[tuple[str, str, str], ...]
    track_radius: float
    track_half_z: float
    particle_barrel_radius: float
    particle_endcap_z: float
    particle_max_length: float
    plot_limit: float
    cluster_size_base: float
    cluster_size_scale: float
    cluster_size_max: float
    cluster_alpha: float


DETECTORS = {
    "cld": DetectorConfig(
        key="cld",
        title="CLD",
        track_collection="SiTracks_Refitted",
        track_label="Reconstructed tracks",
        cluster_collection="PandoraClusters",
        hit_collections=(
            ("VXDTrackerHits", "Tracker hits", "#d62728"),
            ("VXDEndcapTrackerHits", "Tracker hits", "#d62728"),
            ("ITrackerHits", "Tracker hits", "#d62728"),
            ("OTrackerHits", "Tracker hits", "#d62728"),
            ("ITrackerEndcapHits", "Tracker hits", "#d62728"),
            ("OTrackerEndcapHits", "Tracker hits", "#d62728"),
            ("ECALBarrel", "ECAL hits", "#1f77b4"),
            ("ECALEndcap", "ECAL hits", "#1f77b4"),
            ("HCALBarrel", "HCAL hits", "#2ca02c"),
            ("HCALEndcap", "HCAL hits", "#2ca02c"),
            ("HCALOther", "HCAL hits", "#2ca02c"),
            ("MUON", "Muon hits", "#ff7f0e"),
        ),
        track_radius=2300.0,
        track_half_z=3500.0,
        particle_barrel_radius=2050.0,
        particle_endcap_z=2300.0,
        particle_max_length=3600.0,
        plot_limit=4200.0,
        cluster_size_base=5.0,
        cluster_size_scale=2.0,
        cluster_size_max=25.0,
        cluster_alpha=0.9,
    ),
    "idea": DetectorConfig(
        key="idea",
        title="IDEA",
        track_collection="TracksFromGenParticles",
        track_label="Truth-seeded tracks",
        cluster_collection="TopoClusterAll",
        hit_collections=(
            ("VTXBDigis", "Tracker hits", "#d62728"),
            ("VTXDDigis", "Tracker hits", "#d62728"),
            ("DCH_DigiCollection", "Tracker hits", "#d62728"),
            ("SiWrBDigis", "Tracker hits", "#d62728"),
            ("SiWrDDigis", "Tracker hits", "#d62728"),
            ("TopoClusterAllCells", "Dual-readout calorimeter hits", "#1f77b4"),
            ("PreshowerSystemCollection", "Preshower hits", "#2ca02c"),
            ("MuonSystemCollection", "Muon hits", "#ff7f0e"),
        ),
        track_radius=2150.0,
        track_half_z=2450.0,
        particle_barrel_radius=2470.0,
        particle_endcap_z=2500.0,
        particle_max_length=4400.0,
        plot_limit=6500.0,
        cluster_size_base=1.5,
        cluster_size_scale=0.8,
        cluster_size_max=10.0,
        cluster_alpha=0.45,
    ),
}


def _detector_config(tree, detector: str = "auto") -> DetectorConfig:
    if detector != "auto":
        config = DETECTORS[detector]
        if config.track_collection not in tree or config.cluster_collection not in tree:
            raise ValueError(
                f"{detector.upper()} collections are not present: expected " f"{config.track_collection} and {config.cluster_collection}"
            )
        return config
    matches = [config for config in DETECTORS.values() if config.track_collection in tree and config.cluster_collection in tree]
    if len(matches) != 1:
        found = ", ".join(config.key for config in matches) or "none"
        raise ValueError(f"could not infer detector uniquely (matched: {found}); use --detector")
    return matches[0]


def _event(tree, branch: str, event: int) -> np.ndarray:
    return ak.to_numpy(tree[branch].array(entry_start=event, entry_stop=event + 1)[0])


def _open_root(root_file: str | Path):
    """Open large local production files without uproot's file-thread pool."""
    return uproot.open(root_file, handler=uproot.source.file.MemmapSource)


def _track_paths(tree, event: int, config: DetectorConfig) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    collection = config.track_collection
    begin = _event(tree, f"{collection}/{collection}.trackStates_begin", event).astype(int)
    prefix = f"_{collection}_trackStates/_{collection}_trackStates."
    phi = _event(tree, prefix + "phi", event)[begin]
    omega = _event(tree, prefix + "omega", event)[begin]
    tan_lambda = _event(tree, prefix + "tanLambda", event)[begin]
    d0 = _event(tree, prefix + "D0", event)[begin]
    z0 = _event(tree, prefix + "Z0", event)[begin]
    paths = []
    for ph, om, tl, d, z in zip(phi, omega, tan_lambda, d0, z0):
        if not np.isfinite(om) or abs(om) < 1e-10:
            continue
        radius = 1.0 / om
        arc = np.linspace(0, min(4500.0, 1.8 * abs(radius)), 45)
        angle = ph - arc / radius
        # EDM4hep phi is the momentum direction: dx/ds=cos(phi), dy/ds=sin(phi)
        # at the point of closest approach. The minus sign in the phase is the
        # EDM4hep/CLD omega convention, verified against the stored outer track
        # states and by the --debug track-to-hit association plots.
        x = (d + radius) * np.sin(ph) - radius * np.sin(angle)
        y = -(d + radius) * np.cos(ph) + radius * np.cos(angle)
        zz = z + arc * tl
        keep = (np.hypot(x, y) < config.track_radius) & (np.abs(zz) < config.track_half_z)
        paths.append((x[keep], y[keep], zz[keep]))
    return paths


def _track_trajectories(tree, event: int, config: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _track_paths(tree, event, config)
    xs, ys, zs = [], [], []
    for x, y, z in paths:
        xs.extend(x.tolist() + [np.nan])
        ys.extend(y.tolist() + [np.nan])
        zs.extend(z.tolist() + [np.nan])
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _collection_id_map(root_file) -> dict[int, str]:
    metadata = root_file["podio_metadata"]
    legacy_names = "events___idTable/m_names"
    if legacy_names in metadata:
        names = metadata[legacy_names].array()[0]
        ids = metadata["events___idTable/m_collectionIDs"].array()[0]
    else:
        prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
        names = metadata[prefix + "name"].array()[0]
        ids = metadata[prefix + "collectionID"].array()[0]
    return {int(collection_id): str(name) for name, collection_id in zip(names, ids)}


def _hit_positions(tree, event: int, collections: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        collection: tuple(_event(tree, f"{collection}/{collection}.position.{axis}", event) for axis in "xyz")
        for collection in collections
        if collection in tree
    }


def _association_groups(
    tree, event: int, object_collection: str, relation_collection: str, id_to_name: dict[int, str]
) -> list[list[tuple[str, int]]]:
    begin = _event(tree, f"{object_collection}/{object_collection}.{relation_collection}_begin", event).astype(int)
    end = _event(tree, f"{object_collection}/{object_collection}.{relation_collection}_end", event).astype(int)
    relation = f"_{object_collection}_{relation_collection}"
    indices = _event(tree, f"{relation}/{relation}.index", event).astype(int)
    collection_ids = _event(tree, f"{relation}/{relation}.collectionID", event).astype(int)
    return [[(id_to_name.get(int(collection_ids[i]), ""), int(indices[i])) for i in range(first, last)] for first, last in zip(begin, end)]


def _particle_display_length(px: float, py: float, pz: float, energy: float, neutral: bool, config: DetectorConfig) -> float:
    """Length of a generator-particle guide line in the detector envelope.

    Neutral particles reach the first calorimeter surface (approximated by a
    barrel cylinder and endcap planes). A bounded logarithmic energy scaling
    makes energetic particles extend slightly farther without dominating the
    event display. Charged guide lines remain shorter than the fitted tracks.
    """
    momentum = np.sqrt(px * px + py * py + pz * pz)
    if momentum == 0:
        return 0.0
    ux, uy, uz = px / momentum, py / momentum, pz / momentum
    transverse = np.hypot(ux, uy)
    to_barrel = config.particle_barrel_radius / transverse if transverse > 1e-9 else np.inf
    to_endcap = config.particle_endcap_z / abs(uz) if abs(uz) > 1e-9 else np.inf
    calorimeter_distance = min(to_barrel, to_endcap)
    energy_fraction = np.clip(np.log1p(max(energy, 0.0)) / np.log1p(100.0), 0.0, 1.0)
    scale = (1.0 + 0.18 * energy_fraction) if neutral else (0.72 + 0.22 * energy_fraction)
    return float(np.clip(calorimeter_distance * scale, 900.0, config.particle_max_length))


def render_debug_plots(root_file: str | Path, event: int, output_dir: str | Path, detector: str = "auto") -> list[Path]:
    """Render association sanity checks for tracks/hits and clusters/hits."""
    source = _open_root(root_file)
    tree = source["events"]
    config = _detector_config(tree, detector)
    id_to_name = _collection_id_map(source)
    output_dir = Path(output_dir)

    tracker_collections = tuple(name for name, label, _ in config.hit_collections if label == "Tracker hits")
    calo_collections = tuple(name for name, label, _ in config.hit_collections if label != "Tracker hits")
    tracker_positions = _hit_positions(tree, event, tracker_collections)
    track_groups = _association_groups(tree, event, config.track_collection, "trackerHits", id_to_name)
    track_paths = _track_paths(tree, event, config)
    track_indices = sorted(range(min(len(track_groups), len(track_paths))), key=lambda i: len(track_groups[i]), reverse=True)[:9]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for ax, track_index in zip(axes.flat, track_indices):
        x, _, z = track_paths[track_index]
        ax.plot(x, z, color="#2563eb", linewidth=1.4, label="Fitted track")
        hit_x, hit_z = [], []
        for collection, index in track_groups[track_index]:
            if collection in tracker_positions and 0 <= index < len(tracker_positions[collection][0]):
                hit_x.append(tracker_positions[collection][0][index])
                hit_z.append(tracker_positions[collection][2][index])
        ax.scatter(hit_x, hit_z, color="#dc2626", s=12, label="Associated tracker hit", zorder=3)
        ax.set_title(f"track {track_index}: {len(hit_x)} hits", fontsize=10)
        ax.set(xlim=(-config.track_radius, config.track_radius), ylim=(-config.track_half_z, config.track_half_z), aspect="equal")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(track_indices) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{config.title} event {event}: tracks and associated tracker hits (x–z)", fontsize=14)
    track_output = output_dir / f"{config.key}_event_{event}_debug_tracks.png"
    fig.savefig(track_output, dpi=160, facecolor="white")
    plt.close(fig)

    calo_positions = _hit_positions(tree, event, calo_collections)
    cluster = config.cluster_collection
    cluster_groups = _association_groups(tree, event, cluster, "hits", id_to_name)
    cluster_x = _event(tree, f"{cluster}/{cluster}.position.x", event)
    cluster_y = _event(tree, f"{cluster}/{cluster}.position.y", event)
    cluster_energy = _event(tree, f"{cluster}/{cluster}.energy", event)
    candidates = [i for i, group in enumerate(cluster_groups) if group]
    cluster_indices = sorted(candidates, key=lambda i: cluster_energy[i], reverse=True)[:9]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for ax, cluster_index in zip(axes.flat, cluster_indices):
        hit_x, hit_y = [], []
        for collection, index in cluster_groups[cluster_index]:
            if collection in calo_positions and 0 <= index < len(calo_positions[collection][0]):
                hit_x.append(calo_positions[collection][0][index])
                hit_y.append(calo_positions[collection][1][index])
        ax.scatter(hit_x, hit_y, color="#dc2626", s=10, alpha=0.7, label="Associated calo hit")
        ax.scatter(
            [cluster_x[cluster_index]],
            [cluster_y[cluster_index]],
            color="#2563eb",
            marker="x",
            s=70,
            linewidth=2,
            label="Cluster",
            zorder=3,
        )
        ax.set_title(f"cluster {cluster_index}: E={cluster_energy[cluster_index]:.1f} GeV, {len(hit_x)} hits", fontsize=9)
        ax.set(xlim=(-config.plot_limit, config.plot_limit), ylim=(-config.plot_limit, config.plot_limit), aspect="equal")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(cluster_indices) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{config.title} event {event}: clusters and associated calorimeter hits (x–y)", fontsize=14)
    cluster_output = output_dir / f"{config.key}_event_{event}_debug_clusters.png"
    fig.savefig(cluster_output, dpi=160, facecolor="white")
    plt.close(fig)
    return [track_output, cluster_output]


def render_event(
    root_file: str | Path,
    event: int,
    output: str | Path,
    max_hits: int = 800,
    detector: str = "auto",
    plot_limit: float | None = None,
) -> str:
    """Render one CLD or IDEA event as a perspective 3D PNG."""
    tree = _open_root(root_file)["events"]
    if not 0 <= event < tree.num_entries:
        raise IndexError(f"event {event} is outside [0, {tree.num_entries})")
    config = _detector_config(tree, detector)
    display_limit = config.plot_limit if plot_limit is None else plot_limit

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    shown_labels: set[str] = set()
    rng = np.random.default_rng(event)

    def project(x, y, z):
        """Fast perspective projection matching a conventional 3D camera."""
        azimuth, elevation = np.deg2rad(36), np.deg2rad(19)
        horizontal = np.cos(azimuth) * x - np.sin(azimuth) * y
        depth_axis = np.sin(azimuth) * x + np.cos(azimuth) * y
        vertical = np.cos(elevation) * z - np.sin(elevation) * depth_axis
        depth = np.sin(elevation) * z + np.cos(elevation) * depth_axis
        perspective = 1.0 / np.clip(1.0 - depth / 12000.0, 0.55, 1.55)
        return horizontal * perspective, vertical * perspective

    for collection, label, color in config.hit_collections:
        if collection not in tree:
            continue
        x = _event(tree, f"{collection}/{collection}.position.x", event)
        y = _event(tree, f"{collection}/{collection}.position.y", event)
        z = _event(tree, f"{collection}/{collection}.position.z", event)
        if len(x) > max_hits:
            idx = np.sort(rng.choice(len(x), max_hits, replace=False))
            x, y, z = x[idx], y[idx], z[idx]
        sx, sy = project(x, y, z)
        ax.scatter(sx, sy, s=4.0, color=color, alpha=0.5, edgecolors="none", rasterized=True, label=label if label not in shown_labels else None)
        shown_labels.add(label)

    tx, ty, tz = _track_trajectories(tree, event, config)
    sx, sy = project(tx, ty, tz)
    ax.plot(sx, sy, color="#ef4444", linewidth=0.8, alpha=0.78, label=config.track_label)

    cluster = config.cluster_collection
    cx = _event(tree, f"{cluster}/{cluster}.position.x", event)
    cy = _event(tree, f"{cluster}/{cluster}.position.y", event)
    cz = _event(tree, f"{cluster}/{cluster}.position.z", event)
    energy = _event(tree, f"{cluster}/{cluster}.energy", event)
    sx, sy = project(cx, cy, cz)
    ax.scatter(
        sx,
        sy,
        s=np.clip(
            config.cluster_size_base + config.cluster_size_scale * np.sqrt(np.maximum(energy, 0)),
            config.cluster_size_base,
            config.cluster_size_max,
        ),
        c=energy,
        cmap="viridis",
        alpha=config.cluster_alpha,
        edgecolors="none",
        rasterized=True,
        label="Calorimeter clusters",
    )

    status = _event(tree, "MCParticles/MCParticles.generatorStatus", event)
    px = _event(tree, "MCParticles/MCParticles.momentum.x", event)
    py = _event(tree, "MCParticles/MCParticles.momentum.y", event)
    pz = _event(tree, "MCParticles/MCParticles.momentum.z", event)
    pdg = np.abs(_event(tree, "MCParticles/MCParticles.PDG", event)).astype(int)
    charge = _event(tree, "MCParticles/MCParticles.charge", event)
    mass = _event(tree, "MCParticles/MCParticles.mass", event)
    particle_energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    keep = status == 1
    px, py, pz, pdg, charge, particle_energy = (v[keep] for v in (px, py, pz, pdg, charge, particle_energy))
    particle_kind = np.where(np.isin(pdg, [11, 13, 22]), pdg, np.where(np.abs(charge) > 0, 211, 130))
    for code, (name, color) in PARTICLE_STYLES.items():
        selected = particle_kind == code
        particle_x, particle_y, particle_z = [], [], []
        for vx, vy, vz, particle_e, particle_charge in zip(px[selected], py[selected], pz[selected], particle_energy[selected], charge[selected]):
            norm = np.sqrt(vx * vx + vy * vy + vz * vz)
            if norm == 0:
                continue
            length = _particle_display_length(vx, vy, vz, particle_e, abs(particle_charge) < 0.5, config)
            scale = length / norm
            particle_x.extend([0, scale * vx, np.nan])
            particle_y.extend([0, scale * vy, np.nan])
            particle_z.extend([0, scale * vz, np.nan])
        if particle_x:
            sx, sy = project(np.asarray(particle_x), np.asarray(particle_y), np.asarray(particle_z))
            ax.plot(sx, sy, color=color, linewidth=0.85, linestyle="--", alpha=0.62, label=f"Particle: {name}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-display_limit, display_limit)
    ax.set_ylim(-display_limit, display_limit)
    ax.set_title(f"{config.title} — event {event}", fontsize=15)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True, framealpha=0.9)
    ax.axis("off")
    fig.savefig(output, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return config.key


def render_comparison(images: list[tuple[Path, str]], event: int, output: str | Path) -> None:
    """Place already-rendered detector displays side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=(8 * len(images), 8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (path, detector) in zip(axes, images):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    detector_names = " vs ".join(DETECTORS[detector].title for _, detector in images)
    fig.suptitle(f"{detector_names} — event {event}", fontsize=18)
    fig.savefig(output, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", type=Path, nargs="+")
    parser.add_argument("--events", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("event_displays"))
    parser.add_argument("--detector", choices=("auto", *DETECTORS), default="auto")
    parser.add_argument("--max-hits", type=int, default=800, help="maximum displayed hits per collection")
    parser.add_argument("--debug", action="store_true", help="also render track/hit and cluster/hit association checks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for root_file in args.root_files:
        with _open_root(root_file) as source:
            config = _detector_config(source["events"], args.detector)
        inputs.append((root_file, config))
    comparison_limit = max(config.plot_limit for _, config in inputs) if len(inputs) > 1 else None

    for event in args.events:
        event_images = []
        for root_file, config in inputs:
            output = args.output_dir / f"{config.key}_event_{event}.png"
            detector = render_event(root_file, event, output, args.max_hits, args.detector, comparison_limit)
            event_images.append((output, detector))
            print(output)
            if args.debug:
                for debug_output in render_debug_plots(root_file, event, args.output_dir, args.detector):
                    print(debug_output)
        if len(event_images) > 1:
            detectors = "_vs_".join(detector for _, detector in event_images)
            comparison_output = args.output_dir / f"{detectors}_event_{event}.png"
            render_comparison(event_images, event, comparison_output)
            print(comparison_output)


if __name__ == "__main__":
    main()
