#!/usr/bin/env python3
"""Create white/transparent detector event layers and a CLD reconstruction-flow figure.

Reuses the repository's transverse camera, track helices and particle palette.
Scene archives allow rerendering without reopening the original detector files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

from scripts.visualize_key4hep import (
    DETECTORS,
    PARTICLE_STYLES,
    _event,
    _open_root,
    _particle_trajectory,
    _track_paths,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = {
    "cld": "/mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root",
    "clic": "/mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root",
    "idea": "/mnt/work/mlpf/root/idea/ttbar/reco_p8_ee_ttbar_ecm365_300000.root",
    "maia": "local_test_data/maia_smoke/root/ttbar_reco_10000.slcio.edm4hep.root",
    "colliderml": "local_test_data/detector_comparison/colliderml_source",
}
VIEWS = {
    "hits": "Raw detector hits",
    "tracks": "Tracks",
    "clusters": "Calorimeter clusters",
    "tracks_clusters": "Tracks and clusters",
    "particles": "Particles",
    "overlay": "All components",
    "tracks_calo_hits": "Tracks and calorimeter hits",
}
TRACK_COLOR = "#163653"
CLUSTER_COLOR = "#7356a6"
MODEL_COLOR = "#3775A8"
HIT_MODEL_COLOR = "#B76720"


def particle_records(tree, event, config):
    """Prefer actual Pandora candidates; explicitly identify truth-only families."""
    reco = "PandoraPFOs" in tree
    collection = "PandoraPFOs" if reco else config.particle_collection

    def read(field):
        return _event(tree, f"{collection}/{collection}.{field}", event)

    pdg, charge = np.abs(read("PDG")).astype(int), read("charge")
    px, py, pz, mass = (read(k) for k in ["momentum.x", "momentum.y", "momentum.z", "mass"])
    keep = ~np.isin(pdg, [12, 14, 16])
    if not reco:
        keep &= read("generatorStatus") == 1
    keep &= np.isfinite(px + py + pz + mass + charge) & ((px * px + py * py + pz * pz) > 0)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    kind = np.select([pdg == 11, pdg == 13, pdg == 15, pdg == 22], [11, 13, 15, 22], default=np.where(abs(charge) > 0, 211, 130))
    records = [
        dict(px=float(x), py=float(y), pz=float(z), charge=float(q), energy=float(e), kind=int(k))
        for x, y, z, q, e, k in zip(px[keep], py[keep], pz[keep], charge[keep], energy[keep], kind[keep])
    ]
    label = (
        "Pandora reconstructed particles"
        if reco
        else ("Truth proxy: primary leaves" if config.key == "colliderml" else "Truth proxy: visible stable particles")
    )
    return records, collection, label


def extract_scene(tree, detector, event, source, max_hits=1200):
    config = DETECTORS[detector]
    if not 0 <= event < tree.num_entries:
        raise IndexError(f"event {event} outside source with {tree.num_entries} entries")
    rng = np.random.default_rng(event)
    hits = []
    for collection, label, color in config.hit_collections:
        if collection not in tree:
            continue
        x, y, z = (_event(tree, f"{collection}/{collection}.position.{axis}", event) for axis in "xyz")
        finite = np.isfinite(x + y + z)
        x, y = x[finite], y[finite]
        indices = np.arange(len(x)) if len(x) <= max_hits else np.sort(rng.choice(len(x), max_hits, replace=False))
        hits.append(dict(collection=collection, label=label, color=color, count=len(x), xy=np.column_stack([-x[indices], y[indices]]).tolist()))
    tracks = [np.column_stack([-x, y]).tolist() for x, y, _ in _track_paths(tree, event, config) if len(x)]
    c = config.cluster_collection
    x, y, energy = (_event(tree, f"{c}/{c}.{field}", event) for field in ["position.x", "position.y", "energy"])
    keep = np.isfinite(x + y + energy)
    particles, collection, label = particle_records(tree, event, config)
    return dict(
        detector=detector,
        title=config.title,
        event=event,
        source=str(source),
        plot_limit={"idea": 4200.0, "maia": 4200.0, "colliderml": 3500.0}.get(detector, config.plot_limit),
        magnetic_field_tesla=config.magnetic_field_tesla,
        hit_cap_per_collection=max_hits,
        hits=hits,
        tracks=tracks,
        track_label=config.track_label,
        track_collection=config.track_collection,
        cluster_collection=c,
        clusters=np.column_stack([-x[keep], y[keep], energy[keep]]).tolist(),
        particles=particles,
        particle_collection=collection,
        particle_label=label,
        orientation="transverse x-y; +x left, +y up; +z toward viewer",
        trajectory_note="Track and particle curves use nominal fields and approximate display envelopes; particle guides originate at the collision point and omit material interactions.",
    )


def draw_scene(ax, scene, view, scale=1.0):
    """Render every view from the identical, already sampled event scene."""
    limit = scene["plot_limit"]
    overlay = view == "overlay"
    if view in {"hits", "tracks_calo_hits", "overlay"}:
        for h in scene["hits"]:
            if view == "tracks_calo_hits" and h["label"] == "Tracker hits":
                continue
            xy = np.asarray(h["xy"]).reshape(-1, 2)
            ax.scatter(*xy.T, s=2.5 * scale**2, c=h["color"], alpha=0.40 if overlay else 0.75, linewidths=0)
    if view in {"tracks", "tracks_clusters", "tracks_calo_hits", "overlay"}:
        ax.add_collection(LineCollection(scene["tracks"], colors=TRACK_COLOR, linewidths=0.70 * scale, alpha=0.60 if overlay else 0.88))
    if view in {"clusters", "tracks_clusters", "overlay"}:
        clusters = np.asarray(scene["clusters"]).reshape(-1, 3)
        config = DETECTORS[scene["detector"]]
        size = (
            np.clip(
                config.cluster_size_base + config.cluster_size_scale * np.sqrt(np.maximum(clusters[:, 2], 0)),
                config.cluster_size_base,
                config.cluster_size_max,
            )
            * scale**2
        )
        ax.scatter(
            clusters[:, 0],
            clusters[:, 1],
            s=size,
            facecolors="none",
            edgecolors=CLUSTER_COLOR,
            linewidths=0.7 * scale,
            alpha=config.cluster_alpha * (0.7 if overlay else 1),
        )
    if view in {"particles", "overlay"}:
        paths, colors = [], []
        for p in scene["particles"]:
            length = limit * (0.32 + 0.55 * np.clip(np.log1p(p["energy"]) / np.log1p(100), 0, 1))
            x, y, _ = _particle_trajectory(p["px"], p["py"], p["pz"], p["charge"], length, scene["magnetic_field_tesla"])
            paths.append(np.column_stack([-x, y]))
            colors.append(PARTICLE_STYLES[p["kind"]][1])
        ax.add_collection(
            LineCollection(paths, colors=colors, linewidths=0.85 * scale, linestyles="dashed" if overlay else "solid", alpha=0.65 if overlay else 0.9)
        )
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.axis("off")


def legend_handles(scene, view):
    handles = []
    if view in {"hits", "tracks_calo_hits", "overlay"}:
        seen = set()
        for h in scene["hits"]:
            if not h["xy"] or h["label"] in seen or (view == "tracks_calo_hits" and h["label"] == "Tracker hits"):
                continue
            seen.add(h["label"])
            handles.append(Line2D([], [], color=h["color"], marker=".", ls="", label=h["label"]))
    if view in {"tracks", "tracks_clusters", "tracks_calo_hits", "overlay"}:
        handles.append(Line2D([], [], color=TRACK_COLOR, label=scene["track_label"]))
    if view in {"clusters", "tracks_clusters", "overlay"}:
        handles.append(Line2D([], [], marker="o", markerfacecolor="none", color=CLUSTER_COLOR, ls="", label="Clusters (energy-scaled)"))
    if view in {"particles", "overlay"}:
        for kind in sorted({p["kind"] for p in scene["particles"]}):
            name, color, _ = PARTICLE_STYLES[kind]
            handles.append(Line2D([], [], color=color, ls="--" if view == "overlay" else "-", label=name.capitalize()))
    return handles


def compact_svg_circles(path):
    """Store circular scatter markers as SVG circles instead of eight Bezier arcs.

    Matplotlib expands energy-sized markers into one long path per cluster.
    Native circles preserve their centers, radii and styling with much smaller files.
    """
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    for element in tree.findall(f".//{{{namespace}}}path"):
        d = element.get("d", "")
        if re.findall(r"[A-Za-z]", d) != ["M"] + ["C"] * 8 + ["L", "z"]:
            continue
        xy = np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]).reshape(-1, 2)
        low, high = xy.min(axis=0), xy.max(axis=0)
        center = (low + high) / 2
        radii = (high - low) / 2
        if not np.isclose(radii[0], radii[1], rtol=0, atol=1e-6):
            continue
        # Both end points must be the bottom of the circle; reject other paths.
        if not np.allclose(xy[[0, -1]], [center[0], high[1]], rtol=0, atol=1e-6):
            continue
        element.tag = f"{{{namespace}}}circle"
        del element.attrib["d"]
        element.set("cx", f"{center[0]:.6f}")
        element.set("cy", f"{center[1]:.6f}")
        element.set("r", f"{radii[0]:.6f}")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_figure(fig, stem, transparent_png=True):
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), transparent=True)
    compact_svg_circles(stem.with_suffix(".svg"))
    if transparent_png:
        fig.savefig(stem.with_name(stem.name + "_transparent").with_suffix(".png"), dpi=180, transparent=True)
    plt.close(fig)


def render_views(scene, output):
    output.mkdir(parents=True, exist_ok=True)
    for view, title in VIEWS.items():
        handles = legend_handles(scene, view)
        columns = 3 if view == "overlay" else min(2, len(handles))
        legend_rows = (len(handles) + columns - 1) // columns
        plot_height = 5.25
        plot_bottom = 0.22 + 0.18 * legend_rows
        height = plot_bottom + plot_height + 0.58
        fig = plt.figure(figsize=(6.5, height))
        ax = fig.add_axes([0.035, plot_bottom / height, 0.93, plot_height / height])
        draw_scene(ax, scene, view)
        fig.text(0.5, 1 - 0.22 / height, f'{scene["title"]}: {title}', ha="center", fontsize=18, weight="bold")
        subtitle = scene["particle_label"] if view in {"particles", "overlay"} else scene["track_label"] if view == "tracks" else "transverse view"
        fig.text(0.5, 1 - 0.45 / height, f'Event {scene["event"]}, {subtitle}', ha="center", fontsize=10, color="#526175")
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, (plot_bottom - 0.025) / height),
            ncol=columns,
            borderaxespad=0,
            borderpad=0,
            labelspacing=0.25,
            columnspacing=1.0,
            handlelength=1.6,
            handletextpad=0.5,
            frameon=False,
            fontsize=9,
        )
        # Physical scale is shown independently of the sparse or dense content.
        bar_y = -0.88 * scene["plot_limit"]
        ax.plot([-0.9 * scene["plot_limit"], -0.9 * scene["plot_limit"] + 1000], [bar_y, bar_y], color="#526175", lw=1.5)
        ax.text(-0.9 * scene["plot_limit"] + 500, bar_y - 0.04 * scene["plot_limit"], "1 m", ha="center", va="top", fontsize=9, color="#526175")
        fig.text(0.5, 0.055 / height, "+x left, +y up, +z toward viewer", ha="center", fontsize=9, color="#526175")
        save_figure(fig, output / f'event_{scene["event"]}_{view}')


def render_flow(scene, output):
    if scene["particle_collection"] != "PandoraPFOs":
        raise ValueError("The reconstruction flow requires reconstructed reference particles")
    fig = plt.figure(figsize=(16, 8))
    fig.text(0.5, 0.95, "From detector hits to particles", ha="center", fontsize=26, weight="bold", color=TRACK_COLOR)
    fig.text(0.5, 0.92, f'CLD: one simulated ttbar event at 365 GeV, event {scene["event"]}', ha="center", fontsize=14, color="#526175")
    centers = [0.13, 0.375, 0.62, 0.865]
    stages = [
        ("hits", "Raw hits"),
        ("tracks_calo_hits", "Tracks + calorimeter hits"),
        ("tracks_clusters", "Tracks + clusters"),
        ("particles", "Particles"),
    ]
    for cx, (view, title) in zip(centers, stages):
        ax = fig.add_axes([cx - 0.115, 0.33, 0.23, 0.42])
        draw_scene(ax, scene, view, scale=0.78)
        fig.text(cx, 0.755, title, ha="center", fontsize=16, weight="bold", color=TRACK_COLOR)
        if view == "particles":
            fig.text(cx, 0.72, "Pandora reference output", ha="center", fontsize=11, color="#526175")
    for start, end, label in zip(centers[:-1], centers[1:], ["Tracking", "Calorimeter clustering", "Particle-flow reconstruction"]):
        fig.add_artist(
            FancyArrowPatch(
                (start + 0.025, 0.835), (end - 0.025, 0.835), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#526175"
            )
        )
        fig.text((start + end) / 2, 0.855, label, ha="center", fontsize=14, color="#526175")
    # Model routes connect their input stage to the particle-output stage.
    for start, end, height, color, label in [
        (0.62, 0.83, 0.235, MODEL_COLOR, "Track/cluster MLPF"),
        (0.13, 0.90, 0.115, HIT_MODEL_COLOR, "Raw-hit MLPF"),
    ]:
        fig.add_artist(Line2D([start, start, end], [0.265, height, height], transform=fig.transFigure, color=color, lw=2.2))
        fig.add_artist(
            FancyArrowPatch((end, height), (end, 0.325), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=17, lw=2.2, color=color)
        )
        fig.text(
            (start + end) / 2,
            height,
            label,
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec=color, lw=1.4),
        )
    fig.text(0.13, 0.285, "Tracker / ECAL / HCAL / muon hits", ha="center", fontsize=10, color="#526175")
    fig.text(0.375, 0.285, "Track fits + calorimeter measurements", ha="center", fontsize=10, color="#526175")
    fig.text(0.62, 0.285, "Track fits + calorimeter clusters", ha="center", fontsize=10, color="#526175")
    fig.text(
        0.5,
        0.035,
        "Measured hits and reconstructed objects from the same event. Model routes indicate inputs and outputs; displayed particles are the Pandora reference.",
        ha="center",
        fontsize=10,
        color="#526175",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output)


def load_scene(detector, source, event, max_hits, colliderml_parquet):
    if detector != "colliderml":
        with _open_root(source) as file:
            return extract_scene(file["events"], detector, event, source, max_hits)
    import awkward as ak
    import pyarrow.parquet as pq
    from itertools import islice
    from mlpf.data.colliderml.reader import iter_event_records, shard_paths
    from scripts.visualize_colliderml import adapt_event

    paths = [shard_paths(Path(source), "ttbar_pu0", obj)[0] for obj in ("particles", "tracks", "calo_hits", "tracker_hits")]
    count = pq.ParquetFile(paths[0]).metadata.num_rows
    record = next(islice(iter_event_records(*paths, batch_size=1), event, event + 1))
    clusters = None
    if colliderml_parquet:
        data = ak.from_parquet(colliderml_parquet, columns=["event_id", "X_cluster"])
        lookup = {int(i): ak.to_numpy(c) for i, c in zip(data["event_id"], data["X_cluster"])}
        clusters = lookup[record["event_id"]]
    tree = adapt_event(record, event, count, clusters)
    scene = extract_scene(tree, detector, event, source, max_hits)
    scene.update(
        source_event_id=int(record["event_id"]),
        source_shards=[str(p) for p in paths],
        cluster_source=str(colliderml_parquet) if colliderml_parquet else "native clustering from the selected event",
    )
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("local_test_data/reconstruction_event_displays"))
    parser.add_argument("--detectors", nargs="+", choices=DEFAULT_SOURCES, default=list(DEFAULT_SOURCES))
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--max-hits", type=int, default=1200)
    parser.add_argument("--source", action="append", default=[], metavar="DETECTOR=PATH")
    parser.add_argument("--colliderml-parquet", type=Path)
    parser.add_argument("--from-scenes", action="store_true", help="rerender saved scene archives without raw data")
    args = parser.parse_args()
    if args.event < 0 or args.max_hits < 1:
        parser.error("event must be nonnegative and max-hits positive")
    sources = DEFAULT_SOURCES | dict(s.split("=", 1) for s in args.source)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    manifest = {}
    for detector in args.detectors:
        output = args.output_dir / detector
        output.mkdir(parents=True, exist_ok=True)
        archive = output / f"event_{args.event}_scene.json.gz"
        if args.from_scenes:
            scene = json.loads(gzip.decompress(archive.read_bytes()))
        else:
            scene = load_scene(detector, sources[detector], args.event, args.max_hits, args.colliderml_parquet)
            archive.write_bytes(gzip.compress(json.dumps(scene, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
        render_views(scene, output)
        if detector == "cld":
            render_flow(scene, args.output_dir / "pf_event")
        manifest[detector] = {k: v for k, v in scene.items() if k not in {"hits", "tracks", "clusters", "particles"}}
        manifest[detector].update(
            scene_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            hits_total=sum(h["count"] for h in scene["hits"]),
            hits_shown=sum(len(h["xy"]) for h in scene["hits"]),
            tracks=len(scene["tracks"]),
            clusters=len(scene["clusters"]),
            particles=len(scene["particles"]),
        )
        print(detector, manifest[detector], flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
