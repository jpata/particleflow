"""Audit track/calo labels against source truth links and source-particle retention.

Uses the same selected source particles and association weights passed to the
target builder, but scores its stored hit labels against the full link matrix.
This is an assignment audit, not a detector-independent energy calibration.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.sparse import coo_matrix

from mlpf.conf import EDM4HEP, ParticleFeatures
from mlpf.data.target_building import filter_adj, visibility_mask
from mlpf.data.track_calo_targets import _match, _matrix, _ranked
from scripts.studies.plot_track_calo_track_resolution import matrix


ROOTS = {
    "cld": "/local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_300000.root",
    "clic": "/local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm380/root/reco_p8_ee_ttbar_ecm380_300000.root",
    "idea": "/local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_300000.root",
}
PT_BINS = np.array([0, 0.3, 1, 3, 10, 30, np.inf])
ETA_BINS = np.array([0, 0.5, 1, 1.5, 2, 3, np.inf])
COL = {name: i for i, name in enumerate(ParticleFeatures.get_names())}


def json_text(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    return json.dumps(value, indent=2, default=convert)


def particle_class(pdg, charge):
    pdg = abs(int(pdg))
    if pdg == 22:
        return "photon"
    if pdg == 11:
        return "electron"
    if pdg == 13:
        return "muon"
    if charge != 0:
        return "charged_hadron"
    return "neutral_hadron"


def key4hep_events(detector, count, root_path):
    from mlpf.data.key4hep import postprocessing as kp

    root = uproot.open(root_path, handler=uproot.source.file.MemmapSource)
    tree = root["events"]
    info = root["podio_metadata"]
    prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
    ids = dict(zip(info[prefix + "name"].array()[0], info[prefix + "collectionID"].array()[0]))
    reverse = {v: k for k, v in ids.items()}
    cfg = EDM4HEP.DETECTORS[detector]
    branches = [
        kp.mc_coll,
        *[
            f"{kp.mc_coll}.{name}"
            for name in (
                "PDG",
                "momentum.x",
                "momentum.y",
                "momentum.z",
                "mass",
                "charge",
                "generatorStatus",
                "simulatorStatus",
                "daughters_begin",
                "daughters_end",
            )
        ],
        f"_{kp.mc_coll}_daughters/_{kp.mc_coll}_daughters.index",
        kp.track_coll,
        f"_{kp.track_coll}_trackStates",
        "PandoraClusters",
        "_PandoraClusters_hits/_PandoraClusters_hits.index",
        "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
        "PandoraPFOs",
        f"{kp.track_coll}_dQdx",
    ]
    prop = tree.arrays(branches, entry_stop=count)
    calo = tree.arrays(
        [
            "CalohitMCTruthLink.weight",
            *[f"_CalohitMCTruthLink_{end}/_CalohitMCTruthLink_{end}.{field}" for end in ("to", "from") for field in ("collectionID", "index")],
        ],
        entry_stop=count,
    )
    tracks = tree.arrays(
        [
            "SiTracksMCTruthLink.weight",
            *[f"_SiTracksMCTruthLink_{end}/_SiTracksMCTruthLink_{end}.{field}" for end in ("to", "from") for field in ("collectionID", "index")],
        ],
        entry_stop=count,
    )
    tracker_branches = []
    for rel in kp.tracker_hit_relations.values():
        tracker_branches.extend([f"_{rel}_{end}/_{rel}_{end}.index" for end in ("from", "to")])
    for rel in kp.tracker_hit_sim.values():
        tracker_branches.extend([f"_{rel}_particle/_{rel}_particle.{field}" for field in ("index", "collectionID")])
    tracker = tree.arrays(tracker_branches, entry_stop=count)
    hits = {name: tree[name].array(entry_stop=count) for name in cfg.hit_collections}
    encodings = kp.get_cellid_encodings(root)
    muon_ids = tuple(ids[name] for name in cfg.muon_collections)
    for iev in range(count):
        with contextlib.redirect_stdout(io.StringIO()):
            gpdata = kp.get_genparticles_and_adjacencies(
                prop, hits, calo, tracks, tracker, iev, ids, reverse, ids[kp.mc_coll], cfg.b_field, encodings, cfg.require_cellid_encoding, muon_ids
            )
        calo_mask = np.asarray(gpdata.hit_features["subdetector"]) != 3
        yield iev, gpdata.gen_features, gpdata.genparticle_to_hit, gpdata.genparticle_to_track, calo_mask, None


def idea_events(count, root_path):
    from mlpf.data.key4hep import postprocessing as kp
    from mlpf.data.key4hep.idea_track_calo import hit_inputs_and_links, required_branches

    root = uproot.open(root_path, handler=uproot.source.file.MemmapSource)
    tree = root["events"]
    meta = root["podio_metadata"]
    prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
    ids = dict(zip(meta[prefix + "name"].array()[0], meta[prefix + "collectionID"].array()[0]))
    branches = [
        kp.mc_coll,
        *[
            f"{kp.mc_coll}.{name}"
            for name in (
                "PDG",
                "momentum.x",
                "momentum.y",
                "momentum.z",
                "mass",
                "charge",
                "generatorStatus",
                "simulatorStatus",
                "daughters_begin",
                "daughters_end",
            )
        ],
        f"_{kp.mc_coll}_daughters/_{kp.mc_coll}_daughters.index",
        "TracksFromGenParticlesAssociation.weight",
        "_TracksFromGenParticlesAssociation_from/_TracksFromGenParticlesAssociation_from.index",
        "_TracksFromGenParticlesAssociation_to/_TracksFromGenParticlesAssociation_to.index",
        *required_branches(),
    ]
    prop = tree.arrays(branches, entry_stop=count)
    for iev in range(count):
        gen = kp.gen_to_features(prop, iev)
        n_gen = len(gen["PDG"])
        xhit, hit_links = hit_inputs_and_links(prop, iev, ids, n_gen)
        direct = kp._idea_relation(prop, "TracksFromGenParticlesAssociation", iev)
        hit_links, inclusive_tracks = kp.add_daughters_to_status1(gen, hit_links, direct)
        keep = (np.asarray(gen["generatorStatus"]) == 1) & ~np.isin(np.abs(np.asarray(gen["PDG"])), [12, 14, 16])
        mapping = {old: new for new, old in enumerate(np.flatnonzero(keep))}
        hit_links, inclusive_tracks, direct = (filter_adj(edges, mapping) for edges in (hit_links, inclusive_tracks, direct))
        features = {name: np.asarray(gen[name])[keep] for name in ("PDG", "charge", "pt", "eta", "energy")}
        yield iev, features, hit_links, inclusive_tracks, np.ones(len(xhit), dtype=bool), direct


def colliderml_events(count, base):
    from mlpf.data.colliderml.reader import iter_event_records
    from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables

    names = ("particles", "tracks", "calo_hits", "tracker_hits")
    paths = [base / f"ttbar_pu0_{name}" / "data" / f"ttbar_pu0_{name}" / "train-00000-of-01000.parquet" for name in names]
    for iev, event in enumerate(iter_event_records(*paths)):
        if iev >= count:
            break
        particles, tracks, calo, tracker = (event[name] for name in names)
        gen, hit_links, track_links, _ = compute_gen_tables(particles, calo, tracks, DEFAULT_CALIBRATION, tracker_ev=tracker)
        n_hit = len(calo["x"])
        yield iev, gen, hit_links, track_links, np.arange(n_hit + len(tracker["x"])) < n_hit, None


def source_status(features, hit_edges, track_edges, calo_mask, representative_edges, n_track):
    n_gen = len(features["PDG"])
    tracks = _matrix(track_edges, (n_gen, n_track))
    representatives = tracks if representative_edges is None else _matrix(representative_edges, (n_gen, n_track))
    hits = _matrix(hit_edges, (n_gen, len(calo_mask)))[:, calo_mask].tocsr()
    energy = np.asarray(features["energy"], dtype=float)
    track_support = np.asarray(representatives.max(axis=1).toarray()).ravel() >= 0.2 if n_track and n_gen else np.zeros(n_gen, dtype=bool)
    visible = visibility_mask(np.asarray(hits.sum(axis=1)).ravel(), energy, np.asarray(features["charge"]) != 0, track_support)
    candidates = [[] for _ in range(n_gen)]
    for particle in np.flatnonzero(visible):
        candidates[particle] = _ranked(representatives, particle, 0.2) + [n_track + hit for hit in _ranked(hits, particle)]
    order = sorted(np.flatnonzero(visible), key=lambda i: (-energy[i], i))
    anchors, _ = _match(candidates, order, n_track + hits.shape[1])
    return hits, visible, anchors


def audit_event(detector, iev, event, features, hit_edges, track_edges, calo_mask, representative_edges):
    yhit = matrix(event.ytarget_track_calo_hit, len(COL))
    ytrack = matrix(event.ytarget_track_calo_track, len(COL))
    mapping = np.asarray(event.track_calo_original_to_target, dtype=np.int64)
    targets = matrix(event.track_calo_target_particles, len(COL))
    n_track = len(ytrack)
    hits, visible, anchors = source_status(features, hit_edges, track_edges, calo_mask, representative_edges, n_track)
    assert len(mapping) == len(anchors) and hits.shape[1] == len(yhit)
    assert np.count_nonzero(anchors >= 0) == len(targets)
    assert np.array_equal(np.unique(mapping[mapping >= 0]), np.arange(len(targets)))
    status = np.full(len(mapping), "dropped", dtype=object)
    status[~visible] = "invisible"
    status[(anchors < 0) & (mapping >= 0)] = "merged"
    status[anchors >= 0] = "represented"
    assert np.all(mapping[anchors >= 0] >= 0)
    assert not np.any((status == "dropped") & (mapping >= 0))
    assert not np.any((status == "invisible") & (mapping >= 0))
    accounting = np.asarray(event.track_calo_energy_accounting, dtype=float)
    source_energy = np.asarray(features["energy"], dtype=float)
    observed = np.array(
        [
            source_energy.sum(),
            source_energy[status == "invisible"].sum(),
            source_energy[status == "dropped"].sum(),
            source_energy[np.isin(status, ["represented", "merged"])].sum(),
        ]
    )
    assert np.allclose(observed, accounting, rtol=1e-8, atol=1e-6), (detector, iev, observed, accounting)
    labels = np.rint(yhit[:, COL["particle_number"]]).astype(int) - 1
    assert np.all((labels >= -1) & (labels < len(targets)))
    is_rep_hit = yhit[:, COL["PDG"]] != 0
    rep_ids = np.rint(yhit[is_rep_hit, COL["particle_number"]]).astype(int) - 1
    assert len(rep_ids) == len(np.unique(rep_ids))
    coo = hits.tocoo()
    mapped = mapping[coo.row]
    valid = mapped >= 0
    truth = coo_matrix((coo.data[valid], (mapped[valid], coo.col[valid])), shape=(len(targets), hits.shape[1])).tocsc()
    truth.sum_duplicates()
    total_hit_truth = np.asarray(hits.sum(axis=0)).ravel()
    total_target_truth = np.asarray(truth.sum(axis=1)).ravel()
    assigned_truth = np.zeros(len(targets))
    target_truth_on_correct_hits = np.zeros(len(yhit))
    dominant = np.full(len(yhit), -1, dtype=int)
    for hit in range(len(yhit)):
        start, end = truth.indptr[hit : hit + 2]
        if start == end:
            continue
        indices, weights = truth.indices[start:end], truth.data[start:end]
        dominant[hit] = indices[np.argmax(weights)]
        if labels[hit] >= 0:
            match = indices == labels[hit]
            if np.any(match):
                amount = float(weights[match][0])
                target_truth_on_correct_hits[hit] = amount
                assigned_truth[labels[hit]] += amount
    rep_track = np.rint(ytrack[ytrack[:, COL["PDG"]] != 0, COL["particle_number"]]).astype(int) - 1
    rep_type = np.full(len(targets), "calo", dtype=object)
    rep_type[rep_track] = "track"
    hit_rows = []
    for hit in range(len(yhit)):
        if labels[hit] < 0:
            continue
        group = "calo" if labels[hit] in rep_ids else "track"
        hit_rows.append(
            {
                "detector": detector,
                "event": iev,
                "hit": hit,
                "group": group,
                "representative": bool(is_rep_hit[hit]),
                "purity": target_truth_on_correct_hits[hit] / total_hit_truth[hit] if total_hit_truth[hit] > 0 else None,
                "weight": float(total_hit_truth[hit]),
                "override": bool(is_rep_hit[hit] and dominant[hit] >= 0 and dominant[hit] != labels[hit]),
                "dominant_mismatch": dominant[hit] >= 0 and dominant[hit] != labels[hit],
            }
        )
    target_rows = []
    for idx, denominator in enumerate(total_target_truth):
        if denominator > 0:
            target_rows.append(
                {
                    "detector": detector,
                    "event": iev,
                    "target": idx,
                    "group": rep_type[idx],
                    "completeness": assigned_truth[idx] / denominator,
                    "weight": float(denominator),
                    "merged_sources": int(np.count_nonzero(mapping == idx)) - 1,
                }
            )
    particle_rows = []
    for idx in range(len(mapping)):
        particle_rows.append(
            {
                "detector": detector,
                "event": iev,
                "source": idx,
                "class": particle_class(features["PDG"][idx], features["charge"][idx]),
                "status": status[idx],
                "pt": float(features["pt"][idx]),
                "abs_eta": abs(float(features["eta"][idx])),
                "energy": float(features["energy"][idx]),
            }
        )
    return hit_rows, target_rows, particle_rows


def summarize(hits, targets, particles):
    def outcomes(rows, field):
        return {
            key: {"count": sum(r[field] == key for r in rows), "energy": sum(r["energy"] for r in rows if r[field] == key)}
            for key in sorted({r[field] for r in rows})
        }

    def binned(rows, field, edges):
        return [
            {
                "lower": float(low),
                "upper": None if np.isinf(high) else float(high),
                "outcomes": outcomes([r for r in rows if low <= r[field] < high], "status"),
            }
            for low, high in zip(edges[:-1], edges[1:])
        ]

    out = {}
    for detector in ("cld", "clic", "idea", "colliderml"):
        all_hh = [r for r in hits if r["detector"] == detector]
        hh = [r for r in all_hh if r["purity"] is not None]
        tt = [r for r in targets if r["detector"] == detector]
        pp = [r for r in particles if r["detector"] == detector]

        def score(rows, key):
            return {
                "count": len(rows),
                "mean": float(np.mean([r[key] for r in rows])) if rows else None,
                "weighted": (sum(r[key] * r["weight"] for r in rows) / sum(r["weight"] for r in rows)) if rows else None,
                "p10": float(np.quantile([r[key] for r in rows], 0.1)) if rows else None,
                "median": float(np.median([r[key] for r in rows])) if rows else None,
            }

        out[detector] = {
            "events": max(r["event"] for r in pp) + 1,
            "source_particles": len(pp),
            "source_energy": sum(r["energy"] for r in pp),
            "outcomes": outcomes(pp, "status"),
            "by_class": {name: outcomes([r for r in pp if r["class"] == name], "status") for name in sorted({r["class"] for r in pp})},
            "by_pt": binned(pp, "pt", PT_BINS),
            "by_abs_eta": binned(pp, "abs_eta", ETA_BINS),
            "hit_purity": score(hh, "purity"),
            "target_completeness": score(tt, "completeness"),
            "hit_purity_by_target_class": {
                name: score([r for r in hh if r["target_class"] == name], "purity") for name in sorted({r["target_class"] for r in hh})
            },
            "target_completeness_by_target_class": {
                name: score([r for r in tt if r["target_class"] == name], "completeness") for name in sorted({r["target_class"] for r in tt})
            },
            "hit_purity_by_representative": {group: score([r for r in hh if r["group"] == group], "purity") for group in ("track", "calo")},
            "target_completeness_by_representative": {
                group: score([r for r in tt if r["group"] == group], "completeness") for group in ("track", "calo")
            },
            "representative_hit_overrides": sum(r["override"] for r in hh),
            "representative_hits_with_links": sum(r["representative"] for r in hh),
            "all_hit_dominant_mismatches": sum(r["dominant_mismatch"] for r in hh),
            "labelled_hits_without_truth_links": len(all_hh) - len(hh),
            "targets_with_merged_sources": sum(r["merged_sources"] > 0 for r in tt),
        }
    return out


def draw(hits, targets, particles, output):
    detectors = ("cld", "clic", "idea", "colliderml")
    colors = {"cld": "#2563a6", "clic": "#df8530", "idea": "#2c8a65", "colliderml": "#a34ba3"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    thresholds = np.logspace(-6, 0, 120)
    for detector in detectors:
        hh = [r["purity"] for r in hits if r["detector"] == detector and r["purity"] is not None]
        tt = [r["completeness"] for r in targets if r["detector"] == detector]
        for ax, values in ((axes[0, 0], hh), (axes[0, 1], tt)):
            if values:
                impurity = 1 - np.asarray(values)
                survival = np.mean(impurity[:, None] >= thresholds[None, :], axis=0)
                ax.plot(thresholds, np.maximum(survival, 1e-7), label=f"{detector.upper()} (n={len(values)})", color=colors[detector])
    for ax, title in ((axes[0, 0], "Labelled-hit association impurity"), (axes[0, 1], "Target association incompleteness")):
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(1e-6, 1),
            ylim=(1e-5, 1),
            xlabel="Missing or foreign fraction of source association weight",
            ylabel="Fraction of hits / targets ≥ threshold",
            title=title,
        )
        ax.legend(fontsize=8)
    for detector in detectors:
        pp = [r for r in particles if r["detector"] == detector]
        for ax, field, bins in ((axes[1, 0], "pt", PT_BINS), (axes[1, 1], "abs_eta", ETA_BINS)):
            centers, rates = [], []
            for low, high in zip(bins[:-1], bins[1:]):
                selected = [r for r in pp if low <= r[field] < high]
                centers.append(np.sqrt(low * high) if np.isfinite(high) and low > 0 else (high / 2 if low == 0 else low * 1.6))
                rates.append(sum(r["status"] == "represented" for r in selected) / len(selected) if selected else np.nan)
            ax.plot(centers, rates, marker="o", label=detector.upper(), color=colors[detector])
    axes[1, 0].set(
        xscale="log", xlabel="Source particle pT [GeV]", ylabel="Represented fraction", ylim=(0, 1.05), title="One representative per source particle"
    )
    axes[1, 1].set(xlabel="|Source particle η|", ylabel="Represented fraction", ylim=(0, 1.05), title="One representative per source particle")
    for ax in axes[1]:
        ax.legend(fontsize=8)
    fig.savefig(output / "truth_and_retention.png", dpi=160)
    fig.savefig(output / "truth_and_retention.svg")
    plt.close(fig)


def draw_classes(summary, output):
    classes = ("charged_hadron", "electron", "muon", "neutral_hadron", "photon")
    labels = ("ch. hadron", "electron", "muon", "neutral hadron", "photon")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    x = np.arange(len(classes))
    for column, detector in enumerate(("cld", "clic", "idea", "colliderml")):
        report = summary[detector]
        impurity = [1 - report["hit_purity_by_target_class"][name]["weighted"] for name in classes]
        incomplete = [1 - report["target_completeness_by_target_class"][name]["weighted"] for name in classes]
        axes[0, column].bar(x - 0.2, np.maximum(impurity, 1e-6), width=0.4, label="Hit impurity", color="#df8530")
        axes[0, column].bar(x + 0.2, np.maximum(incomplete, 1e-6), width=0.4, label="Target incompleteness", color="#2563a6")
        invisible = []
        for name in classes:
            outcomes = report["by_class"][name]
            total = sum(item["count"] for item in outcomes.values())
            invisible.append(outcomes.get("invisible", {}).get("count", 0) / total if total else 0)
        axes[1, column].bar(x, invisible, color="#2c8a65")
        for row in range(2):
            axes[row, column].set_xticks(x, labels, rotation=35, ha="right")
            axes[row, column].set_title(detector.upper())
        axes[0, column].set(yscale="log", ylim=(1e-6, 0.1))
        axes[1, column].set(ylim=(0, 0.1))
    axes[0, 0].set_ylabel("Foreign or missing association-weight fraction")
    axes[1, 0].set_ylabel("Newly invisible source-particle fraction")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Association quality by output target class; visibility by source particle class")
    fig.savefig(output / "class_breakdown.png", dpi=160)
    fig.savefig(output / "class_breakdown.svg")
    plt.close(fig)


def correct_cached_representatives(hit_rows, event):
    """Identify the single full target row, not every hit with its particle ID."""
    yhit = matrix(event.ytarget_track_calo_hit, len(COL))
    rep_rows = set(np.flatnonzero(yhit[:, COL["PDG"]] != 0))
    for row in hit_rows:
        row["representative"] = row["hit"] in rep_rows
        row["override"] = row["representative"] and row["dominant_mismatch"]


def annotate_target_rows(hit_rows, target_rows, event):
    yhit = matrix(event.ytarget_track_calo_hit, len(COL))
    target = matrix(event.track_calo_target_particles, len(COL))
    classes = [particle_class(row[COL["PDG"]], row[COL["charge"]]) for row in target]
    ids = np.rint(yhit[:, COL["particle_number"]]).astype(int) - 1
    for row in hit_rows:
        row["target_class"] = classes[ids[row["hit"]]]
    for row in target_rows:
        row["target_class"] = classes[row["target"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("local_test_data/track_calo_targets"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/track_calo_targets/truth_validation"))
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--refresh", action="store_true", help="Recompute source links instead of loading cached event audits")
    for detector in ("cld", "clic", "idea"):
        parser.add_argument(f"--{detector}-root", type=Path, default=Path(ROOTS[detector]))
    parser.add_argument("--colliderml-source-dir", type=Path, default=Path("local_test_data/track_calo_targets/colliderml_source"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "event_rows"
    cache_dir.mkdir(exist_ok=True)
    source_paths = [args.cld_root, args.clic_root, args.idea_root]
    source_paths += [
        args.colliderml_source_dir / f"ttbar_pu0_{name}" / "data" / f"ttbar_pu0_{name}" / "train-00000-of-01000.parquet"
        for name in ("particles", "tracks", "calo_hits", "tracker_hits")
    ]
    converted_paths = [next((args.input_dir / detector).glob("*.parquet")) for detector in ("cld", "clic", "idea", "colliderml")]
    signature = {
        "events": args.events,
        "files": {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns] for path in source_paths + converted_paths},
    }
    manifest = cache_dir / "manifest.json"
    use_cache = not args.refresh and (not manifest.exists() or json.loads(manifest.read_text()) == signature)
    all_hits, all_targets, all_particles = [], [], []
    for detector in ("cld", "clic", "idea", "colliderml"):
        files = list((args.input_dir / detector).glob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"Expected one converted file for {detector}: {files}")
        data = ak.from_parquet(files[0])
        if isinstance(data, ak.Record):
            data = ak.Array({name: data[name] for name in data.fields})
        count = min(args.events, len(data))
        cached = [cache_dir / f"{detector}_{iev:04d}.json" for iev in range(count)]
        if use_cache and all(path.exists() for path in cached):
            for iev, path in enumerate(cached):
                hit_rows, target_rows, particle_rows = json.loads(path.read_text())
                correct_cached_representatives(hit_rows, data[iev])
                annotate_target_rows(hit_rows, target_rows, data[iev])
                all_hits.extend(hit_rows)
                all_targets.extend(target_rows)
                all_particles.extend(particle_rows)
            print(f"{detector}: loaded {count} cached event audits", flush=True)
            continue
        source = (
            colliderml_events(count, args.colliderml_source_dir)
            if detector == "colliderml"
            else idea_events(count, args.idea_root) if detector == "idea" else key4hep_events(detector, count, getattr(args, f"{detector}_root"))
        )
        for iev, features, hit_edges, track_edges, calo_mask, representative_edges in source:
            if use_cache and cached[iev].exists():
                hit_rows, target_rows, particle_rows = json.loads(cached[iev].read_text())
                correct_cached_representatives(hit_rows, data[iev])
            else:
                hit_rows, target_rows, particle_rows = audit_event(
                    detector, iev, data[iev], features, hit_edges, track_edges, calo_mask, representative_edges
                )
                cached[iev].write_text(json_text([hit_rows, target_rows, particle_rows]) + "\n")
            annotate_target_rows(hit_rows, target_rows, data[iev])
            all_hits.extend(hit_rows)
            all_targets.extend(target_rows)
            all_particles.extend(particle_rows)
            print(f"{detector} event {iev + 1}/{count}: {len(particle_rows)} source, {len(target_rows)} targets", flush=True)
    summary = summarize(all_hits, all_targets, all_particles)
    (args.output_dir / "summary.json").write_text(json_text(summary) + "\n")
    draw(all_hits, all_targets, all_particles, args.output_dir)
    draw_classes(summary, args.output_dir)
    manifest.write_text(json_text(signature) + "\n")
    print(json_text(summary))


if __name__ == "__main__":
    main()
