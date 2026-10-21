"""Evaluate "columns" (a dataset + a clustering) for the clustering table and plots.

A column is a dict from the specs (cluster_table.yaml, cluster_plots.yaml); `column_result`
evaluates it per event and stores the result in `results_dir`, keyed by a hash of the column's
spec with its clustering config fully resolved (preset names and defaults expanded), the
metrics version and the event-cache tag — so relabelling or restyling a column reuses it, while
changing a preset, a default or a metric definition (bump cluster_metrics.METRICS_VERSION)
recomputes it. Display-only keys (DISPLAY_KEYS) do not enter the key. Metric keys are those of
mlpf/data/cluster_metrics.py (definitions in its header).

Column sources:
  colliderml  ColliderML v1 hits + v20 Pandora (mlpf/data/colliderml/cluster_eval.py): sample
              (hard_scatter | full_pileup), events [first, n], config (a cluster_eval.PRESETS
              name, `pandora`, or a dict of cluster_event kwargs). Full metrics incl. allocator.
  key4hep     EDM4hep ROOT file (CLD / CLIC; jpata/particleflow on Hugging Face), Pandora
              clusters, truth from the key4hep converter's own tables
              (mlpf/data/key4hep/postprocessing.get_genparticles_and_adjacencies): file,
              detector, events [first, n], pileup (optional, for titles; default 0: the
              jpata/particleflow CLD/CLIC samples have no beam-background overlay — no
              MCParticle carries the overlay bit). Full metrics incl. allocator.
  tfds        any MLPF TFDS dataset directory (e.g. CMS): data_dir (the version dir), split,
              events [first, n], track_types / cluster_types (X[:, 0] codes). Counts only
              (clusters, tracks, targets), no hit-level truth.
  cms_ntuple  CMS PF ntuple ROOT file (pfana/pftree, scripts/fetch_test_data_cms.sh): file,
              events [first, n], track_types / cluster_types (element_type codes). Truth
              metrics at PF-cluster level against caloparticles (see run_cms_ntuple).
              `cms_targets: true` also runs CMS's own target building
              (postprocessing2.make_graph + prepare_normalized_table, slow: networkx) for the
              target-count rows; the shared-allocator rows stay empty.
"""

import contextlib
import hashlib
import io
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import mlpf.data.colliderml.cluster_eval as ce
from mlpf.data.cluster_metrics import METRICS_VERSION, aggregate, allocator_metrics, cluster_metrics, pp_metrics

# ---------------------------------------------------------------- sources


def _map(fn, items, workers):
    """[fn(x) for x in items], over a forked Pool if workers > 1."""
    if workers > 1 and len(items) > 1:
        with Pool(min(workers, len(items))) as pool:
            return pool.map(fn, items, chunksize=1)
    return [fn(x) for x in items]


def run_colliderml(col, defaults, workers):
    first, n = col["events"]
    events = ce.load_events(col["sample"], first, n, defaults.get("cache_dir"))
    if col["config"] == "pandora":
        return [ce.pandora_metrics(ev) for ev in events]
    kw = ce.parse_config(col["config"])
    ce.EVENTS = events  # forked workers read the module global
    # compile the numba kernels once here, so forked workers inherit them instead of each paying
    # the JIT cost inside its timed first event
    ev = events[0]
    sl = slice(0, min(len(ev["x"]), 2000))
    ce.cluster_event(ev["x"][sl], ev["y"][sl], ev["z"][sl], ev["e"][sl], ev["det"][sl], **kw)
    res = _map(ce.run_one, [("c", kw, i) for i in range(len(events))], workers)
    return [m for _, _, m in sorted(res, key=lambda r: r[1])]


def _key4hep_event_metrics(gpdata, track_nhits):
    hf = gpdata.hit_features
    sub = np.asarray(hf["subdetector"])
    calo = sub != 3  # 3 = tracker hits, which carry no Pandora cluster
    n_hit = len(sub)
    e = np.asarray(hf["energy"], np.float64)
    cluster_of = np.full(n_hit, -1, np.int64)
    h2c = gpdata.hit_to_cluster
    if len(h2c[0]):
        cluster_of[np.asarray(h2c[0], np.int64)] = np.asarray(h2c[1], np.int64)
    calo_idx = np.full(n_hit, -1, np.int64)
    calo_idx[calo] = np.arange(int(calo.sum()))
    gi, hi, w = (np.asarray(a) for a in gpdata.genparticle_to_hit)
    gi, hi = gi.astype(np.int64), hi.astype(np.int64)
    keep = calo[hi] & (w > 0)
    gen = gpdata.gen_features
    n_gp = len(gen["energy"])
    gen_energy = np.asarray(gen["energy"], np.float64)
    truth = dict(
        hit=calo_idx[hi[keep]],
        leaf=gi[keep],
        w=w[keep].astype(np.float64),
        n_rows=n_gp,
        ispu=np.zeros(n_gp, bool),
        energy=gen_energy,
        target=np.ones(n_gp, bool),  # the links cover the visible particles only
    )
    n_cluster = len(gpdata.cluster_features["type"])
    n_track = len(gpdata.track_features["type"])
    m = cluster_metrics(cluster_of[calo], e[calo], truth, n_tracks=n_track, n_visible=n_gp, dists=True)
    m.update(allocator_metrics(gpdata, n_cluster, n_track))
    trk_hit = ~calo[hi]  # tracker-hit links: one per hit the particle owns
    m.update(
        pp_metrics(
            gen_energy,
            gpdata.genparticle_to_track,
            (truth["leaf"], truth["hit"], truth["w"]),
            np.bincount(gi[trk_hit], minlength=n_gp),
            cluster_of[calo],
            e[calo],
            track_nhits,
            np.asarray(gpdata.track_features["p"], np.float64),
        )
    )
    return m


def run_key4hep(col, defaults, workers):
    import uproot

    import mlpf.data.key4hep.postprocessing as k4
    from mlpf.conf import EDM4HEP

    first, n = col["events"]
    fi = uproot.open(col["file"])
    arrs = fi["events"]
    info = "events___CollectionTypeInfo/events___CollectionTypeInfo"
    meta = fi["podio_metadata"].arrays([f"{info}.name", f"{info}.collectionID"])
    collectionIDs = dict(zip(meta[f"{info}.name"][0], meta[f"{info}.collectionID"][0]))
    collectionIDs_reverse = {v: k for k, v in collectionIDs.items()}
    cellid_encodings = k4.get_cellid_encodings(fi)
    sl = dict(entry_start=first, entry_stop=first + n)
    mc, trk = k4.mc_coll, k4.track_coll
    # the branch lists mirror key4hep/postprocessing.process_one_file (non-IDEA path)
    prop_data = arrs.arrays(
        [
            mc,
            *[
                f"{mc}.{f}"
                for f in (
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
            f"_{mc}_daughters/_{mc}_daughters.index",
            trk,
            f"_{trk}_trackStates",
            "PandoraClusters",
            "_PandoraClusters_hits/_PandoraClusters_hits.index",
            "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
            "PandoraPFOs",
            f"{trk}_dQdx",
        ],
        **sl,
    )

    def links(link):
        ends = [f"_{link}_{end}/_{link}_{end}.{f}" for end in ("to", "from") for f in ("collectionID", "index")]
        return arrs.arrays([f"{link}.weight", *ends], **sl)

    calohit_links, sitrack_links = links("CalohitMCTruthLink"), links("SiTracksMCTruthLink")
    tl = []
    for rel in k4.tracker_hit_relations.values():
        tl += [f"_{rel}_from/_{rel}_from.index", f"_{rel}_to/_{rel}_to.index"]
    for sim in k4.tracker_hit_sim.values():
        tl += [f"_{sim}_particle/_{sim}_particle.index", f"_{sim}_particle/_{sim}_particle.collectionID"]
    tracker_links = arrs.arrays(tl, **sl)
    det = EDM4HEP.DETECTORS[col["detector"]]
    hit_data = {k: arrs[k].array(**sl) for k in det.hit_collections}
    out = []
    for iev in range(n):
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the converter's own numpy where= warnings
            try:
                gpdata = k4.get_genparticles_and_adjacencies(
                    prop_data,
                    hit_data,
                    calohit_links,
                    sitrack_links,
                    tracker_links,
                    iev,
                    collectionIDs,
                    collectionIDs_reverse,
                    collectionIDs[mc],
                    det.b_field,
                    cellid_encodings,
                )
            except ValueError:  # no visible particle (the converter skips these too)
                continue
            trk_arr = prop_data[trk][iev]
            m = _key4hep_event_metrics(gpdata, np.asarray(trk_arr[f"{trk}.trackerHits_end"]) - np.asarray(trk_arr[f"{trk}.trackerHits_begin"]))
        m["time_s"] = np.nan
        out.append(m)
        print(f"  key4hep {col['detector']}: event {first + iev} done ({time.time() - t0:.1f} s)", flush=True)
    return out


def run_tfds(col, defaults, workers):
    import tensorflow_datasets as tfds

    first, n = col["events"]
    ds = tfds.builder_from_directory(col["data_dir"]).as_data_source(split=col.get("split", "test"))
    trk = np.asarray(col["track_types"], np.float32)
    clu = np.asarray(col["cluster_types"], np.float32)
    out = []
    for i in range(first, first + n):
        ev = ds[i]
        t = ev["X"][:, 0]
        n_cl, n_tr = int(np.isin(t, clu).sum()), int(np.isin(t, trk).sum())
        n_tg = int((ev["ytarget"][:, 0] != 0).sum())
        out.append(
            {
                "n_clusters": n_cl,
                "n_tracks": n_tr,
                "n_targets": n_tg,
                "clusters_per_target_num": n_cl,
                "clusters_per_target_den": n_tg,
                "elements_per_target_num": n_cl + n_tr,
                "elements_per_target_den": n_tg,
            }
        )
    return out


def run_cms_ntuple(col, defaults, workers):
    """CMS PF ntuple (pfana/pftree): PF clusters scored against caloparticles at cluster level.

    The ntuple has no per-hit truth, so each PF cluster is one unit: its energy is
    element_energy and caloparticle_to_element_cmp gives each caloparticle's energy in it (GeV;
    they add up to the element energy). A particle's "deposit" is therefore its energy inside
    PF clusters (unclustered deposits are not recorded), and the clustered energy fraction is
    not defined. Superclusters (type 10, built from ECAL clusters) are counted as clusters but
    left out of the truth metrics, to avoid double counting.
    """
    import uproot

    first, n = col["events"]
    clu_types = np.asarray(col["cluster_types"])
    trk_types = np.asarray(col["track_types"])
    br = [
        "element_type",
        "element_energy",
        "element_num_hits",
        "caloparticle_pid",
        "caloparticle_charge",
        "caloparticle_energy",
        "caloparticle_ev",
        "caloparticle_idx_trackingparticle",
        "caloparticle_to_element.first",
        "caloparticle_to_element.second",
        "caloparticle_to_element_cmp",
        "trackingparticle_to_element.first",
        "trackingparticle_to_element.second",
    ]
    tree = uproot.open(col["file"])["pfana/pftree"]
    a = tree.arrays(br, entry_start=first, entry_stop=first + n)
    n_targets = None
    if col.get("cms_targets", False):
        global _CMS_EV
        _CMS_EV = tree.arrays(library="np", entry_start=first, entry_stop=first + n)
        n_targets = _map(_cms_targets_one, list(range(len(a))), workers)
    out = []
    for iev in range(len(a)):
        ev = {k: np.asarray(a[k][iev]) for k in br}
        typ = ev["element_type"]
        is_clu = np.isin(typ, clu_types)
        units = np.nonzero(is_clu & (typ != 10))[0]  # truth-scored clusters
        unit_of = np.full(len(typ), -1, np.int64)
        unit_of[units] = np.arange(len(units))
        cp, el, w = ev["caloparticle_to_element.first"], ev["caloparticle_to_element.second"], ev["caloparticle_to_element_cmp"]
        keep = (unit_of[el] >= 0) & (w > 0)
        n_cp = len(ev["caloparticle_pid"])
        truth = dict(
            hit=unit_of[el[keep]],
            leaf=cp[keep].astype(np.int64),
            w=w[keep].astype(np.float64),
            n_rows=n_cp,
            ispu=ev["caloparticle_ev"] != 0,
            energy=ev["caloparticle_energy"].astype(np.float64),
        )
        # targets: caloparticles with energy in a PF cluster or a track (via their tracking particle)
        tp_el_t, tp_el_e = ev["trackingparticle_to_element.first"], ev["trackingparticle_to_element.second"]
        tps_with_track = set(tp_el_t[np.isin(typ[tp_el_e], trk_types)].tolist())
        has_track = np.array([int(t) in tps_with_track for t in ev["caloparticle_idx_trackingparticle"]], dtype=bool)
        visible = has_track.copy()
        visible[np.unique(cp[keep])] = True
        m = cluster_metrics(
            np.arange(len(units)),
            ev["element_energy"][units].astype(np.float64),
            truth,
            n_tracks=int(np.isin(typ, trk_types).sum()),
            n_visible=int(visible.sum()),
            dists=True,
        )
        m["_dist"]["clu_nhit"] = ev["element_num_hits"][units].astype(np.int32)  # hits of each PF cluster
        # counts over all PF clusters (superclusters included, as in the TFDS column)
        nh, e = ev["element_num_hits"][is_clu], ev["element_energy"][is_clu]
        n_cl = int(is_clu.sum())
        m.update(
            n_clusters=n_cl,
            n_clusters_gt0p1=int(np.sum(e > 0.1)),
            n_clusters_1hit=int(np.sum(nh == 1)),
            clu_nhit_max=int(nh.max()) if n_cl else 0,
            clu_E_max=float(e.max()) if n_cl else 0.0,
            clusters_per_visible_num=n_cl,
            elements_per_visible_num=n_cl + m["n_tracks"],
            time_s=np.nan,
        )
        if n_targets is not None:  # CMS's own targets (as in its TFDS)
            nt = n_targets[iev]
            m.update(
                n_targets=nt,
                clusters_per_target_num=n_cl,
                clusters_per_target_den=nt,
                elements_per_target_num=n_cl + m["n_tracks"],
                elements_per_target_den=nt,
            )
        # each unit is a whole PF cluster: the dropped-energy fraction, the removed hits (none)
        # and the hit-count purity/recall (always 1 at this granularity) are not meaningful
        for k in [k for k in m if k.startswith("dropped_E_frac") or "_hits_" in k or k == "n_removed_hits"]:
            m.pop(k)
        out.append(m)
    return out


_CMS_EV = None  # the ntuple's event arrays, shared with forked workers


def _cms_targets_one(iev):
    """CMS's own target building on one ntuple event (make_graph + prepare_normalized_table of
    mlpf/data/cms/postprocessing2.py), then the TFDS builder's PS/brem element mask: the targets
    per event exactly as they end up in the CMS TFDS."""
    import mlpf.data.cms.postprocessing2 as cms

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xelem, _, ytarget = cms.prepare_normalized_table(cms.make_graph(_CMS_EV, iev), iev)
    typ = np.asarray(Xelem["typ"])
    keep = ~np.isin(typ, (2, 3, 7))  # mlpf/heptfds/cms_pf/cms_utils.py drops PS1/PS2/BREM
    return int((np.asarray(ytarget["pid"])[keep] != 0).sum())


SOURCES = {"colliderml": run_colliderml, "key4hep": run_key4hep, "tfds": run_tfds, "cms_ntuple": run_cms_ntuple}


# ---------------------------------------------------------------- result cache


DISPLAY_KEYS = ("label", "bold", "slot")  # how a column is shown, not what is computed


def _strip_display(col):
    """Column spec without its display-only keys (labels must not key the result cache)."""
    return {k: v for k, v in col.items() if k not in DISPLAY_KEYS}


def _sorted(x):
    """json.dumps(sort_keys=True)-canonical form: dict keys as strings, in json's sort order
    (sorted by the ORIGINAL key, so int-keyed dicts stay numeric-ordered, then stringified)."""
    if isinstance(x, dict):
        return {str(k): _sorted(v) for k, v in sorted(x.items(), key=lambda item: item[0])}
    if isinstance(x, (list, tuple)):
        return [_sorted(v) for v in x]
    return x


def _col_key(col):
    """Result-cache key: the column with its clustering config fully resolved (preset names and
    defaults expanded, so changing a preset or a default invalidates the result), plus the
    metrics version and, for ColliderML, the decoded-event cache tag."""
    spec = dict(col, metrics_version=METRICS_VERSION)
    if col["source"] == "colliderml":
        spec["cache_tag"] = ce.CACHE_TAG
        if col["config"] != "pandora":
            spec["config"] = ce.config_json(ce.parse_config(col["config"]))
    # no sort_keys here: _sorted already orders everything (a second sort would re-order the
    # stringified int keys lexicographically, "10" < "9")
    return hashlib.sha1(json.dumps(_sorted(spec), default=str).encode()).hexdigest()[:12]


def _pack(per_event):
    """Per-event values and distributions as flat arrays (npz): ev_<key> holds one value per
    event for every scalar metric (nan where an event lacks it); each `_dist` array is
    concatenated over events."""
    keys = sorted({k for m in per_event for k, v in m.items() if not k.startswith("_") and np.isscalar(v)})
    out = {f"ev_{k}": np.array([m.get(k, np.nan) for m in per_event], np.float64) for k in keys}
    dists = [m.get("_dist", {}) for m in per_event]
    for k in sorted({k for d in dists for k in d}):
        out[k] = np.concatenate([np.asarray(d[k]) for d in dists if k in d])
    return out


def column_result(col, defaults, workers, recompute, per_event=False):
    """A column's aggregated metrics (and, with per_event, its per-event values and
    distributions under "events"), from results_dir if stored, else computed and stored: the
    aggregates as <source>_<key>.json, the per-event arrays as <source>_<key>.npz."""
    rdir = Path(defaults["results_dir"])
    col = _strip_display(col)
    stem = rdir / f"{col['source']}_{_col_key(col)}"
    path, npz = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if path.exists() and (npz.exists() or not per_event) and not recompute:
        res = json.loads(path.read_text())
        if per_event:
            with np.load(npz) as f:
                res["events"] = dict(f)
        return res
    print(f"computing column: {col}", flush=True)
    per_ev = SOURCES[col["source"]](col, defaults, workers)
    res = {"column": col, "n_events": len(per_ev), "metrics": aggregate(per_ev)}
    rdir.mkdir(parents=True, exist_ok=True)
    packed = _pack(per_ev)
    np.savez_compressed(npz, **packed)
    path.write_text(json.dumps(res, indent=1, default=float))
    if per_event:
        res["events"] = packed
    return res


def dataset_id(col):
    """Identity of the dataset behind a column (everything but the clustering config and the
    display-only keys): only columns on the same dataset are compared (table: 'best' bolding;
    plots: one panel)."""
    return json.dumps(_sorted({k: v for k, v in _strip_display(col).items() if k != "config"}), default=str)


def dataset_title(col):
    """Short name of a column's dataset, e.g. "ColliderML ttbar PU200, events 50-52"."""
    first, n = col.get("events", [None, None])
    span = f", events {first}-{first + n - 1}" if first is not None else ""
    if col["source"] == "colliderml":
        return f"ColliderML ttbar {'PU0' if col['sample'] == 'hard_scatter' else 'PU200'}{span}"
    if col["source"] == "key4hep":
        return f"{col['detector'].upper()} ttbar PU{col.get('pileup', 0)}{span}"
    if col["source"] == "cms_ntuple":
        return f"CMS ttbar (PF ntuple){span}"
    return f"{Path(col['data_dir']).parent.parent.name} ({col.get('split', 'test')}){span}"
