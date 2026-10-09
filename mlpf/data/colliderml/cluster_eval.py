# Evaluate calorimeter clusterings on ColliderML ttbar events, against truth and against Pandora.
#
# Reads the v20 pilot events (Pandora clusters on DDCaloDigi cells) together with the same events
# of release 1 (v1 calo hits = our clusterer's production input; event_id joins 1:1), decodes
# them once into per-event arrays (cached as pickles), and scores any clustering setting with
# mlpf/data/colliderml/cluster_metrics.py. Pandora is scored the same way, at run time, so both
# always use the current metric definitions.
import json
import os
import pickle
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlpf.data.colliderml.cluster_metrics import allocator_metrics as _allocator_metrics
from mlpf.data.colliderml.cluster_metrics import cluster_metrics, hit_truth
from mlpf.data.colliderml.cluster_metrics import pp_metrics as _pp_metrics
from mlpf.data.colliderml.clue import CLUE_PRESETS, resolve_clue_config
from mlpf.data.colliderml.clustering import DEFAULT_MERGE_FRAC, DEFAULT_REGION_RADII_MM, cluster_event
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, calibration_factors, compute_gen_tables
from mlpf.data.target_building import EventData

V1_ROOT = Path("/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1")
V20_ROOT = Path("/mnt/ceph/users/ewulff/data/colliderml/v20")
V1_SAMPLE = {"hard_scatter": "ttbar_pu0", "full_pileup": "ttbar_pu200"}
V1_EVENTS_PER_SHARD = {"hard_scatter": 1000, "full_pileup": 100}
# decoded-event cache format; bump whenever decode_event's output changes
CACHE_TAG = "v4"

_PU200_RADII = {**{k: 16.25 for k in (9, 10, 11)}, **{k: 36.0 for k in (12, 13, 14)}}
PRESETS = {
    # pre-CLUE production settings (scripts/flatiron/colliderml_convert*.sh, 2026-10-05)
    "bfs_merge_pu0": dict(algorithm="bfs_merge", merge_frac=0.25),
    "bfs_pu200": dict(algorithm="bfs", merge_frac=0.0, radii_mm=_PU200_RADII),
    "clue": dict(algorithm="clue"),
    "clue_pu200": dict(algorithm="clue", clue_params=CLUE_PRESETS["pu200"]["params"], clue_options=CLUE_PRESETS["pu200"]["options"]),
    # same, dropping single-hit clusters (isolated soft leftovers; Pandora-like)
    "clue_no1hit": dict(algorithm="clue", clue_options={"drop_1hit": True}),
    "clue_pu200_no1hit": dict(
        algorithm="clue",
        clue_params=CLUE_PRESETS["pu200"]["params"],
        clue_options={**CLUE_PRESETS["pu200"]["options"], "drop_1hit": True},
    ),
    # same, dropping clusters below 0.1 GeV (isolated soft leftovers; Pandora-like)
    "clue_dropE0p1": dict(algorithm="clue", clue_options={"drop_E": 0.1}),
    "clue_pu200_dropE0p1": dict(
        algorithm="clue",
        clue_params=CLUE_PRESETS["pu200"]["params"],
        clue_options={**CLUE_PRESETS["pu200"]["options"], "drop_E": 0.1},
    ),
}

PARTICLE_COLS = ["event_id", "particle_id", "pdg_id", "parent_id", "primary", "energy", "charge", "vertex_primary", "px", "py", "pz"]
TRACK_COLS = ["event_id", "majority_particle_id", "hit_ids", "qop"]
TRACKER_HIT_COLS = ["event_id", "particle_id"]
V1_HIT_COLS = ["event_id", "detector", "total_energy", "x", "y", "z", "contrib_particle_ids", "contrib_energies"]
CELL_COLS = ["event_id", "cell_id", "detector", "energy", "x", "y", "z", "contrib_particle_ids", "contrib_energies"]

EVENTS = []  # decoded events; module-level so forked worker processes share them


def _int_keys(d):
    return {int(k): v for k, v in d.items()} if isinstance(d, dict) else d


def normalize_kwargs(kw):
    """cluster_event kwargs with int region keys (json/yaml give strings or ints)."""
    kw = dict(kw)
    for key in ("radii_mm", "clue_params"):
        if key in kw:
            kw[key] = _int_keys(kw[key])
    return kw


def parse_config(cfg):
    """A preset name or a dict of cluster_event kwargs -> normalized kwargs."""
    return normalize_kwargs(PRESETS[cfg] if isinstance(cfg, str) else cfg)


def resolved_kwargs(kw):
    """The full setting cluster_event will run for kw (every default filled in): what result
    caches must be keyed on, so a changed preset or default invalidates them."""
    kw = normalize_kwargs(kw)
    algo = kw.get("algorithm", "bfs_merge")
    if algo == "clue":
        params, options = resolve_clue_config(params=kw.get("clue_params"), options=kw.get("clue_options"))
        return dict(algorithm="clue", clue_params=params, clue_options=options)
    merge_frac = kw.get("merge_frac")
    if merge_frac is None:
        merge_frac = DEFAULT_MERGE_FRAC if algo == "bfs_merge" else 0.0
    return dict(algorithm=algo, merge_frac=merge_frac, radii_mm=kw.get("radii_mm") or DEFAULT_REGION_RADII_MM)


def _rows(path: Path, cols, start: int, n: int):
    """Yield single-event awkward records for rows [start, start+n) of one parquet file,
    reading only the row groups that hold them."""
    pf = pq.ParquetFile(path)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    groups = [i for i in range(len(sizes)) if edges[i] < start + n and edges[i + 1] > start]
    if not groups:
        return
    seen = int(edges[groups[0]])
    for rb in pf.iter_batches(batch_size=1, columns=cols, row_groups=groups):
        if seen >= start + n:
            return
        if seen >= start:
            yield ak.from_arrow(rb)[0]
        seen += 1


def iter_events(sample: str, first: int, n: int, with_tracks: bool = True):
    """Yield (v1 particles, v1 hits, v20 cells, pandora cell lists[, v1 tracks, v1 tracker hits,
    {"n_tracks": v20 track count}])."""
    s = V1_SAMPLE[sample]
    shard, start = divmod(first, V1_EVENTS_PER_SHARD[sample])
    assert start + n <= V1_EVENTS_PER_SHARD[sample], "events must come from one v1 shard"

    def v1(obj):
        return V1_ROOT / f"{s}_{obj}" / "data" / f"{s}_{obj}" / f"train-{shard:05d}-of-01000.parquet"

    reco = V20_ROOT / sample / "ttbar" / "v20" / "parquet" / "reco"

    def v20(obj):
        for f in sorted((reco / obj).glob(f"{sample}.ttbar.v20.reco.{obj}.events*.parquet")):
            a, b = f.name.split(".events")[1].split(".")[0].split("-")
            if int(a) <= first <= int(b):
                assert first + n - 1 <= int(b), "events must come from one v20 file"
                return f, first - int(a)
        raise FileNotFoundError(f"no v20 {obj} file covers event {first}")

    fc, oc = v20("calo_cells")
    fk, ok = v20("calo_clusters")
    gens = [
        _rows(v1("particles"), PARTICLE_COLS, start, n),
        _rows(v1("calo_hits"), V1_HIT_COLS, start, n),
        _rows(fc, CELL_COLS, oc, n),
        _rows(fk, ["event_id", "cell_ids"], ok, n),
    ]
    v20_ntracks = None
    if with_tracks:
        gens += [_rows(v1("tracks"), TRACK_COLS, start, n), _rows(v1("tracker_hits"), TRACKER_HIT_COLS, start, n)]
        # the v20 tracks files are NOT stored in event order (unlike cells/clusters and all v1
        # tables), so their per-event track count is looked up by event_id, not by row
        ft, _ = v20("tracks")
        t = pq.read_table(ft, columns=["event_id", "track_id"])
        v20_ntracks = dict(zip(t["event_id"].to_numpy().tolist(), pc.list_value_length(t["track_id"]).to_numpy().tolist()))
    for recs in zip(*gens):
        eids = {int(r["event_id"]) for r in recs}
        assert len(eids) == 1, f"event_id mismatch {eids}"
        if v20_ntracks is not None:
            recs = (*recs, {"n_tracks": v20_ntracks[eids.pop()]})
        yield recs


def _np(a, dt=None):
    v = np.asarray(ak.to_numpy(a))
    return v.astype(dt) if dt is not None else v


def decode_event(parts, hits, cells, clus, tracks=None, tracker=None, tracks_v20=None):
    """One event as plain arrays: v1 hits + truth for our clusterers, the converter's truth
    tables for the allocator metrics (with tracks), and Pandora's clustering of the v20 cells
    (cluster assignment + truth, scored at run time by pandora_metrics)."""
    det = _np(hits["detector"], np.int64)
    ev = dict(
        event_id=int(parts["event_id"]),
        x=_np(hits["x"], np.float64),
        y=_np(hits["y"], np.float64),
        z=_np(hits["z"], np.float64),
        e=_np(hits["total_energy"], np.float64) * calibration_factors(det, DEFAULT_CALIBRATION),
        det=det,
        truth=hit_truth(parts, hits),
    )
    if tracks is not None:
        gen, gp_to_hit, gp_to_track, _ = compute_gen_tables(parts, hits, tracks, DEFAULT_CALIBRATION, tracker_ev=tracker)
        ev["alloc"] = dict(
            gen=gen,
            gp_to_hit=gp_to_hit,
            gp_to_track=gp_to_track,
            n_track=len(_np(tracks["majority_particle_id"])),
            n_tracker=len(_np(tracker["particle_id"])),
            track_nhits=np.asarray(ak.to_numpy(ak.num(tracks["hit_ids"])), np.int64),
            track_p=1.0 / np.maximum(np.abs(_np(tracks["qop"], np.float64)), 1e-12),  # ACTS q/p in e/GeV
        )
    # Pandora on the v20 cells: cell energies are DDCaloDigi GeV, the same scale as our
    # calibrated v1 hits (Pandora's own cluster energies are recalibrated and not used)
    cell_id = _np(cells["cell_id"], np.uint64)
    order = np.argsort(cell_id)
    cluster_of = np.full(len(cell_id), -1, np.int64)
    for k, ids in enumerate(clus["cell_ids"]):
        ids = _np(ids, np.uint64)
        pos = order[np.searchsorted(cell_id[order], ids)]
        assert np.all(cell_id[pos] == ids)
        cluster_of[pos] = k
    ev["pandora"] = dict(cluster_of=cluster_of, e=_np(cells["energy"], np.float64), truth=hit_truth(parts, cells))
    if tracks is not None:
        # targets are the converter's (v1) visible set; Pandora's elements are its own v20 tracks
        ev["pandora"].update(n_tracks=int(tracks_v20["n_tracks"]), n_visible=len(gen["energy"]))
    return ev


def pandora_metrics(ev):
    """cluster_metrics of the event's Pandora clustering (computed now, current definitions)."""
    p = ev["pandora"]
    counts = {k: p[k] for k in ("n_tracks", "n_visible") if k in p}
    m = cluster_metrics(p["cluster_of"], p["e"], p["truth"], **counts)
    m["time_s"] = np.nan
    return m


def allocator_metrics(alloc, n_hit: int, hit_to_cluster, n_cluster: int):
    """The shared allocator summary (cluster_metrics.allocator_metrics) on one ColliderML event."""
    gd = EventData(
        alloc["gen"],
        {"type": np.zeros(n_hit + alloc["n_tracker"], np.float32)},
        {"type": np.zeros(n_cluster, np.float32)},
        {"type": np.zeros(alloc["n_track"], np.float32)},
        alloc["gp_to_hit"],
        alloc["gp_to_track"],
        hit_to_cluster,
        (np.array([]), np.array([])),
    )
    return _allocator_metrics(gd, n_cluster, alloc["n_track"])


def training_view_metrics(alloc, n_hit: int, cluster_of, hit_e):
    """cluster_metrics.pp_metrics on one ColliderML event: compute_gen_tables' gp_to_hit holds
    calo hits (index < n_hit, deposit in GeV) and zero-weight tracker-hit links after them."""
    gi, hi, w = (np.asarray(a) for a in alloc["gp_to_hit"])
    gi, hi = gi.astype(np.int64), hi.astype(np.int64)
    n_gp = len(alloc["gen"]["energy"])
    calo = hi < n_hit
    return _pp_metrics(
        np.asarray(alloc["gen"]["energy"], np.float64),
        alloc["gp_to_track"],
        (gi[calo], hi[calo], w[calo]),
        np.bincount(gi[~calo], minlength=n_gp),
        cluster_of,
        hit_e,
        alloc["track_nhits"],
        alloc["track_p"],
    )


def load_events(sample: str, first: int, n: int, cache: str | None, with_tracks: bool = True):
    """Decoded events [first, first + n) of a sample, from cache/<sample>_<first>_<n>_<tag>.pkl
    if present (written atomically on first use)."""
    tag = CACHE_TAG + ("" if with_tracks else "_notracks")
    path = Path(cache) / f"{sample}_{first}_{n}_{tag}.pkl" if cache else None
    if path is not None and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    events = []
    for rec in iter_events(sample, first, n, with_tracks):
        events.append(decode_event(*rec))
        print(f"decoded event {events[-1]['event_id']}", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:  # write-then-rename: a concurrent job never reads a partial file
            pickle.dump(events, f)
        os.replace(tmp, path)
    return events


def run_one(job):
    """(label, cluster_event kwargs, event index into EVENTS) -> (label, index, metrics)."""
    name, kw, i = job
    ev = EVENTS[i]
    t0 = time.time()
    cluster_of, feats, hit_to_cluster, _ = cluster_event(ev["x"], ev["y"], ev["z"], ev["e"], ev["det"], **kw)
    t_cluster = time.time() - t0  # clustering only (metrics and allocator excluded)
    counts = dict(n_tracks=ev["alloc"]["n_track"], n_visible=len(ev["alloc"]["gen"]["energy"])) if "alloc" in ev else {}
    m = cluster_metrics(cluster_of, ev["e"], ev["truth"], **counts)
    m["time_s"] = t_cluster
    if "alloc" in ev:
        m.update(allocator_metrics(ev["alloc"], len(ev["x"]), hit_to_cluster, len(feats)))
        m.update(training_view_metrics(ev["alloc"], len(ev["x"]), cluster_of, ev["e"]))
    return name, i, m


def config_json(kw) -> str:
    """Canonical json of resolved cluster_event kwargs (int keys sorted numerically)."""

    def _canon(x):
        if isinstance(x, dict):
            return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda item: (str(type(item[0])), item[0]))}
        return x

    return json.dumps(_canon(resolved_kwargs(kw)), default=str)
