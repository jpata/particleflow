# Read and join ColliderML release-1 parquet shards into per-event dicts of awkward arrays.
#
# The four object tables (particles, tracks, calo_hits, tracker_hits) share a shard layout:
# each row of a `train-XXXXX-of-01000.parquet` is one event. The join key is `event_id`; the
# reader enforces that the four tables see the event ids in the same order (they do in the
# release, because shards were written event-aligned) and drops the rare event missing from one
# table or carrying corrupt calo hits.
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# calo region codes the release uses (ODD ECAL barrel/endcaps 9-11, HCAL barrel/endcaps 12-14);
# must match the keys of truth.DEFAULT_CALIBRATION
CALO_REGIONS = (9, 10, 11, 12, 13, 14)


def shard_paths(source_dir: Path, sample: str, object_type: str) -> list[Path]:
    # The colliderml download layout is <source_dir>/<process>_<obj>/data/<process>_<obj>/train-*.parquet.
    # `sample` here is the process label ("ttbar_pu0"); the fixture we ship in the repo has
    # the same layout rooted at the directory the caller passes via --input.
    p = Path(source_dir) / f"{sample}_{object_type}" / "data" / f"{sample}_{object_type}"
    return sorted(p.glob("train-*.parquet"))


def _iter_table_batches(table_path: Path, batch_size: int) -> Iterator[ak.Array]:
    """Yield one table in chunks of `batch_size` events (rows) as awkward arrays.

    The whole-shard alternative holds the full table in memory twice at peak (arrow +
    awkward), which is too much for high-pileup samples. Batched iteration bounds the
    resident footprint to ~batch_size/n_events of the shard.
    """
    pf = pq.ParquetFile(table_path)
    for rb in pf.iter_batches(batch_size=batch_size):
        yield ak.from_arrow(pa.Table.from_batches([rb]))


def _read_event_ids(path: Path) -> np.ndarray:
    return pq.read_table(path, columns=["event_id"])["event_id"].to_numpy()


def common_event_ids(paths: Sequence[Path]) -> np.ndarray | None:
    """Event ids present in every table (in table order), or None if all row counts agree.

    A few release-1 shards lack one event in a single table (ttbar_pu0 train-00891 and
    train-00991: tracks hold 999 rows, e.g. event 991463 is absent from the latter); such
    events are dropped. Only shards whose parquet-metadata row counts differ pay for reading
    the id columns; equal-count shards still get the per-event id check in the reader.
    """
    n_rows = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    if len(set(n_rows)) == 1:
        return None
    ids = [_read_event_ids(p) for p in paths]
    for p, i in zip(paths, ids):
        if len(np.unique(i)) != len(i):
            raise RuntimeError(f"duplicate event_id in {p}")
    common = ids[0]
    for i in ids[1:]:
        common = common[np.isin(common, i)]
    for p, i in zip(paths, ids):
        if not np.array_equal(i[np.isin(i, common)], common):
            raise RuntimeError(f"shared event_ids are ordered differently in {p}")
    return common


def _corrupt_calo_event_ids(calo_path: Path) -> np.ndarray:
    """Event ids whose calo hits cannot be attributed, in table order.

    Release-1 revisions before f8ae2aea (2026-11-06) carried blocks of events with corrupt calo
    truth in 39 calo_hits shards (replaced upstream since; the check stays as a guard): hits
    whose region code is 255 instead of one of CALO_REGIONS (dihiggs_pu0, ggf_pu0; positions
    are ordinary ECAL/HCAL ones, the label is lost), and hits whose contrib_particle_ids and
    contrib_energies lists differ in length (whole shards of dihiggs_pu0, ggf_pu0,
    higgs_portal_pu0, hidden_valley_pu0). Such events are dropped whole.
    Cached (keyed on path, size and mtime) because the resume check, the tqdm total and the
    reader all ask for the same shard.
    """
    st = Path(calo_path).stat()
    return _corrupt_calo_event_ids_cached(str(calo_path), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=8)
def _corrupt_calo_event_ids_cached(calo_path: str, size: int, mtime_ns: int) -> np.ndarray:
    pf = pq.ParquetFile(calo_path)
    regions = pa.array(CALO_REGIONS, type=pa.uint8())
    bad = []
    for rb in pf.iter_batches(batch_size=10, columns=["event_id", "detector", "contrib_particle_ids", "contrib_energies"]):
        ev = rb.column("event_id").to_numpy()
        det = rb.column("detector")
        bad_det = ~pc.is_in(pc.list_flatten(det), value_set=regions).to_numpy(zero_copy_only=False)
        ids, en = rb.column("contrib_particle_ids"), rb.column("contrib_energies")
        mismatch = pc.list_value_length(pc.list_flatten(ids)).to_numpy() != pc.list_value_length(pc.list_flatten(en)).to_numpy()
        rows = np.union1d(
            pc.list_parent_indices(det).to_numpy()[bad_det],
            pc.list_parent_indices(ids).to_numpy()[mismatch],
        )
        bad.append(ev[rows.astype(np.int64)])
    return np.concatenate(bad) if bad else np.zeros(0, dtype=np.uint32)


def kept_event_ids(paths: Sequence[Path]) -> np.ndarray | None:
    """Event ids iter_event_records yields for a full shard (table order), or None for all of them.

    Drops events missing from any table (common_event_ids) and events with corrupt calo hits
    (_corrupt_calo_event_ids); paths are (particles, tracks, calo_hits, tracker_hits).
    """
    keep = common_event_ids(paths)
    corrupt = _corrupt_calo_event_ids(paths[2])
    if len(corrupt) == 0:
        return keep
    if keep is None:
        keep = _read_event_ids(paths[0])
    return keep[~np.isin(keep, corrupt)]


def n_shard_events(paths: Sequence[Path]) -> int:
    """Number of events iter_event_records yields for a full shard."""
    keep = kept_event_ids(paths)
    return pq.ParquetFile(paths[0]).metadata.num_rows if keep is None else len(keep)


def _iter_table_rows(table_path: Path, batch_size: int, keep: np.ndarray | None) -> Iterator[tuple[int, ak.Record]]:
    """Yield (event_id, record without event_id) per row, skipping ids not in `keep` (if given)."""
    for t in _iter_table_batches(table_path, batch_size):
        # Strip the parquet option level so downstream sees plain ndarrays, not MaskedArrays
        # (option types make per-element access slow). Assert the tables carry no Nones.
        for f in t.fields:
            if f == "event_id":
                continue
            if ak.any(ak.is_none(t[f])):
                raise ValueError(f"unexpected None entries in field {f}")
            t[f] = ak.fill_none(t[f], 0)
        ids = np.asarray(ak.to_numpy(t["event_id"]))
        fields = [f for f in t.fields if f != "event_id"]
        for i in np.flatnonzero(np.isin(ids, keep)) if keep is not None else range(len(ids)):
            yield int(ids[i]), ak.Record({f: t[f][i] for f in fields})


def iter_event_records(
    particles_path: Path, tracks_path: Path, calo_path: Path, tracker_hits_path: Path, batch_size: int = 5
) -> Iterator[Dict[str, Any]]:
    """Yield events as {event_id, particles, tracks, calo_hits, tracker_hits}.

    All four tables are required (tracker_hits feeds the gp->track hit-fraction links and
    X_hit_tracker) and row-aligned already; we walk them in lockstep batches of `batch_size`
    events and check event-id alignment per event. Events missing from any table or with
    corrupt calo hits are dropped (see kept_event_ids).
    """
    paths = [particles_path, tracks_path, calo_path, tracker_hits_path]
    names = ["particles", "tracks", "calo_hits", "tracker_hits"]

    common = common_event_ids(paths)
    if common is not None:
        ids = [_read_event_ids(p) for p in paths]
        dropped = np.setdiff1d(np.concatenate(ids), common)
        absent_from = [name for name, i in zip(names, ids) if not np.isin(dropped, i).all()]
        print(f"{Path(particles_path).name}: dropping event_ids {dropped.tolist()} (missing from {', '.join(absent_from)})", flush=True)
    corrupt = _corrupt_calo_event_ids(calo_path)
    if len(corrupt):
        print(
            f"WARNING: {Path(calo_path).name}: dropping {len(corrupt)} event(s) with corrupt calo hits "
            f"(unknown region code or contrib id/energy length mismatch), event_ids {corrupt.min()}..{corrupt.max()}",
            flush=True,
        )
    keep = kept_event_ids(paths)

    for rows in zip(*[_iter_table_rows(p, batch_size, keep) for p in paths]):
        ids = [eid for eid, _ in rows]
        if any(eid != ids[0] for eid in ids[1:]):
            raise RuntimeError(f"event_id mismatch between particles/tracks/calo_hits/tracker_hits: {ids}")
        rec = {"event_id": ids[0]}
        rec.update({name: r for name, (_, r) in zip(names, rows)})
        yield rec
