# Read and join ColliderML release-1 parquet shards into per-event dicts of awkward arrays.
#
# The four object tables (particles, tracks, calo_hits, tracker_hits) share a shard layout:
# each row of a `train-XXXXX-of-01000.parquet` is one event. The join key is `event_id`; the
# reader enforces that the four tables see the event ids in the same order (they do in the
# release, because shards were written event-aligned) and drops the rare event missing from one
# table.
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


def n_shard_events(paths: Sequence[Path]) -> int:
    """Number of events iter_event_records yields for a full shard."""
    common = common_event_ids(paths)
    return pq.ParquetFile(paths[0]).metadata.num_rows if common is None else len(common)


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
    events and check event-id alignment per event. Events missing from any table are dropped
    (see common_event_ids).
    """
    paths = [particles_path, tracks_path, calo_path, tracker_hits_path]
    names = ["particles", "tracks", "calo_hits", "tracker_hits"]

    keep = common_event_ids(paths)
    if keep is not None:
        ids = [_read_event_ids(p) for p in paths]
        dropped = np.setdiff1d(np.concatenate(ids), keep)
        absent_from = [name for name, i in zip(names, ids) if not np.isin(dropped, i).all()]
        print(f"{Path(particles_path).name}: dropping event_ids {dropped.tolist()} (missing from {', '.join(absent_from)})", flush=True)

    for rows in zip(*[_iter_table_rows(p, batch_size, keep) for p in paths]):
        ids = [eid for eid, _ in rows]
        if any(eid != ids[0] for eid in ids[1:]):
            raise RuntimeError(f"event_id mismatch between particles/tracks/calo_hits/tracker_hits: {ids}")
        rec = {"event_id": ids[0]}
        rec.update({name: r for name, (_, r) in zip(names, rows)})
        yield rec
