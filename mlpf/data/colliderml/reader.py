# Read and join ColliderML release-1 parquet shards into per-event dicts of awkward arrays.
#
# The four object tables (particles, tracks, calo_hits, tracker_hits) share a shard layout:
# each row of a `train-XXXXX-of-01000.parquet` is one event. The join key is `event_id`; the
# reader enforces that the four tables see the event ids in the same order for the whole shard
# up front (they do in the release, because shards were written event-aligned).
from pathlib import Path
from typing import Any, Dict, Iterator

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


def iter_event_records(
    particles_path: Path, tracks_path: Path, calo_path: Path, tracker_hits_path: Path, batch_size: int = 5
) -> Iterator[Dict[str, Any]]:
    """Yield events as {event_id, particles, tracks, calo_hits, tracker_hits}.

    All four tables are required (tracker_hits feeds the gp->track hit-fraction links and
    X_hit_tracker) and row-aligned already; we walk them in lockstep batches of `batch_size`
    events and slice per event within the batch. Event-id alignment is checked per batch.
    """
    paths = [particles_path, tracks_path, calo_path, tracker_hits_path]
    names = ["particles", "tracks", "calo_hits", "tracker_hits"]

    # cheap up-front row-count check from parquet metadata (no bulk load)
    n_rows = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    if len(set(n_rows)) != 1:
        raise RuntimeError("shard row-count mismatch: " + ", ".join(f"{n}={r}" for n, r in zip(names, n_rows)))

    def _strip_outer(table, i):
        return ak.Record({f: table[f][i] for f in table.fields if f != "event_id"})

    for tables in zip(*[_iter_table_batches(p, batch_size) for p in paths]):
        ids = [np.asarray(ak.to_numpy(t["event_id"])) for t in tables]
        if any(not np.array_equal(i, ids[0]) for i in ids[1:]):
            raise RuntimeError("event_id mismatch between particles/tracks/calo_hits/tracker_hits")
        for i in range(len(tables[0])):
            rec = {"event_id": int(ids[0][i])}
            rec.update({name: _strip_outer(t, i) for name, t in zip(names, tables)})
            yield rec
