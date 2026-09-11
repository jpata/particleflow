"""Dataset utilities for the initial IDEA track/cluster MLPF pipeline.

The current IDEA samples use truth-seeded ``TracksFromGenParticles`` as an
explicitly temporary proxy.  They do not contain baseline PFO collections, so
the loader can synthesize an oracle ``ycand`` by copying the target.  That
candidate is useful only for exercising serialization and evaluation code; it
is never a physics baseline.
"""

from collections import defaultdict
from pathlib import Path
import random

import awkward as ak
import numpy as np

from mlpf.heptfds.edm4hep_utils.utils_pf import (
    N_X_FEATURES,
    N_Y_FEATURES,
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    labels,
)

TRACK_SOURCE = "truth_seeded"
CANDIDATE_SOURCE = "target_oracle"
SPLIT_SEED = 12345


def _matrix(value, width: int) -> np.ndarray:
    """Convert one variable-length event field into a stable 2-D matrix."""
    arr = np.asarray(ak.to_numpy(value), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    return arr.reshape(-1, width)


def make_oracle_candidates(ytarget: np.ndarray) -> np.ndarray:
    """Return an evaluator-plumbing reference, explicitly copied from truth."""
    return np.array(ytarget, dtype=np.float32, copy=True)


def repair_jet_indices(ytarget: np.ndarray, num_target_jets: int) -> tuple[np.ndarray, int]:
    """Set invalid representative jet indices to the schema value ``-1``."""
    ret = np.array(ytarget, dtype=np.float32, copy=True)
    if not len(ret):
        return ret, 0
    representative = ret[:, 0] > 0
    jet_idx = ret[:, Y_FEATURES.index("jet_idx")]
    invalid = representative & ((jet_idx < -1) | (jet_idx >= num_target_jets))
    ret[invalid, Y_FEATURES.index("jet_idx")] = -1
    return ret, int(np.sum(invalid))


def _map_pid_to_class_index(values: np.ndarray) -> np.ndarray:
    mapped = np.array(values, dtype=np.float32, copy=True)
    lookup = {pid: index for index, pid in enumerate(labels)}
    unknown = sorted(set(mapped[:, 0].astype(int)) - set(lookup)) if len(mapped) else []
    if unknown:
        raise ValueError(f"Unsupported IDEA target PID classes: {unknown}")
    if len(mapped):
        mapped[:, 0] = np.asarray([lookup[int(pid)] for pid in mapped[:, 0]], dtype=np.float32)
    return mapped


def prepare_data_idea(filename: str | Path, event_indices=None):
    """Load IDEA parquet events into the common MLPF tensor contract.

    Missing ``ycand_*`` fields are deliberately replaced with an oracle copy
    of ``ytarget_*``.  This makes the initial file consumable by all existing
    training/evaluation code while keeping the provenance explicit here and
    in the TFDS metadata.
    """
    data = ak.from_parquet(filename)
    num_events = len(data["X_track"])
    indices = range(num_events) if event_indices is None else event_indices
    has_candidates = all(field in data.fields for field in ("ycand_track", "ycand_cluster"))
    examples = []

    for event_index in indices:
        xtrack = _matrix(data["X_track"][event_index], len(X_FEATURES_TRK))
        xcluster = _matrix(data["X_cluster"][event_index], len(X_FEATURES_CL))
        ytrack = _matrix(data["ytarget_track"][event_index], N_Y_FEATURES)
        ycluster = _matrix(data["ytarget_cluster"][event_index], N_Y_FEATURES)
        targetjets = _matrix(data["targetjet"][event_index], 4)
        ytrack, _ = repair_jet_indices(ytrack, len(targetjets))
        ycluster, _ = repair_jet_indices(ycluster, len(targetjets))

        if len(xtrack) != len(ytrack) or len(xcluster) != len(ycluster):
            raise ValueError(
                f"IDEA event {event_index} has inconsistent X/ytarget lengths: "
                f"track={len(xtrack)}/{len(ytrack)}, cluster={len(xcluster)}/{len(ycluster)}"
            )
        if len(xtrack) + len(xcluster) == 0:
            continue

        if xtrack.shape[1] < N_X_FEATURES:
            xtrack = np.pad(xtrack, ((0, 0), (0, N_X_FEATURES - xtrack.shape[1])))
        if xcluster.shape[1] < N_X_FEATURES:
            xcluster = np.pad(xcluster, ((0, 0), (0, N_X_FEATURES - xcluster.shape[1])))

        if has_candidates:
            ctrack = _matrix(data["ycand_track"][event_index], N_Y_FEATURES)
            ccluster = _matrix(data["ycand_cluster"][event_index], N_Y_FEATURES)
            ctrack, _ = repair_jet_indices(ctrack, len(targetjets))
            ccluster, _ = repair_jet_indices(ccluster, len(targetjets))
        else:
            ctrack = make_oracle_candidates(ytrack)
            ccluster = make_oracle_candidates(ycluster)

        x = np.concatenate((xtrack, xcluster), axis=0)
        ytarget = _map_pid_to_class_index(np.concatenate((ytrack, ycluster), axis=0))
        ycand = _map_pid_to_class_index(np.concatenate((ctrack, ccluster), axis=0))
        genjets = _matrix(data["genjet"][event_index], 4)
        genmet = np.float32(data["genmet"][event_index])

        for name, value in (
            ("X", x),
            ("ytarget", ytarget),
            ("ycand", ycand),
            ("genjets", genjets),
            ("targetjets", targetjets),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"IDEA event {event_index} contains non-finite values in {name}")

        examples.append(
            {
                "event_index": int(event_index),
                "X": x.astype(np.float32, copy=False),
                "ytarget": ytarget.astype(np.float32, copy=False),
                "ycand": ycand.astype(np.float32, copy=False),
                "genmet": genmet,
                "genjets": genjets,
                "targetjets": targetjets,
            }
        )
    return examples


def find_idea_parquets(manual_dir: str | Path, process_name: str = "p8_ee_qq_ecm365") -> list[Path]:
    """Find one process' production-style or direct IDEA parquet inputs."""
    manual_dir = Path(manual_dir)
    process_dir = manual_dir / process_name
    files = sorted(process_dir.glob("*.parquet")) if process_dir.is_dir() else sorted(manual_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No IDEA parquet files found under {manual_dir}")
    return files


def split_event_references(files, train_fraction: float = 0.8, seed: int = SPLIT_SEED):
    """Create a deterministic, event-disjoint split that works with one file."""
    references = []
    for filename in files:
        data = ak.from_parquet(filename, columns=["X_track"])
        references.extend((str(filename), event_index) for event_index in range(len(data["X_track"])))
    if len(references) < 2:
        raise ValueError("At least two IDEA events are required for train/test splitting")
    random.Random(seed).shuffle(references)
    boundary = min(max(int(train_fraction * len(references)), 1), len(references) - 1)
    return {"train": references[:boundary], "test": references[boundary:]}


def generate_examples(references):
    """Yield TFDS examples while reading each parquet file only once."""
    by_file = defaultdict(list)
    for filename, event_index in references:
        by_file[filename].append(event_index)
    for filename, event_indices in by_file.items():
        for example in prepare_data_idea(filename, event_indices=event_indices):
            event_index = example.pop("event_index")
            yield f"{filename}_{event_index}", example
