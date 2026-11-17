"""ILD PF conversion with explicit provenance and no silently skipped events."""

from pathlib import Path

import awkward as ak
import numpy as np

from mlpf.heptfds.edm4hep_utils.utils_pf import prepare_data_clic
from mlpf.data.key4hep.ild import PROVENANCE


def _root_digest(data, path):
    if "metadata_input_sha256" not in data.fields:
        raise ValueError(f"{path}: missing source ROOT checksum")
    values = set(ak.to_list(data["metadata_input_sha256"]))
    if len(values) != 1 or any(not isinstance(value, str) or len(value) != 64 for value in values):
        raise ValueError(f"{path}: invalid source ROOT checksums")
    return values.pop()


def generate_examples(files):
    for path in files:
        data = ak.from_parquet(path)
        for field, expected in PROVENANCE.items():
            values = ak.to_list(data[field]) if field in data.fields else None
            if not isinstance(values, list) or len(values) != len(data["X_track"]) or any(value != expected for value in values):
                raise ValueError(f"{path}: missing or incompatible {field}")
        _root_digest(data, path)
        rows = prepare_data_clic(path)
        if len(rows[0]) != len(data["X_track"]):
            raise ValueError(f"{path}: PF conversion discarded events")
        for index, (x, target, candidate, met, jets, targetjets) in enumerate(zip(*rows)):
            arrays = dict(
                X=x,
                ytarget=target,
                ycand=candidate,
                genjets=jets,
                targetjets=targetjets,
            )
            for name, value in arrays.items():
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{path}:{index}: nonfinite {name}")
            met = float(np.asarray(met).reshape(-1)[0])
            if not np.isfinite(met):
                raise ValueError(f"{path}:{index}: nonfinite genmet")
            yield (
                f"{Path(path).resolve()}:{index}",
                {
                    **{name: value.astype(np.float32) for name, value in arrays.items()},
                    "genmet": np.float32(met),
                },
            )


def split_sample(path, builder_config):
    if builder_config.name != "1":
        raise ValueError("ILD qualification has one builder configuration")
    files = sorted(Path(path).glob("*.parquet"))
    if len(files) < 2:
        raise ValueError("ILD needs at least two independent ROOT/Parquet jobs for disjoint train/test files")
    digests = [_root_digest(ak.from_parquet(path, columns=["metadata_input_sha256"]), path) for path in files]
    if len(set(digests)) != len(digests):
        raise ValueError("ILD input contains duplicate ROOT jobs; train/test splits must be independent")
    boundary = min(len(files) - 1, max(1, int(0.9 * len(files))))
    return {
        "train": generate_examples(files[:boundary]),
        "test": generate_examples(files[boundary:]),
    }
