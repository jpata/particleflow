"""Attribute retained CMS BF16 particle-count shifts to decision margins and rounding.

Final-head BF16 rounding is a counterfactual diagnostic, not a backend repair:
it cannot restore rounding already omitted inside the network.
"""

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def bf16_round(value):
    """Round finite float32 to BF16, ties to even, then widen exactly."""
    value = np.asarray(value, dtype=np.float32)
    assert np.isfinite(value).all()
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def decode(bid, pid):
    return np.where(bid[:, 1] > bid[:, 0], pid.argmax(axis=-1), 0)


def new_record():
    return {
        "events": 0,
        "elements": 0,
        "reference_particles": 0,
        "candidate_particles": 0,
        "gained": 0,
        "lost": 0,
        "gained_with_reference_bid_tie": 0,
        "gained_with_reference_pid_zero": 0,
        "gained_with_rounded_candidate_bid_tie": 0,
        "reference_bid_ties": 0,
        "candidate_bid_ties": 0,
        "rounded_bid_only_net": 0,
        "rounded_id_only_net": 0,
        "rounded_both_net": 0,
        "rounded_both_gained": 0,
        "rounded_both_lost": 0,
        "gained_by_pid": np.zeros(9, dtype=np.int64),
        "lost_by_pid": np.zeros(9, dtype=np.int64),
        "gained_margins": [],
        "lost_margins": [],
        "event_rows": [],
    }


def add(record, case, ref, candidate):
    rb, ri = ref["bid"], ref["id"]
    cb, ci = candidate["bid"], candidate["id"]
    assert rb.shape == cb.shape and ri.shape == ci.shape
    r, c = decode(rb, ri), decode(cb, ci)
    rounded_bid, rounded_id = bf16_round(cb), bf16_round(ci)
    variants = {
        "rounded_bid_only": decode(rounded_bid, ci),
        "rounded_id_only": decode(cb, rounded_id),
        "rounded_both": decode(rounded_bid, rounded_id),
    }
    gained, lost = (r == 0) & (c != 0), (r != 0) & (c == 0)
    record["events"] += 1
    record["elements"] += len(r)
    record["reference_particles"] += int((r != 0).sum())
    record["candidate_particles"] += int((c != 0).sum())
    record["gained"] += int(gained.sum())
    record["lost"] += int(lost.sum())
    record["gained_with_reference_bid_tie"] += int((gained & (rb[:, 0] == rb[:, 1])).sum())
    record["gained_with_reference_pid_zero"] += int((gained & (ri.argmax(-1) == 0)).sum())
    record["gained_with_rounded_candidate_bid_tie"] += int((gained & (rounded_bid[:, 0] == rounded_bid[:, 1])).sum())
    record["reference_bid_ties"] += int((rb[:, 0] == rb[:, 1]).sum())
    record["candidate_bid_ties"] += int((cb[:, 0] == cb[:, 1]).sum())
    for label, values in variants.items():
        record[label + "_net"] += int((values != 0).sum() - (r != 0).sum())
    rounded = variants["rounded_both"]
    record["rounded_both_gained"] += int(((r == 0) & (rounded != 0)).sum())
    record["rounded_both_lost"] += int(((r != 0) & (rounded == 0)).sum())
    record["gained_by_pid"] += np.bincount(c[gained], minlength=9)
    record["lost_by_pid"] += np.bincount(r[lost], minlength=9)
    record["gained_margins"].append((cb[:, 1] - cb[:, 0])[gained])
    record["lost_margins"].append((cb[:, 1] - cb[:, 0])[lost])
    record["event_rows"].append(
        {
            "case": case,
            "net": int(gained.sum() - lost.sum()),
            "gained": int(gained.sum()),
            "lost": int(lost.sum()),
            "rounded_both_net": int((rounded != 0).sum() - (r != 0).sum()),
        }
    )


def finish(record):
    record["net"] = record["candidate_particles"] - record["reference_particles"]
    for key in ["gained_by_pid", "lost_by_pid"]:
        record[key] = record[key].tolist()
    for key in ["gained_margins", "lost_margins"]:
        values = np.concatenate(record.pop(key))
        record[key + "_quantiles"] = (
            dict(zip(["min", "p10", "p50", "p90", "max"], np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist())) if len(values) else {}
        )
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    cases = json.loads((args.root / "input-manifest.json").read_text())["cases"]
    records = {}
    # Read the archived outputs directly; no extraction or GPU rerun is needed.
    for device in ["amd", "nvidia"]:
        archive = args.root / f"raw-results-{device}.tgz"
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                path = Path(member.name)
                if not member.isfile() or path.suffix != ".npz" or not path.name.startswith("event_"):
                    continue
                parts = path.parts
                if len(parts) != 6 or parts[0] != "results":
                    continue
                _, model, provider, profile, mode, case = parts
                key = f"{model}/{provider}/{mode}"
                records.setdefault(key, new_record())
                raw = tar.extractfile(member).read()
                with np.load(io.BytesIO(raw)) as candidate, np.load(args.root / "reference" / model / case) as ref:
                    add(records[key], case, ref, candidate)
                if records[key]["events"] % 256 == 0:
                    print(key, records[key]["events"], flush=True)
    for model in ["attention", "heptv2"]:
        key = f"{model}/pytorch-amd/native"
        records[key] = new_record()
        for case in cases:
            with (
                np.load(args.root / "reference" / model / case["file"]) as ref,
                np.load(args.root / "reference-amd" / model / case["file"]) as candidate,
            ):
                add(records[key], case["file"], ref, candidate)
    records = {key: finish(value) for key, value in records.items()}
    for record in records.values():
        assert record["events"] == len(cases)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps({"scope": __doc__, "rows": records}, indent=2) + "\n")
    for key, value in records.items():
        print(key, {k: v for k, v in value.items() if k not in ["event_rows"]}, flush=True)


if __name__ == "__main__":
    main()
