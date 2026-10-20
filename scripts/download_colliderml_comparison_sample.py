#!/usr/bin/env python3
"""Download one aligned ttbar/no-pileup shard for detector feature validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "64c3d2f112df3d5d20979d22da7cfdff13e10c4b"
OBJECTS = ("particles", "tracks", "calo_hits", "tracker_hits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    def download(obj):
        config = f"ttbar_pu0_{obj}"
        path = hf_hub_download(
            "CERN/ColliderML-Release-1",
            f"data/{config}/train-00000-of-01000.parquet",
            repo_type="dataset",
            revision=REVISION,
            local_dir=args.output_dir / config,
        )
        print(f"Downloaded {obj}: {path}", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, OBJECTS))
    (args.output_dir / "provenance.json").write_text(
        json.dumps({"repo": "CERN/ColliderML-Release-1", "revision": REVISION, "paths": paths}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
