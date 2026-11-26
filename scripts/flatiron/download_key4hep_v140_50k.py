#!/usr/bin/env python3
"""Download the Key4hep v1.4.0_50k campaign into the Flatiron H100 TFDS roots."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM = REPO_ROOT / "configs/training/platforms/flatiron_h100.yaml"
REMOTE = "davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"
CAMPAIGN = "v1.4.0_50k"
DETECTORS = ("cld", "clic", "idea", "ild")
DOWNLOAD_BYTES_PER_SECOND = 125_000_000  # 1 Gbps across all workers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the transfer commands")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "davix.sif")
    parser.add_argument("--proxy", type=Path, help="override X509_USER_PROXY or ~/x509-cms-2048")
    parser.add_argument("--jobs", type=int, default=8, help="parallel Davix workers per detector")
    args = parser.parse_args()

    with PLATFORM.open() as config:
        data_dirs = yaml.safe_load(config)["data_dir"]
    destinations = {detector: data_dirs[detector] for detector in DETECTORS}
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/tfds_webdav_tree.py"),
        "--resume",
        f"--image={args.image}",
        f"--jobs={args.jobs}",
        f"--limit-rate={DOWNLOAD_BYTES_PER_SECOND}",
    ]
    if args.proxy is not None:
        command.extend(["--proxy", str(args.proxy)])
    if args.dry_run:
        command.append("--dry-run")

    for detector, local_root in destinations.items():
        remote_url = f"{REMOTE}/{detector}/{CAMPAIGN}/"
        print(f"{remote_url} -> {local_root}", flush=True)
        result = subprocess.run([*command, "--remote-url", remote_url, "--local-root", local_root], check=False)
        if result.returncode:
            raise SystemExit(f"Download failed for {detector}. Rerun to resume.")


if __name__ == "__main__":
    main()
