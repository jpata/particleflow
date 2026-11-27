#!/usr/bin/env python3
"""Download the Key4hep v1.4.0_50k campaign into the Flatiron H100 TFDS roots."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "v1.4.0_50k"
DETECTORS = ("cld", "clic", "idea", "ild")
DOWNLOAD_BYTES_PER_SECOND = 125_000_000  # 1 Gbps across all workers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list the transfer plan for each detector")
    parser.add_argument("--proxy", type=Path, help="override X509_USER_PROXY or ~/x509-cms-2048")
    parser.add_argument("--capath", type=Path, help="trusted CA bundle or OpenSSL-hashed directory")
    parser.add_argument("--jobs", type=int, default=8, help="parallel WebDAV workers per detector")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/tfds_webdav.py"),
        "download",
        "--all",
        "--resume",
        "--platform=flatiron_h100",
        f"--campaign={CAMPAIGN}",
        f"--jobs={args.jobs}",
        f"--limit-rate={DOWNLOAD_BYTES_PER_SECOND}",
    ]
    if args.proxy is not None:
        command.extend(["--proxy", str(args.proxy)])
    if args.capath is not None:
        command.extend(["--capath", str(args.capath)])
    if args.dry_run:
        command.append("--dry-run")

    for detector in DETECTORS:
        print(f"{detector} / {CAMPAIGN}", flush=True)
        result = subprocess.run([*command, "--detector", detector], check=False)
        if result.returncode:
            raise SystemExit(f"Download failed for {detector}. Rerun to resume.")


if __name__ == "__main__":
    main()
