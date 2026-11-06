#!/usr/bin/env python3
"""Upload all CLD granularity scan TFDS roots with verified Davix transfers."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/local/joosep/mlpf/cld/granularity_scan_2025-05-29")
POINTS = (
    "ecal_1p5mm",
    "ecal_2mm",
    "ecal_2p5mm",
    "ecal_3mm",
    "ecal_4mm",
    "ecal_5p1mm",
    "hcal_10mm",
    "hcal_12mm",
    "hcal_15mm",
    "hcal_20mm",
    "hcal_25mm",
    "hcal_30mm",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=ROOT, help="directory containing scan-point workspaces")
    parser.add_argument("--image", type=Path, default=Path("davix.sif"))
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--remote-root")
    parser.add_argument("--point", action="append", choices=POINTS, help="repeat to select scan points; default: all twelve")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [point for point in POINTS if not args.point or point in args.point]

    # Validate the entire selection before writing any remote data.
    tasks = []
    total_files = total_bytes = 0
    for point in selected:
        tfds = args.local_root / point / "tfds"
        sample = "z_tautau" if point.startswith("ecal_") else "ttbar"
        expected = {f"cld_edm_{sample}_pf": "3.3.0", f"cld_edm_{sample}_hits": "3.2.2"}
        if not tfds.is_dir():
            parser.error(f"missing TFDS root: {tfds}")
        actual = {p.name for p in tfds.iterdir() if p.is_dir() and p.name.startswith("cld_edm_")}
        if actual != set(expected):
            parser.error(f"unexpected dataset names at {tfds}: {sorted(actual)}")
        for dataset, version in expected.items():
            dataset_dir = tfds / dataset
            for config in map(str, range(1, 11)):
                version_dir = dataset_dir / config / version
                if not (version_dir / "dataset_info.json").is_file():
                    parser.error(f"missing TFDS metadata: {version_dir}/dataset_info.json")
                members = list(version_dir.iterdir())
                if any(not p.is_file() or p.is_symlink() for p in members):
                    parser.error(f"unexpected directory or symlink in {version_dir}")
                total_files += len(members)
                total_bytes += sum(p.stat().st_size for p in members)
            tasks.append((point, tfds, dataset, version))
    print(f"Selected {len(selected)} scan points, {len(tasks)} datasets, " f"{total_files} files, {total_bytes:,} bytes", flush=True)
    if args.dry_run:
        for point, _, dataset, version in tasks:
            print(f"cld/granularity_scan_2025-05-29/{point}/{dataset}/*/{version}")
        return

    uploader = Path(__file__).with_name("tfds_webdav_upload.py")
    for index, (point, tfds, dataset, version) in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {point}/{dataset} {version}", flush=True)
        command = [
            sys.executable,
            str(uploader),
            "--local-root",
            str(tfds),
            "--detector",
            "cld",
            "--campaign",
            "granularity_scan_2025-05-29",
            "--variant",
            point,
            "--dataset",
            dataset,
            "--version",
            version,
            "--jobs",
            str(args.jobs),
            "--image",
            str(args.image),
        ]
        if args.proxy:
            command += ["--proxy", str(args.proxy)]
        if args.remote_root:
            command += ["--remote-root", args.remote_root]
        subprocess.run(command, check=True)
    print(f"Completed {len(tasks)} datasets, {total_files} files, {total_bytes:,} bytes.", flush=True)


if __name__ == "__main__":
    main()
