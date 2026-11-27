#!/usr/bin/env python3
"""Compare CLD, CLIC, IDEA PF and CLD scan files with WebDAV copies.

Run with the ParticleFlow Python environment and an X.509 proxy containing the private key.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tfds_webdav_client import DEFAULT_REMOTE, WebDAV, child_url, file_digest, retry
from tfds_webdav_transfer import METADATA


def selected_versions(root):
    campaign = "v1.2.5_key4hep_2025-05-29"
    for detector, names in (
        ("cld", ("ttbar", "ww_fullhad", "qq", "zz")),
        ("clic", ("ttbar", "ww_fullhad", "qq")),
    ):
        for name in names:
            dataset = f"{detector}_edm_{name}_pf"
            for config in range(1, 11):
                rel = f"{detector}/{campaign}/{dataset}/{config}/3.3.0"
                local = root / detector / campaign / "tfds" / dataset / str(config) / "3.3.0"
                yield rel, local
    scan = "granularity_scan_2025-05-29"
    for kind, sizes, sample in (
        ("ecal", ("1p5", "2", "2p5", "3", "4", "5p1"), "z_tautau"),
        ("hcal", ("10", "12", "15", "20", "25", "30"), "ttbar"),
    ):
        for size in sizes:
            point = f"{kind}_{size}mm"
            for suffix, version in (("pf", "3.3.0"), ("hits", "3.2.2")):
                dataset = f"cld_edm_{sample}_{suffix}"
                for config in range(1, 11):
                    rel = f"cld/{scan}/{point}/{dataset}/{config}/{version}"
                    local = root / "cld" / scan / point / "tfds" / dataset / str(config) / version
                    yield rel, local
    idea = "IDEA_o1_v03_fccconfig_a05a3a9"
    for name in ("ttbar", "ww_fullhad", "qq"):
        dataset = f"idea_edm_{name}_pf"
        for version in ("0.2.0", "0.2.1"):
            rel = f"idea/{idea}/{dataset}/1/{version}"
            local = root / "idea" / idea / "tfds" / dataset / "1" / version
            yield rel, local


def remote_listing(client, rel):
    url = child_url(client.root, *rel.split("/"))
    entries = client.list(url)
    if METADATA not in entries or any(entry.size is None for entry in entries.values()):
        raise RuntimeError(f"incomplete or unexpected remote TFDS version: {rel}")
    return url, {name: entry.size for name, entry in entries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--capath", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scope", choices=("all", "cld-existing", "scan", "new"), default="all")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 8:
        parser.error("--jobs must be between 1 and 8")
    with WebDAV(DEFAULT_REMOTE, proxy=args.proxy, capath=args.capath) as client:
        verify(args, client)


def verify(args, client):
    versions = list(selected_versions(args.local_root))
    if args.scope == "cld-existing":
        versions = [
            (rel, local)
            for rel, local in versions
            if rel.startswith("cld/v1.2.5_key4hep_2025-05-29/") and ("/cld_edm_ttbar_pf/" in rel or "/cld_edm_qq_pf/" in rel)
        ]
    elif args.scope == "scan":
        versions = [(rel, local) for rel, local in versions if rel.startswith("cld/granularity_scan_2025-05-29/")]
    elif args.scope == "new":
        versions = [
            (rel, local)
            for rel, local in versions
            if rel.startswith("clic/")
            or rel.startswith("idea/")
            or (rel.startswith("cld/v1.2.5_key4hep_2025-05-29/") and ("/cld_edm_ww_fullhad_pf/" in rel or "/cld_edm_zz_pf/" in rel))
        ]
    manifest = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(remote_listing, client, rel): (rel, local) for rel, local in versions}
        for done, future in enumerate(as_completed(futures), 1):
            rel, local = futures[future]
            try:
                url, expected = future.result()
                actual = {p.name: p.stat().st_size for p in local.iterdir() if p.is_file()} if local.is_dir() else {}
                if actual != expected:
                    missing = sorted(expected.keys() - actual.keys())
                    extra = sorted(actual.keys() - expected.keys())
                    wrong = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
                    raise RuntimeError(f"file manifest differs: missing={missing}, extra={extra}, wrong_size={wrong}")
                manifest.extend((child_url(url, name), local / name, size) for name, size in expected.items())
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
            if done % 40 == 0 or done == len(versions):
                print(f"Listed {done}/{len(versions)} versions; {len(errors)} errors", flush=True)
    if errors:
        for error in errors:
            print("ERROR", error, file=sys.stderr)
        raise SystemExit(f"Manifest check failed for {len(errors)} version(s)")
    print(f"Comparing {len(manifest)} files, {sum(item[2] for item in manifest):,} bytes using server MD5", flush=True)

    def compare(item):
        url, path, expected_size = item
        local_size, local_hash = file_digest(path, "md5")
        if local_size != expected_size:
            raise RuntimeError(f"local size changed: {path}")
        remote_size, remote_hash = retry(lambda: client.checksum(url), client.attempts)
        if remote_size != expected_size or local_hash != remote_hash:
            raise RuntimeError(f"MD5 mismatch: {path} local={local_hash.hex()} remote={remote_hash.hex()}")
        return expected_size

    verified = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compare, item): item for item in manifest}
        for future in as_completed(futures):
            try:
                total_bytes += future.result()
                verified += 1
            except Exception as exc:
                errors.append(str(exc))
                print("ERROR", exc, file=sys.stderr, flush=True)
            if (verified + len(errors)) % 100 == 0:
                print(f"Verified {verified}/{len(manifest)} files, {total_bytes:,} bytes; {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(f"Checksum comparison failed for {len(errors)} file(s)")
    print(f"PASS: {verified} files and {total_bytes:,} bytes match remote using server MD5", flush=True)


if __name__ == "__main__":
    main()
