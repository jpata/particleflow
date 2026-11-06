#!/usr/bin/env python3
"""Upload a TFDS campaign to Grid WebDAV with parallel copies and readback checks."""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_REMOTE = "davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"


def component(value):
    if not value or value in {".", ".."} or any(c in value for c in "/\\"):
        raise argparse.ArgumentTypeError("must be one path component")
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True, help="campaign TFDS directory containing dataset-name directories")
    parser.add_argument("--detector", type=component, required=True)
    parser.add_argument("--campaign", type=component, required=True)
    parser.add_argument("--variant", type=component, help="optional scan point within the campaign")
    parser.add_argument("--dataset", type=component, action="append", required=True, help="repeat for each dataset")
    parser.add_argument("--version", type=component, required=True)
    parser.add_argument("--config", type=component, action="append", help="repeat to select configurations; default: all with this version")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE)
    parser.add_argument("--image", type=Path, default=Path("davix.sif"))
    parser.add_argument("--proxy", type=Path, default=Path(os.environ.get("X509_USER_PROXY", str(Path.home() / "x509-cms-2048"))))
    parser.add_argument("--capath", default="/etc/grid-security/certificates")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.attempts < 1:
        parser.error("--jobs and --attempts must be positive")
    url = urlsplit(args.remote_root)
    if url.scheme != "davs" or not url.netloc or url.query or url.fragment:
        parser.error("--remote-root must be a davs:// directory URL")
    args.local_root = args.local_root.expanduser().resolve()
    args.image = args.image.expanduser().resolve()
    args.proxy = args.proxy.expanduser().resolve()
    if not args.local_root.is_dir():
        parser.error(f"local root not found: {args.local_root}")
    if not args.dry_run and (not args.image.is_file() or not args.proxy.is_file()):
        parser.error("Davix image and X.509 proxy must be readable")

    root = args.remote_root.rstrip("/")
    campaign_url = "/".join((root, quote(args.detector, safe=""), quote(args.campaign, safe="")))
    if args.variant:
        campaign_url += "/" + quote(args.variant, safe="")
    versions = {}
    files = []
    for dataset in dict.fromkeys(args.dataset):
        dataset_dir = args.local_root / dataset
        configs = args.config or [p.name for p in dataset_dir.iterdir() if p.is_dir() and (p / args.version / "dataset_info.json").is_file()]
        for config in sorted(set(configs), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            version_dir = dataset_dir / config / args.version
            if not (version_dir / "dataset_info.json").is_file():
                parser.error(f"missing TFDS metadata: {version_dir}/dataset_info.json")
            version_url = "/".join((campaign_url, quote(dataset, safe=""), quote(config, safe=""), quote(args.version, safe="")))
            members = sorted(version_dir.iterdir())
            if any(not p.is_file() or p.is_symlink() for p in members):
                parser.error(f"unexpected directory or symlink in {version_dir}")
            versions[version_url] = members
            files.extend((p, version_url + "/" + quote(p.name, safe="")) for p in members)
    if not files:
        parser.error("no files selected")
    print(
        f"Selected {len(versions)} TFDS versions, {len(files)} files, " f"{sum(p.stat().st_size for p, _ in files):,} bytes -> {campaign_url}",
        flush=True,
    )
    if args.dry_run:
        for version_url, members in versions.items():
            print(f"{version_url}: {len(members)} files, {sum(p.stat().st_size for p in members):,} bytes")
        return

    common = ["apptainer", "exec", "-B", f"{args.proxy}:/tmp/x509:ro", "-B", f"{args.local_root}:/tfds:ro", str(args.image)]
    auth = ["-P", "grid", "-E", "/tmp/x509", "--key", "/tmp/x509", "--capath", args.capath, "--retry", "5", "--retry-delay", "2"]

    def run(program, *tail):
        return subprocess.run([*common, program, *auth, *tail], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    def ensure_dir(address):
        if run("davix-ls", address).returncode == 0:
            return
        result = run("davix-mkdir", address)
        if result.returncode and run("davix-ls", address).returncode:
            raise RuntimeError(f"cannot create remote directory {address}: {result.stderr.strip()}")
        print(f"Created {address}", flush=True)

    # The parent /store/user/jpata/mlpf exists. Create collections serially.
    addresses = [
        root,
        root + "/" + quote(args.detector, safe=""),
        root + "/" + quote(args.detector, safe="") + "/" + quote(args.campaign, safe=""),
        campaign_url,
    ]
    for version_url in versions:
        addresses.extend((version_url.rsplit("/", 2)[0], version_url.rsplit("/", 1)[0], version_url))
    for address in dict.fromkeys(addresses):
        ensure_dir(address)

    def listed_sizes(version_url):
        result = run("davix-ls", "-l", version_url)
        if result.returncode:
            raise RuntimeError(f"cannot list {version_url}: {result.stderr.strip()}")
        sizes = {}
        for line in result.stdout.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[0].startswith("-"):
                continue
            sizes[fields[5]] = int(fields[2])
        return sizes

    remote_sizes = {url: listed_sizes(url) for url in versions}

    def remote_digest(address):
        process = subprocess.Popen([*common, "davix-get", *auth, address], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        digest = hashlib.sha256()
        size = 0
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"readback failed: {stderr.strip()}")
        return size, digest.digest()

    def transfer(item):
        path, address = item
        version_url = address.rsplit("/", 1)[0]
        expected_size = path.stat().st_size
        expected_digest = None
        force_upload = False
        for attempt in range(1, args.attempts + 1):
            try:
                present = not force_upload and remote_sizes[version_url].get(path.name) == expected_size
                if not present:
                    local_path = "/tfds/" + path.relative_to(args.local_root).as_posix()
                    result = run("davix-put", local_path, address)
                    if result.returncode:
                        raise RuntimeError(f"upload failed: {result.stderr.strip()}")
                actual_size = listed_sizes(version_url).get(path.name)
                if actual_size != expected_size:
                    raise RuntimeError(f"size mismatch: local {expected_size}, remote {actual_size}")
                if expected_digest is None:
                    expected_digest = digest_file(path)
                read_size, read_digest = remote_digest(address)
                if read_size != expected_size or read_digest != expected_digest:
                    force_upload = True
                    raise RuntimeError(f"SHA-256 readback mismatch ({read_size} bytes)")
                return path, "verified" if present else "uploaded and verified"
            except (OSError, RuntimeError) as exc:
                if attempt == args.attempts:
                    raise RuntimeError(f"{path}: failed after {attempt} attempts: {exc}") from exc
                time.sleep(min(2**attempt, 30))

    errors = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(transfer, item): item for item in files}
        for future in as_completed(futures):
            try:
                path, state = future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(files):
                    print(f"{completed}/{len(files)} files verified (latest: {path.name}, {state})", flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f"ERROR: {exc}", file=sys.stderr, flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} file(s) failed; rerun the same command to retry")
    print(f"Verified all {len(files)} files by SHA-256 readback.", flush=True)


if __name__ == "__main__":
    main()
