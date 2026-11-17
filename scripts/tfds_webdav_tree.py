#!/usr/bin/env python3
"""Download a complete WebDAV TFDS data root with Davix and live progress."""

import argparse
import atexit
import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from tfds_webdav_status import TransferStatus, new_status_path


def manifest(listing, remote, datasets=()):
    """Convert a recursive davix-ls listing to selected relative paths and sizes."""
    prefix = unquote(urlsplit(remote).path).lstrip("/").rstrip("/") + "/"
    selected = set(datasets)
    files = {}
    for line in listing.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[0].startswith("-"):
            continue
        path = fields[5].lstrip("/")
        if not path.startswith(prefix):
            raise ValueError(f"listing contains a file outside {remote}: {path}")
        relative = PurePosixPath(path.removeprefix(prefix))
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe path in listing: {path}")
        if selected and relative.parts[0] not in selected:
            continue
        if relative in files:
            raise ValueError(f"duplicate file in listing: {path}")
        files[relative] = int(fields[2])
    missing = selected - {path.parts[0] for path in files}
    if missing:
        raise ValueError(f"datasets not found below {remote}: {', '.join(sorted(missing))}")
    if not files:
        raise ValueError(f"no files found below {remote}")
    return files


def local_progress(root, expected):
    """Count received bytes for manifest paths without walking other datasets."""
    files = 0
    bytes_received = 0
    for relative, size in expected.items():
        try:
            actual = root.joinpath(*relative.parts).stat().st_size
        except FileNotFoundError:
            continue
        bytes_received += min(actual, size)
        files += actual == size
    return files, bytes_received


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-url", required=True, help="WebDAV directory that is the TFDS data root")
    parser.add_argument("--local-root", type=Path, required=True, help="local TFDS data root; must be empty unless --resume is set")
    parser.add_argument("--dataset", action="append", default=[], help="dataset directory to download; repeat to select multiple")
    parser.add_argument("--image", type=Path, default=Path("davix.sif"))
    parser.add_argument("--proxy", type=Path, default=Path(os.environ.get("X509_USER_PROXY", str(Path.home() / "x509-cms-2048"))))
    parser.add_argument("--capath", default="/etc/grid-security/certificates")
    parser.add_argument("--jobs", type=int, default=8, help="Davix parallel listing and download threads")
    parser.add_argument("--status-file", type=Path, help="write atomic JSON progress here; default: ~/.local/state/mlpf-tfds-webdav/")
    parser.add_argument("--resume", action="store_true", help="download only missing or wrong-size files into an existing data root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    url = urlsplit(args.remote_url)
    if url.scheme != "davs" or not url.netloc or url.query or url.fragment:
        parser.error("--remote-url must be a davs:// directory URL without query or fragment")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if any(name in {"", ".", ".."} or "/" in name or "\\" in name for name in args.dataset):
        parser.error("--dataset must be one directory name")
    args.remote_url = args.remote_url.rstrip("/") + "/"
    args.local_root = args.local_root.expanduser().resolve()
    args.image = args.image.expanduser().resolve()
    args.proxy = args.proxy.expanduser().resolve()
    if args.local_root.exists() and (not args.local_root.is_dir() or (not args.resume and any(args.local_root.iterdir()))):
        parser.error(f"destination exists and is nonempty: {args.local_root}; choose an empty directory")
    if not args.dry_run and (not args.image.is_file() or not args.proxy.is_file()):
        parser.error("Davix image and X.509 proxy must be readable")

    common = ["apptainer", "exec", "-B", f"{args.proxy}:/tmp/x509:ro", "-B", f"{args.local_root}:/tfds", str(args.image)]
    auth = ["-P", "grid", "-E", "/tmp/x509", "--key", "/tmp/x509", "--capath", args.capath]
    listing_command = [*common, "davix-ls", *auth, "-l", "-r", str(args.jobs), "--retry", "5", args.remote_url]
    download_command = [*common, "davix-get", *auth, "-r", str(args.jobs), "--retry", "5", args.remote_url, "/tfds"]
    if args.dry_run:
        print(shlex.join(listing_command))
        if args.dataset:
            print(f"Selected datasets: {', '.join(dict.fromkeys(args.dataset))}")
        if args.resume or args.dataset:
            print("Inventory the selected files, then run parallel davix-get for missing or wrong-size files")
        else:
            print(shlex.join(download_command))
        return

    args.local_root.mkdir(parents=True, exist_ok=True)
    print(f"Listing {args.remote_url} recursively...", flush=True)
    listing = subprocess.run(listing_command, capture_output=True, text=True, check=True)
    try:
        expected = manifest(listing.stdout, args.remote_url, args.dataset)
    except ValueError as exc:
        if "no files found" in str(exc) and listing.stderr.strip():
            raise SystemExit(f"davix-ls found no files and reported: {listing.stderr.strip()}") from exc
        raise
    total_bytes = sum(expected.values())
    print(f"Selected {len(expected):,} files, {total_bytes:,} bytes -> {args.local_root}", flush=True)
    label = "download-tree/" + url.netloc + url.path.rstrip("/")
    reporter = TransferStatus(
        args.status_file or new_status_path(label),
        direction="download",
        label=label,
        local=args.local_root,
        remote=args.remote_url,
        total_files=len(expected),
        total_bytes=total_bytes,
        progress_every=1000,
    )
    atexit.register(lambda: reporter.update(force=True, phase="failed") if reporter.status["phase"] != "complete" else None)
    reporter.update(force=True, phase="transferring")
    if args.resume or args.dataset:
        missing = []
        existing_files = 0
        existing_bytes = 0
        for relative, size in expected.items():
            path = args.local_root.joinpath(*relative.parts)
            if path.is_file() and path.stat().st_size == size:
                existing_files += 1
                existing_bytes += size
            else:
                missing.append((relative, size))
        reporter.update(force=True, completed_files=existing_files, completed_bytes=existing_bytes)
        print(f"Selected: {existing_files:,} size-matched files; {len(missing):,} to download.", flush=True)

        def download_one(item):
            relative, size = item
            destination = args.local_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + f".part.{os.getpid()}")
            source = args.remote_url + quote(relative.as_posix(), safe="/")
            target = "/tfds/" + relative.as_posix() + f".part.{os.getpid()}"
            try:
                result = subprocess.run([*common, "davix-get", *auth, "--retry", "5", source, target], capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError(f"davix-get failed: {result.stderr.strip()}")
                if temporary.stat().st_size != size:
                    raise RuntimeError(f"size mismatch: {temporary.stat().st_size} != {size}")
                temporary.replace(destination)
                return size
            finally:
                temporary.unlink(missing_ok=True)

        errors = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(download_one, item): item for item in missing}
            for future in as_completed(futures):
                try:
                    reporter.complete_file(future.result(), transferred=True)
                except Exception as exc:
                    errors.append(f"{futures[future][0]}: {exc}")
                    reporter.error()
                    print(f"ERROR: {errors[-1]}", flush=True)
        if errors:
            raise SystemExit(f"{len(errors):,} files failed; rerun with --resume to retry")
    else:
        print(shlex.join(download_command), flush=True)
        with tempfile.TemporaryFile(mode="w+t") as errors:
            process = subprocess.Popen(download_command, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                while process.poll() is None:
                    files, bytes_received = local_progress(args.local_root, expected)
                    reporter.update(
                        force=True, completed_files=files, completed_bytes=bytes_received, transferred_files=files, transferred_bytes=bytes_received
                    )
                    time.sleep(5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            if process.returncode:
                errors.seek(0)
                raise SystemExit(f"davix-get failed ({process.returncode}): {errors.read()[-4000:]}")

    reporter.update(force=True, phase="verifying")
    files, bytes_received = local_progress(args.local_root, expected)
    if files != len(expected) or bytes_received != total_bytes:
        raise SystemExit(f"download incomplete: {files:,}/{len(expected):,} files, {bytes_received:,}/{total_bytes:,} bytes; inspect destination")
    reporter.update(
        force=True,
        phase="complete",
        completed_files=files,
        completed_bytes=bytes_received,
        transferred_files=reporter.status["transferred_files"] if args.resume or args.dataset else files,
        transferred_bytes=reporter.status["transferred_bytes"] if args.resume or args.dataset else bytes_received,
    )
    print(f"Verified sizes of all {files:,} downloaded files.", flush=True)


if __name__ == "__main__":
    main()
