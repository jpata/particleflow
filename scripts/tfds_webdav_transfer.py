"""TFDS inventories, transfer plans, and resumable copies."""

import fcntl
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from tfds_webdav_client import Entry, WebDAVError, child_url, file_digest, retry, safe_component
from tfds_webdav_status import TransferStatus, new_status_path

METADATA = "dataset_info.json"
PART_SUFFIX = ".tfds-webdav.part"


@dataclass(frozen=True)
class Selection:
    datasets: tuple = ()
    configs: tuple = ()
    versions: tuple = ()

    def includes(self, path, *, dataset_root=False):
        if self.versions and path.name not in self.versions:
            return False
        candidates = ((path.parent.name, None), (path.parent.parent.name, path.parent.name))
        if dataset_root:
            if len(path.parts) not in (2, 3):
                return False
            candidates = (candidates[len(path.parts) - 2],)
        for dataset, config in candidates:
            if dataset and (not self.datasets or dataset in self.datasets) and (not self.configs or config in self.configs):
                return True
        return False


@dataclass
class Version:
    path: PurePosixPath
    files: dict[str, Entry]

    @property
    def size(self):
        return sum(entry.size for entry in self.files.values())

    def describe(self, include_files=False):
        result = {"path": self.path.as_posix(), "version": self.path.name, "file_count": len(self.files), "bytes": self.size}
        if include_files:
            result["files"] = {name: self.files[name].size for name in sorted(self.files)}
        return result


def check_selection(versions, selection, root):
    if not versions:
        raise ValueError(f"no matching TFDS versions below {root}. Check the filters and select the TFDS data root above the dataset directories.")
    observed = (
        ("datasets", selection.datasets, {version.path.parts[0] for version in versions}),
        ("configurations", selection.configs, {version.path.parts[1] for version in versions if len(version.path.parts) == 3}),
        ("versions", selection.versions, {version.path.name for version in versions}),
    )
    for field, requested, available in observed:
        missing = set(requested) - available
        if missing:
            raise ValueError(f"no matching {field} below {root}: {', '.join(sorted(missing))}")


def local_versions(root, selection, *, strict=False):
    versions = []
    if not root.is_dir():
        return versions

    def inaccessible(error):
        raise error

    for directory, dirs, names in os.walk(root, onerror=inaccessible):
        path = Path(directory)
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if METADATA not in names:
            if strict and len(relative.parts) >= 3:
                dirs.clear()
            continue
        if not selection.includes(relative, dataset_root=strict):
            dirs.clear()
            continue
        files = {}
        if strict and dirs:
            raise ValueError(f"unexpected subdirectory in {path}")
        for name in names:
            source = path / name
            if name.startswith(".") and (name.endswith(PART_SUFFIX) or name.endswith(PART_SUFFIX + ".json")):
                if strict:
                    raise ValueError(f"download is incomplete in {path}; finish it before uploading")
                continue
            if source.is_symlink() or not source.is_file() or not safe_component(name):
                if strict:
                    raise ValueError(f"unexpected file or symlink: {source}")
                continue
            files[name] = Entry(source.stat().st_size)
        if METADATA in files:
            versions.append(Version(relative, files))
        dirs.clear()
    return sorted(versions, key=lambda version: version.path)


def remote_versions(client, root, selection, jobs, *, dataset_root=False):
    """Walk only Depth: 1 collections and collect directories with TFDS metadata."""
    initial = [PurePosixPath(name) for name in selection.datasets] if dataset_root and selection.datasets else [PurePosixPath()]
    versions = []

    def listing(path):
        return client.list(child_url(root, *path.parts))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(listing, path): path for path in initial}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                path = pending.pop(future)
                try:
                    entries = future.result()
                except WebDAVError as exc:
                    if exc.status == 404 and path in initial and path.parts:
                        continue
                    raise
                files = {name: entry for name, entry in entries.items() if entry.size is not None}
                if METADATA in files and len(path.parts) >= 2:
                    if selection.includes(path, dataset_root=dataset_root):
                        versions.append(Version(path, files))
                    continue
                if dataset_root and len(path.parts) >= 3:
                    continue
                for name, entry in entries.items():
                    if entry.size is None:
                        child = path / name
                        pending[pool.submit(listing, child)] = child
    return sorted(versions, key=lambda version: version.path)


def file_is_current(path, expected):
    return not path.is_symlink() and path.is_file() and path.stat().st_size == expected


def destination_path(root, relative):
    target = root
    for name in relative.parts:
        target /= name
        if target.is_symlink():
            raise ValueError(f"destination contains a symlink: {target}")
    return target


@contextmanager
def download_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another WebDAV download is using {root}") from exc
        yield
    finally:
        os.close(descriptor)


def partial_paths(destination):
    part = destination.with_name("." + destination.name + PART_SUFFIX)
    state = part.with_name(part.name + ".json")
    for path in (part, state):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"invalid partial download path: {path}")
    return part, state


def download_file(client, url, destination, entry, bytes_per_second=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part, state = partial_paths(destination)
    validator = entry.etag or entry.modified
    identity = {"url": url, "size": entry.size, "etag": entry.etag, "modified": entry.modified}
    try:
        valid_partial = validator and part.is_file() and json.loads(state.read_text()) == identity
    except (FileNotFoundError, ValueError):
        valid_partial = False
    offset = part.stat().st_size if valid_partial else 0
    if not valid_partial or offset > entry.size:
        part.unlink(missing_ok=True)
        offset = 0
    if not offset:
        state.write_text(json.dumps(identity))

    if offset != entry.size or not part.is_file():
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        if offset and entry.etag and not entry.etag.startswith("W/"):
            headers["If-Range"] = entry.etag
        expected = (200, 206, 416) if offset else (200, 206)
        with client.request("GET", url, expected, headers=headers, stream=True) as response:
            if response.status_code == 416:
                part.unlink(missing_ok=True)
                raise WebDAVError(f"server rejected the range for {url}; restarting the file", retryable=True)
            if response.status_code == 206:
                expected_range = f"bytes {offset}-{entry.size - 1}/{entry.size}"
                if not offset or response.headers.get("Content-Range") != expected_range:
                    part.unlink(missing_ok=True)
                    raise WebDAVError(f"unexpected Content-Range for {url}", retryable=True)
            else:
                offset = 0
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise WebDAVError(f"unexpected Content-Encoding for {url}")
            advertised = response.headers.get("Content-Length")
            if advertised is not None and advertised != str(entry.size - offset):
                part.unlink(missing_ok=True)
                raise WebDAVError(f"unexpected Content-Length for {url}", retryable=True)
            received = offset
            start = time.monotonic()
            with part.open("ab" if offset else "wb") as stream:
                for chunk in client.blocks(response, url):
                    received += len(chunk)
                    if received > entry.size:
                        break
                    stream.write(chunk)
                    if bytes_per_second:
                        delay = (received - offset) / bytes_per_second - (time.monotonic() - start)
                        if delay > 0:
                            client.pace(delay)
            if received != entry.size:
                if received > entry.size:
                    part.unlink(missing_ok=True)
                raise WebDAVError(f"incomplete or oversized download for {url}: expected {entry.size} bytes", retryable=True)
    part.replace(destination)
    state.unlink(missing_ok=True)


def upload_file(client, url, source, size, present, mode, attempts):
    expected = file_digest(source, "md5" if mode == "md5-server" else "sha256", client.check_running)
    if expected[0] != size:
        raise ValueError(f"local file size changed: {source}")
    uploaded = False

    def matches():
        try:
            return client.checksum(url, mode) == expected
        except WebDAVError as exc:
            if exc.status != 404:
                raise
            return False

    def copy_and_verify():
        nonlocal present, uploaded
        if present and matches():
            return uploaded
        client.put(url, source, size)
        present = True
        uploaded = True
        if not matches():
            present = False
            raise WebDAVError(f"{mode} mismatch for {url}", retryable=True)
        return True

    return retry(copy_and_verify, attempts)


def transfer_batch(pool, files, work, reporter):
    failures = 0
    futures = {pool.submit(work, path, entry): (path, entry.size) for path, entry in files}
    for future in as_completed(futures):
        path, size = futures[future]
        try:
            transferred = future.result()
        except (OSError, ValueError, WebDAVError) as exc:
            failures += 1
            reporter.error()
            print(f"ERROR {path}: {exc}", file=sys.stderr, flush=True)
        else:
            reporter.complete_file(size, transferred=transferred)
    if failures:
        raise WebDAVError(f"{failures} file(s) failed. Rerun with --resume.")


def run_transfer(direction, root, remote, versions, work, jobs, *, status_file=None, progress_every=20, prepare=None, verify=None, cancel):
    label = f"{direction}/{urlsplit(remote).netloc}{urlsplit(remote).path}"
    reporter = TransferStatus(
        status_file or new_status_path(label),
        direction=direction,
        label=label,
        local=root,
        remote=remote,
        total_files=sum(len(version.files) for version in versions),
        total_bytes=sum(version.size for version in versions),
        progress_every=progress_every,
        stream=sys.stderr,
    )
    try:
        if prepare:
            prepare()
        reporter.update(force=True, phase="transferring")
        payload = [(version.path / name, entry) for version in versions for name, entry in version.files.items() if name != METADATA]
        metadata = [(version.path / METADATA, version.files[METADATA]) for version in versions]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            try:
                transfer_batch(pool, payload, work, reporter)
                transfer_batch(pool, metadata, work, reporter)
            except KeyboardInterrupt:
                cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
        if verify:
            reporter.update(force=True, phase="verifying")
            verify()
    except BaseException:
        reporter.update(force=True, phase="failed")
        raise
    reporter.update(force=True, phase="complete")
    return {"status_file": str(reporter.path), **reporter.status}
