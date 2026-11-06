#!/usr/bin/env python3
"""Show progress for TFDS WebDAV copy jobs, or provide their status JSON to an agent."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def state_dir():
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "mlpf-tfds-webdav"


def new_status_path(label):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return state_dir() / f"{safe}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}.json"


def human_bytes(value):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def human_duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def summary(status):
    total = status["total_bytes"]
    done = status["completed_bytes"]
    fraction = done / total if total else 1
    bar_width = 20
    bar = "#" * round(bar_width * fraction) + "-" * (bar_width - round(bar_width * fraction))
    speed = human_bytes(status["rate_bytes_per_second"]) + "/s"
    return (
        f"{status['phase']:>10} [{bar}] {fraction:5.1%} "
        f"{status['completed_files']:,}/{status['total_files']:,} files "
        f"{human_bytes(done)}/{human_bytes(total)} "
        f"{speed} ETA {human_duration(status['eta_seconds'])} "
        f"errors {status['error_count']}"
    )


class TransferStatus:
    def __init__(self, path, *, direction, label, local, remote, total_files, total_bytes, progress_every=20):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.last_write = 0.0
        self.last_display = 0.0
        self.progress_every = progress_every
        self.status = {
            "schema_version": 1,
            "pid": os.getpid(),
            "direction": direction,
            "label": label,
            "local": str(local),
            "remote": str(remote),
            "phase": "preparing",
            "total_files": total_files,
            "total_bytes": total_bytes,
            "completed_files": 0,
            "completed_bytes": 0,
            "transferred_files": 0,
            "transferred_bytes": 0,
            "error_count": 0,
            "elapsed_seconds": 0,
            "rate_bytes_per_second": 0,
            "eta_seconds": None,
            "updated_at": None,
        }
        self.update(force=True)
        print(f"Transfer status: {self.path}", flush=True)

    def update(self, *, force=False, **changes):
        self.status.update(changes)
        now = time.monotonic()
        if not force and now - self.last_write < 1:
            return
        elapsed = max(now - self.started, 0.001)
        self.status["elapsed_seconds"] = round(elapsed, 1)
        self.status["rate_bytes_per_second"] = round(self.status["completed_bytes"] / elapsed)
        rate = self.status["rate_bytes_per_second"]
        self.status["eta_seconds"] = (
            round((self.status["total_bytes"] - self.status["completed_bytes"]) / rate) if rate and self.status["phase"] == "transferring" else None
        )
        self.status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(self.status, indent=2) + "\n")
        temp.replace(self.path)
        self.last_write = now
        if sys.stderr.isatty():
            print(
                "\r" + summary(self.status) + "\033[K",
                end="\n" if self.status["phase"] in {"complete", "failed"} else "",
                file=sys.stderr,
                flush=True,
            )
        elif force or self.status["completed_files"] % self.progress_every == 0:
            print(summary(self.status), file=sys.stderr, flush=True)

    def complete_file(self, size, *, transferred=False):
        changes = {"completed_files": self.status["completed_files"] + 1, "completed_bytes": self.status["completed_bytes"] + size}
        if transferred:
            changes["transferred_files"] = self.status["transferred_files"] + 1
            changes["transferred_bytes"] = self.status["transferred_bytes"] + size
        self.update(**changes)

    def error(self):
        self.update(force=True, error_count=self.status["error_count"] + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", help="status JSON path; default: newest job")
    parser.add_argument("--watch", action="store_true", help="refresh until the job completes")
    parser.add_argument("--json", action="store_true", help="print the complete JSON status")
    args = parser.parse_args()
    if args.path is None:
        candidates = list(state_dir().glob("*.json"))
        if not candidates:
            parser.error(f"no transfer status files in {state_dir()}")
        args.path = max(candidates, key=lambda p: p.stat().st_mtime)
    while True:
        status = json.loads(args.path.read_text())
        line = json.dumps(status, indent=2) if args.json else f"{status['label']}: {summary(status)}"
        print(
            "\r" + line + ("\033[K" if args.watch and sys.stdout.isatty() else ""), end="" if args.watch and sys.stdout.isatty() else "\n", flush=True
        )
        if not args.watch or status["phase"] in {"complete", "failed"}:
            if args.watch and sys.stdout.isatty():
                print()
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
