"""Run a Python command while sampling shared-memory and cgroup memory usage."""

import argparse
from contextlib import contextmanager
import threading
import json
from pathlib import Path
import subprocess
import sys
import time
import os


def snapshot():
    result = {"unix_time": time.time()}
    try:
        stat = os.statvfs("/dev/shm")
        result["shm"] = {
            "total_bytes": stat.f_blocks * stat.f_frsize,
            "available_bytes": stat.f_bavail * stat.f_frsize,
            "available_inodes": stat.f_favail,
        }
        # Read the current cgroup and its ancestors: Slurm can enforce limits
        # at the step or job level. Support both unified and legacy hierarchies.
        cgroups = []
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers == "":
                cgroups.append((Path("/sys/fs/cgroup"), relative))
            elif "memory" in controllers.split(","):
                cgroups.append((Path("/sys/fs/cgroup/memory"), relative))
        result["cgroups"] = {}
        for root, relative in cgroups:
            current = root / relative.lstrip("/")
            while current == root or root in current.parents:
                values = {}
                for name in (
                    "memory.current",
                    "memory.peak",
                    "memory.max",
                    "memory.events",
                    "memory.stat",
                    "memory.usage_in_bytes",
                    "memory.max_usage_in_bytes",
                    "memory.limit_in_bytes",
                    "memory.failcnt",
                    "memory.oom_control",
                ):
                    path = current / name
                    if path.is_file():
                        values[name] = path.read_text().strip()
                if values:
                    result["cgroups"][str(current)] = values
                if current == root:
                    break
                current = current.parent
    except OSError as exc:
        result["error"] = str(exc)
    return result


@contextmanager
def monitor_memory(path, interval=5):
    """Sample the allocation during a sweep and always stop the monitor on failure."""
    stop = threading.Event()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:

        def record():
            output.write(json.dumps(snapshot()) + "\n")
            output.flush()

        def poll():
            while not stop.wait(interval):
                record()

        record()
        thread = threading.Thread(target=poll, name="mlpf-sweep-memory", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python command is required after --")
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with args.diagnostics.open("x") as output:

        def record():
            output.write(json.dumps(snapshot()) + "\n")
            output.flush()

        record()
        child = subprocess.Popen([sys.executable, *command])
        while True:
            try:
                returncode = child.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                record()
        record()
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    sys.exit(main())
