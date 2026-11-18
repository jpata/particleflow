#!/usr/bin/env python3
"""Copy one complete TFDS version between a local detector root and Grid WebDAV."""

import argparse
import atexit
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

from tfds_webdav_status import TransferStatus, new_status_path

DEFAULT_REMOTE = "davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"


def component(value):
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be one path component")
    return value


def command_argv(args, program, *operands):
    base = ["apptainer", "exec", "-B", f"{args.proxy}:/tmp/x509:ro"]
    # Bind only the selected local data tree; the image need not know host paths.
    base += ["-B", f"{args.local_root}:/tfds"]
    base += [str(args.image), program, "-P", "grid", "-E", "/tmp/x509", "--key", "/tmp/x509"]
    base += ["--capath", args.capath, *operands]
    return base


def command(args, program, *operands):
    base = command_argv(args, program, *operands)
    print(shlex.join(base), flush=True)
    print(" ".join(shlex.quote(arg) for arg in base), flush=True)
    if not args.dry_run:
        subprocess.run(base, check=True)


def listed_sizes(args, remote):
    result = subprocess.run(command_argv(args, "davix-ls", "-l", remote), capture_output=True, text=True, check=True)
    sizes = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[0].startswith("-"):
            sizes[fields[5]] = int(fields[2])
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("upload", "download"))
    parser.add_argument("detector", type=component)
    parser.add_argument("campaign", type=component)
    parser.add_argument("--variant", type=component, help="optional scan point within the campaign")
    parser.add_argument("dataset", type=component)
    parser.add_argument("config", type=component)
    parser.add_argument("version", type=component)
    parser.add_argument("--local-root", type=Path, required=True, help="directory containing dataset-name directories (TFDS --data-dir)")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE, help="WebDAV directory containing detector directories")
    parser.add_argument("--image", type=Path, default=Path("davix.sif"))
    parser.add_argument("--proxy", type=Path, default=Path(os.environ.get("X509_USER_PROXY", str(Path.home() / "x509-cms-2048"))))
    parser.add_argument("--capath", default="/etc/grid-security/certificates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status-file", type=Path, help="write atomic JSON progress here; default: ~/.local/state/mlpf-tfds-webdav/")
    args = parser.parse_args()

    parts = urlsplit(args.remote_root)
    if parts.scheme != "davs" or not parts.netloc or parts.query or parts.fragment:
        parser.error("--remote-root must be a davs:// directory URL without query or fragment")
    args.local_root = args.local_root.expanduser().resolve()
    args.image = args.image.expanduser().resolve()
    args.proxy = args.proxy.expanduser().resolve()
    if not args.image.is_file() and not args.dry_run:
        parser.error(f"Davix image not found: {args.image}")
    if not args.proxy.is_file() and not args.dry_run:
        parser.error(f"X.509 proxy not found: {args.proxy}")
    if args.direction == "download" and not args.dry_run:
        args.local_root.mkdir(parents=True, exist_ok=True)
    if not args.local_root.is_dir() and not args.dry_run:
        parser.error(f"local TFDS root not found: {args.local_root}")

    if args.direction == "upload":
        uploader = Path(__file__).with_name("tfds_webdav_upload.py")
        upload_command = [
            sys.executable,
            str(uploader),
            "--local-root",
            str(args.local_root),
            "--detector",
            args.detector,
            "--campaign",
            args.campaign,
            "--dataset",
            args.dataset,
            "--config",
            args.config,
            "--version",
            args.version,
            "--remote-root",
            args.remote_root,
            "--image",
            str(args.image),
            "--proxy",
            str(args.proxy),
            "--capath",
            args.capath,
        ]
        if args.variant:
            upload_command += ["--variant", args.variant]
        if args.status_file:
            upload_command += ["--status-file", str(args.status_file)]
        if args.dry_run:
            upload_command.append("--dry-run")
        raise SystemExit(subprocess.call(upload_command))

    remote_root = args.remote_root.rstrip("/")
    remote = remote_root
    parts = [args.detector, args.campaign]
    if args.variant:
        parts.append(args.variant)
    parts.extend((args.dataset, args.config))
    for name in parts:
        remote += "/" + quote(name, safe="")
    remote += "/" + quote(args.version, safe="")
    local = Path("/tfds") / args.dataset / args.config / args.version
    host_local = args.local_root / args.dataset / args.config / args.version
    if args.direction == "download":
        if host_local.exists() and (not host_local.is_dir() or any(host_local.iterdir())):
            parser.error(f"destination exists and is nonempty: {host_local}; choose an empty directory")
        print(f"Downloading {remote} -> {host_local}")
        if not args.dry_run:
            host_local.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            command(args, "davix-get", "-r", "8", "--retry", "5", remote, str(local))
            return
        sizes = listed_sizes(args, remote)
        if not sizes or "dataset_info.json" not in sizes:
            raise SystemExit(f"remote TFDS version appears empty or incomplete: {remote}")
        label = "/".join(filter(None, ("download", args.detector, args.campaign, args.variant, args.dataset, args.config, args.version)))
        reporter = TransferStatus(
            args.status_file or new_status_path(label),
            direction="download",
            label=label,
            local=host_local,
            remote=remote,
            total_files=len(sizes),
            total_bytes=sum(sizes.values()),
        )
        atexit.register(lambda: reporter.update(force=True, phase="failed") if reporter.status["phase"] != "complete" else None)
        reporter.update(force=True, phase="transferring")
        argv = command_argv(args, "davix-get", "-r", "8", "--retry", "5", remote, str(local))
        print(shlex.join(argv), flush=True)
        with tempfile.TemporaryFile(mode="w+t") as errors:
            process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=errors)
            while process.poll() is None:
                present = {name: (host_local / name).stat().st_size for name in sizes if (host_local / name).is_file()}
                reporter.update(
                    completed_files=sum(present.get(name) == size for name, size in sizes.items()),
                    completed_bytes=sum(min(present.get(name, 0), size) for name, size in sizes.items()),
                    transferred_files=sum(present.get(name) == size for name, size in sizes.items()),
                    transferred_bytes=sum(min(present.get(name, 0), size) for name, size in sizes.items()),
                )
                time.sleep(2)
            if process.returncode:
                errors.seek(0)
                raise SystemExit(f"davix-get failed ({process.returncode}): {errors.read()[-4000:]}")
        reporter.update(force=True, phase="verifying")
        for name, size in sizes.items():
            if not (host_local / name).is_file() or (host_local / name).stat().st_size != size:
                raise SystemExit(f"download size mismatch: {host_local / name}")
        reporter.update(
            force=True,
            phase="complete",
            completed_files=len(sizes),
            completed_bytes=sum(sizes.values()),
            transferred_files=len(sizes),
            transferred_bytes=sum(sizes.values()),
        )


if __name__ == "__main__":
    main()
