#!/usr/bin/env python3
"""Copy one complete TFDS version between a local detector root and Grid WebDAV."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_REMOTE = "davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"


def component(value):
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be one path component")
    return value


def command(args, program, *operands):
    base = ["apptainer", "exec", "-B", f"{args.proxy}:/tmp/x509:ro"]
    # Bind only the selected local data tree; the image need not know host paths.
    base += ["-B", f"{args.local_root}:/tfds"]
    base += [str(args.image), program, "-P", "grid", "-E", "/tmp/x509", "--key", "/tmp/x509"]
    base += ["--capath", args.capath, *operands]
    print(shlex.join(base), flush=True)
    if not args.dry_run:
        subprocess.run(base, check=True)


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

    remote_root = args.remote_root.rstrip("/")
    remote_dirs = [remote_root]
    remote = remote_root
    parts = [args.detector, args.campaign]
    if args.variant:
        parts.append(args.variant)
    parts.extend((args.dataset, args.config))
    for name in parts:
        remote += "/" + quote(name, safe="")
        remote_dirs.append(remote)
    remote += "/" + quote(args.version, safe="")
    local = Path("/tfds") / args.dataset / args.config / args.version
    host_local = args.local_root / args.dataset / args.config / args.version
    if args.direction == "upload":
        if not host_local.is_dir():
            parser.error(f"TFDS version directory not found: {host_local}")
        if not (host_local / "dataset_info.json").is_file():
            parser.error(f"dataset_info.json missing from: {host_local}")
        files = [p for p in host_local.rglob("*") if p.is_file()]
        print(f"Uploading {len(files)} files, {sum(p.stat().st_size for p in files)} bytes: {host_local} -> {remote}")
        # davix-put creates the version collection; make its ancestors first.
        for url in remote_dirs:
            # Existing collections cause mkdir to fail, so list before creating.
            if args.dry_run:
                print(f"Ensure WebDAV collection exists: {url}")
            else:
                probe = [
                    "apptainer",
                    "exec",
                    "-B",
                    f"{args.proxy}:/tmp/x509:ro",
                    str(args.image),
                    "davix-ls",
                    "-P",
                    "grid",
                    "-E",
                    "/tmp/x509",
                    "--key",
                    "/tmp/x509",
                    "--capath",
                    args.capath,
                    url,
                ]
                if subprocess.run(probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                    command(args, "davix-mkdir", url)
        command(args, "davix-put", "-r", "8", "--retry", "5", str(local), remote)
    else:
        if host_local.exists() and (not host_local.is_dir() or any(host_local.iterdir())):
            parser.error(f"destination exists and is nonempty: {host_local}; choose an empty directory")
        print(f"Downloading {remote} -> {host_local}")
        if not args.dry_run:
            host_local.parent.mkdir(parents=True, exist_ok=True)
        command(args, "davix-get", "-r", "8", "--retry", "5", remote, str(local))


if __name__ == "__main__":
    main()
