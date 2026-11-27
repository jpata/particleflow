#!/usr/bin/env python3
"""List and transfer TFDS datasets on local disks and Grid WebDAV."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from tfds_webdav_client import DEFAULT_REMOTE, WebDAV, WebDAVError, child_url, https_url, retry, safe_component
from tfds_webdav_status import human_bytes
from tfds_webdav_status import main as status_main
from tfds_webdav_transfer import (
    Selection,
    check_selection,
    destination_path,
    download_file,
    download_lock,
    file_is_current,
    local_versions,
    remote_versions,
    run_transfer,
    upload_file,
)

PLATFORMS = Path(__file__).resolve().parents[1] / "configs/training/platforms"


def component(value):
    if not safe_component(value):
        raise argparse.ArgumentTypeError("must be one path component")
    return value


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def platform_roots(platform):
    config = PLATFORMS / f"{platform}.yaml"
    raw = yaml.safe_load(config.read_text())["data_dir"]
    mapping = {"default": raw} if isinstance(raw, str) else raw
    return {key: {"path": os.path.expandvars(os.path.expanduser(path)), "template": path} for key, path in mapping.items()}


def local_roots(args, *, transfer=False):
    if args.local_root:
        return {args.data_key or args.detector or "explicit": args.local_root.expanduser().resolve()}
    configured = platform_roots(args.platform)
    key = args.data_key or args.detector
    if key is None and ("default" in configured or (transfer and len(configured) == 1)):
        key = next(iter(configured))
    if key:
        if key not in configured and "default" in configured:
            key = "default"
        if key not in configured:
            raise ValueError(f"platform {args.platform} has no path for {key}. Choose {', '.join(configured)} with --data-key or set --local-root.")
        configured = {key: configured[key]}
    if transfer and args.variant:
        raise ValueError("pass --local-root to select the variant's local TFDS data root")
    if transfer and len(configured) != 1:
        raise ValueError("choose one TFDS data root with --data-key or --local-root")
    paths = {}
    for key, entry in configured.items():
        path = entry["path"]
        if "$" in path:
            if not transfer:
                paths[key] = None
                continue
            raise ValueError(f"set the environment variable in {entry['template']} for {key} or pass --local-root")
        paths[key] = Path(path).resolve()
    return paths


def print_json(data):
    print(json.dumps(data, indent=2))


def show_platforms(args):
    names = [args.platform] if args.platform else sorted(path.stem for path in PLATFORMS.glob("*.yaml"))
    profiles = {name: platform_roots(name) for name in names}
    if args.json:
        print_json({"platforms": profiles})
        return
    for name, roots in profiles.items():
        for key, entry in roots.items():
            print(f"{name:28} {key:18} {entry['path']}")


def remote_scope(args):
    if args.remote_url:
        if args.campaign or args.variant:
            raise ValueError("--remote-url already specifies the scope. Use --data-key to choose a local path.")
        return https_url(args.remote_url)
    if (args.campaign or args.variant) and not args.detector:
        raise ValueError("--campaign and --variant need --detector")
    return child_url(https_url(args.remote_root), *filter(None, (args.detector, args.campaign, args.variant)))


def client_for(args, root):
    return WebDAV(root, proxy=args.proxy, capath=args.capath, timeout=args.timeout, attempts=args.attempts, redirect_hosts=args.redirect_host)


def selection_for(args):
    return Selection(tuple(dict.fromkeys(args.dataset)), tuple(args.config), tuple(args.version))


def describe_inventory(root, versions, args, **fields):
    return {"root": str(root) if root is not None else None, **fields, "datasets": [version.describe(args.files) for version in versions]}


def show_inventory(args):
    selection = selection_for(args)
    if args.source == "local":
        inventories = [
            describe_inventory(path, local_versions(path, selection) if path else [], args, data_key=key, available=bool(path and path.is_dir()))
            for key, path in local_roots(args).items()
        ]
    else:
        root = remote_scope(args)
        with client_for(args, root) as client:
            versions = remote_versions(client, root, selection, args.jobs, dataset_root=bool(args.variant))
        inventories = [describe_inventory(root, versions, args, available=True)]
    if args.json:
        result = {"source": args.source, "roots": inventories}
        if args.source == "local":
            result["platform"] = args.platform
        print_json(result)
        return
    for inventory in inventories:
        heading = inventory["root"] or "path has an unset environment variable"
        if "data_key" in inventory:
            heading = f"{inventory['data_key']}  {heading}"
        print(heading + ("" if inventory["available"] else "  [unavailable]"))
        if inventory["available"] and not inventory["datasets"]:
            print("  No matching TFDS versions")
        for version in inventory["datasets"]:
            print(f"  {version['path']}  {version['file_count']:,} files  {human_bytes(version['bytes'])}")
            for name, size in version.get("files", {}).items():
                print(f"    {name}  {size:,} bytes")


def show_plan(args, root, remote, versions):
    plan = {
        "direction": args.command,
        "dry_run": True,
        "local_root": str(root),
        "remote_url": remote,
        "total_files": sum(len(version.files) for version in versions),
        "total_bytes": sum(version.size for version in versions),
        "datasets": [version.describe(args.files) for version in versions],
    }
    if args.command == "download":
        present = [
            entry.size
            for version in versions
            for name, entry in version.files.items()
            if file_is_current(destination_path(root, version.path / name), entry.size)
        ]
        plan.update(present_files=len(present), present_bytes=sum(present), pending_files=plan["total_files"] - len(present))
    if args.json:
        print_json(plan)
        return
    source, target = (remote, root) if args.command == "download" else (root, remote)
    print(f"Plan: {source} -> {target}")
    print(f"{len(versions):,} TFDS versions, {plan['total_files']:,} files, {human_bytes(plan['total_bytes'])}")
    if "present_files" in plan:
        print(f"{plan['present_files']:,} files already match by size, {plan['pending_files']:,} files pending")
    for version in plan["datasets"]:
        print(f"  {version['path']}  {version['file_count']:,} files  {human_bytes(version['bytes'])}")
        for name, size in version.get("files", {}).items():
            print(f"    {name}  {size:,} bytes")


def prepare_download(root, versions, resume, empty_root):
    if resume:
        return
    if empty_root and root.exists() and any(root.iterdir()):
        raise ValueError(f"destination is nonempty: {root}. Use --resume.")
    for version in versions:
        path = destination_path(root, version.path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"destination is nonempty: {path}. Use --resume.")


def download(args, client, root, remote, selection, *, empty_root):
    versions = remote_versions(client, remote, selection, args.jobs, dataset_root=True)
    check_selection(versions, selection, remote)
    if args.dry_run:
        show_plan(args, root, remote, versions)
        return None

    def work(relative, entry):
        target = destination_path(root, relative)
        if file_is_current(target, entry.size):
            return False
        address = child_url(remote, *relative.parts)
        rate = args.limit_rate / args.jobs if args.limit_rate else None
        retry(lambda: download_file(client, address, target, entry, rate), args.attempts)
        return True

    if root.exists() and not root.is_dir():
        raise ValueError(f"local root is not a directory: {root}")
    with download_lock(root):
        prepare_download(root, versions, args.resume, empty_root)
        return run_transfer("download", root, remote, versions, work, args.jobs, status_file=args.status_file, cancel=client.cancel)


def upload(args, root, remote, selection):
    versions = local_versions(root, selection, strict=True)
    check_selection(versions, selection, root)
    if args.dry_run:
        show_plan(args, root, remote, versions)
        return None

    for version in versions:
        for name in version.files:
            destination_path(root, version.path / name)
    with client_for(args, remote) as client:
        existing = {}

        def prepare():
            for version in versions:
                address = child_url(remote, *version.path.parts)
                client.ensure_collection(address)
                existing[version.path] = client.list(address)

        def work(relative, entry):
            previous = existing[relative.parent].get(relative.name)
            return upload_file(
                client,
                child_url(remote, *relative.parts),
                root.joinpath(*relative.parts),
                entry.size,
                previous is not None and previous.size == entry.size,
                args.verify_mode,
                args.attempts,
            )

        def verify():
            for version in versions:
                sizes = client.list(child_url(remote, *version.path.parts))
                for name, entry in version.files.items():
                    if name not in sizes or sizes[name].size != entry.size:
                        raise WebDAVError(f"final remote size mismatch: {version.path / name}")

        return run_transfer(
            "upload",
            root,
            remote,
            versions,
            work,
            args.jobs,
            status_file=args.status_file,
            progress_every=args.progress_every,
            prepare=prepare,
            verify=verify,
            cancel=client.cancel,
        )


def transfer(args, *, empty_root=False):
    if not args.dataset and not args.all:
        raise ValueError("select --dataset NAME or explicitly use --all")
    if not args.detector and not args.remote_url:
        raise ValueError("set --detector or use --remote-url for a TFDS data root")
    selection = selection_for(args)
    remote = remote_scope(args)
    root = next(iter(local_roots(args, transfer=True).values()))
    if args.command == "download":
        if args.limit_rate is not None and args.limit_rate < args.jobs:
            raise ValueError("--limit-rate must be at least --jobs bytes per second")
        with client_for(args, remote) as client:
            result = download(args, client, root, remote, selection, empty_root=empty_root)
    else:
        result = upload(args, root, remote, selection)
    if result is not None:
        if args.json:
            print_json(result)
        else:
            print(f"Verified {result['completed_files']:,} files, {human_bytes(result['completed_bytes'])}.")


def add_remote_arguments(parser):
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--remote-root", default=DEFAULT_REMOTE, help="WebDAV base URL; accepts https:// and davs://")
    scope.add_argument("--remote-url", help="full WebDAV URL of the selected collection")
    parser.add_argument("--campaign", type=component)
    parser.add_argument("--variant", type=component)
    parser.add_argument("--proxy", type=Path, help="proxy PEM; default: X509_USER_PROXY or ~/x509-cms-2048")
    parser.add_argument("--capath", type=Path, help="trusted CA bundle or OpenSSL-hashed directory on this host")
    parser.add_argument("--redirect-host", action="append", default=[], help="allow an additional HTTPS storage hostname; repeat as needed")
    parser.add_argument("--attempts", type=positive, default=5)
    parser.add_argument("--timeout", type=positive, default=120, help="HTTP read timeout in seconds")
    parser.add_argument("--jobs", type=positive, default=8, help="parallel WebDAV workers")


def add_local_arguments(parser):
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--platform", type=component, default="local", help="training platform profile; default: local")
    local.add_argument("--local-root", type=Path, help="explicit local TFDS data directory")
    parser.add_argument("--data-key", type=component, help="key in the platform's data_dir mapping")


def add_selection_arguments(parser, *, transfer=False):
    parser.add_argument("--detector", type=component, help="remote detector directory and default platform data key")
    selection = parser.add_mutually_exclusive_group() if transfer else parser
    selection.add_argument("--dataset", type=component, action="append", default=[], help="repeat to select multiple datasets")
    if transfer:
        selection.add_argument("--all", action="store_true", help="select all datasets under the specified TFDS root")
    parser.add_argument("--config", type=component, action="append", default=[], help="repeat to select multiple configurations")
    parser.add_argument("--version", type=component, action="append", default=[], help="repeat to select multiple versions")
    parser.add_argument("--files", action="store_true", help="include each file name and byte size in listings and dry runs")
    parser.add_argument("--json", action="store_true", help="write a single JSON result to stdout")


def legacy_args(argv):
    if len(argv) < 2 or argv[0] not in ("download", "upload") or argv[1].startswith("-"):
        return argv
    parser = argparse.ArgumentParser(prog=f"tfds_webdav.py {argv[0]}")
    for name in ("detector", "campaign", "dataset", "config", "version"):
        parser.add_argument(name)
    parser.add_argument("--variant")
    options, remaining = parser.parse_known_args(argv[1:])
    flags = [arg for key, value in vars(options).items() if value is not None for arg in (f"--{key}", value)]
    return [argv[0], *flags, *remaining]


def main(argv=None, *, empty_root=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "status":
        return status_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use COMMAND --help for options.")
    commands = parser.add_subparsers(dest="command", required=True)
    platform = commands.add_parser("platforms", help="show configured data roots")
    platform.add_argument("--platform", type=component, help="show only this profile")
    platform.add_argument("--json", action="store_true")
    inventory = commands.add_parser("list", help="list TFDS versions on disk or on WebDAV")
    sources = inventory.add_subparsers(dest="source", required=True)
    local = sources.add_parser("local", help="scan the configured or specified local roots")
    remote = sources.add_parser("remote", help="scan a WebDAV collection")
    for source in (local, remote):
        add_selection_arguments(source)
    add_local_arguments(local)
    add_remote_arguments(remote)
    for direction in ("download", "upload"):
        copy = commands.add_parser(direction, help=f"{direction.capitalize()} selected datasets")
        add_local_arguments(copy)
        add_remote_arguments(copy)
        add_selection_arguments(copy, transfer=True)
        copy.add_argument("--dry-run", action="store_true", help="print the transfer plan without copying")
        copy.add_argument("--resume", action="store_true", help="continue an interrupted transfer; uploads already resume by default")
        copy.add_argument("--status-file", type=Path, help="path for live JSON progress")
        if direction == "download":
            copy.add_argument("--limit-rate", type=positive, help="aggregate download limit in bytes per second")
        else:
            copy.add_argument("--verify-mode", choices=("sha256-readback", "md5-server"), default="sha256-readback")
            copy.add_argument("--progress-every", type=positive, default=20)
    commands.add_parser("status", help="show live transfer progress")
    args = parser.parse_args(legacy_args(argv))
    try:
        if args.command == "platforms":
            show_platforms(args)
        elif args.command == "list":
            show_inventory(args)
        else:
            transfer(args, empty_root=empty_root)
    except (OSError, ValueError, WebDAVError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Rerun the transfer with --resume.\n")


if __name__ == "__main__":
    main()
