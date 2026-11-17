"""Publish a checksummed benchmark package to a Hub model repository, then tag its commit."""

import argparse
import hashlib
import json
from pathlib import Path


def validate_package(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    records = manifest["files"]
    expected = {entry["path"] for entry in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate paths in release manifest")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path != directory / "manifest.json"}
    if actual != expected:
        raise ValueError("Package files differ from the release manifest")
    for entry in records:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest paths must stay inside the package")
        path = directory / relative
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != directory.parent):
            raise ValueError("Release packages must not use symlinks")
        if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
    return sorted([directory / entry for entry in expected] + [directory / "manifest.json"])


def publish(directory, repo_id, tag, api):
    directory = Path(directory)
    files = validate_package(directory)
    if not tag or tag in {".", ".."} or Path(tag).name != tag:
        raise ValueError("Use a single release name as the tag")
    destination = f"benchmarks/{tag}/{directory.name}"
    refs = api.list_repo_refs(repo_id, repo_type="model")
    if any(ref.name == tag for ref in refs.tags):
        raise ValueError(f"Tag {tag} already exists; refusing to move or replace a release tag")
    parent = api.repo_info(repo_id, repo_type="model", revision="main").sha
    # Refuse path collisions. A new release is additive to older benchmark packages.
    destinations = [f"{destination}/{path.relative_to(directory).as_posix()}" for path in files]
    existing = {entry.path for entry in api.get_paths_info(repo_id, paths=destinations, repo_type="model", revision=parent)}
    conflicts = [
        f"{destination}/{path.relative_to(directory).as_posix()}"
        for path in files
        if f"{destination}/{path.relative_to(directory).as_posix()}" in existing
    ]
    if conflicts:
        raise ValueError(f"Remote benchmark paths already exist: {conflicts[:5]}")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        parent_commit=parent,
        folder_path=str(directory),
        path_in_repo=destination,
        commit_message=f"Add portable CLD benchmark evidence for {tag}",
    )
    print(f"Uploaded commit: {commit.oid}", flush=True)
    remote = {
        entry.path: entry
        for entry in api.list_repo_tree(repo_id, path_in_repo=destination, repo_type="model", revision=commit.oid, recursive=True)
        if hasattr(entry, "size")
    }
    for path in files:
        key = f"{destination}/{path.relative_to(directory).as_posix()}"
        item = remote[key]
        data = path.read_bytes()
        if item.size != len(data):
            raise RuntimeError(f"Remote size mismatch: {key}; release remains untagged")
        lfs = getattr(item, "lfs", None)
        actual = lfs.sha256 if lfs else item.blob_id
        expected = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Remote content mismatch: {key}; release remains untagged")
    api.create_tag(
        repo_id,
        repo_type="model",
        tag=tag,
        revision=commit.oid,
        tag_message=f"Benchmark evidence release {tag}; historical platform differences documented",
    )
    # Annotated tag refs identify a tag object, not necessarily the commit itself.
    resolved = api.repo_info(repo_id, repo_type="model", revision=tag).sha
    if resolved != commit.oid:
        raise RuntimeError("Release tag did not resolve to the verified upload commit")
    return commit.oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Package to add under benchmarks/TAG/PACKAGE_NAME/")
    parser.add_argument("--repo-id", default="jpata/particleflow")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--upload", action="store_true", help="Upload and tag after local checks; otherwise only print the package plan")
    args = parser.parse_args(argv)
    files = validate_package(args.directory)
    print(
        f"{len(files)} files, {sum(path.stat().st_size for path in files):,} bytes -> {args.repo_id}/benchmarks/{args.tag}/{args.directory.name}, tag {args.tag}"
    )
    if args.upload:
        from huggingface_hub import HfApi

        revision = publish(args.directory, args.repo_id, args.tag, HfApi())
        print(f"Verified commit: {revision}")
        print(f"https://huggingface.co/{args.repo_id}/tree/{args.tag}/benchmarks")


if __name__ == "__main__":
    main()
