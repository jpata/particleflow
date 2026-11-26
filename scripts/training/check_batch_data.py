"""Check that a TFDS root contains the exact copied CLD subset used by the sweep, or write its manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def check_data(root, manifest):
    directory = Path(root) / manifest["relative_directory"]
    errors = []
    for name, expected_size in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            errors.append(f"missing: {path}")
        elif path.stat().st_size != expected_size:
            errors.append(f"size mismatch: {path}")
    for name, expected_digest in manifest["metadata_sha256"].items():
        path = directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            errors.append(f"metadata checksum mismatch: {path}")
    return errors


def write_manifest(root, dataset, config, version):
    """Manifest of a reference TFDS configuration: shard sizes, metadata checksums and split events."""
    relative = f"{dataset}/{config}/{version}"
    directory = Path(root) / relative
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    info = json.loads((directory / "dataset_info.json").read_text())
    return {
        "dataset": dataset,
        "config": str(config),
        "version": version,
        "relative_directory": relative,
        "files": {path.name: path.stat().st_size for path in sorted(directory.glob("*.array_record-*"))},
        "metadata_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("dataset_info.json", "features.json", "metadata.json")
            if (directory / name).is_file()
        },
        "split_events": {split["name"]: sum(int(n) for n in split["shardLengths"]) for split in info["splits"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tfds_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest to check against, or to create with --write")
    parser.add_argument(
        "--write",
        nargs=3,
        metavar=("DATASET", "CONFIG", "VERSION"),
        help="Write the manifest of the reference copy (for example cld_edm_ttbar_full 1 1.0.0) instead of checking",
    )
    args = parser.parse_args(argv)
    if args.write:
        manifest = write_manifest(args.tfds_root, *args.write)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {args.manifest}: {len(manifest['files'])} files, events {manifest['split_events']}")
        return
    manifest = json.loads(args.manifest.read_text())
    errors = check_data(args.tfds_root, manifest)
    if errors:
        parser.exit(1, "\n".join(errors[:20]) + f"\n{len(errors)} dataset check(s) failed\n")
    print(f"Verified {len(manifest['files'])} files, {sum(manifest['files'].values()):,} bytes; metadata checksums match.")
    print("Events:", manifest["split_events"], "(shard payload checksums not checked)")


if __name__ == "__main__":
    main()
