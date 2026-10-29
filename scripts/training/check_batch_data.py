"""Check that a TFDS root contains the exact copied CLD subset used by the sweep."""

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tfds_root", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "configs/training/data/cld_ttbar_hits_config1_3.2.2.json"
    )
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    errors = check_data(args.tfds_root, manifest)
    if errors:
        parser.exit(1, "\n".join(errors[:20]) + f"\n{len(errors)} dataset check(s) failed\n")
    print(f"Verified {len(manifest['files'])} files, {sum(manifest['files'].values()):,} bytes; metadata checksums match.")
    print("Events:", manifest["split_events"], "(shard payload checksums not checked)")


if __name__ == "__main__":
    main()
