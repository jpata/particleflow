#!/usr/bin/env python3

import os
import re
import yaml
import argparse
from pathlib import Path
from huggingface_hub import HfApi

# Set timeout for large file uploads
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"


def resolve_variables(value, data):
    """
    Recursively resolve variables in the form of ${key.path} in strings.
    """
    if isinstance(value, list):
        return [resolve_variables(v, data) for v in value]
    if isinstance(value, dict):
        return {k: resolve_variables(v, data) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    matches = re.findall(r"\${([^}]+)}", value)
    for match in matches:
        keys = match.split(".")
        ref = data
        try:
            for k in keys:
                ref = ref[k]
            # Recursively resolve the reference itself
            resolved_ref = resolve_variables(ref, data)
            value = value.replace(f"${{{match}}}", str(resolved_ref))
        except (KeyError, TypeError):
            continue
    return value


def get_dir_size(path):
    """Calculate the total size of a directory in bytes."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_size(num, suffix="B"):
    """Convert bytes to human-readable format."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Upload TFDS datasets to HuggingFace.")
    parser.add_argument("scenario", help="Scenario name from particleflow_spec.yaml (e.g., cms_run3)")
    parser.add_argument("split", help="Dataset split to upload (e.g., 1)")
    parser.add_argument("--repo", default="jpata/particleflow", help="HF repository ID")
    parser.add_argument("--spec", default="particleflow_spec.yaml", help="Path to particleflow_spec.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing upload")
    parser.add_argument("--site", default="tallinn", help="Site profile to use (default: tallinn)")

    args = parser.parse_args()

    if not os.path.exists(args.spec):
        print(f"Error: Spec file {args.spec} not found.")
        return

    with open(args.spec, "r") as f:
        spec = yaml.safe_load(f)

    # Apply site override logic
    if "project" in spec and "sites" in spec["project"]:
        if args.site in spec["project"]["sites"]:
            spec["project"].update(spec["project"]["sites"][args.site])

    if args.scenario not in spec.get("productions", {}):
        print(f"Error: Scenario '{args.scenario}' not found in 'productions'.")
        return

    prod = spec["productions"][args.scenario]
    workspace_dir = resolve_variables(prod.get("workspace_dir", ""), spec)

    tfds_root = Path(workspace_dir) / "tfds"
    if not tfds_root.is_dir():
        print(f"Error: TFDS root directory {tfds_root} not found.")
        return

    print(f"Searching for datasets in {tfds_root} for split {args.split}...")

    api = HfApi() if not args.dry_run else None

    # Discover datasets on disk
    count = 0
    total_size = 0
    for dataset_dir in tfds_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Skip internal TFDS/system directories
        if dataset_dir.name in ["downloads"] or dataset_dir.name.startswith("torchinductor"):
            continue

        dataset_name = dataset_dir.name
        split_dir = dataset_dir / args.split

        if split_dir.is_dir():
            # Find version directory inside the split directory
            for version_dir in split_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                version = version_dir.name
                # Local structure: tfds/dataset_name/split/version
                local_dir = version_dir

                # Remote structure: tensorflow_datasets/scenario/dataset_name/split/version
                remote_path = f"tensorflow_datasets/{args.scenario}/{dataset_name}/{args.split}/{version}"

                size = get_dir_size(local_dir)
                total_size += size

                if args.dry_run:
                    print(f"[DRY-RUN] Would upload {local_dir} ({format_size(size)}) to {args.repo} at {remote_path}")
                else:
                    print(f"Uploading {local_dir} ({format_size(size)}) to {args.repo}/{remote_path}...")
                    api.upload_folder(folder_path=str(local_dir), path_in_repo=remote_path, repo_id=args.repo, repo_type="dataset")
                count += 1

    if count == 0:
        print(f"No datasets found for split {args.split} in {tfds_root}")
    else:
        print(f"Processed {count} dataset(s). Total size: {format_size(total_size)}")


if __name__ == "__main__":
    main()
