#!/usr/bin/env python3

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

# Set timeout for large file uploads
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1000"

MAIN_MODEL_REPO = "jpata/particleflow"
MAIN_REPO_WORKING_CEILING = 16_000
MAIN_REPO_FILE_LIMIT = 20_000


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


def folder_files(folder, ignore_patterns=None):
    """Enumerate the relative paths selected by the Hub's folder upload."""
    candidates = (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    return list(filter_repo_objects(candidates, ignore_patterns=ignore_patterns))


def preflight_file_count(api, repo_id, repo_type, destinations):
    existing = set(api.list_repo_files(repo_id, repo_type=repo_type))
    additions = len(destinations - existing)
    projected = len(existing) + additions
    print(f"Hub file count is {len(existing):,}. This upload adds {additions:,}, for {projected:,} total.")
    if repo_id != MAIN_MODEL_REPO or repo_type != "model":
        return
    if additions and projected > MAIN_REPO_FILE_LIMIT:
        raise SystemExit(f"Upload would exceed the {MAIN_REPO_FILE_LIMIT:,}-file limit for {repo_id}.")
    if projected >= MAIN_REPO_WORKING_CEILING:
        print(
            f"Warning: projected count reaches the {MAIN_REPO_WORKING_CEILING:,}-file working ceiling for {repo_id}.",
            file=sys.stderr,
        )


def format_size(num, suffix="B"):
    """Convert bytes to human-readable format."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Upload ParticleFlow model checkpoints to HuggingFace.")
    parser.add_argument("experiment_dir", help="Path to the experiment directory in 'experiments/'")
    parser.add_argument("--repo", default=MAIN_MODEL_REPO, help="HF repository ID")
    parser.add_argument("--name", help="Custom descriptive name for the experiment in the repo")
    parser.add_argument("--version", help="Custom version string (e.g., v1.0.0)")
    parser.add_argument("--step", type=int, help="Checkpoint step to upload as best_weights.pth (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--spec", default="particleflow_spec.yaml", help="Path to particleflow_spec.yaml")
    parser.add_argument("--repo-type", default="model", help="HF repository type (default: model)")

    args = parser.parse_args()

    exp_path = Path(args.experiment_dir)
    if not exp_path.is_dir():
        print(f"Error: Experiment directory {exp_path} not found.")
        return

    # Load spec to get model metadata
    if not os.path.exists(args.spec):
        print(f"Error: Spec file {args.spec} not found.")
        return

    with open(args.spec, "r") as f:
        spec = yaml.safe_load(f)

    exp_name = exp_path.name
    parts = exp_name.split("_")
    model_id = parts[0]  # e.g., pyg-cld-hits-v1
    scenario = parts[1]  # e.g., cld

    if args.version:
        version = args.version
    elif model_id not in spec.get("models", {}):
        print(f"Warning: Model '{model_id}' not found in 'models' section of {args.spec}. Using defaults for structure.")
        dtype = "clusters"
        version = "v1"
    else:
        model_spec = spec["models"][model_id]
        dataset_name = model_spec.get("dataset", "")
        if "hits" in dataset_name:
            dtype = "hits"
        else:
            dtype = "clusters"

        # Try to extract version from model_id (e.g., v1 from pyg-cld-hits-v1)
        version_match = re.search(r"v\d+(\.\d+)*", model_id)
        version = version_match.group(0) if version_match else "v1"

    if model_id in spec.get("models", {}):
        model_spec = spec["models"][model_id]
        dataset_name = model_spec.get("dataset", "")
        if "hits" in dataset_name:
            dtype = "hits"
        else:
            dtype = "clusters"
    else:
        # If model_id not in spec, we need to guess dtype if it wasn't already set
        dtype = "hits" if "hits" in model_id else "clusters"

    final_name = args.name if args.name else exp_name
    remote_path = f"{scenario}/{dtype}/{version}/{final_name}"

    print(f"Preparing upload of {exp_path} to {args.repo}/{remote_path}...")

    # Find the checkpoint to use as best_weights.pth
    checkpoint_dir = exp_path / "checkpoints"
    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoints directory {checkpoint_dir} not found.")
        return

    checkpoints = sorted(list(checkpoint_dir.glob("checkpoint-*.pth")), key=lambda x: int(x.stem.split("-")[1]))

    if not checkpoints:
        print("Error: No checkpoints found.")
        return

    if args.step:
        best_checkpoint = checkpoint_dir / f"checkpoint-{args.step}.pth"
        if not best_checkpoint.exists():
            print(f"Error: Checkpoint step {args.step} not found.")
            return
        step_num = str(args.step)
    else:
        best_checkpoint = checkpoints[-1]
        step_num = best_checkpoint.stem.split("-")[1]

    print(f"Using {best_checkpoint.name} as best_weights.pth")

    # Files and folders to upload
    files_to_upload = [
        ("hyperparameters.json", "hyperparameters.json"),
        ("model_kwargs.pkl", "model_kwargs.pkl"),
        ("train-config.yaml", "train-config.yaml"),
        ("test-config.yaml", "test-config.yaml"),
        ("train.log", "train.log"),
        ("test.log", "test.log"),
        ("particleflow_spec.yaml", "particleflow_spec.yaml"),
    ]

    # Directories to upload
    dirs_to_upload = [
        ("history", "history"),
        ("runs", "runs"),
        ("validation", "validation", ["*.parquet"]),
    ]

    # Handle plots and preds for the chosen step
    plots_dir = exp_path / f"plots_step_{step_num}"

    # Fallback to plots_test if step-specific ones don't exist
    if not plots_dir.exists():
        plots_dir = exp_path / "plots_test"

    if plots_dir.exists():
        dirs_to_upload.append((plots_dir.name, "plots_best_weights"))

    destinations = {f"{remote_path}/checkpoints/best_weights.pth"}
    destinations.update(f"{remote_path}/{remote_file}" for local_file, remote_file in files_to_upload if (exp_path / local_file).exists())
    folder_uploads = []
    for local_dir, remote_dir, *patterns in dirs_to_upload:
        folder = exp_path / local_dir
        if not folder.is_dir():
            continue
        ignore = patterns[0] if patterns else None
        relative_paths = folder_files(folder, ignore)
        remote_folder = f"{remote_path}/{remote_dir}"
        destinations.update(f"{remote_folder}/{relative}" for relative in relative_paths)
        folder_uploads.append((folder, remote_folder, ignore, relative_paths))

    api = HfApi() if not args.dry_run else None
    if args.dry_run:
        print(f"[DRY-RUN] Selected {len(destinations):,} files.")
    else:
        preflight_file_count(api, args.repo, args.repo_type, destinations)

    total_size = 0

    # Upload best weights
    if args.dry_run:
        print(f"[DRY-RUN] Would upload {best_checkpoint} as {remote_path}/checkpoints/best_weights.pth")
    else:
        print(f"Uploading {best_checkpoint} as {remote_path}/checkpoints/best_weights.pth...")
        api.upload_file(
            path_or_fileobj=str(best_checkpoint),
            path_in_repo=f"{remote_path}/checkpoints/best_weights.pth",
            repo_id=args.repo,
            repo_type=args.repo_type,
        )
    total_size += best_checkpoint.stat().st_size

    # Upload files
    for local_file, remote_file in files_to_upload:
        local_path = exp_path / local_file
        if local_path.exists():
            if args.dry_run:
                print(f"[DRY-RUN] Would upload {local_path} as {remote_path}/{remote_file}")
            else:
                print(f"Uploading {local_path} as {remote_path}/{remote_file}...")
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"{remote_path}/{remote_file}",
                    repo_id=args.repo,
                    repo_type=args.repo_type,
                )
            total_size += local_path.stat().st_size

    # Upload folders
    for folder, remote_folder, ignore, relative_paths in folder_uploads:
        size = sum((folder / relative).stat().st_size for relative in relative_paths)
        total_size += size
        if args.dry_run:
            print(f"[DRY-RUN] Would upload folder {folder} ({format_size(size)}) to {remote_folder}")
            if ignore:
                print(f"  Ignoring: {ignore}")
        else:
            print(f"Uploading folder {folder} ({format_size(size)}) to {remote_folder}...")
            api.upload_folder(
                folder_path=str(folder),
                path_in_repo=remote_folder,
                repo_id=args.repo,
                repo_type=args.repo_type,
                ignore_patterns=ignore,
            )

    print(f"Total size processed: {format_size(total_size)}")


if __name__ == "__main__":
    main()
