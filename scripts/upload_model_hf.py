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
    parser = argparse.ArgumentParser(description="Upload ParticleFlow model checkpoints to HuggingFace.")
    parser.add_argument("experiment_dir", help="Path to the experiment directory in 'experiments/'")
    parser.add_argument("--repo", default="jpata/particleflow", help="HF repository ID")
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

    api = HfApi() if not args.dry_run else None

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
    ]

    # Handle plots and preds for the chosen step
    plots_dir = exp_path / f"plots_step_{step_num}"
    preds_dir = exp_path / f"preds_step_{step_num}"

    # Fallback to plots_test/preds_test if step-specific ones don't exist
    if not plots_dir.exists():
        plots_dir = exp_path / "plots_test"
    if not preds_dir.exists():
        preds_dir = exp_path / "preds_test"

    if plots_dir.exists():
        dirs_to_upload.append((plots_dir.name, "plots_best_weights"))
    if preds_dir.exists():
        dirs_to_upload.append((preds_dir.name, "preds_best_weights"))

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
    for local_dir_name, remote_dir_name in dirs_to_upload:
        local_dir_path = exp_path / local_dir_name
        if local_dir_path.exists():
            size = get_dir_size(local_dir_path)
            total_size += size
            if args.dry_run:
                print(f"[DRY-RUN] Would upload folder {local_dir_path} ({format_size(size)}) to {remote_path}/{remote_dir_name}")
            else:
                print(f"Uploading folder {local_dir_path} ({format_size(size)}) to {remote_path}/{remote_dir_name}...")
                api.upload_folder(
                    folder_path=str(local_dir_path),
                    path_in_repo=f"{remote_path}/{remote_dir_name}",
                    repo_id=args.repo,
                    repo_type=args.repo_type,
                )

    print(f"Total size processed: {format_size(total_size)}")


if __name__ == "__main__":
    main()
