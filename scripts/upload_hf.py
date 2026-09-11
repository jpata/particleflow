#!/usr/bin/env python3

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow enough time to inspect large Hub repositories on slower connections.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "1000")

import yaml  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

DEFAULT_REPO = "jpata/particleflow"
REMOTE_INFO_BATCH_SIZE = 100


def resolve_variables(value, data):
    """Recursively resolve ${key.path} references in the production spec."""
    if isinstance(value, list):
        return [resolve_variables(item, data) for item in value]
    if isinstance(value, dict):
        return {key: resolve_variables(item, data) for key, item in value.items()}
    if not isinstance(value, str):
        return value

    for match in re.findall(r"\${([^}]+)}", value):
        ref = data
        try:
            for key in match.split("."):
                ref = ref[key]
            value = value.replace(f"${{{match}}}", str(resolve_variables(ref, data)))
        except (KeyError, TypeError):
            continue
    return value


def get_dir_size(path):
    """Calculate the total size of a directory in bytes."""
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def format_size(num, suffix="B"):
    """Convert bytes to a human-readable size."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def remote_sizes(api, repo_id, paths):
    """Return the sizes of the requested paths that exist on the Hub."""
    result = []
    for start in range(0, len(paths), REMOTE_INFO_BATCH_SIZE):
        result.extend(
            api.get_paths_info(
                repo_id,
                paths[start : start + REMOTE_INFO_BATCH_SIZE],
                repo_type="dataset",
            )
        )
    return {item.path: item.size for item in result}


def check_remote_files(api, repo_id, files, force=False):
    """Split (local path, remote path) pairs into matching and pending files."""
    destinations = [remote_path for _, remote_path in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Multiple local files map to the same Hub destination")

    sizes = remote_sizes(api, repo_id, destinations)
    matching = []
    pending = []
    collisions = []
    for local_path, remote_path in files:
        remote_size = sizes.get(remote_path)
        if remote_size is None:
            pending.append((local_path, remote_path))
        elif remote_size == local_path.stat().st_size:
            matching.append((local_path, remote_path))
        elif force:
            pending.append((local_path, remote_path))
        else:
            collisions.append((local_path, remote_path, remote_size))

    if collisions:
        details = "\n".join(
            f"  {remote_path}: local={local_path.stat().st_size}, remote={remote_size}" for local_path, remote_path, remote_size in collisions
        )
        raise RuntimeError("Refusing to overwrite Hub files with different sizes. " f"Use --force to replace them:\n{details}")
    return matching, pending


def upload_files(api, files, repo_id, dry_run=False, force=False):
    """Upload individual files, skipping remote files with matching sizes."""
    if not files:
        print("No files selected.")
        return 0, 0

    total_size = sum(local_path.stat().st_size for local_path, _ in files)
    if dry_run:
        for local_path, remote_path in files:
            print(f"[DRY-RUN] {local_path} ({format_size(local_path.stat().st_size)}) " f"-> {repo_id}/{remote_path}")
        print(f"Selected {len(files)} file(s), {format_size(total_size)} total.")
        return 0, 0

    matching, pending = check_remote_files(api, repo_id, files, force=force)
    print(f"Selected {len(files)} file(s); skipping {len(matching)} matching Hub file(s).")
    uploaded_size = 0
    for index, (local_path, remote_path) in enumerate(pending, start=1):
        size = local_path.stat().st_size
        print(f"Uploading {index}/{len(pending)}: {local_path} ({format_size(size)}) -> {remote_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {remote_path}",
        )
        uploaded_size += size

    print(f"Uploaded {len(pending)} file(s), {format_size(uploaded_size)}; skipped {len(matching)}.")
    return len(pending), len(matching)


def upload_folder_batched(api, local_dir, remote_path, repo_id, batch_size, force=False):
    """Upload a directory in small, resumable commits."""
    local_files = sorted(path for path in local_dir.rglob("*") if path.is_file())
    files = [(local_path, f"{remote_path}/{local_path.relative_to(local_dir).as_posix()}") for local_path in local_files]
    matching, pending = check_remote_files(api, repo_id, files, force=force)
    pending_relpaths = [local_path.relative_to(local_dir).as_posix() for local_path, _ in pending]

    print(f"Found {len(files)} local file(s); skipping {len(matching)} matching Hub file(s).")
    num_parts = (len(pending_relpaths) + batch_size - 1) // batch_size
    for start in range(0, len(pending_relpaths), batch_size):
        batch = pending_relpaths[start : start + batch_size]
        part = start // batch_size + 1
        print(f"Uploading batch {part}/{num_parts} ({len(batch)} file(s))...")
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=batch,
            commit_message=f"Upload {remote_path} (batch {part}/{num_parts})",
        )
    return len(pending_relpaths), len(matching)


def load_production(args, parser):
    """Load one production and resolve its workspace directory."""
    spec_path = Path(args.spec)
    if not spec_path.is_file():
        parser.error(f"spec file not found: {spec_path}")

    with spec_path.open() as handle:
        spec = yaml.safe_load(handle)

    project = spec.get("project", {})
    sites = project.get("sites", {})
    if args.site in sites:
        project.update(sites[args.site])

    production = spec.get("productions", {}).get(args.scenario)
    if production is None:
        if args.workspace_dir is None:
            parser.error(f"scenario '{args.scenario}' is not in {spec_path}; " "provide --workspace-dir for an external campaign")
        production = {"samples": {}}

    workspace_value = args.workspace_dir or resolve_variables(production.get("workspace_dir", ""), spec)
    if not workspace_value:
        parser.error(f"no workspace directory configured for scenario '{args.scenario}'")
    workspace_dir = Path(workspace_value)
    if not workspace_dir.is_dir():
        parser.error(f"workspace directory not found: {workspace_dir}")
    return production, workspace_dir


def selected_samples(production, requested, parser):
    """Return configured samples, optionally filtered by sample name."""
    samples = production.get("samples", {}) or {}
    if requested:
        unknown = sorted(set(requested) - set(samples))
        if unknown:
            parser.error(f"unknown sample(s): {', '.join(unknown)}")
        return {name: samples[name] for name in requested}
    return samples


def find_root_dir(workspace_dir, sample):
    """Locate the directory containing raw ROOT files for a configured sample."""
    process_name = sample.get("process_name")
    if not process_name:
        return None

    candidates = [Path(workspace_dir) / "gen" / process_name / "root"]
    if sample.get("output_subdir"):
        candidates.extend(
            [
                Path(workspace_dir) / sample["output_subdir"] / process_name / "root",
                Path(workspace_dir) / "gen" / sample["output_subdir"] / process_name / "root",
            ]
        )
    candidates.append(Path(workspace_dir) / process_name / "root")
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def find_parquet_dir(workspace_dir, sample):
    """Locate the directory containing Parquet files for a configured sample."""
    process_name = sample.get("process_name")
    if not process_name:
        return None

    post_dir = Path(workspace_dir) / "post"
    candidates = [post_dir / process_name]
    if sample.get("output_subdir"):
        candidates.insert(0, post_dir / sample["output_subdir"] / process_name)
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def inferred_sample_name(process_name):
    """Infer the conventional Hub sample name from a Key4hep process name."""
    match = re.fullmatch(r"p8_ee_(.+)_ecm\d+", process_name)
    return match.group(1).lower() if match else process_name


def discover_root_samples(workspace_dir):
    """Discover Key4hep-style ROOT directories for an external campaign."""
    result = {}
    gen_dir = Path(workspace_dir) / "gen"
    if not gen_dir.is_dir():
        return result
    for root_dir in sorted(gen_dir.glob("*/root")):
        process_name = root_dir.parent.name
        sample_name = inferred_sample_name(process_name)
        if sample_name in result:
            raise ValueError(f"multiple processes map to inferred sample '{sample_name}'")
        result[sample_name] = {"process_name": process_name}
    return result


def choose_files(paths, selection, num_files):
    """Apply the common first-N/all selection to a sorted path sequence."""
    paths = sorted(paths)
    if selection == "all":
        return paths
    return paths[:num_files]


def upload_tfds(args, production, workspace_dir, api):
    tfds_root = workspace_dir / "tfds"
    if not tfds_root.is_dir():
        print(f"TFDS root directory not found: {tfds_root}")
        return

    datasets = []
    for dataset_dir in sorted(tfds_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "downloads" or dataset_dir.name.startswith("torchinductor"):
            continue
        if args.dataset and dataset_dir.name not in args.dataset:
            continue
        split_dir = dataset_dir / args.split
        if not split_dir.is_dir():
            continue
        for version_dir in sorted(split_dir.iterdir()):
            if not version_dir.is_dir() or (args.version and version_dir.name != args.version):
                continue
            remote_path = f"tensorflow_datasets/{args.scenario}/{dataset_dir.name}/{args.split}/{version_dir.name}"
            datasets.append((version_dir, remote_path))

    if not datasets:
        print(f"No TFDS datasets found for split {args.split} in {tfds_root}.")
        return

    total_size = sum(get_dir_size(local_dir) for local_dir, _ in datasets)
    print(f"Selected {len(datasets)} TFDS dataset(s), {format_size(total_size)} total.")
    for local_dir, remote_path in datasets:
        size = get_dir_size(local_dir)
        if args.dry_run:
            print(f"[DRY-RUN] {local_dir} ({format_size(size)}) -> {args.repo}/{remote_path}")
        else:
            print(f"Uploading {local_dir} ({format_size(size)}) -> {remote_path}")
            upload_folder_batched(
                api,
                local_dir,
                remote_path,
                args.repo,
                args.batch_size,
                force=args.force,
            )


def upload_root(args, production, workspace_dir, api, parser):
    samples = production.get("samples", {}) or discover_root_samples(workspace_dir)
    production = {**production, "samples": samples}
    samples = selected_samples(production, args.sample, parser)

    files = []
    for sample_name, sample in samples.items():
        root_dir = find_root_dir(workspace_dir, sample)
        if root_dir is None:
            print(f"ROOT directory not found for sample '{sample_name}', skipping.")
            continue
        selected = choose_files(root_dir.glob("*.root"), args.selection, args.num_files)
        if not selected:
            print(f"No ROOT files found for sample '{sample_name}', skipping.")
        files.extend((path, f"root/{args.scenario}/{sample_name}/{path.name}") for path in selected)

    upload_files(api, files, args.repo, dry_run=args.dry_run, force=args.force)


def hub_root_paths(api, repo_id, scenario, requested_samples):
    """List uploaded ROOT files for a scenario, optionally filtered by sample."""
    prefix = f"root/{scenario}/"
    result = []
    try:
        tree = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=f"root/{scenario}",
            recursive=True,
            repo_type="dataset",
        )
        for item in tree:
            path = item.path
            if not path.startswith(prefix) or not path.endswith(".root"):
                continue
            relative = path[len(prefix) :]
            sample_name, separator, _ = relative.partition("/")
            if separator and (not requested_samples or sample_name in requested_samples):
                result.append(path)
    except EntryNotFoundError:
        return []
    return sorted(result)


def parquets_matching_hub_root(args, workspace_dir, api):
    """Find local Parquet files whose basenames match ROOT files on the Hub."""
    root_paths = hub_root_paths(api, args.repo, args.scenario, set(args.sample))
    if not root_paths:
        print(f"No matching ROOT files found at {args.repo}/root/{args.scenario}/.")
        return []

    by_name = defaultdict(list)
    for path in (workspace_dir / "post").rglob("*.parquet"):
        by_name[path.name].append(path)

    files = []
    missing = []
    for root_path in root_paths:
        parquet_name = f"{Path(root_path).stem}.parquet"
        matches = by_name.get(parquet_name, [])
        if not matches:
            missing.append(root_path)
            continue
        if len(matches) > 1:
            locations = ", ".join(str(path) for path in matches)
            raise RuntimeError(f"multiple local Parquet files match {root_path}: {locations}")
        remote_path = f"parquet/{root_path.removeprefix('root/')}"
        remote_path = str(Path(remote_path).with_suffix(".parquet"))
        files.append((matches[0], remote_path))

    if missing:
        print(f"Skipping {len(missing)} Hub ROOT file(s) without a local Parquet counterpart:")
        for path in missing:
            print(f"  {path}")
    return files


def local_parquet_files(args, production, workspace_dir, parser):
    samples = selected_samples(production, args.sample, parser)
    if not samples:
        parser.error("local Parquet selection requires samples configured in the production spec")

    files = []
    for sample_name, sample in samples.items():
        parquet_dir = find_parquet_dir(workspace_dir, sample)
        if parquet_dir is None:
            print(f"Parquet directory not found for sample '{sample_name}', skipping.")
            continue
        selected = choose_files(parquet_dir.glob("*.parquet"), args.selection, args.num_files)
        if not selected:
            print(f"No Parquet files found for sample '{sample_name}', skipping.")
        files.extend((path, f"parquet/{args.scenario}/{sample_name}/{path.name}") for path in selected)
    return files


def upload_parquet(args, production, workspace_dir, api, parser):
    if args.selection == "matching-root":
        files = parquets_matching_hub_root(args, workspace_dir, api)
    else:
        files = local_parquet_files(args, production, workspace_dir, parser)
    upload_files(api, files, args.repo, dry_run=args.dry_run, force=args.force)


def add_common_arguments(parser):
    parser.add_argument("scenario", help="Production scenario / Hub directory name")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repository (default: {DEFAULT_REPO})")
    parser.add_argument("--spec", default="particleflow_spec.yaml", help="Production spec path")
    parser.add_argument("--site", default="tallinn", help="Site profile used to resolve workspace paths")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace path; also permits scenarios absent from the spec",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show selected files without uploading")
    parser.add_argument("--force", action="store_true", help="Replace Hub files whose sizes differ")


def build_parser():
    parser = argparse.ArgumentParser(description="Upload ParticleFlow ROOT, Parquet, or TFDS data to Hugging Face.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tfds_parser = subparsers.add_parser("tfds", help="Upload one TFDS split")
    add_common_arguments(tfds_parser)
    tfds_parser.add_argument("split", help="TFDS configuration split (for example: 1)")
    tfds_parser.add_argument("--version", help="Only upload this dataset version")
    tfds_parser.add_argument("--dataset", action="append", default=[], help="Only upload this dataset (repeatable)")
    tfds_parser.add_argument("--batch-size", type=int, default=100, help="Files per upload commit (default: 100)")

    root_parser = subparsers.add_parser("root", help="Upload raw ROOT files")
    add_common_arguments(root_parser)
    root_parser.add_argument("--sample", action="append", default=[], help="Only upload this sample (repeatable)")
    root_parser.add_argument("--selection", choices=("first", "all"), default="first")
    root_parser.add_argument(
        "--num-files",
        type=int,
        default=2,
        help="Files per sample with --selection first (default: 2)",
    )

    parquet_parser = subparsers.add_parser("parquet", help="Upload postprocessed Parquet files")
    add_common_arguments(parquet_parser)
    parquet_parser.add_argument("--sample", action="append", default=[], help="Only upload this sample (repeatable)")
    parquet_parser.add_argument(
        "--selection",
        choices=("matching-root", "first", "all"),
        default="matching-root",
        help="File selection policy (default: matching-root)",
    )
    parquet_parser.add_argument(
        "--num-files",
        type=int,
        default=2,
        help="Files per sample with --selection first (default: 2)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be at least 1")
    if getattr(args, "num_files", 1) < 1:
        parser.error("--num-files must be at least 1")

    production, workspace_dir = load_production(args, parser)
    api = HfApi()
    print(f"Scenario: {args.scenario}")
    print(f"Workspace: {workspace_dir}")
    print(f"Repository: {args.repo}")

    try:
        if args.command == "tfds":
            upload_tfds(args, production, workspace_dir, api)
        elif args.command == "root":
            upload_root(args, production, workspace_dir, api, parser)
        elif args.command == "parquet":
            upload_parquet(args, production, workspace_dir, api, parser)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
