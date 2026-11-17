import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.training.export_benchmark import export, write_checksums
from scripts.training.publish_benchmark import publish, validate_package


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def sweep(tmp_path):
    root = tmp_path / "raw"
    run = root / "runs/recipe/model_seed12345"
    session = run / "benchmark/worker"
    write_json(
        root / "sweep-state.json",
        {"cases": [{"variant": "model", "global_batch_size": 2, "seed": 12345, "status": "completed", "started_at": 1, "finished_at": 5}]},
    )
    write_json(run / "scenario-manifest.json", {"job": {"variant_name": "model", "global_batch_size": 2, "seed": 12345}})
    write_json(
        session / "metadata.json",
        {
            "config": {"train_dataset": {}, "valid_dataset": {}, "test_dataset": {}, "model": {}, "data_dir": "/private/data"},
            "world_size": 1,
            "device": {"name": "future GPU", "uuid": "private-uuid", "total_memory_bytes": 1234},
            "architecture": "new_arch",
            "python": "3.11",
            "torch": "2.11",
            "git": {"revision": "abc", "status": " M source.py"},
        },
    )
    write_json(
        session / "summary.json",
        {
            "rank": 0,
            "status": "completed",
            "steps": 2,
            "warmup_steps_excluded": 1,
            "events": 4,
            "hits": 40,
            "started_at_unix_time": 2,
            "finished_at_unix_time": 4,
            "worker_elapsed_seconds": 2,
            "steady_events_per_training_second": 2,
            "steady_hits_per_training_second": 20,
            "steady_training_step_seconds": {"p50": 1, "p99": 1},
            "phase_exclusive_seconds": {"training_loop": 2},
            "framework_memory_bytes": {"process_peak_allocated_bytes": 10},
        },
    )
    (session / "steps.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "events": 2,
                "hits": 20,
                "padded_hits": 24,
                "duration_seconds": 1,
                "data_wait_seconds": 0.01,
                "losses": {"Total_unweighted": 3},
            }
        )
        + "\n"
        + json.dumps({"event": "cuda_timing", "step": 1, "cuda_stream_elapsed_seconds": 0.8})
        + "\n"
    )
    with (session / "steps.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "step": 2,
                    "events": 2,
                    "hits": 20,
                    "padded_hits": 24,
                    "duration_seconds": 1,
                    "data_wait_seconds": 0.01,
                    "losses": {"Total_unweighted": 2.8},
                }
            )
            + "\n"
        )
    write_json(run / "history/step_2.json", {"train": {"Total_unweighted": 3}, "valid": {"Total_unweighted": 2.9}})
    (run / "checkpoint.pth").write_bytes(b"must not export")
    return root


def test_export_keeps_provenance_and_omits_private_paths_and_large_artifacts(sweep, tmp_path):
    output = tmp_path / "package"
    cases = export([("Future", sweep)], output, "v3.2.0")
    write_checksums(output)
    validate_package(output)
    assert len(cases) == 1 and cases[0]["dirty"]
    text = (output / "results.json").read_text()
    assert "abc" in text and "future GPU" in text
    assert "private-uuid" not in text and "/private/data" not in text
    assert len((output / "steps.csv").read_text().splitlines()) == 3
    assert not list(output.glob("*.pth"))
    with pytest.raises(ValueError, match="empty output"):
        export([("Future", sweep)], output, "v3.2.0")


def test_package_validation_rejects_edits_and_unlisted_files(tmp_path):
    (tmp_path / "README.md").write_text("original")
    write_checksums(tmp_path)
    validate_package(tmp_path)
    (tmp_path / "README.md").write_text("modified")
    with pytest.raises(ValueError, match="Checksum"):
        validate_package(tmp_path)
    write_checksums(tmp_path)
    (tmp_path / "unexpected.txt").write_text("new")
    with pytest.raises(ValueError, match="files differ"):
        validate_package(tmp_path)


def test_publish_verifies_contents_before_tagging_and_pins_parent(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "README.md").write_text("release")
    write_checksums(package)
    api = Mock()
    api.list_repo_refs.return_value.tags = []
    # Hub annotations may have their own object ID; resolve the tag through repo_info.
    api.repo_info.side_effect = [SimpleNamespace(sha="parent"), SimpleNamespace(sha="new")]
    api.get_paths_info.return_value = []
    api.upload_folder.return_value.oid = "new"
    remote = []
    for path in package.iterdir():
        data = path.read_bytes()
        remote.append(
            SimpleNamespace(
                path=f"benchmarks/v3.2.0/package/{path.name}",
                size=len(data),
                lfs=None,
                blob_id=hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
            )
        )
    api.list_repo_tree.return_value = remote
    assert publish(package, "owner/repo", "v3.2.0", api) == "new"
    assert api.upload_folder.call_args.kwargs["parent_commit"] == "parent"
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == "benchmarks/v3.2.0/package"
    api.create_tag.assert_called_once()
    assert api.create_tag.call_args.kwargs["revision"] == "new"


def test_existing_tag_never_moves(tmp_path):
    (tmp_path / "README.md").write_text("release")
    write_checksums(tmp_path)
    api = Mock()
    api.list_repo_refs.return_value.tags = [SimpleNamespace(name="v3.2.0")]
    with pytest.raises(ValueError, match="already exists"):
        publish(tmp_path, "owner/repo", "v3.2.0", api)
    api.upload_folder.assert_not_called()
    api.create_tag.assert_not_called()


def test_export_rejects_summary_raw_disagreement(sweep, tmp_path):
    summary_path = next(sweep.glob("runs/*/*/benchmark/*/summary.json"))
    summary = json.loads(summary_path.read_text())
    summary["steady_events_per_training_second"] = 123
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="Raw throughput disagrees"):
        export([("Future", sweep)], tmp_path / "bad_export", "v3.2.0")


def test_remote_checksum_failure_leaves_commit_untagged(tmp_path):
    (tmp_path / "README.md").write_text("release")
    write_checksums(tmp_path)
    api = Mock()
    api.list_repo_refs.return_value.tags = []
    api.repo_info.return_value.sha = "parent"
    api.get_paths_info.return_value = []
    api.upload_folder.return_value.oid = "new"
    api.list_repo_tree.return_value = [
        SimpleNamespace(path=f"benchmarks/v3.2.0/{tmp_path.name}/{path.name}", size=path.stat().st_size, lfs=SimpleNamespace(sha256="wrong"))
        for path in tmp_path.iterdir()
    ]
    with pytest.raises(RuntimeError, match="Remote content mismatch"):
        publish(tmp_path, "owner/repo", "v3.2.0", api)
    api.create_tag.assert_not_called()
