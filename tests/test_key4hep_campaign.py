"""The joint release campaign must build complete, isolated multi-detector datasets."""

import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from mlpf.snakemake import produce_key4hep_snakemake

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("cld", "clic", "idea", "ild")
SAMPLES = {"ttbar", "qq", "ww_fullhad", "z_qq", "z_tautau"}
VERSIONS = {"cld": "3.4.0", "clic": "3.4.0", "idea": "0.5.0", "ild": "0.2.0"}
EVENTS_PER_SHARD = 10
SHARDS_PER_SAMPLE = {"cld": 5000, "clic": 5000, "idea": 1000, "ild": 5000}
GEN_MEMORY_MB = {"cld": 4500, "clic": 4500, "idea": 48000, "ild": 5000}
POST_MEMORY_MB = {"cld": 1000, "clic": 1000, "idea": 1000, "ild": 1200}


def local_spec(tmp_path):
    spec = yaml.safe_load((ROOT / "particleflow_spec.yaml").read_text())
    spec["project"]["paths"] = {
        "project_root": str(ROOT),
        "storage_root": str(tmp_path / "storage"),
        "scratch_root": str(tmp_path / "scratch"),
        "tmpdir": str(tmp_path / "tmp"),
    }
    return spec


def test_campaign_event_budget_and_builder_versions(tmp_path, monkeypatch):
    import tensorflow_datasets as tfds

    monkeypatch.delenv("TFDS_VERSION", raising=False)
    spec = local_spec(tmp_path)
    total = 0
    for detector in DETECTORS:
        production = spec["productions"][detector]
        assert set(production["samples"]) == set(production["tfds_mapping"]) == SAMPLES
        assert not production["allow_partial"] and production["validate_events"]
        assert production["workspace_dir"].endswith("/v1.4.0/50k")
        for sample, settings in production["samples"].items():
            first, last = settings["seed_range"]
            count = (last - first) * settings["events_per_job"]
            assert settings["events_per_job"] == EVENTS_PER_SHARD
            assert last - first == SHARDS_PER_SAMPLE[detector]
            total += count
        for kind in ("tfds_mapping", "tfds_hit_mapping"):
            for sample, settings in production.get(kind, {}).items():
                module = importlib.import_module(settings["builder_path"].replace("/", "."))
                classes = [
                    c
                    for c in vars(module).values()
                    if isinstance(c, type) and c.__module__ == module.__name__ and issubclass(c, tfds.core.GeneratorBasedBuilder)
                ]
                assert len(classes) == 1
                builder = classes[0](config="1", data_dir=tmp_path / "tfds")
                assert str(builder.version) == settings["version"]
                assert settings["version"] == ("3.3.0" if kind == "tfds_hit_mapping" else VERSIONS[detector])
    assert total == 800000


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("channel", ("qq", "tautau"))
def test_z_builder_selects_only_91gev_process(tmp_path, monkeypatch, detector, channel):
    module = importlib.import_module(f"mlpf.heptfds.{detector}_pf_edm4hep.z_{channel}")
    cls = next(c for c in vars(module).values() if isinstance(c, type) and c.__module__ == module.__name__)
    base = importlib.import_module(f"mlpf.heptfds.{detector}_pf_edm4hep.qq")
    selected = []
    if detector == "idea":
        monkeypatch.setattr(base, "find_idea_parquets", lambda path, process_name: selected.append(path / process_name) or [])
        monkeypatch.setattr(base, "split_event_references", lambda files: {"train": [], "test": []})
    else:
        monkeypatch.setattr(base, "split_sample", lambda path, *args, **kwargs: selected.append(path) or {})
    builder = cls(config="1", data_dir=tmp_path / "tfds")
    builder._split_generators(SimpleNamespace(manual_dir=tmp_path))
    assert selected == [tmp_path / f"p8_ee_Z_{channel}_ecm91"]
    assert "91 GeV" in builder.info.description


def test_combined_graph_is_complete_without_training_or_local_execution(tmp_path, monkeypatch):
    pytest.importorskip("snakemake")
    spec = local_spec(tmp_path)
    (tmp_path / "particleflow_spec.yaml").write_text(yaml.safe_dump(spec))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PF_SITE", raising=False)
    produce_key4hep_snakemake.main([])
    jobs = tmp_path / spec["campaigns"]["key4hep_50k"]["jobs_dir"]
    for detector in DETECTORS:
        graph = (jobs / detector / "Snakefile").read_text()
        assert "rule train_" not in graph and "glob_wildcards" not in graph
        gen_rule = graph.split("rule gen_task:", 1)[1].split("\nrule ", 1)[0]
        post_rule = graph.split("rule post_task:", 1)[1].split("\nrule ", 1)[0]
        assert f"mem_mb={GEN_MEMORY_MB[detector]}," in gen_rule
        assert f"mem_mb={POST_MEMORY_MB[detector]}," in post_rule
        for sample in SAMPLES:
            gen = jobs / detector / "gen" / f"gen_{sample}.sh"
            subprocess.run(["bash", "-n", str(gen)], check=True)
            text = gen.read_text()
            assert f"export NEV={EVENTS_PER_SHARD} &&" in text and f"t.GetEntries()=={EVENTS_PER_SHARD}," in text
            assert f"scratch/{detector}_" in text
            post = jobs / detector / "post" / f"post_{sample}.sh"
            assert "X_track)==" not in post.read_text()
            tfds = (jobs / detector / "tfds" / f"tfds_{sample}.sh").read_text()
            assert f"export TFDS_VERSION={VERSIONS[detector]}" in tfds
            assert "num_examples" not in tfds
    run = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", str(jobs / "Snakefile"), "--executor", "local", "--cores", "1", "--dry-run", "--quiet", "rules"],
        cwd=tmp_path,
        env=os.environ | {"XDG_CACHE_HOME": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
    )
    output = run.stdout + run.stderr
    assert run.returncode == 0, output[-6000:]
    for detector in DETECTORS:
        shard_count = len(SAMPLES) * SHARDS_PER_SAMPLE[detector]
        tfds_count = 50 if detector in ("cld", "clic") else 5
        for stage, count in (("gen_task", shard_count), ("post_task", shard_count), ("tfds_task", tfds_count)):
            assert re.search(rf"^\s*{detector}_{stage}\s+{count}\s*$", output, re.MULTILINE), output[-6000:]
    assert not re.search(r"train_\S+\s+\d", output)


@pytest.mark.parametrize("detector", DETECTORS)
def test_new_z_dataset_arrayrecord_roundtrip(tmp_path, detector):
    import awkward as ak
    import tensorflow_datasets as tfds
    from tests.test_ild_pipeline import write_event
    from mlpf.heptfds.edm4hep_utils import utils_idea

    manual = tmp_path / "manual"
    directory = manual / "p8_ee_Z_qq_ecm91"
    directory.mkdir(parents=True)
    for index in range(100 if detector in ("cld", "clic") else 2):
        path = directory / f"seed{index:03d}.parquet"
        write_event(path, energy=10.0 + index)
        if detector == "idea":
            data = ak.from_parquet(path)
            fields = {name: data[name] for name in data.fields}
            fields["X_track"] = data.X_track[:, :, : len(utils_idea.X_FEATURES_TRK)]
            for field, value in {
                "metadata_track_source": utils_idea.TRACK_SOURCE,
                "metadata_track_collection": utils_idea.TRACK_COLLECTION,
                "metadata_track_quality_policy": utils_idea.TRACK_QUALITY_POLICY,
                "metadata_track_truth_policy": utils_idea.TRACK_TRUTH_POLICY,
                "metadata_track_truth_min_purity": utils_idea.TRACK_TRUTH_MIN_PURITY,
            }.items():
                fields[field] = ak.Array([value])
            ak.to_parquet(ak.Record(fields), path)
    module = importlib.import_module(f"mlpf.heptfds.{detector}_pf_edm4hep.z_qq")
    cls = next(c for c in vars(module).values() if isinstance(c, type) and c.__module__ == module.__name__)
    builder = cls(config="1", data_dir=tmp_path / "tfds")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual))
    assert builder.info.splits["train"].num_examples == (9 if detector in ("cld", "clic") else 1)
    assert builder.info.splits["test"].num_examples == 1
    example = builder.as_data_source(split="test")[0]
    assert example["X"].shape == (1, 19)
    assert example["ytarget"].shape == (1, 14)
    assert example["ytarget"][0, 0] == 1  # charged-hadron class, rather than a raw PDG ID


@pytest.mark.parametrize("detector", ("cld", "clic"))
def test_z_hit_dataset_arrayrecord_roundtrip(tmp_path, detector):
    import awkward as ak
    import numpy as np
    import tensorflow_datasets as tfds
    from mlpf.heptfds.edm4hep_utils.utils_hits import X_FEATURES, Y_FEATURES
    from tests.test_ild_pipeline import write_event

    manual = tmp_path / "manual"
    directory = manual / "p8_ee_Z_tautau_ecm91"
    directory.mkdir(parents=True)
    for index in range(100):
        path = directory / f"seed{index:03d}.parquet"
        write_event(path)
        data = ak.from_parquet(path)
        fields = {name: data[name] for name in data.fields}
        fields.update(
            {
                "X_hit_tracker": ak.Array([np.ones((1, len(X_FEATURES)), dtype=np.float32)]),
                "X_hit_calo": ak.Array([np.zeros((0, len(X_FEATURES)), dtype=np.float32)]),
                "ytarget_hit_tracker": data.ytarget_track,
                "ytarget_hit_calo": data.ytarget_cluster,
            }
        )
        ak.to_parquet(ak.Record(fields), path)
    module = importlib.import_module(f"mlpf.heptfds.{detector}_pf_edm4hep_hits.z_tautau")
    cls = next(c for c in vars(module).values() if isinstance(c, type) and c.__module__ == module.__name__)
    builder = cls(config="1", data_dir=tmp_path / "tfds")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual))
    assert builder.info.splits["train"].num_examples == 9
    assert builder.info.splits["test"].num_examples == 1
    example = builder.as_data_source(split="test")[0]
    assert example["X"].shape == (1, len(X_FEATURES))
    assert example["ytarget"].shape == (1, len(Y_FEATURES))
    assert not example["ycand"].any()


def test_smoke_graph_runs_ten_generation_jobs_per_sample(tmp_path, monkeypatch):
    pytest.importorskip("snakemake")
    spec = local_spec(tmp_path)
    (tmp_path / "particleflow_spec.yaml").write_text(yaml.safe_dump(spec))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PF_SITE", raising=False)
    produce_key4hep_snakemake.main(["--smoke"])
    jobs = tmp_path / (spec["campaigns"]["key4hep_50k"]["jobs_dir"] + "_smoke_10ev")
    for detector in DETECTORS:
        graph = (jobs / detector / "Snakefile").read_text()
        assert "rule post_task" not in graph
        assert "rule tfds_task" not in graph
        for sample, settings in spec["productions"][detector]["samples"].items():
            first, _ = settings["seed_range"]
            assert f"gen_{sample}_{{seed}}.done', seed=range({first}, {first + 10}, 1)" in graph
            script = (jobs / detector / "gen" / f"gen_{sample}.sh").read_text()
            assert "/50k/smoke_10ev/gen/" in script
            assert "export NEV=10 &&" in script
            assert "t.GetEntries()==10," in script

    run = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", str(jobs / "Snakefile"), "--cores", "1", "--dry-run", "--quiet", "rules"],
        env=os.environ | {"XDG_CACHE_HOME": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
    )
    output = run.stdout + run.stderr
    assert run.returncode == 0, output
    for detector in DETECTORS:
        assert re.search(rf"^\s*{detector}_gen_task\s+50\s*$", output, re.MULTILINE), output
