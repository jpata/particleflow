"""ILD integration contracts: native track states, metadata and disjoint TFDS jobs."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import shutil

import awkward as ak
import numpy as np
import pytest
import yaml

from mlpf.conf import Dataset, EDM4HEP, MLPFConfig
from mlpf.data.key4hep.postprocessing import track_to_features, _assign_target_jets
from mlpf.data.key4hep.ild import pfo_representatives, track_truth
from mlpf.heptfds.edm4hep_utils.utils_ild import (
    PROVENANCE,
    generate_examples,
    split_sample,
)


def fitted_track(locations=(2, 1), ndof=10):
    name = "MarlinTrkTracks"
    properties = {
        "type": [0],
        "chi2": [2.0],
        "ndf": [ndof],
        "trackStates_begin": [0],
        "trackStates_end": [2],
    }
    track = ak.Array({f"{name}.{key}": [value] for key, value in properties.items()})
    states = {
        "location": locations,
        "referencePoint.x": [50.0, 0.0],
        "referencePoint.y": [0.0, 0.0],
        "omega": [0.01, 0.00006],
        "tanLambda": [1.0, 0.2],
        "phi": [2.0, 0.3],
        "D0": [1.0, 0.0],
        "Z0": [2.0, 0.0],
        "time": [0.0, 0.0],
    }
    return {
        name: track,
        f"_{name}_trackStates": ak.Array({f"_{name}_trackStates.{key}": [list(value)] for key, value in states.items()}),
    }


def test_ild_uses_explicit_ip_state_and_measured_first_hit():
    result = track_to_features(fitted_track(), 0, 2.0, "MarlinTrkTracks", False)
    np.testing.assert_allclose(result["pt"], [10.0])
    np.testing.assert_allclose(result["phi"], [0.3])
    np.testing.assert_allclose(result["radiusOfInnermostHit"], [50.0])
    np.testing.assert_array_equal(result["dEdx"], [0.0])


def test_multitrack_pfos_have_one_baseline_representative_without_double_counting():
    reco = {"charge": [1, 0, 0], "tracks_begin": [0, 2, 4], "tracks_end": [2, 4, 6], "clusters_begin": [0, 0, 2], "clusters_end": [0, 2, 2]}
    tracks, clusters = pfo_representatives(reco, np.arange(6), np.arange(2), [4, 10, 8, 9, 6, 6], [5, 20])
    assert tracks == {1: 0, 4: 2}
    assert clusters == {1: 1}
    assert sorted([*tracks.values(), *clusters.values()]) == [0, 1, 2]


def test_track_labels_count_distinct_measurements_and_reject_secondary_owners():
    relation, simulated = "TrackerRelations", "TrackerSim"
    values = {
        f"_{relation}_from/_{relation}_from.index": [0, 0, 1, 2, 3, 4, 5, 5],
        f"_{relation}_from/_{relation}_from.collectionID": [10] * 8,
        f"_{relation}_to/_{relation}_to.index": [0, 0, 1, 2, 3, 4, 0, 3],
        f"_{relation}_to/_{relation}_to.collectionID": [20] * 8,
        f"_{simulated}_particle/_{simulated}_particle.index": [0, 0, 0, 1, 1],
        f"_{simulated}_particle/_{simulated}_particle.collectionID": [1] * 5,
    }
    links = {key: ak.Array([value]) for key, value in values.items()}
    measurements = [0, 0, 1, 2, 9, 3, 4, 0, 0, 1, 9, 10, 0, 1, 0, 5]
    prefix = "_MarlinTrkTracks_trackerHits/_MarlinTrkTracks_trackerHits"
    prop = {
        prefix + ".index": ak.Array([measurements]),
        prefix + ".collectionID": ak.Array([[10] * len(measurements)]),
        "MarlinTrkTracks": ak.Array(
            {"MarlinTrkTracks.trackerHits_begin": [[0, 5, 8, 12, 14]], "MarlinTrkTracks.trackerHits_end": [[5, 8, 12, 14, 16]]}
        ),
    }
    gen = {"PDG": [211, 211], "generatorStatus": [1, 0], "charge": [1, 1]}
    ids = {"MCParticles": 1, "TrackerHits": 10, simulated: 20}
    relations = {"TrackerHits": (relation, simulated)}
    gp, track, purity = track_truth(prop, links, 0, ids, relations, gen)
    np.testing.assert_array_equal(gp, [0, 0, 0])
    np.testing.assert_array_equal(track, [0, 3, 4])
    np.testing.assert_allclose(purity, [0.75, 1.0, 0.75])
    gen["charge"][0] = 0
    assert len(track_truth(prop, links, 0, ids, relations, gen)[0]) == 0


def test_target_jet_constituents_close_with_leading_unlabelled_inputs():
    tracks = np.zeros((3, 14), dtype=np.float32)
    clusters = np.zeros((2, 14), dtype=np.float32)
    tracks[1, [0, 2, 5, 6]] = [211, 20, 1, 20]
    clusters[1, [0, 2, 5, 6]] = [22, 15, -1, 15]
    jets = _assign_target_jets(tracks, clusters)
    target = np.concatenate([tracks, clusters])
    assert len(jets) == 2
    assert np.all(target[target[:, 0] == 0, 12] == -1)
    for index, jet in enumerate(jets):
        rows = target[(target[:, 0] != 0) & (target[:, 12] == index)]
        assert len(rows) == 1
        assert np.sum(rows[:, 6]) == pytest.approx(float(jet.energy))
        assert np.sum(rows[:, 2] * rows[:, 5]) == pytest.approx(float(jet.pt * np.cos(jet.phi)))


def test_cluster_direction_diagnostic_uses_original_source_charge():
    from scripts.studies.qualify_ild import target_direction_metrics

    clusters = np.zeros((2, 19))
    clusters[:, 6:8] = [[np.cos(1.0), np.sin(1.0)], [1.0, 0.0]]
    targets = np.zeros((2, 14))
    targets[:, [0, 2, 5, 6, 13]] = [[130, 5, 1, 5, 1], [22, 5, 1, 5, 2]]
    hits = targets.copy()
    hits[0, 1] = 1  # Both cluster targets have q=0; the hit keeps the original charge.
    data = ak.Record(
        {
            "X_track": ak.Array([np.zeros((0, 16))]),
            "ytarget_track": ak.Array([np.zeros((0, 14))]),
            "X_cluster": ak.Array([clusters]),
            "ytarget_cluster": ak.Array([targets]),
            "ytarget_hit_tracker": ak.Array([np.zeros((0, 14))]),
            "ytarget_hit_calo": ak.Array([hits]),
        }
    )
    metrics = target_direction_metrics(data)
    assert metrics["cluster_charged_source"]["fraction_aligned"] == 0.0
    assert metrics["cluster_neutral_source"]["fraction_aligned"] == 1.0
    assert metrics["cluster_unknown_source"]["count"] == 0


@pytest.mark.parametrize("locations,ndof", [((2, 2), 10), ((1, 1), 10), ((2, 1), 0)])
def test_ild_rejects_invalid_fits_instead_of_fabricating_ip_states(locations, ndof):
    with pytest.raises(ValueError, match="ILD track"):
        track_to_features(fitted_track(locations, ndof), 0, 2.0, "MarlinTrkTracks", False)


def write_event(path, energy=10.0):
    x = np.zeros((1, 19), dtype=np.float32)
    x[0, [0, 1, 4, 5]] = [1, energy, 1, energy]
    y = np.zeros((1, 14), dtype=np.float32)
    y[0, [0, 1, 2, 5, 6, 12, 13]] = [211, 1, energy, 1, energy, -1, 1]
    zero = np.zeros((0, 14), dtype=np.float32)
    data = {
        "X_track": [x],
        "X_cluster": [np.zeros((0, 19))],
        "ytarget_track": [y],
        "ytarget_cluster": [zero],
        "ycand_track": [y],
        "ycand_cluster": [zero],
        "genmet": [energy],
        "genjet": [np.zeros((0, 4))],
        "targetjet": [np.zeros((0, 4))],
    }
    data.update({field: [value] for field, value in PROVENANCE.items()})
    data["metadata_input_sha256"] = [hashlib.sha256(path.name.encode()).hexdigest()]
    ak.to_parquet(ak.Record({field: ak.Array(value) for field, value in data.items()}), path)


def test_tfds_requires_provenance_and_independent_files(tmp_path):
    first, second = tmp_path / "seed1.parquet", tmp_path / "seed2.parquet"
    write_event(first)
    with pytest.raises(ValueError, match="two independent"):
        split_sample(tmp_path, SimpleNamespace(name="1"))
    write_event(second, 20.0)
    splits = split_sample(tmp_path, SimpleNamespace(name="1"))
    train, test = list(splits["train"]), list(splits["test"])
    assert train[0][0] != test[0][0]
    assert train[0][1]["ytarget"][0, 2] == 10.0
    assert test[0][1]["ytarget"][0, 2] == 20.0
    shutil.copy(first, tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError, match="duplicate ROOT jobs"):
        split_sample(tmp_path, SimpleNamespace(name="1"))
    data = ak.from_parquet(first)
    data = ak.Record(
        {
            **{field: data[field] for field in data.fields},
            "metadata_track_source": ak.Array(["truth_seeded"]),
        }
    )
    ak.to_parquet(data, first)
    with pytest.raises(ValueError, match="metadata_track_source"):
        list(generate_examples([first]))


def test_tfds_array_record_roundtrip(tmp_path):
    import tensorflow_datasets as tfds
    from mlpf.heptfds.ild_pf_edm4hep.qq import IldEdmQqPf

    manual = tmp_path / "manual"
    process = manual / "p8_ee_qq_ecm365"
    process.mkdir(parents=True)
    write_event(process / "seed1.parquet")
    write_event(process / "seed2.parquet", 20.0)
    builder = IldEdmQqPf(config="1", data_dir=tmp_path / "tfds")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual))
    assert builder.info.splits["train"].num_examples == 1
    assert builder.info.splits["test"].num_examples == 1
    example = builder.as_data_source(split="test")[0]
    assert example["X"].shape == (1, 19)
    assert example["ytarget"].shape == (1, 14)
    assert example["ytarget"][0, 2] == 20.0


def test_real_spec_and_detector_contract():
    root = Path(__file__).resolve().parents[1]
    config = MLPFConfig.from_spec(root / "particleflow_spec.yaml", "pyg-ild-v1", "ild")
    assert config.dataset is Dataset.ILD
    assert config.input_dim == 19 and config.num_classes == 6
    detector = EDM4HEP.DETECTORS["ild"]
    assert detector.b_field == 2.0
    assert detector.tracker_relations["TPCTrackerHits"] == (
        "TPCTrackerHitRelations",
        "TPCCollection",
    )
    assert detector.track_collection == "MarlinTrkTracks"


def test_snakemake_routes_ild_generation_postprocessing_and_builders(tmp_path, monkeypatch):
    from mlpf.snakemake import produce_snakemake

    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "particleflow_spec.yaml").read_text())
    spec["project"]["executor"] = "local"
    spec["project"]["paths"] = {
        "storage_root": str(tmp_path / "storage"),
        "scratch_root": str(tmp_path / "scratch"),
        "tmpdir": str(tmp_path / "tmp"),
        "project_root": str(root),
    }
    (tmp_path / "particleflow_spec.yaml").write_text(yaml.safe_dump(spec))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PF_SITE", raising=False)
    monkeypatch.setattr("sys.argv", ["produce_snakemake", "--production", "ild", "--allow-partial"])
    produce_snakemake.main()
    jobs = tmp_path / "snakemake_jobs/ild"
    assert "gen/ild/run_sim.sh p8_ee_qq_ecm365" in (jobs / "gen/gen_qq.sh").read_text()
    assert "--detector ild" in (jobs / "post/post_qq.sh").read_text()
    assert "ild_pf_edm4hep/qq" in (jobs / "tfds/tfds_qq.sh").read_text()
    assert "gun_" not in " ".join(path.name for path in (jobs / "tfds").glob("*.sh"))
    assert "export LD_PRELOAD=/usr/lib64/libffi.so.8" in (jobs / "gen/gen_qq.sh").read_text()
    assert "export KEY4HEP_RELEASE=2026-04-08" in (jobs / "gen/gen_qq.sh").read_text()
    assert set(spec["productions"]["ild"]["samples"]) == {"ttbar", "qq", "ww_fullhad", "z_qq", "z_tautau"}
    for sample in spec["productions"]["ild"]["samples"].values():
        begin, end = sample["seed_range"]
        assert (end - begin) * sample["events_per_job"] == 50000
    graph = (jobs / "Snakefile").read_text()
    assert "glob_wildcards" not in graph  # ILD forbids partial datasets, including with --allow-partial.
    assert all(f"rule {stage}" in graph for stage in ("gen_task", "post_task", "tfds_task"))

    # An existing generation marker must not shrink the requested dataset.
    pytest.importorskip("snakemake")
    import os
    import subprocess
    import sys
    import re

    (jobs / "gen/gen_qq_500000.done").touch()
    target = (jobs / "tfds/tfds_qq_tfds_1.done").relative_to(tmp_path)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-s",
            str(jobs / "Snakefile"),
            "--cores",
            "1",
            "--executor",
            "local",
            "--dry-run",
            str(target),
            "--quiet",
            "rules",
        ],
        cwd=tmp_path,
        env=os.environ | {"XDG_CACHE_HOME": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
        check=True,
    )
    output = run.stdout + run.stderr
    for rule, count in (("gen_task", 4999), ("post_task", 5000), ("tfds_task", 1)):
        assert re.search(rf"^\s*{rule}\s+{count}\s*$", output, re.MULTILINE), output[-5000:]
