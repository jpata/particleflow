"""
Spec: Validates the input normalization (mlpf/model/input_normalization.py), which
standardizes each track/cluster input column with the statistics of its dataset version,
after clipping to CLIP_BOUNDS and log1p(x / scale) for LOG1P_SCALES, and its statistics script
(scripts/compute_input_normalization_stats.py). Key tests: the log1p quantities are input
column names, committed statistics match the current column names and transforms while
stale ones are refused, a column is standardized with its own dataset's statistics in
log1p or linear space and clipped, constant columns are zeroed while the element type and padding rows
are kept, the configuration takes one version with statistics per track/cluster dataset, a
normalized MLPF runs on a batch mixing datasets, and the shards of the script reduce to the
statistics of all train events, with the clipped and non-finite values counted.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from mlpf import conf
from mlpf.conf import ELEMENT_TYPES, LOG1P_SCALES, SOURCE_IDS, MLPFConfig, column_names, column_transform, load_stats, stats_path
from mlpf.model.input_normalization import InputNormalization
from mlpf.model.mlpf import MLPF
from scripts import compute_input_normalization_stats as stats_script

VERSIONS = {"cld": "3.2.1", "clic": "3.2.1", "colliderml": "1.1.0", "idea": "0.1.0"}
TRACK, CLUSTER = 1, 2


@pytest.fixture
def synthetic_stats(tmp_path, monkeypatch):
    """Statistics of VERSIONS with mean = column + source id and std = 2, except a constant track column 15."""
    monkeypatch.setattr(conf, "STATS_DIR", tmp_path)
    for dataset, version in VERSIONS.items():
        stats = {}
        for elemtype, name in ELEMENT_TYPES.items():
            names = column_names(dataset, elemtype)
            stats[name] = [
                {
                    "column": column,
                    "name": names[column],
                    **column_transform(names[column]),
                    "mean": column + SOURCE_IDS[dataset],
                    "std": 0.0 if (elemtype, column) == (TRACK, 15) else 2.0,
                }
                for column in range(1, 17)
            ]
        stats_path(dataset, version).write_text(yaml.safe_dump(stats))


def test_log1p_scales_are_given_for_input_column_names():
    assert set(LOG1P_SCALES) <= {name for dataset in conf.X_FEATURES for elemtype in ELEMENT_TYPES for name in column_names(dataset, elemtype)}


@pytest.mark.parametrize("path", sorted(conf.STATS_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_committed_statistics_match_the_column_names_and_transforms(path):
    stats = load_stats(*path.stem.rsplit("_", 1))
    for name in ELEMENT_TYPES.values():
        assert [column["column"] for column in stats[name]] == list(range(1, len(stats[name]) + 1))


def test_stale_statistics_are_refused(synthetic_stats):
    path = stats_path("cld", "3.2.1")
    stats = yaml.safe_load(path.read_text())
    stats["cluster"][4]["log1p_scale"] = 1.0
    path.write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="cluster column 5 as .*'log1p_scale': 1.0.*, not .*'log1p_scale': 0.01"):
        load_stats("cld", "3.2.1")

    stats["cluster"][4]["log1p_scale"] = 0.01
    stats["track"][10]["clip"] = None
    path.write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="track column 11 as .*'clip': None.*, not .*'clip': \\[-20.0, 20.0\\]"):
        load_stats("cld", "3.2.1")


def test_statistics_that_are_not_finite_are_refused(synthetic_stats):
    path = stats_path("cld", "3.2.1")
    stats = yaml.safe_load(path.read_text())
    stats["track"][5]["mean"] = float("nan")
    path.write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="track column 6 mean or std that is not finite"):
        load_stats("cld", "3.2.1")


@pytest.mark.parametrize("dataset", VERSIONS)
def test_columns_are_standardized_with_their_dataset_statistics(synthetic_stats, dataset):
    # Cluster column 2 (eta) is linear and column 5 (energy) log1p(x / 0.01) in every dataset.
    shift = SOURCE_IDS[dataset]
    X = torch.zeros(1, 1, 17)
    X[..., 0] = CLUSTER
    X[..., 2] = 2 + shift + 4.0
    X[..., 5] = 0.01 * math.expm1(5 + shift - 2.0)
    out = InputNormalization(17, VERSIONS)(X, torch.tensor([shift]))[0, 0]
    assert out[2].item() == pytest.approx(2.0, abs=1e-4)
    assert out[5].item() == pytest.approx(-1.0, abs=1e-4)


def test_columns_are_clipped_before_standardization(synthetic_stats):
    # CLD track column 11 (tanLambda) is clipped to [-20, 20], with mean 11 + 3 and std 2.
    X = torch.zeros(1, 2, 17)
    X[..., 0] = TRACK
    X[0, :, 11] = torch.tensor([1.0e8, -1.0e8])
    out = InputNormalization(17, VERSIONS)(X, torch.tensor([SOURCE_IDS["cld"]]))[0, :, 11]
    assert out.tolist() == pytest.approx([(20.0 - 14.0) / 2.0, (-20.0 - 14.0) / 2.0])


def test_constant_columns_are_zeroed_and_element_type_and_padding_kept(synthetic_stats):
    X = torch.rand(2, 4, 17) * 10.0
    X[..., 0] = torch.tensor([[1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0]])
    X[X[..., 0] == 0] = 0.0
    out = InputNormalization(17, VERSIONS)(X, torch.tensor([SOURCE_IDS["cld"], SOURCE_IDS["idea"]]))
    assert torch.equal(out[..., 0], X[..., 0])
    assert torch.equal(out[X[..., 0] == 0], X[X[..., 0] == 0])
    assert torch.all(out[X[..., 0] == TRACK][:, 15] == 0)


def make_config(samples):
    return MLPFConfig.model_validate(
        {
            "dataset": "cld",
            "data_dir": "/tmp",
            "input_dim": 17,
            "model": {
                "type": "attention",
                "input_normalization": True,
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "simple"},
            },
            "conv_type": "attention",
            "test_dataset": {name: {"version": version, "splits": ["1"]} for name, version in samples.items()},
        }
    )


def test_normalized_model_runs_on_a_batch_mixing_datasets(synthetic_stats):
    config = make_config({"cld_edm_ttbar_pf": "3.2.1", "clic_edm_qq_pf": "3.2.1", "idea_edm_qq_pf": "0.1.0"})
    assert config.dataset_versions() == {"cld": "3.2.1", "clic": "3.2.1", "idea": "0.1.0"}
    model = MLPF(config)

    X = torch.rand(3, 16, 17) * 1000.0
    X[..., 0] = torch.randint(1, 3, (3, 16)).float()
    mask = torch.ones(3, 16, dtype=torch.bool)
    source_id = torch.tensor([SOURCE_IDS["cld"], SOURCE_IDS["clic"], SOURCE_IDS["idea"]])
    preds_binary, preds_pid, preds_momentum, _ = model(X, mask, source_id=source_id)

    assert preds_binary.shape == (3, 16, 2)
    assert torch.isfinite(preds_pid).all()
    assert torch.isfinite(preds_momentum).all()


def test_input_normalization_needs_one_version_with_statistics_per_track_cluster_dataset(synthetic_stats):
    with pytest.raises(ValueError, match="cld samples have versions"):
        make_config({"cld_edm_ttbar_pf": "3.2.1", "cld_edm_qq_pf": "3.3.0"})
    with pytest.raises(ValueError, match="No input normalization statistics"):
        make_config({"cld_edm_ttbar_pf": "0.0.0"})
    with pytest.raises(ValueError, match="supports track/cluster samples"):
        make_config({"clic_edm_ttbar_hits": "3.2.1"})


class FakeBuilder:
    """TFDS builder of in-memory train events."""

    def __init__(self, name, events):
        self.events = events
        splits = {"train": SimpleNamespace(num_examples=len(events))}
        self.info = SimpleNamespace(full_name=name, splits=splits, features={"X": SimpleNamespace(shape=(None, 17))})

    def as_data_source(self, split):
        return [{"X": X} for X in self.events]


def random_events(rng):
    """Seven events of random tracks and clusters with positive features."""
    return [np.column_stack([rng.integers(1, 3, n), rng.exponential(10.0, (n, 16))]).astype(np.float32) for n in rng.integers(1, 50, 7)]


def patch_builds(monkeypatch, tmp_path, events):
    monkeypatch.setattr(conf, "STATS_DIR", tmp_path)
    builders = [FakeBuilder("a/1/0.0.0", events[:3]), FakeBuilder("b/1/0.0.0", events[3:])]
    monkeypatch.setattr(stats_script, "open_builds", lambda dataset, version, data_dir, samples: builders)
    monkeypatch.setattr(stats_script, "CHUNK_EVENTS", 2)


def run_step(step, tmp_path, **shard):
    step(SimpleNamespace(dataset="cld", version="0.0.0", data_dir=None, samples=None, work_dir=tmp_path / "work", **shard))


def test_statistics_script_refuses_statistics_that_are_not_finite(tmp_path, monkeypatch):
    events = random_events(np.random.default_rng(0))
    # A NaN track pt makes the mean and std of CLD track column 1 NaN.
    events[0][np.flatnonzero(events[0][:, 0] == TRACK)[0], 1] = np.nan
    patch_builds(monkeypatch, tmp_path, events)
    run_step(stats_script.map_shard, tmp_path, shard=0, num_shards=1)
    with pytest.raises(SystemExit, match="not finite, not written:\ntrack .*'name': 'pt'.*'nonfinite': 1"):
        run_step(stats_script.reduce_shards, tmp_path)
    assert not stats_path("cld", "0.0.0").exists()


def test_statistics_script_reduces_the_shards_to_the_statistics_of_all_train_events(tmp_path, monkeypatch):
    events = random_events(np.random.default_rng(0))
    patch_builds(monkeypatch, tmp_path, events)

    def run(step, **shard):
        run_step(step, tmp_path, **shard)

    # Shard 1 of 3 (events 2 and 3) spans both builds.
    for shard in (0, 2):
        run(stats_script.map_shard, shard=shard, num_shards=3)
    with pytest.raises(SystemExit, match="Missing"):
        run(stats_script.reduce_shards)
    run(stats_script.map_shard, shard=1, num_shards=3)
    run(stats_script.reduce_shards)

    stats = load_stats("cld", "0.0.0")
    assert stats["events"] == len(events)
    X = np.concatenate(events).astype(np.float64)
    for elemtype, name in ELEMENT_TYPES.items():
        for column in stats[name]:
            raw = X[X[:, 0] == elemtype, column["column"]]
            bounds = column["clip"] or (-np.inf, np.inf)
            values = np.clip(raw, *bounds)
            values = np.log1p(values / column["log1p_scale"]) if column["log1p_scale"] else values
            # Sums of squares leave a std of about 1e-9 on columns made constant by clipping, like omega here.
            assert (column["mean"], column["std"]) == pytest.approx((values.mean(), values.std()), abs=1e-6)
            clipped = int(((raw < bounds[0]) | (raw > bounds[1])).sum())
            assert (column["min"], column["max"], column["clipped"], column["nonfinite"]) == (raw.min(), raw.max(), clipped, 0)
    # Random values up to about 100 exceed the tanLambda bounds of CLD track column 11.
    assert stats["track"][10]["clipped"] > 0


def test_statistics_script_reads_the_selected_samples(tmp_path, monkeypatch):
    for sample in ("ttbar_nopu", "ttbar_pu200", "zee_pu200"):
        for config in ("1", "2"):
            (tmp_path / f"colliderml_{sample}_pf" / config / "1.5.0").mkdir(parents=True)
    monkeypatch.setattr(stats_script.tfds, "builder", lambda name, data_dir: name)
    selected = stats_script.open_builds("colliderml", "1.5.0", tmp_path, ["ttbar_nopu", "ttbar_pu200"])
    assert selected == [f"colliderml_{sample}_pf/{config}:1.5.0" for sample in ("ttbar_nopu", "ttbar_pu200") for config in ("1", "2")]
    assert len(stats_script.open_builds("colliderml", "1.5.0", tmp_path)) == 6
    with pytest.raises(SystemExit, match="No colliderml_ggf_nopu_pf samples of version 1.5.0"):
        stats_script.open_builds("colliderml", "1.5.0", tmp_path, ["ttbar_nopu", "ggf_nopu"])


def test_statistics_script_discovers_cms_sample_configs(tmp_path, monkeypatch):
    for sample in ("qcd", "qcd_nopu", "ttbar", "ttbar_nopu", "ztt", "ztt_nopu"):
        (tmp_path / f"cms_pf_{sample}" / "1" / "2.8.0").mkdir(parents=True)
    monkeypatch.setattr(stats_script.tfds, "builder", lambda name, data_dir: name)
    builds = stats_script.open_builds("cms", "2.8.0", tmp_path)
    assert builds == [f"cms_pf_{sample}/1:2.8.0" for sample in ("qcd", "qcd_nopu", "ttbar", "ttbar_nopu", "ztt", "ztt_nopu")]
    assert stats_script.open_builds("cms", "2.8.0", tmp_path, ["ttbar_nopu"]) == ["cms_pf_ttbar_nopu/1:2.8.0"]


def test_cms_statistics_combine_all_calo_element_types(tmp_path, monkeypatch):
    monkeypatch.setattr(conf, "STATS_DIR", tmp_path)
    track = np.zeros((1, len(conf.X_FEATURES["cms"])), dtype=np.float32)
    ps1 = track.copy()
    hfhad = track.copy()
    track[0, :2] = [1, 1]
    ps1[0, :2] = [2, 2]
    hfhad[0, :2] = [9, 8]
    builder = FakeBuilder("cms_pf_qcd/1/2.8.0", [np.concatenate([track, ps1]), hfhad])
    builder.info.features["X"].shape = (None, len(conf.X_FEATURES["cms"]))
    monkeypatch.setattr(stats_script, "open_builds", lambda *args: [builder])
    args = SimpleNamespace(dataset="cms", version="2.8.0", data_dir=None, samples=None, work_dir=tmp_path / "work", shard=0, num_shards=1)
    stats_script.map_shard(args)
    stats_script.reduce_shards(args)
    stats = load_stats("cms", "2.8.0")
    pt = stats["cluster"][0]
    values = np.log1p(np.array([2.0, 8.0]) / 0.01)
    assert (pt["mean"], pt["std"]) == pytest.approx((values.mean(), values.std()))
