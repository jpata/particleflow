"""Track/calo feature mapping, target integrity, split and ArrayRecord contracts."""

import copy
import importlib

import awkward as ak
import numpy as np
import pytest
import tensorflow_datasets as tfds

from mlpf.conf import INPUT_TYPE_IDS, dataset_input_type_id, dataset_source_id
from mlpf.heptfds.track_calo_utils.utils import X_FEATURES, feature_availability, generate_examples, split_files

DETECTORS = {"cld": "CldTrackCaloTtbar", "clic": "ClicTrackCaloTtbar", "maia": "MaiaTrackCaloTtbar", "colliderml": "CollidermlTrackCaloTtbar"}


def example(detector):
    width = 17 if detector == "colliderml" else 16
    track = np.arange(1, width + 1, dtype=np.float32)[None, :]
    hit = np.arange(2, 17, dtype=np.float32)[None, :]
    target = np.zeros((2, 14), dtype=np.float32)
    target[:, 0] = [211, 22]
    target[:, 1] = [1, 0]
    target[:, 2] = [2, 3]
    target[:, 5] = 1
    target[:, 6] = [5, 10]
    target[:, 12] = -1
    target[:, 13] = [1, 2]
    return dict(
        X_track=track,
        X_hit_calo=hit,
        ytarget_track_calo_track=target[:1],
        ytarget_track_calo_hit=target[1:],
        track_calo_target_particles=target,
        track_calo_original_to_target=np.array([0, 1]),
        track_calo_energy_accounting=np.array([15, 0, 0, 15]),
        targetjet_track_calo=np.zeros((0, 4)),
        genmet=2.0,
        genjet=np.zeros((0, 4)),
    )


def write(path, events, record=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ak.Record({name: ak.from_iter([event[name] for event in events]) for name in events[0]}) if record else ak.Array(events)
    ak.to_parquet(data, path)


@pytest.mark.parametrize("detector", DETECTORS)
def test_features_and_targets(tmp_path, detector):
    path = tmp_path / "events.parquet"
    raw = example(detector)
    write(path, [raw], record=detector != "colliderml")
    key, event = next(generate_examples([path], detector))
    assert key == "events.parquet:0"
    x = event["X"]
    assert x.shape == (2, 25)
    np.testing.assert_array_equal(x[1, :15], raw["X_hit_calo"][0])
    assert not x[0, 6:9].any()  # helix parameters must not become xyz
    assert not x[0, 10:15].any()  # no tracker-hit geometry on a fitted track
    if detector == "colliderml":
        assert x[0, X_FEATURES.index("D0")] == 7
        assert x[0, X_FEATURES.index("n_meas")] == 14
        assert x[0, X_FEATURES.index("ndf")] == 0
        assert not feature_availability(detector)["track"][X_FEATURES.index("ndf")]
    else:
        assert x[0, X_FEATURES.index("D0")] == 13
        assert x[0, X_FEATURES.index("ndf")] == 8
        assert x[0, X_FEATURES.index("time")] == 16
        assert not feature_availability(detector)["track"][X_FEATURES.index("n_meas")]
    assert event["ytarget"][:, 0].tolist() == [1, 3]
    assert event["ytarget"][:, 13].tolist() == [1, 2]
    assert not event["ycand"].any()
    assert event["genmet"] == 2
    assert event["targetjets"].shape == (0, 4)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "--track-calo-targets"),
        ("representatives", "representative"),
        ("alignment", "not aligned"),
        ("width", "expected"),
        ("nan", "non-finite"),
        ("jets", "jet indices"),
        ("energy", "accounting"),
        ("class", "PDG"),
        ("membership", "membership"),
    ],
)
def test_rejects_bad_data(tmp_path, problem, match):
    raw = copy.deepcopy(example("cld"))
    if problem == "missing":
        del raw["track_calo_target_particles"]
    elif problem == "representatives":
        raw["ytarget_track_calo_hit"][0, 13] = 1
    elif problem == "alignment":
        raw["X_track"] = np.zeros((0, 16))
    elif problem == "width":
        raw["X_track"] = raw["X_track"][:, :15]
    elif problem == "nan":
        raw["X_track"][0, 2] = np.nan
    elif problem == "jets":
        raw["track_calo_target_particles"][0, 12] = 5
        raw["ytarget_track_calo_track"][0, 12] = 5
    elif problem == "energy":
        raw["track_calo_energy_accounting"][0] = 20
    elif problem == "class":
        raw["track_calo_target_particles"][0, 0] = 999
        raw["ytarget_track_calo_track"][0, 0] = 999
    elif problem == "membership":
        raw["ytarget_track_calo_hit"][0, 13] = 0.5
    path = tmp_path / "bad.parquet"
    write(path, [raw])
    with pytest.raises(ValueError, match=match):
        list(generate_examples([path], "cld"))


def test_empty_event_preserved(tmp_path):
    raw = example("cld")
    for name in ("X_track", "X_hit_calo", "ytarget_track_calo_track", "ytarget_track_calo_hit", "track_calo_target_particles"):
        raw[name] = raw[name][:0]
    raw["track_calo_original_to_target"] = np.array([], dtype=int)
    raw["track_calo_energy_accounting"] = np.zeros(4)
    path = tmp_path / "events.parquet"
    write(path, [raw, example("cld")], record=True)
    result = list(generate_examples([path], "cld"))
    assert [key for key, _ in result] == ["events.parquet:0", "events.parquet:1"]
    assert result[0][1]["X"].shape == (0, 25)


def test_split_files(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.parquet").touch()
    assert split_files(tmp_path, "10") == {"train": [tmp_path / "a.parquet"], "test": [tmp_path / "b.parquet"]}
    with pytest.raises(ValueError, match="config 10"):
        split_files(tmp_path, "1")
    for name in ("train", "test"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "same.parquet").symlink_to(tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="same source"):
        split_files(tmp_path, "10")


@pytest.mark.parametrize("detector", DETECTORS)
def test_array_record_roundtrip(tmp_path, detector):
    module = importlib.import_module(f"mlpf.heptfds.{detector}_track_calo.ttbar")
    builder = getattr(module, DETECTORS[detector])(config="10", data_dir=tmp_path / "tfds")
    manual = tmp_path / "manual"
    write(manual / "train" / "a.parquet", [example(detector)], record=detector != "colliderml")
    write(manual / "test" / "b.parquet", [example(detector)], record=detector != "colliderml")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual, try_download_gcs=False))
    loaded = tfds.builder_from_directory(builder.data_dir)
    assert loaded.info.metadata["input_type"] == "track_calo"
    assert loaded.info.metadata["x_features"] == X_FEATURES
    assert loaded.info.splits["train"].num_examples == 1
    source = loaded.as_data_source(split="test")
    assert source[0]["X"].shape == (2, 25)
    assert source[0]["ytarget"][:, 13].tolist() == [1, 2]
    assert dataset_input_type_id(builder.name) == INPUT_TYPE_IDS["track_calo"]
    assert dataset_source_id(builder.name) != 0
    # Existing set-target extraction, sorting and padding can consume the schema.
    from mlpf.model.PFDataset import TFDSDataSource

    adapted = TFDSDataSource(source, sort=True, pad_to_multiple=8, build_target_set=True)[0]
    assert adapted["X"].shape == (8, 25)
    assert adapted["ytarget_set"].shape == (2, 14)
    assert adapted["input_type_id"] == INPUT_TYPE_IDS["track_calo"]


def test_smoke_builder_checks_real_scalar_roundtrip(tmp_path):
    from scripts.build_track_calo_tfds import build

    path = tmp_path / "input.parquet"
    write(path, [example("cld"), example("cld"), example("cld")], record=True)
    directory = build(path, "cld", tmp_path / "out")
    loaded = tfds.builder_from_directory(directory)
    assert loaded.info.splits["train"].num_examples == 2
    assert loaded.info.splits["test"].num_examples == 1


def test_preserves_background_and_inclusive_membership(tmp_path):
    raw = example("cld")
    raw["X_hit_calo"] = np.repeat(raw["X_hit_calo"], 3, axis=0)
    rows = np.zeros((3, 14), dtype=np.float32)
    rows[0] = raw["ytarget_track_calo_hit"][0]
    rows[1, 13] = 2  # Supporting hit, not a second full target.
    raw["ytarget_track_calo_hit"] = rows
    path = tmp_path / "input.parquet"
    write(path, [raw])
    _, event = next(generate_examples([path], "cld"))
    assert event["ytarget"][:, 0].tolist() == [1, 3, 0, 0]
    assert event["ytarget"][:, 13].tolist() == [1, 2, 2, 0]
