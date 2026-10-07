from pathlib import Path
from types import SimpleNamespace

import numpy as np
import awkward as ak
import pytest

from mlpf.conf import Dataset, MLPFConfig, dataset_source_id
from mlpf.heptfds.maia_pf_edm4hep.ttbar import MaiaEdmTtbarPf
from scripts.build_colliderml_comparison_tfds import write_smoke_shards


def test_maia_spec_and_pipeline_use_native_maia_dataset():
    spec = Path(__file__).resolve().parents[1] / "particleflow_spec.yaml"
    config = MLPFConfig.from_spec(spec, "pyg-maia-v1", "maia", SimpleNamespace(pipeline=True))
    assert config.dataset is Dataset.MAIA
    assert config.input_dim == 17 and config.num_classes == 6
    assert config.elemtypes_nonzero == [1, 2]
    assert config.model.attention.num_convs == 1
    assert config.train_dataset["maia"]["physical"].samples["maia_edm_ttbar_pf"].splits == ["10"]
    assert config.valid_dataset["maia"]["physical"].samples["maia_edm_ttbar_pf"].splits == ["10"]
    assert config.test_dataset["maia_edm_ttbar_pf"].splits == ["10"]
    assert dataset_source_id("maia_edm_ttbar_pf") != 0


def test_maia_builder_serializes_common_tensor_contract(tmp_path):
    builder = MaiaEdmTtbarPf(config="10", data_dir=str(tmp_path))
    features = builder.info.features
    event = {
        "X": np.ones((2, 17), dtype=np.float32),
        "ytarget": np.ones((2, 14), dtype=np.float32),
        "ycand": np.ones((2, 14), dtype=np.float32),
        "genmet": np.float32(2),
        "genjets": np.ones((1, 4), dtype=np.float32),
        "targetjets": np.ones((1, 4), dtype=np.float32),
    }
    restored = features.deserialize_example_np(features.serialize_example(event))
    for key, value in event.items():
        np.testing.assert_array_equal(value, restored[key])
    assert builder.name == "maia_edm_ttbar_pf"


@pytest.mark.parametrize("record_layout", [False, True])
def test_smoke_shards_split_events_not_outer_parquet_rows(tmp_path, record_layout):
    columns = {"X_track": [[[i, 1.0]] for i in range(5)], "event_id": list(range(5))}
    data = ak.Record(columns) if record_layout else ak.Array(columns)
    source = tmp_path / "source.parquet"
    manual = tmp_path / "manual"
    ak.to_parquet(data, source)
    write_smoke_shards(source, manual)
    shards = [ak.from_parquet(path)["event_id"].to_list() for path in sorted(manual.glob("*.parquet"))]
    assert shards == [[0, 1], [2, 3, 4]]


def test_real_shards_use_disjoint_ninety_ten_split_and_omit_hits(tmp_path):
    source = tmp_path / "source.parquet"
    manual = tmp_path / "manual"
    ak.to_parquet(ak.Array({"event_id": list(range(10)), "X_track": [[[1.0]]] * 10, "X_hit_tracker": [[[2.0]]] * 10}), source)
    write_smoke_shards(source, manual, train_fraction=0.9)
    train, test = [ak.from_parquet(path) for path in sorted(manual.glob("*.parquet"))]
    assert train.event_id.to_list() == list(range(9))
    assert test.event_id.to_list() == [9]
    assert "X_hit_tracker" not in train.fields
