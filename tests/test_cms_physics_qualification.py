"""Regression checks for production decoding and physics-closure selections."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_rescales_momenta_and_removes_absent_particles():
    pytest.importorskip("torch")
    module = load_script("analyze-cms-physics-closure")
    features = np.zeros((2, 55), dtype=np.float32)
    features[:, 1] = [4, 10]
    features[:, 5] = [8, 20]
    raw = {
        "bid": np.array([[0, 1], [1, 0]], dtype=np.float32),
        "id": np.eye(9, dtype=np.float32)[[1, 5]],
        "momentum": np.array([[np.log(2), 0, 0, 1, np.log(3)], [0, 0, 0, 1, 0]], dtype=np.float32),
        "pu": np.zeros((2, 2), dtype=np.float32),
    }
    particles = module.decode(raw, features)
    np.testing.assert_array_equal(particles["cls_id"], [1, 0])
    np.testing.assert_allclose(particles["pt"], [8, 0])
    np.testing.assert_allclose(particles["energy"], [24, 0])
    np.testing.assert_allclose(particles["ispu"], [0.5, 0.5])
    # Decoding must not mutate the retained numerical reference.
    np.testing.assert_allclose(raw["momentum"][0, 0], np.log(2))


def test_jet_matching_is_one_to_one_and_order_independent():
    pytest.importorskip("awkward")
    module = load_script("analyze-cms-physics-closure")
    reference = {
        "pt": np.array([100.0, 50.0]),
        "energy": np.array([120.0, 70.0]),
        "eta": np.array([0.0, 1.0]),
        "phi": np.array([0.0, 2.0]),
        "mass": np.array([5.0, 2.0]),
    }
    candidate = {k: v[::-1].copy() for k, v in reference.items()}
    result = module.jet_residuals(reference, candidate)
    np.testing.assert_array_equal(result["pt_response"], [1.0, 1.0])
    np.testing.assert_array_equal(result["delta_r"], [0.0, 0.0])
    assert result["unmatched_reference"] == result["unmatched_candidate"] == 0


def test_clustering_uses_particle_selection_and_pt_threshold():
    pytest.importorskip("fastjet")
    module = load_script("analyze-cms-physics-closure")
    particles = {
        "cls_id": np.array([1, 0, 1]),
        "pt": np.array([10.0, 1000.0, 2.0]),
        "energy": np.array([10.0, 1000.0, 2.0]),
        "eta": np.array([0.0, 2.0, -2.0]),
        "phi": np.zeros(3),
        "ispu": np.array([0.9, 0.0, 0.0]),
    }
    np.testing.assert_allclose(module.cluster(particles)["pt"], [10.0])
    assert len(module.cluster(particles, no_pu=True)["pt"]) == 0


def test_reference_padding_is_observed_but_export_padding_is_enforced(tmp_path):
    module = load_script("qualify-cms-physics")
    outputs = [np.ones((1, 4, width), dtype=np.float32) for width in [2, 9, 5, 2]]
    case = {"file": "event.npz", "length": 2, "sha256": "input"}
    feeds = {"mask": np.array([[1, 1, 0, 0]], dtype=np.float32)}
    with pytest.raises(RuntimeError, match="padding_zero=False"):
        module.retain_outputs(tmp_path / "export", case, outputs, feeds)
    result = module.retain_outputs(tmp_path / "reference", case, outputs, feeds, enforce_padding=False)
    assert result["finite"] and not result["padding_zero"]
    with np.load(tmp_path / "reference/event.npz") as data:
        assert data["momentum"].shape == (2, 5)


def test_columnar_serialization_preserves_variable_event_lengths(tmp_path):
    ak = pytest.importorskip("awkward")
    module = load_script("analyze-cms-physics-closure")
    rows = [
        {"index": 4, "particles": {"pt": np.array([1.0, 2.0])}, "genmet": np.array(3.0)},
        {"index": 8, "particles": {"pt": np.array([], dtype=np.float64)}, "genmet": np.array(5.0)},
    ]
    path = tmp_path / "events.parquet"
    ak.to_parquet(module.columnar_events(rows), path)
    result = ak.from_parquet(path)
    assert ak.to_list(result.particles.pt) == [[1.0, 2.0], []]
    assert ak.to_list(result.index) == [4, 8]


def test_heptv2_runtime_retains_qualified_integer_and_compiler_settings():
    module = load_script("qualify-cms-physics")
    settings = module.migraphx_environment("heptv2", "fp32", "on")
    assert settings["MIGRAPHX_DISABLE_PASSES"] == "eliminate_data_type"
    assert settings["MIGRAPHX_GPU_HIP_FLAGS"] == "-mllvm -enable-gvn-memdep=false"
    assert settings["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == "!attention"
    assert module.migraphx_environment("attention", "decomposed-fp16", "on")["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == "attention"
    assert module.migraphx_environment("attention", "decomposed-fp16", "off")["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == "!attention"


@pytest.mark.parametrize("layout", ["inputs/manifest.json", "input-manifest.json"])
def test_prepared_inputs_feed_reference_selection_and_bf16_audit(tmp_path, monkeypatch, layout):
    """Preparation output is consumed directly, including by the BF16 reports."""
    import json
    import sys
    from types import SimpleNamespace
    from mlpf.model import PFDataset

    qualification = load_script("qualify-cms-physics")
    directory = tmp_path / "data/cms_pf_ttbar/1/2.8.0"
    directory.mkdir(parents=True)
    (directory / "dataset_info.json").write_text("{}")
    raw = SimpleNamespace(data_source=SimpleNamespace(__getitems__=lambda indices: [bytes(indices)]))

    class Raw:
        data_source = raw.data_source

        def __len__(self):
            return 2

    builder = SimpleNamespace(as_data_source=lambda **kwargs: Raw())
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", SimpleNamespace(builder_from_directory=lambda path: builder))
    monkeypatch.setattr(qualification, "config_for", lambda *args: SimpleNamespace(input_dim=55, sort_data=False))

    events = []
    for index in range(2):
        x = np.zeros((2, 55), np.float32)
        x[0, :2] = [1, index + 1]
        events.append(
            dict(
                X=x,
                ytarget=np.zeros((2, 14), np.float32),
                ycand=np.zeros((2, 14), np.float32),
                genjets=np.zeros((0, 4), np.float32),
                genmet=np.float32(0),
                ytarget_pt_orig=np.array([2, 0], np.float32),
                ytarget_e_orig=np.array([3, 0], np.float32),
            )
        )
    monkeypatch.setattr(PFDataset, "TFDSDataSource", lambda *args, **kwargs: events)
    args = SimpleNamespace(root=tmp_path, data_dir=tmp_path / "data", seed=1, events=2, bucket_width=100, case=None, limit=0, bucket=None)
    qualification.prepare(args)
    prepared = tmp_path / "inputs/manifest.json"
    if layout != "inputs/manifest.json":
        prepared.rename(tmp_path / layout)
    cases = qualification.selected_cases(args)
    assert len(cases) == 2
    # The throughput consumer reaches its sample-size guard without a GPU.
    with pytest.raises(ValueError, match="Need 64 distinct events"):
        load_script("benchmark-cms-throughput").worker(SimpleNamespace(root=tmp_path, provider="cuda", engine="native", slots=100))
    for case in cases:
        feeds = qualification.feeds_for(tmp_path, case)
        assert feeds["mask"].sum() == 1
        with np.load(tmp_path / "inputs" / case["file"]) as event:
            assert event["target"][0, 2] == 2 and event["target"][0, 6] == 3
        for model in ["attention", "heptv2"]:
            for relative in [f"reference/{model}", f"results/{model}/cuda/attention24-bf16/on", f"results/{model}/cuda/attention24-bf16/off"]:
                folder = tmp_path / relative
                folder.mkdir(parents=True, exist_ok=True)
                np.savez(folder / case["file"], bid=np.array([[0, 1]], np.float32), id=np.array([[1, 0]], np.float32))
    load_script("audit-cms-bf16-rounding").audit(tmp_path, "cuda")
    audit = json.loads((tmp_path / "bf16-rounding-audit-cuda.json").read_text())
    assert len(audit["rows"]) == 6
    assert all(row["events"] == 2 and row["heads"]["id"]["non_bf16_values"] == 0 for row in audit["rows"])


def test_input_manifest_prefers_prepared_data_without_hiding_corruption(tmp_path):
    import json
    from mlpf.standalone_eval.cms_evidence import input_manifest_path

    (tmp_path / "inputs").mkdir()
    canonical = tmp_path / "inputs/manifest.json"
    canonical.write_text("invalid JSON")
    (tmp_path / "input-manifest.json").write_text('{"cases": []}')
    with pytest.raises(json.JSONDecodeError):
        json.loads(input_manifest_path(tmp_path).read_text())
