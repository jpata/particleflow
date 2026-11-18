"""Normalized ONNX predictions retain dynamic dataset routing and raw correction baselines."""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.conf import MLPFConfig, SOURCE_IDS
from mlpf.model.mlpf import MLPF
from mlpf.model.onnx_attention import export_mlpf


VERSIONS = {"cld": "3.3.0", "clic": "3.3.0", "colliderml": "1.5.0", "idea": "0.1.0"}


def test_particle_decoding_does_not_require_pandas():
    # A fresh interpreter prevents other tests' imports from hiding this dependency.
    code = """
import sys
import torch
from types import SimpleNamespace
sys.modules['pandas'] = None
from mlpf.conf import OutputMode
from mlpf.model.mlpf import MLPF
raw = (torch.tensor([[[0., 1.]]]), torch.tensor([[[0., 1., 0., 0., 0., 0.]]]),
       torch.tensor([[[0., 0., 0., 1., 0.]]]), torch.zeros(1, 1, 2))
proxy = SimpleNamespace(output_mode=OutputMode.ELEMENTWISE, forward=lambda *args, **kwargs: raw)
features = torch.tensor([[[1., 3., 0., 0., 1., 4.]]])
particles = MLPF.predict_particles(proxy, features, torch.ones(1, 1, dtype=torch.bool))
assert particles['pt'].item() == 3.
assert particles['energy'].item() == 4.
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


def make_model(backbone="shared"):
    torch.manual_seed(569)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {
                "type": "attention",
                "input_normalization": True,
                "backbone": {"mode": backbone},
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"},
            },
            "test_dataset": {f"{name}_edm_qq_pf": {"version": version, "splits": ["1"]} for name, version in VERSIONS.items()},
        }
    )
    return MLPF(config).eval()


def make_batch(lengths, size):
    gen = torch.Generator().manual_seed(31)
    x = torch.rand(len(lengths), size, 19, generator=gen) * 10
    x[..., 0] = torch.arange(size) % 2 + 1
    x[..., 2] = torch.linspace(-1, 1, size)
    x[..., 3] = x[..., 2].sin()
    x[..., 4] = x[..., 2].cos()
    # Include negative unlogged columns and outliers subject to clipping.
    tracks = x[..., 0] == 1
    x[..., 6][tracks] = -100
    x[..., 12][tracks] = 1e5
    mask = torch.arange(size)[None, :] < torch.tensor(lengths)[:, None]
    x[~mask] = 0
    return x, mask.float()


@pytest.mark.parametrize("backbone", ["shared", "split"])
@pytest.mark.parametrize("attention_format", ["decomposed", "attention24"])
@pytest.mark.parametrize("attention_dtype", [None, torch.float16])
def test_normalized_export_routes_dynamic_sources_and_preserves_predictions(tmp_path, backbone, attention_format, attention_dtype):
    torch.set_num_threads(1)
    model = make_model(backbone)
    x, mask = make_batch([7, 3], 9)
    path = tmp_path / "normalized.onnx"
    # Only CLD occurs at export: other sources must still work at runtime.
    source = torch.full((2,), SOURCE_IDS["cld"], dtype=torch.int64)
    export_mlpf(model, x, mask, path, source_id=source, attention_format=attention_format, attention_dtype=attention_dtype)
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    assert [v.name for v in graph.graph.input] == ["Xfeat_normed", "mask", "source_id"]
    assert graph.graph.input[2].type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert graph.graph.input[2].type.tensor_type.shape.dim[0].dim_param == "num_batch"
    metadata = {p.key: p.value for p in graph.metadata_props}
    assert json.loads(metadata["input_normalization"]) == {str(SOURCE_IDS[k]): {"dataset": k, "version": v} for k, v in VERSIONS.items()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for lengths, size, names in [([5], 5, ["idea"]), ([11, 6, 3, 1], 15, list(VERSIONS))]:
        x, mask = make_batch(lengths, size)
        original = x.clone()
        source = torch.tensor([SOURCE_IDS[name] for name in names])
        feeds = {"Xfeat_normed": x.numpy(), "mask": mask.numpy(), "source_id": source.numpy()}
        actual = session.run(None, feeds)
        with torch.no_grad():
            expected = model(x, mask, source_id=source)
        valid = mask.bool().numpy()
        for a, b in zip(actual, expected, strict=True):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(a[valid], b.numpy()[valid], atol=0.003, rtol=0.003)
            assert np.all(a[~valid] == 0)
        assert torch.equal(x, original)
        for i, length in enumerate(lengths):
            single = session.run(
                None,
                {"Xfeat_normed": x[i : i + 1, :length].numpy(), "mask": mask[i : i + 1, :length].numpy(), "source_id": source[i : i + 1].numpy()},
            )
            for a, b in zip(actual, single, strict=True):
                np.testing.assert_allclose(a[i, :length], b[0], atol=0.003, rtol=0.003)
        poisoned = x.clone()
        poisoned[~mask.bool()] = -100
        second = session.run(None, {**feeds, "Xfeat_normed": poisoned.numpy()})
        for a, b in zip(actual, second, strict=True):
            np.testing.assert_array_equal(a, b)
        # Check physical predictions too: decode ONNX heads through the real
        # predict_particles path, whose pT/E baselines must remain raw features.
        with torch.no_grad():
            physical = model.predict_particles(x, mask, source_id=source)
        proxy = SimpleNamespace(output_mode=model.output_mode, forward=lambda *args, **kwargs: tuple(torch.from_numpy(a.copy()) for a in actual))
        decoded = MLPF.predict_particles(proxy, x, mask, source_id=source)
        for feature in ["pt", "energy", "eta", "sin_phi", "cos_phi"]:
            torch.testing.assert_close(decoded[feature][mask.bool()], physical[feature][mask.bool()], atol=0.03, rtol=0.01)


@pytest.mark.parametrize("source", [None, torch.tensor([3.0, 3.0]), torch.tensor([3]), torch.tensor([[3], [3]]), torch.tensor([-1, 999])])
def test_normalized_export_rejects_invalid_sources(tmp_path, source):
    model = make_model()
    x, mask = make_batch([7, 3], 9)
    with pytest.raises(ValueError, match="source_id"):
        export_mlpf(model, x, mask, tmp_path / "invalid.onnx", source_id=source)


def test_float_output_wrapper_forwards_sources():
    from mlpf.model.onnx_attention import _FloatOutputs

    model = make_model()
    x, mask = make_batch([7, 3], 9)
    source = torch.tensor([SOURCE_IDS["cld"], SOURCE_IDS["idea"]])
    with torch.no_grad():
        expected = model(x, mask, source_id=source)
        actual = _FloatOutputs(model)(x, mask, source)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_feature_bag_export_routes_unseen_sources(tmp_path):
    """Feature gather tables must use the runtime source, not the export example."""
    torch.set_num_threads(1)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {
                "type": "attention",
                "input_encoding": "split",
                "input_normalization": True,
                "feature_bag": True,
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"},
            },
            "test_dataset": {f"{name}_edm_qq_pf": {"version": version, "splits": ["1"]} for name, version in VERSIONS.items()},
        }
    )
    model = MLPF(config).eval()
    x, mask = make_batch([7, 3], 9)
    path = tmp_path / "feature_bag.onnx"
    export_mlpf(model, x, mask, path, source_id=torch.full((2,), SOURCE_IDS["cld"], dtype=torch.int64))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_inputs()] == ["Xfeat_normed", "mask", "source_id"]
    for lengths, size, names in [([5], 5, ["clic"]), ([11, 6, 3, 1], 15, list(VERSIONS))]:
        x, mask = make_batch(lengths, size)
        source = torch.tensor([SOURCE_IDS[name] for name in names], dtype=torch.int64)
        actual = session.run(None, {"Xfeat_normed": x.numpy(), "mask": mask.numpy(), "source_id": source.numpy()})
        with torch.no_grad():
            expected = model(x, mask, source_id=source)
        valid = mask.bool().numpy()
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value[valid], reference.numpy()[valid], atol=3e-3, rtol=3e-3)


def test_normalized_export_keeps_known_raw_correction_baselines(tmp_path):
    model = make_model()

    def constant(net, values):
        for p in net.parameters():
            p.data.zero_()
        linear = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
        linear[-1].bias.data.copy_(torch.tensor(values))

    for head in model.nn_pt.nn:
        constant(head, [math.log(2)])
    for head in model.nn_energy.nn:
        constant(head, [math.log(1.25)])
    constant(model.nn_eta.nn, [1.0, 0.1])
    constant(model.nn_sin_phi.nn, [1.0, 0.0])
    constant(model.nn_cos_phi.nn, [1.0, 0.0])
    x, mask = make_batch([7], 7)
    source = torch.tensor([SOURCE_IDS["cld"]])
    path = tmp_path / "corrections.onnx"
    export_mlpf(model, x, mask, path, source_id=source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    momentum = session.run(["momentum"], {"Xfeat_normed": x.numpy(), "mask": mask.numpy(), "source_id": source.numpy()})[0]
    np.testing.assert_allclose(np.exp(momentum[..., 0]) * x[..., 1].numpy(), 2 * x[..., 1].numpy(), rtol=1e-5)
    eta = x[..., 2].numpy() + 0.1
    np.testing.assert_allclose(momentum[..., 1], eta, atol=1e-6)
    np.testing.assert_allclose(momentum[..., 2:4], x[..., 3:5].numpy(), atol=1e-6)
    energy = 1.25 * 2 * x[..., 1].numpy() * np.cosh(eta)
    np.testing.assert_allclose(np.exp(momentum[..., 4]) * x[..., 5].numpy(), energy, rtol=1e-5)


def test_runtime_replay_validates_normalization_sources(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate-onnx-runtime.py"
    spec = importlib.util.spec_from_file_location("validate_runtime", script)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    model = make_model()
    x, mask = make_batch([7, 3], 9)
    source = torch.tensor([SOURCE_IDS["cld"], SOURCE_IDS["idea"]])
    path = tmp_path / "replay.onnx"
    export_mlpf(model, x, mask, path, source_id=source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = tmp_path / "inputs.npz"
    for values in [source.numpy(), np.array([999, 999]), np.array([SOURCE_IDS["cld"]]), source.numpy().astype(np.float32)]:
        np.savez(inputs, Xfeat_normed=x.numpy(), mask=mask.numpy(), source_id=values)
        if np.array_equal(values, source.numpy()) and values.dtype == np.int64:
            feeds, valid = replay.load_inputs(inputs, session)
            assert valid.shape == mask.shape
            assert all(np.isfinite(a).all() for a in session.run(None, feeds))
        else:
            with pytest.raises(ValueError, match="source_id"):
                replay.load_inputs(inputs, session)
