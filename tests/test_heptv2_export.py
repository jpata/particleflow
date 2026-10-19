"""HEPTv2 export must retain numerical behavior at unseen batch/sequence sizes."""

import os

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.conf import MLPFConfig  # noqa: E402
from mlpf.model.mlpf import MLPF  # noqa: E402
from mlpf.model.onnx_attention import export_mlpf  # noqa: E402


def make_heptv2():
    torch.manual_seed(42)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "heptv2",
            "model": {
                "type": "heptv2",
                "heptv2": {
                    "num_convs": 1,
                    "num_heads": 2,
                    "embedding_dim": 16,
                    "width": 16,
                    "block_size": 8,
                    "n_hashes": 2,
                },
            },
        }
    )
    return MLPF(config).eval(), config


def batch(config, lengths, size):
    gen = torch.Generator().manual_seed(17)
    x = torch.randn(len(lengths), size, config.input_dim, generator=gen)
    x[..., 0] = config.elemtypes_nonzero[0]
    x[..., 1] = x[..., 1].exp()
    x[..., 5] = x[..., 5].exp()
    # Real detector elements can have identical coordinates: sorting ties must
    # use the same ordering in PyTorch and ONNX.
    x[:, 1:6, 2:5] = x[:, :1, 2:5]
    mask = torch.arange(size)[None, :] < torch.tensor(lengths)[:, None]
    x[~mask] = 0
    return x, mask.float()


def test_heptv2_dynamic_export(tmp_path):
    torch.set_num_threads(1)
    model, config = make_heptv2()
    x, mask = batch(config, [13, 9], 16)
    path = tmp_path / "heptv2.onnx"
    export_mlpf(model, x, mask, path, attention_format="decomposed")
    onnx.checker.check_model(onnx.load(path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for lengths, size in [([13, 9], 16), ([19], 24), ([25, 18, 9], 32)]:
        x, mask = batch(config, lengths, size)
        with torch.no_grad():
            expected = model(x, mask)
        actual = session.run(None, {"Xfeat_normed": x.numpy(), "mask": mask.numpy()})
        valid = mask.bool().numpy()
        for value, reference in zip(actual, expected, strict=True):
            assert value.shape == tuple(reference.shape)
            np.testing.assert_allclose(value[valid], reference.numpy()[valid], atol=2e-4, rtol=2e-4)
            assert np.isfinite(value).all()
            assert np.all(value[~valid] == 0)
        poisoned = x.clone()
        poisoned[~mask.bool()] = 100
        second = session.run(None, {"Xfeat_normed": poisoned.numpy(), "mask": mask.numpy()})
        for first, value in zip(actual, second, strict=True):
            np.testing.assert_array_equal(first, value)


@pytest.mark.parametrize("batch_size,size", [(1, 24), (2, 16), (3, 32)])
def test_quantile_partition_matches_table_gather(batch_size, size):
    from mlpf.model.heptv2 import quantile_partition

    generator = torch.Generator().manual_seed(7)
    positions = torch.argsort(torch.randn(batch_size, size, generator=generator), dim=-1)
    regions = torch.tensor([[3.0], [7.0], [11.0], [17.0]])
    region_size = torch.ceil(size / regions)
    table = torch.arange(size)[None] // region_size + 1
    inverse = torch.argsort(positions, dim=-1)
    expected = torch.gather(table[:, None, :].expand(-1, batch_size, -1), -1, inverse[None].expand(len(regions), -1, -1))
    torch.testing.assert_close(quantile_partition(positions, regions), expected.to(torch.int64), rtol=0, atol=0)


def test_stable_hash_sort_preserves_input_order_for_ties():
    from mlpf.model.heptv2 import stable_argsort

    values = torch.tensor([[2.0, 1.0, 2.0, 1.0, 0.0]])
    torch.testing.assert_close(stable_argsort(values), torch.tensor([[4, 1, 3, 0, 2]]))


def test_export_materializes_distinct_output_heads(tmp_path):
    """Mixed output widths must not alias or exchange data at the EP boundary."""
    from mlpf.model.onnx_runtime import create_session, finish_profile

    class Heads(torch.nn.Module):
        def forward(self, features, mask):
            features = features * mask.unsqueeze(-1)
            return features[..., :2], features * 2, features[..., :5] - 3, features[..., :2] + 7

    model = Heads().eval()
    x = torch.randn(3, 400, 9, generator=torch.Generator().manual_seed(19))
    mask = (torch.arange(400)[None, :] < torch.tensor([399, 349, 299])[:, None]).float()
    path = tmp_path / "heads.onnx"
    export_mlpf(model, x, mask, path)
    graph = onnx.load(path)
    for value in [*graph.graph.input, *graph.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = 3
        value.type.tensor_type.shape.dim[1].dim_value = 400
    del graph.graph.value_info[:]
    onnx.save(onnx.shape_inference.infer_shapes(graph), path)
    provider = os.environ.get("MLPF_TEST_PROVIDER", "cpu")
    session = create_session(path, provider, profile_prefix=tmp_path / "heads", require_full_gpu=provider != "cpu")
    actual = session.run(["bid", "id", "momentum", "pu"], {"Xfeat_normed": x.numpy(), "mask": mask.numpy()})
    valid = mask.bool().numpy()
    for value, expected in zip(actual, model(x, mask), strict=True):
        np.testing.assert_array_equal(value[valid], expected.numpy()[valid])
        assert np.all(value[~valid] == 0)
    finish_profile(session, provider, require_full_gpu=provider != "cpu")


@pytest.mark.parametrize("size", [1, 24, 3900, 4100])
def test_exported_sort_resolves_large_tied_integer_keys(tmp_path, size):
    from mlpf.model.heptv2 import HASH_PADDING_KEY, stable_argsort
    from mlpf.model.onnx_runtime import create_session, finish_profile

    class Sort(torch.nn.Module):
        def forward(self, values):
            return stable_argsort(values)

    generator = torch.Generator().manual_seed(19)
    values = torch.randint(-20, 20, (2, size), generator=generator) * (1 << 40)
    values[:, -max(1, size // 8) :] = HASH_PADDING_KEY
    path = tmp_path / "sort.onnx"
    torch.onnx.export(Sort(), (values,), path, opset_version=20, input_names=["values"], dynamo=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    used = {v.name for v in graph.graph.output} | {v for n in graph.graph.node for v in n.input}
    assert all(output in used for node in graph.graph.node if node.op_type == "TopK" for output in node.output)
    provider = os.environ.get("MLPF_TEST_PROVIDER", "cpu")
    session = create_session(path, provider, profile_prefix=tmp_path / "sort", require_full_gpu=provider != "cpu")
    actual = session.run(None, {"values": values.numpy()})[0]
    np.testing.assert_array_equal(actual, torch.argsort(values, stable=True).numpy())
    finish_profile(session, provider, require_full_gpu=provider != "cpu")


def test_integer_coordinates_export_exactly(tmp_path):
    from mlpf.model.heptv2 import HASH_COORD_SCALE, fixed_point_coordinates

    class Coordinates(torch.nn.Module):
        def forward(self, features):
            return fixed_point_coordinates(features)

    angles = torch.linspace(-torch.pi, torch.pi, 4097)
    features = torch.zeros(len(angles) + 1, 5)
    features[:-1, 2] = torch.linspace(-40, 40, len(angles))
    features[:-1, 3] = angles.sin()
    features[:-1, 4] = angles.cos()
    expected = fixed_point_coordinates(features)
    # Include the zero vector used for padded elements, all quadrants, and
    # both sides of the branch cut. Integer hash inputs must match bitwise.
    assert expected[-1].tolist() == [0, 0]
    reference = torch.atan2(features[:-1, 3], features[:-1, 4])
    torch.testing.assert_close(expected[:-1, 1].float() / HASH_COORD_SCALE, reference, atol=2e-6, rtol=0)
    path = tmp_path / "coordinates.onnx"
    torch.onnx.export(Coordinates(), (features,), path, opset_version=20, input_names=["features"], dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"features": features.numpy()})[0], expected.numpy())
