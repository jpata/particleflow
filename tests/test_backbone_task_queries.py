import torch

import mlpf.model.mlpf as mlpf_module
from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF, is_jagged_tensor


def make_config(backbone_mode="shared", task_queries=True, num_convs=2, use_jagged_attention=False):
    return MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {
                "type": "attention",
                "input_encoding": "split",
                "task_queries": task_queries,
                "backbone": {"mode": backbone_mode, "num_convs": num_convs},
                "attention": {
                    "num_convs": num_convs,
                    "num_heads": 2,
                    "head_dim": 4,
                    "attention_type": "math",
                    "use_jagged_attention": use_jagged_attention,
                    "dropout_ff": 0.0,
                },
            },
        }
    )


def make_inputs(config, batch_size=2, seq_len=8):
    X = torch.randn(batch_size, seq_len, config.input_dim)
    elem_types = torch.tensor(config.elemtypes_nonzero)
    X[..., 0] = elem_types[torch.randint(0, len(elem_types), (batch_size, seq_len))].to(X.dtype)
    X[..., 1] = torch.exp(X[..., 1])
    X[..., 5] = torch.exp(X[..., 5])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return X, mask


def assert_output_shapes(outputs, config, batch_size=2, seq_len=8):
    preds_binary_particle, preds_pid, preds_momentum, preds_pu = outputs
    assert preds_binary_particle.shape == (batch_size, seq_len, 2)
    assert preds_pid.shape == (batch_size, seq_len, config.num_classes)
    assert preds_momentum.shape == (batch_size, seq_len, 5)
    assert preds_pu.shape == (batch_size, seq_len, 2)


def test_shared_backbone_uses_task_query_readouts():
    config = make_config(backbone_mode="shared", task_queries=True, num_convs=2)
    model = MLPF(config).eval()
    X, mask = make_inputs(config)

    with torch.no_grad():
        outputs = model(X, mask)
        embedding = model.encode_backbone(X, mask)

    assert_output_shapes(outputs, config)
    assert embedding.shape == (2, 8, 8)
    assert model.classification_query is not None
    assert model.regression_query is not None
    assert model.classification_readout is not None
    assert model.regression_readout is not None


def test_split_backbone_keeps_separate_id_and_regression_paths():
    config = make_config(backbone_mode="split", task_queries=False, num_convs=2)
    model = MLPF(config).eval()
    X, mask = make_inputs(config)

    with torch.no_grad():
        outputs = model(X, mask)
        embedding = model.encode_backbone(X, mask)

    assert_output_shapes(outputs, config)
    assert embedding.shape == (2, 8, 8)
    assert model.classification_query is None
    assert model.regression_query is None
    assert len(model.conv_id) == 2
    assert len(model.conv_reg) == 2
    assert model.conv_id is not model.conv_reg


def test_shared_backbone_ignores_padded_elements():
    torch.manual_seed(7)
    config = make_config(backbone_mode="shared", task_queries=False, num_convs=2, use_jagged_attention=True)
    model = MLPF(config).eval()
    X, mask = make_inputs(config, batch_size=1, seq_len=6)

    jagged_backbone_inputs = []
    hooks = [
        layer.register_forward_pre_hook(lambda _module, inputs: jagged_backbone_inputs.append(is_jagged_tensor(inputs[0])))
        for layer in model.backbone
    ]

    padded_X = torch.nn.functional.pad(X, (0, 0, 0, 10))
    padded_mask = torch.nn.functional.pad(mask, (0, 10))

    with torch.no_grad():
        outputs = model(X, mask)
        padded_outputs = model(padded_X, padded_mask)
        embedding = model.encode_backbone(X, mask)
        padded_embedding = model.encode_backbone(padded_X, padded_mask)

    for hook in hooks:
        hook.remove()

    assert model.use_jagged_attention
    assert jagged_backbone_inputs == [True] * (4 * len(model.backbone))
    torch.testing.assert_close(embedding, padded_embedding[:, : X.shape[1]], rtol=1e-5, atol=1e-6)
    for output, padded_output in zip(outputs, padded_outputs):
        torch.testing.assert_close(output, padded_output[:, : X.shape[1]], rtol=1e-5, atol=1e-6)


def test_dense_unmasked_attention_remains_the_default():
    config = make_config(task_queries=False)
    model = MLPF(config)

    assert not config.model.attention.use_jagged_attention
    assert not model.use_jagged_attention


def test_shared_jagged_backbone_packs_and_unpacks_once(monkeypatch):
    config = make_config(task_queries=False, num_convs=2, use_jagged_attention=True)
    model = MLPF(config).eval()
    X, mask = make_inputs(config, batch_size=2, seq_len=8)
    mask[0, 6:] = False

    pack_calls = 0
    unpack_calls = 0
    original_pack = mlpf_module.dense_to_jagged
    original_unpack = mlpf_module.jagged_to_dense

    def counted_pack(*args, **kwargs):
        nonlocal pack_calls
        pack_calls += 1
        return original_pack(*args, **kwargs)

    def counted_unpack(*args, **kwargs):
        nonlocal unpack_calls
        unpack_calls += 1
        return original_unpack(*args, **kwargs)

    monkeypatch.setattr(mlpf_module, "dense_to_jagged", counted_pack)
    monkeypatch.setattr(mlpf_module, "jagged_to_dense", counted_unpack)

    with torch.no_grad():
        model(X, mask)

    assert pack_calls == 1
    assert unpack_calls == 1
