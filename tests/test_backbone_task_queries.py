import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF


def make_config(backbone_mode="shared", task_queries=True, num_convs=2):
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
