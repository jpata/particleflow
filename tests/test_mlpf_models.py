import torch
import pytest
from mlpf.model.mlpf import MLPF
from mlpf.conf import MLPFConfig


def test_mlpf_attention():
    config_dict = {
        "dataset": "cms",
        "data_dir": "/tmp",
        "model": {
            "type": "attention",
            "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math", "use_simplified_attention": True},
        },
        "conv_type": "attention",
    }
    config = MLPFConfig.model_validate(config_dict)
    model = MLPF(config)

    batch_size = 2
    seq_len = 32
    input_dim = config.input_dim

    X = torch.randn(batch_size, seq_len, input_dim)
    # Set realistic type indices (from elemtypes_nonzero)
    elem_types = torch.tensor(config.elemtypes_nonzero)
    X[..., 0] = elem_types[torch.randint(0, len(elem_types), (batch_size, seq_len))].to(X.dtype)

    # Ensure pt and energy are positive for log/sqrt in forward pass
    X[..., 1] = torch.exp(X[..., 1])  # pt
    X[..., 5] = torch.exp(X[..., 5])  # energy

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    preds_binary_particle, preds_pid, preds_momentum, preds_pu = model(X, mask)

    assert preds_binary_particle.shape == (batch_size, seq_len, 2)
    assert preds_pid.shape == (batch_size, seq_len, config.num_classes)
    assert preds_momentum.shape == (batch_size, seq_len, 5)
    assert preds_pu.shape == (batch_size, seq_len, 2)


def test_mlpf_gnn_lsh():
    config_dict = {
        "dataset": "cms",
        "data_dir": "/tmp",
        "model": {
            "type": "gnn_lsh",
            "gnn_lsh": {
                "num_convs": 1,
                "bin_size": 16,
                "max_num_bins": 10,
                "distance_dim": 16,
            },
        },
        "conv_type": "gnn_lsh",
    }
    config = MLPFConfig.model_validate(config_dict)
    model = MLPF(config)

    batch_size = 2
    seq_len = 32
    input_dim = config.input_dim

    X = torch.randn(batch_size, seq_len, input_dim)
    elem_types = torch.tensor(config.elemtypes_nonzero)
    X[..., 0] = elem_types[torch.randint(0, len(elem_types), (batch_size, seq_len))].to(X.dtype)
    X[..., 1] = torch.exp(X[..., 1])
    X[..., 5] = torch.exp(X[..., 5])

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    preds_binary_particle, preds_pid, preds_momentum, preds_pu = model(X, mask)

    assert preds_binary_particle.shape == (batch_size, seq_len, 2)
    assert preds_pid.shape == (batch_size, seq_len, config.num_classes)
    assert preds_momentum.shape == (batch_size, seq_len, 5)
    assert preds_pu.shape == (batch_size, seq_len, 2)


@pytest.mark.parametrize("dataset", ["cms", "clic", "cld"])
def test_mlpf_datasets(dataset):
    config_dict = {
        "dataset": dataset,
        "data_dir": "/tmp",
        "model": {
            "type": "attention",
            "attention": {
                "num_convs": 1,
                "num_heads": 2,
                "head_dim": 8,
                "attention_type": "math",
            },
        },
        "conv_type": "attention",
    }
    config = MLPFConfig.model_validate(config_dict)
    model = MLPF(config)

    batch_size = 1
    seq_len = 16
    input_dim = config.input_dim

    X = torch.randn(batch_size, seq_len, input_dim)
    elem_types = torch.tensor(config.elemtypes_nonzero)
    X[..., 0] = elem_types[torch.randint(0, len(elem_types), (batch_size, seq_len))].to(X.dtype)
    X[..., 1] = torch.exp(X[..., 1])
    X[..., 5] = torch.exp(X[..., 5])

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    preds_binary_particle, preds_pid, preds_momentum, preds_pu = model(X, mask)

    assert preds_binary_particle.shape == (batch_size, seq_len, 2)
    assert preds_pid.shape == (batch_size, seq_len, config.num_classes)
    assert preds_momentum.shape == (batch_size, seq_len, 5)
    assert preds_pu.shape == (batch_size, seq_len, 2)
