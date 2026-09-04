import random

import numpy as np
import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.training import seed_everything


def sample_rngs():
    return random.random(), np.random.random(), torch.rand(3)


def test_seed_everything_reproduces_python_numpy_and_torch_streams():
    seed_everything(2468)
    first = sample_rngs()
    seed_everything(2468)
    second = sample_rngs()

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_seed_is_exposed_and_validated_in_mlpf_config():
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "model": {"type": "attention", "attention": {}},
            "conv_type": "attention",
            "seed": 17,
        }
    )
    assert config.seed == 17

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MLPFConfig.model_validate(
            {
                "dataset": "cld_hits",
                "data_dir": "/tmp",
                "model": {"type": "attention", "attention": {}},
                "conv_type": "attention",
                "seed": -1,
            }
        )
