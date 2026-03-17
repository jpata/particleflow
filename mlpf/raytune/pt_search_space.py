from ray.tune import choice  # grid_search, choice, loguniform, quniform
from mlpf.utils import set_nested_dict


raytune_num_samples = 10  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = choice

search_space = {
    # dataset parameters
    "ntest": samp([1000]),
    "nvalid": samp([1000]),
    # training length parameters
    "num_steps": samp([10000]),
    "val_freq": samp([10000]),
    "checkpoint_freq": samp([10000]),
    # optimizer parameters
    "optimizer": samp(["adamw", "lamb"]),
    "lr": samp([2e-4, 4e-4, 8e-4]),
    # "weight_decay": samp([0.001, 0.01, 0.03, 0.1]),
    # "lr_schedule": samp(["cosinedecay"]),
    # "pct_start": samp([0.05, 0.1, 0.2]),
    "gpu_batch_multiplier": samp([1, 4, 8]),
    # "patience": samp([9999]),
    # model arch parameters
    "conv_type": samp(["attention"]),  # can be "gnn_lsh", "attention"
    # attention parameters
    "model.attention.num_convs": samp([1, 2, 3, 4, 5]),
    "model.attention.num_heads": samp([8, 16, 32]),
    "model.attention.head_dim": samp([8, 16, 32]),
    "model.attention.dropout_ff": samp([0.0, 0.01, 0.1, 0.2]),
    "model.attention.activation": samp(["elu", "relu", "relu6", "leakyrelu"]),
}


def set_hps_from_search_space(search_space, config):
    for key, value in search_space.items():
        set_nested_dict(config, key, value)
    return config
