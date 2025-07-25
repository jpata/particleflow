from ray.tune import grid_search  # grid_search, choice, loguniform, quniform

raytune_num_samples = 1  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = grid_search

# gnn scan
search_space = {
    # dataset parameters
    # "ntrain": samp([500]),
    # "ntest": samp([10000]),
    # "nvalid": samp([500]),
    # "num_epochs": samp([10]),
    # optimizer parameters
    "optimizer": samp(["adamw", "lamb"]),
    "lr": samp([2e-4, 4e-4, 8e-4]),
    "weight_decay": samp([0.001, 0.01, 0.03, 0.1]),
    "lr_schedule": samp(["cosinedecay"]),
    # "pct_start": samp([0.05, 0.1, 0.2]),
    # "gpu_batch_multiplier": samp([256]),
    # "patience": samp([9999]),
    # model arch parameters
    # "activation": samp(["elu", "relu", "relu6", "leakyrelu"]),
    # "conv_type": samp(["attention"]),  # can be "gnn_lsh", "gravnet", "attention"
    # "embedding_dim": samp([32, 64, 128, 252, 512, 1024]),
    # "width": samp([32, 64, 128, 256, 512, 1024]),
    # "num_convs": samp([1, 2, 3, 4, 5]),
    # "dropout": samp([0.0, 0.01, 0.1, 0.4]),
    # only for gnn-lsh
    # "bin_size": samp([80, 160, 320, 640]),
    # "max_num_bins": samp([200]),
    # "distance_dim": samp([128]),
    # "layernorm": samp([True, False]),
    # "num_node_messages": samp([2]),
    # "ffn_dist_hidden_dim": samp([64]),
    # "ffn_dist_num_layers": samp([3]),
}


def set_hps_from_search_space(search_space, config):
    varaible_names = ["optimizer", "lr", "weight_decay", "lr_schedule", "gpu_batch_multiplier", "ntrain", "ntest", "nvalid", "num_epochs", "patience"]
    for var in varaible_names:
        if var in search_space.keys():
            config[var] = search_space[var]

    if "conv_type" in search_space.keys():
        conv_type = search_space["conv_type"]
        config["conv_type"] = conv_type

        common_varaible_names = ["num_convs", "activation"]
        if conv_type == "gnn_lsh" or conv_type == "gravnet" or conv_type == "attention":
            for var in common_varaible_names:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        attention_variables = ["head_dim", "num_heads"]
        if conv_type == "attention":
            for var in attention_variables:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        gnn_lsh_varaible_names = [
            "width",
            "embedding_dim",
            "bin_size",
            "max_num_bins",
            "distance_dim",
            "layernorm",
            "num_node_messages",
            "ffn_dist_hidden_dim",
        ]
        if conv_type == "gnn_lsh":
            for var in gnn_lsh_varaible_names:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        if "pct_start" in search_space.keys():
            config["lr_schedule_config"]["onecycle"]["pct_start"] = search_space["pct_start"]

    return config
