from ray.tune import grid_search  # grid_search, choice, loguniform, quniform

raytune_num_samples = 1  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = grid_search

# gnn scan
search_space = {
    # dataset parameters
    "ntrain": samp([500]),
    # "ntest": samp([10000]),
    "nvalid": samp([500]),
    "num_epochs": samp([10]),
    # optimizer parameters
    "lr": samp([1e-4, 3e-4, 1e-3, 3e-3]),
    "lr_schedule": samp(["onecycle"]),
    "pct_start": samp([0.05]),
    # "gpu_batch_multiplier": samp([1, 4, 8, 16]),
    # "patience": samp([9999]),
    # model arch parameters
    # "activation": samp(["elu", "relu", "relu6", "leakyrelu"]),
    "conv_type": samp(["attention"]),  # can be "gnn_lsh", "gravnet", "attention"
    # "embedding_dim": samp([32, 64, 128, 252, 512, 1024]),
    # "width": samp([32, 64, 128, 256, 512, 1024]),
    # "num_convs": samp([1, 2, 3, 4, 5, 6]),
    # "dropout": samp([0.0, 0.01, 0.1, 0.4]),
    # only for gravnet
    # "k": samp([8, 16, 32]),
    # "propagate_dimensions": samp([8, 16, 32, 64, 128]),
    # "space_dimensions": samp([4]),
    # only for gnn-lsh
    # "bin_size": samp([160, 320, 640]),
    # "max_num_bins": samp([200]),
    # "distance_dim": samp([16, 32, 64, 128, 256]),
    # "layernorm": samp([True, False]),
    # "num_node_messages": samp([1, 2, 3, 4, 5]),
    # "ffn_dist_hidden_dim": samp([16, 32, 64, 128, 256]),
    # "ffn_dist_num_layers": samp([1, 2, 3, 4, 5, 6]),
    # mamba specific parameters
    # "d_state": samp([16]),
    # "d_conv": samp([4]),
    # "expand": samp([2]),
    # "num_heads": samp([2, 4, 6, 8, 10, 12]),
    # attention specifica parameters
    "num_heads": samp([2, 4, 8, 16]),
    # "attention_type": samp(["flash"]),  # flash, efficient, math
}


def set_hps_from_search_space(search_space, config):
    varaible_names = ["lr", "lr_schedule", "gpu_batch_multiplier", "ntrain", "ntest", "nvalid", "num_epochs", "patience"]
    for var in varaible_names:
        if var in search_space.keys():
            config[var] = search_space[var]

    if "conv_type" in search_space.keys():
        conv_type = search_space["conv_type"]
        config["conv_type"] = conv_type

        common_varaible_names = ["embedding_dim", "width", "num_convs", "activation"]
        if conv_type == "gnn_lsh" or conv_type == "gravnet" or conv_type == "attention":
            for var in common_varaible_names:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        gravnet_variable_names = ["k", "propagate_dimensions", "space_dimensions"]
        if conv_type == "gravnet":
            for var in gravnet_variable_names:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        attention_variables = ["num_heads"]
        if conv_type == "attention":
            for var in attention_variables:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        mamba_variables = ["num_heads", "d_state", "d_conv", "expand"]
        if conv_type == "mamba":
            for var in mamba_variables:
                if var in search_space.keys():
                    config["model"][conv_type][var] = search_space[var]

        gnn_lsh_varaible_names = [
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
