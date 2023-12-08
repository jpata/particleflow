from ray.tune import choice  # grid_search, choice, loguniform, quniform

raytune_num_samples = 16  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = choice

# gnn scan
search_space = {
    # optimizer parameters
    "lr": samp([1e-4, 1e-3, 1e-2]),
    "gpu_batch_multiplier": samp([4]),
    # model arch parameters
    "activation": samp(["elu"]),
    "conv_type": samp(["gravnet"]),  # can be "gnn_lsh", "gravnet", "attention"
    "embedding_dim": samp([128, 252, 512]),
    "width": samp([256, 512]),
    "num_convs": samp([3]),
    "dropout": samp([0.0]),
    "patience": samp([20]),
    # only for gravnet
    "k": samp([8, 16]),
    "propagate_dimensions": samp([16, 32]),
    "space_dimensions": samp([4]),
    # only for gnn-lsh
    "bin_size": samp([640]),
    "max_num_bins": samp([200]),
    "distance_dim": samp([128]),
    "layernorm": samp([True]),
    "num_node_messages": samp([2]),
    "ffn_dist_hidden_dim": samp([128]),
}


def set_hps_from_search_space(search_space, config):
    varaible_names = ["lr", "gpu_batch_multiplier"]
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

    return config
