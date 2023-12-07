from ray.tune import choice  # grid_search, choice, loguniform, quniform

raytune_num_samples = 16  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = choice

# gnn scan
search_space = {
    # optimizer parameters
    "lr": samp([1e-4, 1e-3, 1e-2]),
    "gpu_batch_multiplier": samp([4]),
    # model arch parameters
    "conv_type": samp(["gravnet"]),  # can be "gnn_lsh", "gravnet", "attention"
    "embedding_dim": samp([128, 252, 512]),
    "width": samp([256, 512]),
    "num_convs": samp([3]),
    "dropout": samp([0.0]),
    "patience": samp([20]),
    # only for gravnet
    "gravnet_k": samp([8, 16]),
    "propagate_dimensions": samp([16, 32]),
    "space_dimensions": samp([4]),
}


def set_hps_from_search_space(search_space, config):
    if "lr" in search_space.keys():
        config["lr"] = search_space["lr"]

    if "gpu_batch_multiplier" in search_space.keys():
        config["gpu_batch_multiplier"] = search_space["gpu_batch_multiplier"]

    if "conv_type" in search_space.keys():
        conv_type = search_space["conv_type"]
        config["conv_type"] = conv_type

        if conv_type == "gnn_lsh" or conv_type == "gravnet" or conv_type == "attention":
            if "embedding_dim" in search_space.keys():
                config["model"][conv_type]["embedding_dim"] = search_space["embedding_dim"]

            if "width" in search_space.keys():
                config["model"][conv_type]["width"] = search_space["width"]

            if "num_convs" in search_space.keys():
                config["model"][conv_type]["num_convs"] = search_space["num_convs"]

            if "num_convs" in search_space.keys():
                config["model"][conv_type]["num_convs"] = search_space["num_convs"]

        if conv_type == "gravnet":
            if "gravnet_k" in search_space.keys():
                config["model"][conv_type]["k"] = search_space["gravnet_k"]

            if "propagate_dimensions" in search_space.keys():
                config["model"][conv_type]["propagate_dimensions"] = search_space["propagate_dimensions"]

            if "space_dimensions" in search_space.keys():
                config["model"][conv_type]["space_dimensions"] = search_space["space_dimensions"]

    return config
