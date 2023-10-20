from ray.tune import choice  # grid_search, choice, loguniform, quniform

raytune_num_samples = 4  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = choice

# gnn scan
search_space = {
    # optimizer parameters
    "lr": samp([1e-4, 1e-3, 1e-2]),
    # "gpu_batch_multiplier": samp([10, 20, 40]),
    # model arch parameters
    "conv_type": samp(["gnn_lsh"]),
    "embedding_dim": samp([252, 512]),
    # "width": samp([512]),
    # "num_convs": samp([3]),
    # "dropout": samp([0.0]),
    # "patience": samp([20])
}


def set_hps_from_search_space(search_space, config):
    if "lr" in search_space.keys():
        config["lr"] = search_space["lr"]

    if "gpu_batch_multiplier" in search_space.keys():
        config["gpu_batch_multiplier"] = search_space["gpu_batch_multiplier"]

    if "conv_type" in search_space.keys():
        conv_type = search_space["conv_type"]
        config["conv_type"] = conv_type

        if conv_type == "gnn_lsh" or conv_type == "transformer":
            if "embedding_dim" in search_space.keys():
                config["model"][conv_type]["embedding_dim"] = search_space["embedding_dim"]

            if "width" in search_space.keys():
                config["model"][conv_type]["width"] = search_space["width"]

            if "num_convs" in search_space.keys():
                config["model"][conv_type]["num_convs"] = search_space["num_convs"]

            if "num_convs" in search_space.keys():
                config["model"][conv_type]["num_convs"] = search_space["num_convs"]

    if "embedding_dim" in search_space.keys():
        config["embedding_dim"] = search_space["embedding_dim"]

    return config
