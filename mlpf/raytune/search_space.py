from ray.tune import grid_search

raytune_num_samples = 1  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = grid_search
search_space = {
    # Optimizer parameters
    "lr": samp([1e-3]),
    "activation": samp(["elu"]),
    "batch_size_physical": samp([32]),
    # "batch_size_gun": samp([600]),
    "expdecay_decay_steps": samp([2000]),
    # Model parameters
    "layernorm": samp([False]),
    "ffn_dist_hidden_dim": samp([64, 256]),
    "ffn_dist_num_layers": samp([1]),
    "distance_dim": samp([128]),
    "num_node_messages": samp([1]),
    "num_graph_layers_common": samp([3]),
    "num_graph_layers_energy": samp([3]),
    "dropout": samp([0.0]),
    "bin_size": samp([160]),
    "clip_value_low": samp([0.0]),
    "normalize_degrees": samp([True]),
    "output_dim": samp([128]),
    # "event_loss": samp(["none", "sliced_wasserstein", "gen_jet_logcosh", "hist_2d"]),
    # "met_loss": samp([
    #         "none",
    #         {"type": "Huber", "delta": 10.0}
    #     ]),
    "event_and_met_loss": samp(
        [
            ("none", "none"),
            ("sliced_wasserstein", "none"),
            ("gen_jet_logcosh", "none"),
            ("hist_2d", "none"),
            ("none", "met"),
        ]
    ),
    # "mask_reg_cls0": samp([False, True]),
}

# search_space = {
# Optimizer parameters
# "lr": loguniform(1e-4, 3e-2),
# "activation": "elu",
# "batch_size_physical": quniform(4, 32, 4),
# "batch_size_gun": quniform(100, 800, 100),
# "expdecay_decay_steps": quniform(10, 2000, 10),
# "expdecay_decay_rate": uniform(0.9, 1),
# Model parameters
# "layernorm": quniform(0, 1, 1),
# "ffn_dist_hidden_dim": quniform(64, 256, 64),
# "ffn_dist_num_layers": quniform(1, 3, 1),
# "distance_dim": quniform(64, 512, 64),
# "num_node_messages": quniform(1, 3, 1),
# "num_graph_layers_common": quniform(2, 4, 1),
# "num_graph_layers_energy": quniform(2, 4, 1),
# "dropout": uniform(0.0, 0.5),
# "bin_size": quniform(160, 320, 160),
# "clip_value_low": uniform(0.0, 0.2),
# "dist_mult": uniform(0.01, 0.2),
# "normalize_degrees": quniform(0, 1, 1),
# "output_dim": quniform(64, 512, 64),
# }


def set_raytune_search_parameters(search_space, config):
    if "layernorm" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["layernorm"] = bool(search_space["layernorm"])
    if "ffn_dist_hidden_dim" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["ffn_dist_hidden_dim"] = int(search_space["ffn_dist_hidden_dim"])
    if "ffn_dist_num_layers" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["ffn_dist_num_layers"] = int(search_space["ffn_dist_num_layers"])
    if "distance_dim" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["distance_dim"] = int(search_space["distance_dim"])
    if "num_node_messages" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["num_node_messages"] = int(search_space["num_node_messages"])
    if "normalize_degrees" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["node_message"]["normalize_degrees"] = bool(
            search_space["normalize_degrees"]
        )
    if "output_dim" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["node_message"]["output_dim"] = int(search_space["output_dim"])

    if "activation" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["node_message"]["activation"] = search_space["activation"]
        config["parameters"]["combined_graph_layer"]["dist_activation"] = search_space["activation"]
        config["parameters"]["combined_graph_layer"]["activation"] = search_space["activation"]

    if "num_graph_layers_id" in search_space.keys():
        config["parameters"]["num_graph_layers_id"] = int(search_space["num_graph_layers_id"])
    if "num_graph_layers_reg" in search_space.keys():
        config["parameters"]["num_graph_layers_reg"] = int(search_space["num_graph_layers_reg"])
    if "bin_size" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["bin_size"] = int(search_space["bin_size"])
    if "clip_value_low" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["kernel"]["clip_value_low"] = search_space["clip_value_low"]
    if "dist_mult" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["kernel"]["dist_mult"] = search_space["dist_mult"]

    if "dist_norm" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["kernel"]["dist_norm"] = search_space["dist_norm"]

    if "dropout" in search_space.keys():
        config["parameters"]["combined_graph_layer"]["dropout"] = search_space["dropout"] / 2
        config["parameters"]["output_decoding"]["dropout"] = search_space["dropout"]

    if "lr" in search_space.keys():
        config["setup"]["lr"] = search_space["lr"]

    if "batch_size_physical" in search_space.keys():
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = int(search_space["batch_size_physical"])

    if "batch_size_delphes" in search_space.keys():
        config["train_test_datasets"]["delphes"]["batch_per_gpu"] = int(search_space["batch_size_physical"])

    if "batch_size_gun" in search_space.keys():
        config["train_test_datasets"]["gun"]["batch_per_gpu"] = int(search_space["batch_size_gun"])

    if "expdecay_decay_steps" in search_space.keys():
        config["exponentialdecay"]["decay_steps"] = search_space["expdecay_decay_steps"]

    if "expdecay_decay_rate" in search_space.keys():
        config["exponentialdecay"]["decay_rate"] = search_space["expdecay_decay_rate"]

    if "event_loss" in search_space.keys():
        config["loss"]["event_loss"] = search_space["event_loss"]
        if search_space["event_loss"] == "none":
            config["loss"]["event_loss_coef"] = 0.0
        else:
            config["loss"]["event_loss_coef"] = 1.0

    if "met_loss" in search_space.keys():
        config["loss"]["met_loss"] = search_space["event_loss"]
        if search_space["met_loss"] == "none":
            config["loss"]["met_loss_coef"] = 0.0
        else:
            config["loss"]["met_loss_coef"] = 1.0

    if "event_and_met_loss" in search_space.keys():
        event_l, met_l = search_space["event_and_met_loss"]

        config["loss"]["event_loss"] = event_l

        if event_l == "none":
            config["loss"]["event_loss_coef"] = 0.0
        else:
            config["loss"]["event_loss_coef"] = 1.0

        if met_l == "none":
            config["loss"]["met_loss"] = met_l
            config["loss"]["met_loss_coef"] = 0.0
        else:
            config["loss"]["met_loss"] = {"type": "Huber", "delta": 10.0}
            config["loss"]["met_loss_coef"] = 1.0

    if "mask_reg_cls0" in search_space.keys():
        config["parameters"]["output_decoding"]["mask_reg_cls0"] = search_space["mask_reg_cls0"]

    if "lr_schedule" in search_space.keys():
        config["setup"]["lr_schedule"] = search_space["lr_schedule"]

    if "weight_decay" in search_space.keys():
        config["optimizer"]["adamw"]["weight_decay"] = search_space["weight_decay"]

    return config
