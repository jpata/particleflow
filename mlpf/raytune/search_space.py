from ray.tune import choice  # grid_search, choice, loguniform, quniform

raytune_num_samples = 300  # Number of random samples to draw from search space. Set to 1 for grid search.
samp = choice
# search_space = {
# Optimizer parameters
# "lr": samp([1e-4, 1e-3]),
# "activation": samp(["elu"]),
# "batch_size_physical": samp([32, 40]),
# "batch_size_delphes": samp([32, 40]),
# "batch_size_gun": samp([600]),
# "expdecay_decay_steps": samp([2000]),

# Model parameters
# "layernorm": samp([False]),
# "ffn_dist_hidden_dim": samp([64, 256]),
# "ffn_dist_num_layers": samp([1]),
# "distance_dim": samp([128]),
# "num_node_messages": samp([1]),
# "num_graph_layers_id": samp([0, 1, 2, 3, 4]),
# "num_graph_layers_reg": samp([0, 1, 2, 3, 4]),
# "dropout": samp([0.0]),
# "bin_size": samp([32, 64, 128]),
# "clip_value_low": samp([0.0]),
# "dist_norm": samp(["l1", "l2"]),
# "normalize_degrees": samp([True]),
# "output_dim": samp([64, 128, 256]),
# # none, sliced_wasserstein, gen_jet_logcosh, gen_jet_mse, hist_2d
# "event_loss": samp(["none", "sliced_wasserstein", "gen_jet_logcosh", "hist_2d"]),
# "met_loss": samp([
#         "none",
#         {"type": "Huber", "delta": 10.0}
#     ]),
# "event_and_met_loss": samp([
#         ("none", "none"),
#         ("sliced_wasserstein", "none"),
#         ("gen_jet_logcosh", "none"),
#         # ("hist_2d", "none"),
#         ("none", "met"),
# ]),
# "mask_reg_cls0": samp([False, True]),
# }

# search_space = {
#     # Optimizer parameters
#     "lr": loguniform(1e-4, 1e-2),
#     # "activation": "elu",
#     "batch_size_physical": samp([24, 40]),
#     # "batch_size_gun": quniform(100, 800, 100),
#     # "batch_size_delphes": samp([8, 16, 24]),
#     # "expdecay_decay_steps": quniform(10, 2000, 10),
#     # "expdecay_decay_rate": uniform(0.9, 1),
#     # Model parameters
#     "out_hidden_dim": samp([32, 64, 128, 256]),
#     "out_num_layers": samp([1, 2, 3]),
#     "node_encoding_hidden_dim": samp([32, 64, 128, 256]),
#     # "layernorm": quniform(0, 1, 1),
#     "ffn_dist_hidden_dim": quniform(64, 256, 64),
#     "ffn_dist_num_layers": quniform(1, 3, 1),
#     "distance_dim": quniform(32, 256, 32),
#     "num_node_messages": quniform(1, 3, 1),
#     "num_graph_layers_id": quniform(0, 4, 1),
#     "num_graph_layers_reg": quniform(0, 4, 1),
#     # "dropout": uniform(0.0, 0.5),
#     "bin_size": choice([32, 64, 128, 256]),
#     # "clip_value_low": uniform(0.0, 0.1),
#     # "dist_mult": uniform(0.01, 0.2),
#     # "normalize_degrees": quniform(0, 1, 1),
#     "output_dim": choice([8, 16, 32, 64, 128, 256]),
#     "lr_schedule": choice(["none", "cosinedecay"])  # exponentialdecay, cosinedecay, onecycle, none
#     # "weight_decay": loguniform(1e-6, 1e-1),
#     # "event_loss": choice([None, "sliced_wasserstein", "gen_jet_logcosh", "gen_jet_mse", "hist_2d"]),
#     # "mask_reg_cls0": choice([False, True]),
# }

# # onecycle scan
# search_space = {
#     # "lr": samp([1e-4, 1e-3, 1e-2]),
#     # "batch_size_physical": samp([24, 40]),
#     "batch_multiplier": samp([1, 5, 10]),
#     # "model": samp(["gnn_dense", "transformer"]),
#     # "lr_schedule": samp(["none", "cosinedecay", "onecycle"]),
#     # "optimizer": samp(["pcgrad_adam", "adam", "sgd"]),
# }

# transformer scan
# search_space = {
#     # optimizer parameters
#     "lr": samp([1e-5, 1e-4, 1e-3]),
#     "batch_multiplier": samp([10, 20, 40]),
#     # model arch parameters
#     "num_layers_encoder": samp([1, 2, 3, 4]),  # default is 1
#     "num_layers_decoder_reg": samp([1, 2, 3, 4]),  # default is 1
#     "num_layers_decoder_cls": samp([1, 2, 3, 4]),  # default is 1
#     "hidden_dim": samp([32, 64, 128]),  # default is 64
#     "num_heads": samp([8, 16, 32, 64]),  # default is 16
#     "num_random_features": samp([16, 32, 64, 128]),  # default is 32
#     # output_decoding parameters
#     "out_hidden_dim": samp([128, 256, 512]),  # default is ~256
#     "out_num_layers": samp([1, 2, 3, 4]),  # default is ~2
# }

# gnn scan
search_space = {
    # optimizer parameters
    "lr": samp([1e-4, 1e-3, 1e-2]),
    # "batch_multiplier": samp([10, 20, 40]),
    # model arch parameters
    "num_graph_layers_id": samp([1, 2, 3, 4, 5, 6]),
    "num_graph_layers_reg": samp([1, 2, 3, 4, 5, 6]),
    "bin_size": samp([16, 32, 64, 128, 256]),
    "output_dim": samp([64, 128, 256]),
    "ffn_dist_hidden_dim": samp([64, 128, 256]),
    "ffn_dist_num_layers": samp([1, 2, 3, 4, 5]),
    # output_decoding parameters
    "out_hidden_dim": samp([64, 128, 256, 512]),
    "out_num_layers": samp([1, 2, 3, 4, 5]),
}


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
        config["parameters"]["combined_graph_layer"]["node_message"]["normalize_degrees"] = bool(search_space["normalize_degrees"])
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

    if "batch_multiplier" in search_space.keys():
        if not config["batching"]["bucket_by_sequence_length"]:
            raise ValueError("batch_multiplier given but bucket_by_sequence_length is set to False. Check config.")
        config["batching"]["batch_multiplier"] = search_space["batch_multiplier"]

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

    if "optimizer" in search_space.keys():
        if search_space["optimizer"] == "pcgrad_adam":
            config["setup"]["optimizer"] = "adam"
            config["optimizer"]["adam"]["pcgrad"] = True
        elif search_space["optimizer"] == "adam":
            config["setup"]["optimizer"] = "adam"
            config["optimizer"]["adam"]["pcgrad"] = False
        else:
            config["setup"]["optimizer"] = search_space["optimizer"]

    if "node_encoding_hidden_dim" in search_space.keys():
        config["parameters"]["node_encoding_hidden_dim"] = search_space["node_encoding_hidden_dim"]

    if "out_hidden_dim" in search_space.keys():
        config["parameters"]["output_decoding"]["id_hidden_dim"] = search_space["out_hidden_dim"]
        config["parameters"]["output_decoding"]["charge_hidden_dim"] = search_space["out_hidden_dim"]
        config["parameters"]["output_decoding"]["pt_hidden_dim"] = search_space["out_hidden_dim"]
        config["parameters"]["output_decoding"]["eta_hidden_dim"] = search_space["out_hidden_dim"]
        config["parameters"]["output_decoding"]["phi_hidden_dim"] = search_space["out_hidden_dim"]
        config["parameters"]["output_decoding"]["energy_hidden_dim"] = search_space["out_hidden_dim"]

    if "out_num_layers" in search_space.keys():
        config["parameters"]["output_decoding"]["id_num_layers"] = search_space["out_num_layers"]
        config["parameters"]["output_decoding"]["charge_num_layers"] = search_space["out_num_layers"]
        config["parameters"]["output_decoding"]["pt_num_layers"] = search_space["out_num_layers"]
        config["parameters"]["output_decoding"]["eta_num_layers"] = search_space["out_num_layers"]
        config["parameters"]["output_decoding"]["phi_num_layers"] = search_space["out_num_layers"]
        config["parameters"]["output_decoding"]["energy_num_layers"] = search_space["out_num_layers"]

    # transformer specific parameters
    if "num_layers_encoder" in search_space.keys():
        config["parameters"]["num_layers_encoder"] = search_space["num_layers_encoder"]
    if "num_layers_decoder_reg" in search_space.keys():
        config["parameters"]["num_layers_decoder_reg"] = search_space["num_layers_decoder_reg"]
    if "num_layers_decoder_cls" in search_space.keys():
        config["parameters"]["num_layers_decoder_cls"] = search_space["num_layers_decoder_cls"]
    if "hidden_dim" in search_space.keys():
        config["parameters"]["hidden_dim"] = search_space["hidden_dim"]
    if "num_heads" in search_space.keys():
        config["parameters"]["num_heads"] = search_space["num_heads"]
    if "num_random_features" in search_space.keys():
        config["parameters"]["num_random_features"] = search_space["num_random_features"]

    return config
