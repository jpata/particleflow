from tfmodel.model_setup import make_model
from tfmodel.utils import get_loss_dict, get_lr_schedule, get_optimizer


def get_model_builder(config, total_steps):
    lr_schedule, optim_callbacks, lr = get_lr_schedule(config, steps=total_steps)

    def model_builder(hp):
        node_encoding_hidden_dim = hp.Choice("node_dim", values=[128, 256, 512, 1024])

        config["parameters"]["node_encoding_hidden_dim"] = node_encoding_hidden_dim

        config["parameters"]["num_graph_layers_id"] = hp.Choice("num_graph_layers_id", [1, 2, 3, 4, 5])
        config["parameters"]["num_graph_layers_reg"] = hp.Choice("num_graph_layers_reg", [1, 2, 3, 4, 5])

        config["parameters"]["combined_graph_layer"]["dropout"] = hp.Choice("cg_dropout", values=[0.0, 0.1, 0.2])
        config["parameters"]["combined_graph_layer"]["num_node_messages"] = hp.Choice("num_node_messages", [1, 2])
        config["parameters"]["combined_graph_layer"]["bin_size"] = hp.Choice("bin_size", values=[160, 320, 640])
        config["parameters"]["combined_graph_layer"]["ffn_dist_hidden_dim"] = hp.Choice("ffn_dist_hidden_dim", values=[64, 128, 256])
        config["parameters"]["combined_graph_layer"]["ffn_dist_num_layers"] = hp.Choice("ffn_dist_num_layers", values=[1, 2])
        config["parameters"]["combined_graph_layer"]["kernel"]["dist_mult"] = hp.Choice("dist_mult", values=[0.01, 0.1, 1.0])

        config["parameters"]["combined_graph_layer"]["node_message"]["output_dim"] = node_encoding_hidden_dim
        config["parameters"]["combined_graph_layer"]["node_message"]["normalize_degrees"] = hp.Choice("normalize_degrees", values=[True, False])
        config["parameters"]["output_decoding"]["dropout"] = hp.Choice("output_dropout", values=[0.0, 0.1, 0.2])
        config["parameters"]["output_decoding"]["layernorm"] = hp.Choice("output_layernorm", values=[True, False])
        config["parameters"]["output_decoding"]["mask_reg_cls0"] = hp.Choice("output_mask_reg_cls0", values=[True, False])
        config["parameters"]["skip_connection"] = hp.Choice("skip_connection", values=[True, False])
        config["parameters"]["node_update_mode"] = hp.Choice("node_update_mode", values=["additive", "concat"])

        model = make_model(config, dtype="float32")
        model.build(
            (
                1,
                None,
                config["dataset"]["num_input_features"],
            )
        )

        opt = get_optimizer(config, lr_schedule)

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
        )
        return model

    return model_builder, optim_callbacks
