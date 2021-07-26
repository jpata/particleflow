from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import keras_tuner as kt

from tfmodel.model_setup import make_model, FlattenedCategoricalAccuracy
from tfmodel.model import PFNetDense

from tfmodel.utils import (
    get_lr_schedule,
    get_optimizer,
    load_config,
    set_config_loss,
    get_loss_dict,
    parse_config,
)


def get_model_builder(config, total_steps):
    _, optim_callbacks = get_lr_schedule(config, steps=total_steps)
    def model_builder(hp):
        config["parameters"]["hidden_dim"] = hp.Choice("hidden_dim", values=[256])
        config["parameters"]["distance_dim"] = hp.Choice("distance_dim", values=[128])
        config["parameters"]["num_conv"] = hp.Choice("num_conv", [2, 3])
        config["parameters"]["num_gsl"] = hp.Choice("num_gsl", [2, 3])
        config["parameters"]["dropout"] = hp.Choice("dropout", values=[0.2])
        config["parameters"]["bin_size"] = hp.Choice("bin_size", values=[640])

        config["setup"]["lr"] = hp.Choice("lr", values=[1e-4, 3e-4])
        config["setup"]["lr_schedule"] = hp.Choice("lr_schedule", values=["exponentialdecay"])
        config["setup"]["optimizer"] = hp.Choice("optimizer", values=["adam"])


        model = make_model(config, dtype="float32")
        model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

        lr_schedule, _ = get_lr_schedule(config, steps=total_steps)
        opt = get_optimizer(config, lr_schedule)

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
            metrics={
                "cls": [
                    FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                    FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                ]
            },
        )
        return model

    return model_builder, optim_callbacks
