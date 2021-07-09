import yaml
import tfmodel
import tfmodel.model_setup

def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f)
    return config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spec", type=str, default="parameters/delphes-gnn-skipconn.yaml", help="the model specification")
    parser.add_argument("--action", type=str, choices=["data", "train", "eval", "time"], help="Run training, validation or timing", default="train")
    parser.add_argument("--modifier", type=str, choices=["retrain_energy", None], help="Apply a modification on the standard training", default=None)
    parser.add_argument("--weights", type=str, help="weight file to load", default=None)
    parser.add_argument("--ntrain", type=int, help="override the number of training events", default=None)
    parser.add_argument("--ntest", type=int, help="override the number of testing events", default=None)
    parser.add_argument("--recreate", action="store_true", help="recreate a new output dir", default=None)
    parser.add_argument("--raw-path", type=str, help="Override the dataset raw files path", default=None)
    parser.add_argument("--processed-path", type=str, help="Override the dataset processed files path", default=None)
    args = parser.parse_args()
    return args

def apply_modifier_retrain_energy(config):
    assert(config["parameters"]["model"] == "gnn_dense")
    config["setup"]["trainable"] = "ffn_momentum4"
    for loss in [
        "classification_loss_coef",
        "charge_loss_coef",
        "pt_loss_coef",
        "eta_loss_coef",
        "sin_phi_loss_coef",
        "cos_phi_loss_coef"]:
        config["dataset"][loss] = 0.0

    config["dataset"]["energy_loss_coef"] = 1.0
    config["setup"]["batch_size"] = 20
    return config

modifiers = {
    "retrain_energy": apply_modifier_retrain_energy
}

if __name__ == "__main__":
    args = parse_args()
    yaml_path = args.model_spec

    config = load_config(yaml_path)

    if args.modifier:
        config = modifiers[args.modifier](config)

    if config["backend"] == "tensorflow":
        tfmodel.model_setup.main(args, yaml_path, config)
    elif config["backend"] == "pytorch":
        pass
