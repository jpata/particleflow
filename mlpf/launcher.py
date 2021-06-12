import yaml

def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f)
    return config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spec", type=str, default="parameters/delphes-gnn-skipconn.yaml", help="the model specification")
    parser.add_argument("--action", type=str, choices=["data", "train", "eval", "time"], help="Run training, validation or timing", default="train")
    parser.add_argument("--weights", type=str, help="weight file to load", default=None)
    parser.add_argument("--ntrain", type=int, help="override the number of training events", default=None)
    parser.add_argument("--ntest", type=int, help="override the number of testing events", default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    yaml_path = args.model_spec

    config = load_config(yaml_path)

    if config["backend"] == "tensorflow":
        import tfmodel
        from tfmodel.model_setup import main
        main(args, yaml_path, config)
    elif config["backend"] == "pytorch":
        pass
