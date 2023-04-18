import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # for saving the model
    parser.add_argument("--outpath", type=str, default="../experiments/", help="output folder")
    parser.add_argument("--prefix", type=str, default="MLPF_model", help="directory to hold the model and all plots")

    # for loading the data
    parser.add_argument("--dataset", type=str, required=True, help="CMS or DELPHES?")
    parser.add_argument("--data_path", type=str, default="../data/", help="path which contains the samples")
    parser.add_argument("--sample", type=str, default="QCD", help="sample to test on")
    parser.add_argument("--n_train", type=int, default=2, help="number of files to use for training")
    parser.add_argument("--n_valid", type=int, default=2, help="number of data files to use for validation")
    parser.add_argument("--n_test", type=int, default=2, help="number of data files to use for testing")

    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrites the model if True")

    # for loading a pre-trained model
    parser.add_argument("--load", dest="load", action="store_true", help="Load the model (no training)")

    # for training hyperparameters
    parser.add_argument("--alpha", type=float, default=-1, help="hyperparameter for null reconstruction")

    parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--bs", type=int, default=100, help="training minibatch size in number of events")
    parser.add_argument("--patience", type=int, default=50, help="patience before early stopping")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # MLPF architecture
    parser.add_argument("--width", type=int, default=256, help="hidden dimension of mlpf")
    parser.add_argument("--embedding_dim", type=int, default=256, help="first embedding of mlpf")
    parser.add_argument("--num_convs", type=int, default=3, help="number of graph layers for mlpf")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout for MLPF model")
    parser.add_argument("--space_dim", type=int, default=4, help="Gravnet hyperparameter")
    parser.add_argument("--propagate_dim", type=int, default=22, help="Gravnet hyperparameter")
    parser.add_argument("--nearest", type=int, default=32, help="k nearest neighbors in gravnet layer")

    # for testing the model
    parser.add_argument(
        "--make_predictions", dest="make_predictions", action="store_true", help="run inference on the test data"
    )
    parser.add_argument("--make_plots", dest="make_plots", action="store_true", help="makes plots of the test predictions")

    args = parser.parse_args()

    return args
