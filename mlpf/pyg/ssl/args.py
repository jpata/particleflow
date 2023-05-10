import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--penalize_NCH", dest="penalize_NCH", action="store_true", help="penalize null charged hadron predictions"
    )

    parser.add_argument("--outpath", type=str, default="../experiments/", help="output folder")
    parser.add_argument("--data_split_mode", type=str, default="mix", help="choices: ['quick', 'domain_adaptation', 'mix']")

    # samples to be used
    parser.add_argument("--samples", default=-1, help="specifies samples to use")

    # directory containing datafiles
    parser.add_argument("--dataset", type=str, default="CLIC", help="currently only CLIC is supported")
    parser.add_argument("--data_path", type=str, default="../data/", help="path which contains the CLIC samples")
    parser.add_argument("--n_train", type=int, default=-1, help="number of files to use for training")
    parser.add_argument("--n_valid", type=int, default=-1, help="number of data files to use for validation")
    parser.add_argument("--n_test", type=int, default=-1, help="number of data files to use for testing")

    # flag to load a pre-trained model
    parser.add_argument("--load_VICReg", dest="load_VICReg", action="store_true", help="loads the model without training")

    # flag to train mlpf
    parser.add_argument("--train_mlpf", dest="train_mlpf", action="store_true", help="Train MLPF")
    parser.add_argument("--ssl", dest="ssl", action="store_true", help="Train ssl-based MLPF")
    parser.add_argument("--native", dest="native", action="store_true", help="Train native")

    parser.add_argument("--prefix_VICReg", type=str, default=None, help="directory to hold the VICReg model")
    parser.add_argument("--prefix", type=str, default="MLPF_model", help="directory to hold the mlpf model")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="overwrites the model if True")

    # training hyperparameters
    parser.add_argument("--alpha", type=float, default=-1, help="hyperparameter for null reconstruction")

    parser.add_argument("--lmbd", type=float, default=1, help="the lambda term in the VICReg loss")
    parser.add_argument("--mu", type=float, default=0.1, help="the mu term in the VICReg loss")
    parser.add_argument("--nu", type=float, default=1e-9, help="the nu term in the VICReg loss")
    parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs for mlpf")
    parser.add_argument("--n_epochs_VICReg", type=int, default=3, help="number of training epochs for VICReg")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--bs", type=int, default=500, help="number of events to process at once")
    parser.add_argument("--bs_VICReg", type=int, default=2000, help="number of events to process at once")
    parser.add_argument("--patience", type=int, default=50, help="patience before early stopping")

    # VICReg encoder architecture
    parser.add_argument("--width_encoder", type=int, default=256, help="hidden dimension of the encoder")
    parser.add_argument("--embedding_dim_VICReg", type=int, default=256, help="encoded element dimension")
    parser.add_argument("--num_convs_VICReg", type=int, default=3, help="number of graph convolutions")

    # VICReg decoder architecture
    parser.add_argument("--width_decoder", type=int, default=256, help="hidden dimension of the decoder")
    parser.add_argument("--expand_dim", type=int, default=512, help="dimension of the output of the decoder")

    # MLPF architecture
    parser.add_argument("--width", type=int, default=256, help="hidden dimension of mlpf")
    parser.add_argument("--embedding_dim", type=int, default=256, help="first embedding of mlpf")
    parser.add_argument("--num_convs", type=int, default=3, help="number of graph layers for mlpf")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout for MLPF model")

    # shared architecture
    parser.add_argument("--space_dim", type=int, default=4, help="Gravnet hyperparameter")
    parser.add_argument("--propagate_dim", type=int, default=22, help="Gravnet hyperparameter")
    parser.add_argument("--nearest", type=int, default=32, help="k nearest neighbors")

    args = parser.parse_args()

    return args
