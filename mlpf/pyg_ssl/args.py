import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outpath", type=str, default="../experiments/", help="output folder")
    parser.add_argument(
        "--data_split_mode", type=str, default="mix", help="choices are 'quick', 'domain_adaptation', 'mix'."
    )

    # samples to be used
    parser.add_argument("--samples", default=-1, help="specefies samples to use")

    # directory containing datafiles
    parser.add_argument("--dataset", type=str, default="../data/clic_edm4hep/", help="dataset path")

    # flag to load a pre-trained model
    parser.add_argument("--load_VICReg", dest="load_VICReg", action="store_true", help="loads the model without training")

    # flag to train mlpf
    parser.add_argument("--train_mlpf", dest="train_mlpf", action="store_true", help="Train MLPF")
    parser.add_argument("--ssl", dest="ssl", action="store_true", help="Train ssl-based MLPF")
    parser.add_argument("--native", dest="native", action="store_true", help="Train native")

    parser.add_argument("--prefix_VICReg", type=str, default="VICReg_model", help="directory to hold the VICReg model")
    parser.add_argument("--prefix_mlpf", type=str, default="MLPF_model", help="directory to hold the mlpf model")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="overwrites the model if True")

    # training hyperparameters
    parser.add_argument("--lmbd", type=float, default=0.01, help="the lambda term in the VICReg loss")
    parser.add_argument("--u", type=float, default=0.1, help="the mu term in the VICReg loss")
    parser.add_argument("--v", type=float, default=0.1, help="the nu term in the VICReg loss")
    parser.add_argument("--n_epochs_mlpf", type=int, default=3, help="number of training epochs for mlpf")
    parser.add_argument("--n_epochs_VICReg", type=int, default=3, help="number of training epochs for VICReg")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size_mlpf", type=int, default=500, help="number of events to process at once")
    parser.add_argument("--batch_size_VICReg", type=int, default=10000, help="number of events to process at once")
    parser.add_argument("--patience", type=int, default=30, help="patience before early stopping")
    parser.add_argument(
        "--FineTune_VICReg", dest="FineTune_VICReg", action="store_true", help="FineTune VICReg during MLPFtraining"
    )

    # VICReg encoder architecture
    parser.add_argument("--width_encoder", type=int, default=128, help="hidden dimension of the encoder")
    parser.add_argument("--embedding_dim_VICReg", type=int, default=34, help="encoded element dimension")

    # VICReg decoder architecture
    parser.add_argument("--width_decoder", type=int, default=128, help="hidden dimension of the decoder")
    parser.add_argument("--expand_dim", type=int, default=256, help="dimension of the output of the decoder")

    # MLPF architecture
    parser.add_argument("--width_mlpf", type=int, default=256, help="hidden dimension of mlpf")
    parser.add_argument("--embedding_dim_mlpf", type=int, default=128, help="first embedding of mlpf")

    # shared architecture
    parser.add_argument("--num_convs", type=int, default=3, help="number of graph convolutions")
    parser.add_argument("--space_dim", type=int, default=4, help="Gravnet hyperparameter")
    parser.add_argument("--propagate_dim", type=int, default=22, help="Gravnet hyperparameter")
    parser.add_argument("--nearest", type=int, default=8, help="k nearest neighbors")

    args = parser.parse_args()

    return args
