import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # for loading a pre-trained model
    parser.add_argument(
        "--load_VICReg",
        dest="load_VICReg",
        action="store_true",
        help="Load the model (no training)",
    )

    # for training mlpf
    parser.add_argument(
        "--train_mlpf",
        dest="train_mlpf",
        action="store_true",
        help="Train MLPF",
    )
    # for saving the model
    parser.add_argument(
        "--outpath", type=str, default="../experiments/", help="output folder"
    )
    parser.add_argument(
        "--model_prefix_VICReg",
        type=str,
        default="VICReg_model",
        help="directory to hold the VICReg model and all plots under args.outpath/args.model_prefix_VICReg",
    )
    parser.add_argument(
        "--model_prefix_mlpf",
        type=str,
        default="MLPF_model",
        help="directory to hold the mlpf model and all plots under args.outpath/args.model_prefix_VICReg/args.model_prefix_mlpf",
    )

    # for loading the data
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/clic/",
        help="dataset path",
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrites the model if True",
    )

    # for training hyperparameters
    parser.add_argument(
        "--n_epochs", type=int, default=3, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="number of events to run inference on before updating the loss",
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="patience before early stopping"
    )

    # VICReg encoder architecture
    parser.add_argument(
        "--width_encoder",
        type=int,
        default=126,
        help="hidden dimension of layer of the encoder",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=34, help="encoded element dimension"
    )
    parser.add_argument(
        "--num_convs", type=int, default=2, help="number of graph convolutions"
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=4,
        help="Spatial dimension for clustering in gravnet layer",
    )
    parser.add_argument(
        "--propagate_dim",
        type=int,
        default=22,
        help="The number of features to be propagated between the vertices",
    )
    parser.add_argument(
        "--nearest", type=int, default=8, help="k nearest neighbors in gravnet layer"
    )

    # VICReg decoder architecture
    parser.add_argument(
        "--width_decoder",
        type=int,
        default=126,
        help="hidden dimension of layers of the decoder/expander",
    )
    parser.add_argument(
        "--expand_dim",
        type=int,
        default=200,
        help="dimension of the output of the expander",
    )

    # MLPF architecture
    parser.add_argument(
        "--width_mlpf",
        type=int,
        default=126,
        help="hidden dimension of layers of mlpf",
    )
    args = parser.parse_args()

    return args
