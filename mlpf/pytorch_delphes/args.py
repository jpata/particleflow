import argparse
from math import inf


def parse_args():
    parser = argparse.ArgumentParser()

    # for saving the model
    parser.add_argument("--outpath",        type=str,           default='../data/test_tmp_delphes/experiments/',         help="output folder")

    # for loading the data
    parser.add_argument("--dataset",        type=str,           default='../data/test_tmp_delphes/data/pythia8_ttbar',   help="training dataset path")
    parser.add_argument("--dataset_qcd",    type=str,           default='../data/test_tmp_delphes/data/pythia8_qcd',     help="testing dataset path")
    parser.add_argument("--n_train",        type=int,           default=1,      help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_valid",        type=int,           default=1,      help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_test",         type=int,           default=1,      help="number of data files to use for testing.. each file contains 100 events")

    parser.add_argument("--title",          type=str,           default=None,   help="Appends this title to the model's name")
    parser.add_argument("--overwrite",      dest='overwrite',   action='store_true', help="Overwrites the model if True")

    # for loading a pre-trained model
    parser.add_argument("--load",           dest='load',        action='store_true', help="Load the model (no training)")
    parser.add_argument("--load_model",     type=str,           default="",     help="Which model to load")
    parser.add_argument("--load_epoch",     type=int,           default=0,      help="Which epoch of the model to load for evaluation")

    # for training hyperparameters
    parser.add_argument("--n_epochs",       type=int,           default=1,      help="number of training epochs")
    parser.add_argument("--batch_size",     type=int,           default=1,      help="Number of .pt files to load in parallel")
    parser.add_argument("--patience",       type=int,           default=100,    help="patience before early stopping")
    parser.add_argument("--target",         type=str,           default="gen",  choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", )
    parser.add_argument("--lr",             type=float,         default=1e-4,   help="learning rate")
    parser.add_argument("--alpha",          type=float,         default=2e-4,   help="loss = clf + alpha*reg.. if set to 0 model only does trains for classification")

    # for model architecture
    parser.add_argument("--hidden_dim1",    type=int,           default=120,    help="hidden dimension of layers before the graph convolutions")
    parser.add_argument("--hidden_dim2",    type=int,           default=256,     help="hidden dimension of layers after the graph convolutions")
    parser.add_argument("--embedding_dim",  type=int,           default=64,     help="encoded element dimension")
    parser.add_argument("--num_convs",      type=int,           default=2,      help="number of graph convolutions")
    parser.add_argument("--space_dim",      type=int,           default=4,      help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--propagate_dim",  type=int,           default=22,     help="The number of features to be propagated between the vertices")
    parser.add_argument("--nearest",        type=int,           default=16,     help="k nearest neighbors in gravnet layer")

    args = parser.parse_args()

    return args
