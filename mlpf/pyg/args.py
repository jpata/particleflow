import argparse
from math import inf


def parse_args():
    parser = argparse.ArgumentParser()

    # for data loading
    parser.add_argument("--num_workers",        type=int,  default=2,      help="number of subprocesses used for data loading")
    parser.add_argument("--prefetch_factor",    type=int,  default=4,      help="number of samples loaded in advance by each worker")

    # for saving the model
    parser.add_argument("--outpath",        type=str,           default='../experiments/',         help="output folder")
    parser.add_argument("--model_prefix",   type=str,           default='MLPF_model',   help="directory to hold the model and all plots under args.outpath/args.model_prefix")

    # for loading the data
    parser.add_argument("--data",           type=str,           required=True,   help="cms or delphes?")
    parser.add_argument("--dataset",        type=str,           default='../data/delphes/pythia8_ttbar',   help="training dataset path")
    parser.add_argument("--dataset_test",   type=str,           default='../data/delphes/pythia8_qcd',     help="testing dataset path")
    parser.add_argument("--sample",         type=str,           default='QCD',     help="sample to test on")
    parser.add_argument("--n_train",        type=int,           default=1,      help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_valid",        type=int,           default=1,      help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_test",         type=int,           default=1,      help="number of data files to use for testing.. each file contains 100 events")

    parser.add_argument("--overwrite",      dest='overwrite',   action='store_true', help="Overwrites the model if True")

    # for loading a pre-trained model
    parser.add_argument("--load",           dest='load',        action='store_true', help="Load the model (no training)")
    parser.add_argument("--load_epoch",     type=int,           default=-1,      help="Which epoch of the model to load for evaluation")

    # for training hyperparameters
    parser.add_argument("--n_epochs",       type=int,           default=3,      help="number of training epochs")
    parser.add_argument("--batch_size",     type=int,           default=1,      help="number of events to run inference on before updating the loss")
    parser.add_argument("--patience",       type=int,           default=100,    help="patience before early stopping")
    parser.add_argument("--target",         type=str,           default="gen",  choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", )
    parser.add_argument("--lr",             type=float,         default=1e-4,   help="learning rate")
    parser.add_argument("--alpha",          type=float,         default=2e-5,   help="loss = clf + alpha*reg.. if set to 0 model only does trains for classification")
    parser.add_argument("--batch_events",   dest='batch_events',        action='store_true', help="batches the event in eta,phi space to build the graphs")

    # for model architecture
    parser.add_argument("--hidden_dim1",    type=int,           default=126,    help="hidden dimension of layers before the graph convolutions")
    parser.add_argument("--hidden_dim2",    type=int,           default=256,    help="hidden dimension of layers after the graph convolutions")
    parser.add_argument("--embedding_dim",  type=int,           default=32,     help="encoded element dimension")
    parser.add_argument("--num_convs",      type=int,           default=3,      help="number of graph convolutions")
    parser.add_argument("--space_dim",      type=int,           default=4,      help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--propagate_dim",  type=int,           default=8,     help="The number of features to be propagated between the vertices")
    parser.add_argument("--nearest",        type=int,           default=4,     help="k nearest neighbors in gravnet layer")

    # for testing the model
    parser.add_argument("--make_predictions",   dest='make_predictions',        action='store_true', help="run inference on the test data")
    parser.add_argument("--make_plots",   dest='make_plots',        action='store_true', help="makes plots of the test predictions")

    args = parser.parse_args()

    return args
