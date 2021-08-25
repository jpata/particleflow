import argparse
from math import inf

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='../test_tmp_delphes/data/pythia8_ttbar', help="training dataset path", required=True)
    parser.add_argument("--dataset_qcd", type=str, default='../test_tmp_delphes/data/pythia8_qcd', help="testing dataset path", required=True)
    parser.add_argument("--outpath", type=str, default = '../test_tmp_delphes/experiments/', help="output folder", required=True)
    parser.add_argument("--title", type=str, default='', help="Appends this title to the model's name")

    parser.add_argument("--overwrite", action=BoolArg, default=False, help="Overwrites the model if True")
    parser.add_argument("--optimized", action=BoolArg, default=False, help="Uses the optimized version of knn")

    # for training
    parser.add_argument("--train", action=BoolArg, default=True, help="Trains the model")
    parser.add_argument("--n_train", type=int, default=3, help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_valid", type=int, default=1, help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_test", type=int, default=2, help="number of data files to use for testing.. each file contains 100 events")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")

    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--hidden_dim_nn1", type=int, default=64, help="hidden dimension")
    parser.add_argument("--input_encoding", type=int, default=12, help="use an input encoding layer")
    parser.add_argument("--encoding_dim", type=int, default=64, help="encoded element dimension")
    parser.add_argument("--space_dim", type=int, default=4, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--propagate_dimensions", type=int, default=22, help="The number of features to be propagated between the vertices")
    parser.add_argument("--nearest", type=int, default=16, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--nn1", action=BoolArg, default=True, help="Adds an encoder/decoder step before gravnet..")
    parser.add_argument("--nn3", action=BoolArg, default=True, help="Adds the network to regress p4..")
    parser.add_argument("--nn4", action=BoolArg, default=True, help="Adds an extra network for the dnn model..")

    parser.add_argument("--patience", type=int, default=100, help="patience before early stopping")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    parser.add_argument("--optimizer", type=str, default='adam', choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=2e-4, help="Loss multiplier for pdg-id classification.. recall: loss = clf + alpha*reg")

    parser.add_argument("--classification_only", action=BoolArg, default=False, help="Check to train for classification only (no regression)")
    parser.add_argument("--regression_only", action=BoolArg, default=False, help="Check to train for regression only (no classification)")

    # for loading a pre-trained model
    parser.add_argument("--load", action=BoolArg, default=False, help="Load the model (no training)")
    parser.add_argument("--load_model", type=str, help="Which model to load", default="/PFNet7_cand_ntrain_2")
    parser.add_argument("--load_epoch", type=int, default=0, help="Which epoch of the model to load for evaluation")

    # for evaluation: making predictions & making plots
    parser.add_argument("--make_predictions_train", action=BoolArg, default=False, help="make predictions on training data..")
    parser.add_argument("--make_predictions_valid", action=BoolArg, default=False, help="make predictions on validation data..")
    parser.add_argument("--make_predictions_test", action=BoolArg, default=True, help="make predictions on testing data..")
    parser.add_argument("--make_plots_train", action=BoolArg, default=False, help="make plots on training data..")
    parser.add_argument("--make_plots_valid", action=BoolArg, default=False, help="make plots on validation data..")
    parser.add_argument("--make_plots_test", action=BoolArg, default=True, help="make plots on testing data..")

    args = parser.parse_args()

    return args


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
