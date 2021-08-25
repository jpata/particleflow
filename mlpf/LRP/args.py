import argparse
from math import inf

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--LRP_dataset_qcd", type=str, default='../test_tmp_delphes/data/pythia8_qcd', help="dataset path", required=True)
    parser.add_argument("--LRP_outpath", type=str, default = '../test_tmp_delphes/experiments/LRP/', help="Output folder for the LRP relevance scores and heatmaps", required=True)

    # usual specs
    parser.add_argument("--n_test", type=int, default=2, help="number of data files to use for testing LRP.. each file contains 100 events")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")

    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--hidden_dim_nn1", type=int, default=64, help="hidden dimension")
    parser.add_argument("--input_encoding", type=int, default=12, help="use an input encoding layer")
    parser.add_argument("--encoding_dim", type=int, default=64, help="encoded element dimension")
    parser.add_argument("--space_dim", type=int, default=4, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--propagate_dimensions", type=int, default=22, help="The number of features to be propagated between the vertices")
    parser.add_argument("--nearest", type=int, default=16, help="k nearest neighbors in gravnet layer")

    # extras for LRP
    parser.add_argument("--explain", action=BoolArg, default=True, help="General setup mode: if True then you want to explain.. if False then you will load an already explained model (already made R-scores)..")
    parser.add_argument("--LRP_reg", action=BoolArg, default=True, help="Works only if --explain is True.. Runs LRP for interpreting the regression part..")
    parser.add_argument("--LRP_clf", action=BoolArg, default=True, help="Works only if --explain is True.. Runs LRP for interpreting the classification part..")

    parser.add_argument("--LRP_load_model", type=str, default="/PFNet7_cand_ntrain_2", help="Loads the model to explain (or only make_heatmaps)", required=False)
    parser.add_argument("--LRP_load_epoch", type=int, default=0, help="Loads the epoch after which to explain (or only make_heatmaps)")

    parser.add_argument("--make_heatmaps_reg", action=BoolArg, default=True, help="Constructs heatmaps for the regressed p4 (must run with explain=True or else you load a pre-explained model with explain=False)..")
    parser.add_argument("--make_heatmaps_clf", action=BoolArg, default=True, help="Constructs heatmaps for the classified pid (must run with explain=True or else you load a pre-explained model with explain=False)..")

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
