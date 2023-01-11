import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from torch_geometric.data import Data

from .delphes_plots import (
    draw_efficiency_fakerate,
    plot_confusion_matrix,
    plot_distributions_all,
    plot_distributions_pid,
    plot_particle_multiplicity,
    plot_reso,
)

matplotlib.use("Agg")


X_FEATURES_DELPHES = [
    "Track|cluster",
    "$p_{T}|E_{T}$",
    r"$\eta$",
    r"$Sin(\phi)$",
    r"$Cos(\phi)$",
    "P|E",
    r"$\eta_\mathrm{out}|E_{em}$",
    r"$Sin(\(phi)_\mathrm{out}|E_{had}$",
    r"$Cos(\phi)_\mathrm{out}|E_{had}$",
    "charge",
    "is_gen_mu",
    "is_gen_el",
]


def prepare_data_delphes(fn):

    """
    Takes as input a pkl file that contains the delphes raw information, and returns a list of PyG Data() objects.
    Each element of the list looks like this ~ Data(x=[#, 12], ygen=[#, 6], ygen_id=[#, 6], ycand=[#, 6], ycand_id=[#, 6])

    Args
        raw_file_name: raw parquet data file.
    Returns
        list of Data() objects.
    """

    with open(fn, "rb") as fi:
        data = pickle.load(fi, encoding="iso-8859-1")

    batched_data = []
    for i in range(len(data["X"])):
        # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
        d = Data(
            x=torch.tensor(data["X"][i], dtype=torch.float),
            ygen=torch.tensor(data["ygen"][i], dtype=torch.float)[:, 1:],
            ygen_id=torch.tensor(data["ygen"][i], dtype=torch.float)[
                :, 0
            ].long(),
            ycand=torch.tensor(data["ycand"][i], dtype=torch.float)[:, 1:],
            ycand_id=torch.tensor(data["ycand"][i], dtype=torch.float)[
                :, 0
            ].long(),
        )

        batched_data.append(d)
    return batched_data
