import yaml
import argparse
import csv

import matplotlib.pyplot as plt

from mlpf.model.mlpf import MLPF
from mlpf.model.training import override_config
from mlpf.model.utils import (
    CLASS_LABELS,
    X_FEATURES,
    count_parameters,
    ELEM_TYPES_NONZERO,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default=None, help="yaml config")
parser.add_argument(
    "--attention-type",
    type=str,
    default=None,
    help="attention type for self-attention layer",
    choices=["math", "efficient", "flash"],
)
args = parser.parse_args()

with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
    config = yaml.safe_load(stream)


nconvs_width_list = [
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (2, 32),
    (2, 64),
    (2, 128),
    (2, 256),
    (4, 32),
    (4, 64),
    (4, 128),
    (4, 256),
]
summary = [["num_convs", "width", "Trainable parameters", "Non-trainable parameters", "Total parameters"]]

for nconvs, width in nconvs_width_list:
    args.num_convs = nconvs
    args.width = width
    args.embedding_dim = width
    args.test_datasets = []

    override_config(config, args)

    model_kwargs = {
        "input_dim": len(X_FEATURES[config["dataset"]]),
        "num_classes": len(CLASS_LABELS[config["dataset"]]),
        "input_encoding": config["model"]["input_encoding"],
        "pt_mode": config["model"]["pt_mode"],
        "feature_mean": None,  # Not needed for parameter counting
        "feature_std": None,  # Not needed for parameter counting
        "eta_mode": config["model"]["eta_mode"],
        "sin_phi_mode": config["model"]["sin_phi_mode"],
        "cos_phi_mode": config["model"]["cos_phi_mode"],
        "energy_mode": config["model"]["energy_mode"],
        "elemtypes_nonzero": ELEM_TYPES_NONZERO[config["dataset"]],
        "learned_representation_mode": config["model"]["learned_representation_mode"],
        **config["model"][config["conv_type"]],
    }
    model = MLPF(**model_kwargs)

    trainable_params, nontrainable_params, table = count_parameters(model)

    summary.append([nconvs, width, trainable_params, nontrainable_params, trainable_params + nontrainable_params])

    # print(table)

    print("Model conv type:", model.conv_type)
    print("conv_type HPs", config["model"][config["conv_type"]])
    print("Trainable parameters:", trainable_params)
    print("Non-trainable parameters:", nontrainable_params)
    print("Total parameters:", trainable_params + nontrainable_params)

# File path
file_path = "count_summary.csv"

# Writing to CSV file one row at a time
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(summary)

nconvs_width_array = [(summary[ii][0], summary[ii][1]) for ii in range(1, len(summary))]
total_array = [summary[ii][4] for ii in range(1, len(summary))]
print(total_array)
plt.figure()
plt.scatter(total_array, total_array)

for ii, label in enumerate(nconvs_width_array):
    plt.annotate(
        "{}, {}".format(label[0], label[1]),
        (total_array[ii], total_array[ii]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
    )

plt.yscale("log")
plt.xscale("log")
plt.savefig("count_plot.png")
