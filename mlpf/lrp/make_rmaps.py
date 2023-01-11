import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# this script makes Rmaps from a processed list of R_tensors

label_to_class = {
    0: "null",
    1: "chhadron",
    2: "nhadron",
    3: "photon",
    4: "electron",
    5: "muon",
}

label_to_p4 = {
    6: "charge",
    7: "pt",
    8: "eta",
    9: "sin phi",
    10: "cos phi",
    11: "energy",
}


def indexing_by_relevance(num, pid):
    labels = []
    labels.append(pid.capitalize())
    for i in range(num - 1):
        if i == 0:
            labels.append("Most relevant neighbor")
        elif i == 1:
            labels.append("2nd most relevant neighbor")
        elif i == 2:
            labels.append("3rd most relevant neighbor")
        else:
            labels.append(str(i + 1) + "th most relevant neighbor")
    return labels


def process_Rtensor(node, Rtensor, neighbors):
    """
    Given an Rtensor ~ (nodes, in_features) does some preprocessing on it

    Args
        node: an index for the node we're prcoessing the Rmap for
        Rtensor: the tensor/graph of Rscores for that node
        neighbors: # of neighbors to keep when processing the Rmap

    Returns
        an absolutized, normalized, and sorted Rtensor (sorted the
        rows/neighbors by relevance aside from the first row which
        is always the node itself)
    """
    Rtensor = Rtensor.absolute()
    Rtensor = Rtensor / Rtensor.sum()

    # put node itself as the first one
    tmp = Rtensor[0]
    Rtensor[0] = Rtensor[node]
    Rtensor[node] = tmp

    # rank all the others by relevance
    rank_relevance_msk = (
        Rtensor[1:].sum(axis=1).sort(descending=True)[1]
    )  # the index ":1" is to skip the node itself when sorting
    Rtensor[1:] = Rtensor[1:][rank_relevance_msk]

    # Rtensor[Rtensor.sum(axis=1).bool()]   # remove zero rows
    return Rtensor[: neighbors + 1]


def make_Rmaps(
    outpath,
    Rtensors,
    inputs,
    preds,
    pid="chhadron",
    neighbors=2,
    out_neuron=0,
):  # noqa C901
    """
    Recall each event has a corresponding Rmap per node in the event.
    This function process the Rmaps for a given pid.

    Args
        Rtensors: a list of len()=events processed. Each element is an Rtensor ~ (nodes, nodes, in_features)
        pid: class label to process (choices are ['null', 'chhadron', 'nhadron', photon', electron', muon'])
        neighbors: how many neighbors to show in the Rmap
    """
    in_features = Rtensors[0].shape[-1]

    Rtensor_correct, Rtensor_incorrect = torch.zeros(neighbors + 1, in_features), torch.zeros(neighbors + 1, in_features)
    num_Rtensors_correct, num_Rtensors_incorrect = 0, 0

    for event, event_Rscores in enumerate(Rtensors):
        for node, node_Rtensor in enumerate(event_Rscores):
            true_class = torch.argmax(inputs[event]["ygen_id"][node]).item()
            pred_class = torch.argmax(preds[event][node][:6]).item()

            # plot for a particular pid
            if label_to_class[true_class] == pid:
                # check if the node was correctly classified
                if pred_class == true_class:
                    Rtensor_correct = Rtensor_correct + process_Rtensor(node, node_Rtensor, neighbors)
                    num_Rtensors_correct = num_Rtensors_correct + 1
                else:
                    Rtensor_incorrect = Rtensor_incorrect + process_Rtensor(node, node_Rtensor, neighbors)
                    num_Rtensors_incorrect = num_Rtensors_incorrect + 1

    Rtensor_correct = Rtensor_correct / num_Rtensors_correct
    Rtensor_incorrect = Rtensor_incorrect / num_Rtensors_incorrect
    tot_num = num_Rtensors_correct + num_Rtensors_incorrect

    features = [
        "Track|cluster",
        "$p_{T}|E_{T}$",
        r"$\eta$",
        r"$\phi$",
        "P|E",
        r"$\eta_\mathrm{out}|E_{em}$",
        r"$\phi_\mathrm{out}|E_{had}$",
        "charge",
        "is_gen_mu",
        "is_gen_el",
    ]

    node_types = indexing_by_relevance(neighbors + 1, pid)  # only plot 6 rows/neighbors in Rmap

    for status, var in {
        "correct": Rtensor_correct,
        "incorrect": Rtensor_incorrect,
    }.items():
        print(f"Making Rmaps for {status}ly classified {pid}")
        if status == "correct":
            num = num_Rtensors_correct
        else:
            num = num_Rtensors_incorrect

        print(f"fraction is: {num}/{tot_num}")

        if num == 0:
            continue

        _, ax = plt.subplots(figsize=(20, 10))
        if out_neuron < 6:
            ax.set_title(
                f"Average relevance score matrix for {pid}s's classification score "
                + f"of {num}/{tot_num} {status}ly classified elements",
                fontsize=26,
            )
        else:
            ax.set_title(
                f"Average relevance score matrix for {pid}'s {label_to_p4[out_neuron]} "
                + f"of {num}/{tot_num} {status}ly classified elements",
                fontsize=26,
            )

        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(node_types)))
        ax.set_xticklabels(features, fontsize=22)
        ax.set_yticklabels(node_types, fontsize=20)
        for col in range(len(features)):
            for row in range(len(node_types)):
                ax.text(
                    col,
                    row,
                    round(var[row, col].item(), 5),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=14,
                )

        plt.imshow(
            (var[: neighbors + 1] + 1e-12).numpy(),
            cmap="copper",
            aspect="auto",
            norm=matplotlib.colors.LogNorm(vmin=1e-3),
        )

        # create directory to hold Rmaps
        rmap_dir = outpath + "/rmaps/"
        if not os.path.exists(rmap_dir):
            os.makedirs(rmap_dir)

        plt.savefig(f"{rmap_dir}/Rmap_{pid}_{status}_neuron_{out_neuron}.pdf")
