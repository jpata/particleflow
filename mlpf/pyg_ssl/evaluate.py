import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool

from .utils import CLASS_NAMES_CLIC_LATEX, NUM_CLASSES, combine_PFelements, distinguish_PFelements

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

CLASS_TO_ID = {
    "charged_hadron": 1,
    "neutral_hadron": 2,
    "photon": 3,
    "electron": 4,
    "muon": 5,
}


def evaluate(device, encoder, decoder, mlpf, batch_size_mlpf, mode, data, save_as, outpath):

    npred, ngen, ncand = {}, {}, {}
    for class_ in CLASS_TO_ID.keys():
        npred[class_], ngen[class_], ncand[class_] = [], [], []

    test_loader = torch_geometric.loader.DataLoader(data, batch_size_mlpf)

    mlpf.eval()
    encoder.eval()
    decoder.eval()
    conf_matrix = np.zeros((6, 6))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 500 == 0:
                print(f"making predictions: {i+1}/{len(test_loader)}")

            if mode == "ssl":
                # make transformation
                tracks, clusters = distinguish_PFelements(batch.to(device))

                # ENCODE
                embedding_tracks, embedding_clusters = encoder(tracks, clusters)
                # POOLING
                pooled_tracks = global_mean_pool(embedding_tracks, tracks.batch)
                pooled_clusters = global_mean_pool(embedding_clusters, clusters.batch)
                # DECODE
                out_tracks, out_clusters = decoder(pooled_tracks, pooled_clusters)

                # use the learnt representation as your input as well as the global feature vector
                tracks.x = embedding_tracks
                clusters.x = embedding_clusters

                event = combine_PFelements(tracks, clusters)

            elif mode == "native":
                event = batch

            # make mlpf forward pass
            pred_ids_one_hot = mlpf(event.to(device))

            pred_ids = torch.argmax(pred_ids_one_hot, axis=1)
            target_ids = event.ygen_id
            cand_ids = event.ycand_id

            conf_matrix += sklearn.metrics.confusion_matrix(
                target_ids.detach().cpu(),
                pred_ids.detach().cpu(),
                labels=range(NUM_CLASSES),
            )

            for class_, id_ in CLASS_TO_ID.items():
                npred[class_].append((pred_ids == id_).sum().item())
                ngen[class_].append((target_ids == id_).sum().item())
                ncand[class_].append((cand_ids == id_).sum().item())

        make_conf_matrix(conf_matrix, outpath, mode, save_as)
        make_multiplicity_plots(npred, ngen, ncand, outpath, mode, save_as)


def make_conf_matrix(cm, outpath, mode, save_as):
    import itertools

    cmap = plt.get_cmap("Blues")
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    plt.figure(figsize=(8, 6))
    plt.axes()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:0.2f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=15,
        )
    if mode == "ssl":
        plt.title(f"{mode} based MLPF", fontsize=25)
    else:
        plt.title(f"{mode} MLPF", fontsize=25)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)

    plt.xticks(
        range(len(CLASS_NAMES_CLIC_LATEX)),
        CLASS_NAMES_CLIC_LATEX,
        rotation=45,
        fontsize=15,
    )
    plt.yticks(range(len(CLASS_NAMES_CLIC_LATEX)), CLASS_NAMES_CLIC_LATEX, fontsize=15)

    plt.tight_layout()

    plt.savefig(f"{outpath}/conf_matrix_{mode}_{save_as}.pdf")
    with open(f"{outpath}/conf_matrix_{mode}_{save_as}.pkl", "wb") as f:
        pkl.dump(cm, f)
    plt.close()


def make_multiplicity_plots(npred, ngen, ncand, outpath, mode, save_as):
    for class_ in CLASS_TO_ID.keys():

        # Plot the particle multiplicities
        plt.figure()
        plt.axes()
        plt.scatter(ngen[class_], ncand[class_], marker=".", alpha=0.4, label="PF")
        plt.scatter(ngen[class_], npred[class_], marker=".", alpha=0.4, label="MLPF")
        a = 0.5 * min(np.min(npred[class_]), np.min(ngen[class_]))
        b = 1.5 * max(np.max(npred[class_]), np.max(ngen[class_]))
        # plt.xlim(a, b)
        # plt.ylim(a, b)
        plt.plot([a, b], [a, b], color="black", ls="--")
        plt.title(class_)
        plt.xlabel("number of truth particles")
        plt.ylabel("number of reconstructed particles")
        plt.legend(loc=4)
        plt.savefig(f"{outpath}/multiplicity_plots_{CLASS_TO_ID[class_]}_{mode}_{save_as}.pdf")
        plt.close()
