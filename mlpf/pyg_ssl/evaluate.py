import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
from torch_geometric.nn import global_mean_pool

from .utils import (
    CLASS_NAMES_CLIC_LATEX,
    NUM_CLASSES,
    combine_PFelements,
    distinguish_PFelements,
)

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def evaluate(device, encoder, decoder, mlpf, test_loader):

    mlpf.eval()
    encoder.eval()
    decoder.eval()
    conf_matrix = np.zeros((6, 6))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"making predictions: {i+1}/{len(test_loader)}")
            # make transformation
            tracks, clusters = distinguish_PFelements(batch.to(device))

            ### ENCODE
            embedding_tracks, embedding_clusters = encoder(tracks, clusters)
            ### POOLING
            pooled_tracks = global_mean_pool(embedding_tracks, tracks.batch)
            pooled_clusters = global_mean_pool(
                embedding_clusters, clusters.batch
            )
            ### DECODE
            out_tracks, out_clusters = decoder(pooled_tracks, pooled_clusters)

            # use the learnt representation as your input as well as the global feature vector
            tracks.x = embedding_tracks
            clusters.x = embedding_clusters

            event = combine_PFelements(tracks, clusters)

            # make mlpf forward pass
            pred_ids_one_hot = mlpf(event.to(device))
            pred_ids = torch.argmax(pred_ids_one_hot, axis=1)
            target_ids = event.ygen_id

            conf_matrix += sklearn.metrics.confusion_matrix(
                target_ids.detach().cpu(),
                pred_ids.detach().cpu(),
                labels=range(NUM_CLASSES),
            )
    return conf_matrix


def plot_conf_matrix(cm, title, outpath):
    import itertools

    cmap = plt.get_cmap("Blues")
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    fig = plt.figure(figsize=(8, 6))

    ax = plt.axes()
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
    plt.title(title, fontsize=25)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)

    plt.xticks(
        range(len(CLASS_NAMES_CLIC_LATEX)),
        CLASS_NAMES_CLIC_LATEX,
        rotation=45,
        fontsize=15,
    )
    plt.yticks(
        range(len(CLASS_NAMES_CLIC_LATEX)), CLASS_NAMES_CLIC_LATEX, fontsize=15
    )

    plt.tight_layout()

    plt.savefig(f"{outpath}/conf_matrix_test.pdf")
