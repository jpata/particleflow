import json
import os
import os.path as osp
import pickle as pkl
import shutil
import sys
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataListLoader, DataLoader

matplotlib.use("Agg")

# define input/output dimensions
CLUSTERS_X = 6
TRACKS_X = 11
COMMON_X = 11
NUM_CLASSES = 6
CLASS_NAMES_CLIC_LATEX = ["none", "chhad", "nhad", "$\gamma$", "$e^\pm$", "$\mu^\pm$"]

# function that takes an event~Batch() and splits it into two Batch() objects representing the tracks/clusters
def distinguish_PFelements(batch):

    track_id = 1
    cluster_id = 2

    tracks = Batch(
        x=batch.x[batch.x[:, 0] == track_id][
            :, 1:
        ].float(),  # remove the first input feature which is not needed anymore
        ygen=batch.ygen[batch.x[:, 0] == track_id],
        ygen_id=batch.ygen_id[batch.x[:, 0] == track_id],
        ycand=batch.ycand[batch.x[:, 0] == track_id],
        ycand_id=batch.ycand_id[batch.x[:, 0] == track_id],
        batch=batch.batch[batch.x[:, 0] == track_id],
    )
    clusters = Batch(
        x=batch.x[batch.x[:, 0] == cluster_id][:, 1:].float()[
            :, :CLUSTERS_X
        ],  # remove the first input feature which is not needed anymore
        ygen=batch.ygen[batch.x[:, 0] == cluster_id],
        ygen_id=batch.ygen_id[batch.x[:, 0] == cluster_id],
        ycand=batch.ycand[batch.x[:, 0] == cluster_id],
        ycand_id=batch.ycand_id[batch.x[:, 0] == cluster_id],
        batch=batch.batch[batch.x[:, 0] == cluster_id],
    )

    return tracks, clusters


# conversly, function that combines the learned latent representations back into one Batch() object
def combine_PFelements(tracks, clusters):

    #     zero padding
    #     clusters.x = torch.cat([clusters.x, torch.from_numpy(np.zeros([clusters.x.shape[0],TRACKS_X-CLUSTERS_X]))], axis=1)

    event = Batch(
        x=torch.cat([tracks.x, clusters.x]),
        ygen=torch.cat([tracks.ygen, clusters.ygen]),
        ygen_id=torch.cat([tracks.ygen_id, clusters.ygen_id]),
        ycand=torch.cat([tracks.ycand, clusters.ycand]),
        ycand_id=torch.cat([tracks.ycand_id, clusters.ycand_id]),
        batch=torch.cat([tracks.batch, clusters.batch]),
    )

    return event


def load_VICReg(device, outpath):

    encoder_state_dict = torch.load(
        f"{outpath}/encoder_best_epoch_weights.pth", map_location=device
    )
    decoder_state_dict = torch.load(
        f"{outpath}/decoder_best_epoch_weights.pth", map_location=device
    )

    print("Loading a previously trained model..")
    with open(f"{outpath}/encoder_model_kwargs.pkl", "rb") as f:
        encoder_model_kwargs = pkl.load(f)
    with open(f"{outpath}/decoder_model_kwargs.pkl", "rb") as f:
        decoder_model_kwargs = pkl.load(f)

    return (
        encoder_state_dict,
        encoder_model_kwargs,
        decoder_state_dict,
        decoder_model_kwargs,
    )


def save_VICReg(args, outpath, encoder_model_kwargs, decoder_model_kwargs):

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        if not args.overwrite:  # if not overwrite then exit
            print("model already exists, please delete it")
            sys.exit(0)

        print("model already exists, deleting it")

        filelist = [
            f for f in os.listdir(outpath) if not f.endswith(".txt")
        ]  # don't remove the newly created logs.txt
        for f in filelist:
            try:
                shutil.rmtree(os.path.join(outpath, f))
            except:
                os.remove(os.path.join(outpath, f))

    with open(
        f"{outpath}/encoder_model_kwargs.pkl", "wb"
    ) as f:  # dump model architecture
        pkl.dump(encoder_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(
        f"{outpath}/decoder_model_kwargs.pkl", "wb"
    ) as f:  # dump model architecture
        pkl.dump(decoder_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "width": args.width_encoder,
                "embedding_dim": args.embedding_dim,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "input_dim": args.embedding_dim,
                "width": args.width_decoder,
                "output_dim": args.expand_dim,
            },
            fp,
        )


def save_MLPF(args, outpath, mlpf_model_kwargs):

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        filelist = [
            f for f in os.listdir(outpath) if not f.endswith(".txt")
        ]  # don't remove the newly created logs.txt
        for f in filelist:
            try:
                shutil.rmtree(os.path.join(outpath, f))
            except:
                os.remove(os.path.join(outpath, f))

    with open(f"{outpath}/mlpf_model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(mlpf_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "width": args.width_mlpf,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
            },
            fp,
        )
