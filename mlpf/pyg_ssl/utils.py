import glob
import json
import os
import os.path as osp
import pickle as pkl
import random
import shutil
import sys

import matplotlib
import torch
from torch_geometric.data import Batch

matplotlib.use("Agg")

# define input dimensions
X_FEATURES_TRK = [
    "type",
    "pt",
    "eta",
    "phi",
    "p",
    "chi2",
    "ndf",
    "dEdx",
    "dEdxError",
    "radiusOfInnermostHit",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
]
X_FEATURES_CL = [
    "type",
    "et",
    "eta",
    "phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "iTheta",
    "energy_ecal",
    "energy_hcal",
    "energy_other",
    "num_hits",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]

CLUSTERS_X = len(X_FEATURES_CL) - 1  # remove the `type` feature
TRACKS_X = len(X_FEATURES_TRK) - 1  # remove the `type` feature

# define regression output
Y_FEATURES = ["PDG", "charge", "pt", "eta", "phi", "energy"]

# define classification output
CLASS_NAMES_CLIC_LATEX = [
    "none",
    "chhad",
    "nhad",
    r"$\gamma$",
    r"$e^\pm$",
    r"$\mu^\pm$",
]
NUM_CLASSES = len(CLASS_NAMES_CLIC_LATEX)


# function that takes an event~Batch() and splits it into two Batch() objects representing the tracks/clusters
def distinguish_PFelements(batch):

    track_id = 1
    cluster_id = 2

    tracks = Batch(
        x=batch.x[batch.x[:, 0] == track_id][:, 1:].float()[
            :, :TRACKS_X
        ],  # remove the first input feature which is not needed anymore
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

    encoder_state_dict = torch.load(f"{outpath}/encoder_best_epoch_weights.pth", map_location=device)
    decoder_state_dict = torch.load(f"{outpath}/decoder_best_epoch_weights.pth", map_location=device)

    print("Loading a previously trained model..")
    with open(f"{outpath}/encoder_model_kwargs.pkl", "rb") as f:
        encoder_model_kwargs = pkl.load(f)
    with open(f"{outpath}/decoder_model_kwargs.pkl", "rb") as f:
        decoder_model_kwargs = pkl.load(f)

    return encoder_state_dict, encoder_model_kwargs, decoder_state_dict, decoder_model_kwargs


def save_VICReg(args, outpath, encoder, encoder_model_kwargs, decoder, decoder_model_kwargs):

    num_encoder_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    num_decoder_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print(f"Num of 'encoder' parameters: {num_encoder_parameters}")
    print(f"Num of 'decoder' parameters: {num_decoder_parameters}")

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        if not args.overwrite:  # if not overwrite then exit
            print("model already exists, please delete it")
            sys.exit(0)

        print("model already exists, deleting it")

        filelist = [f for f in os.listdir(outpath) if not f.endswith(".txt")]  # don't remove the newly created logs.txt
        for f in filelist:
            try:
                shutil.rmtree(os.path.join(outpath, f))
            except Exception:
                os.remove(os.path.join(outpath, f))

    with open(f"{outpath}/encoder_model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(encoder_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f"{outpath}/decoder_model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(decoder_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "data_split_mode": args.data_split_mode,
                "n_epochs": args.n_epochs_VICReg,
                "lr": args.lr,
                "batch_size_VICReg": args.batch_size_VICReg,
                "width_encoder": args.width_encoder,
                "embedding_dim": args.embedding_dim,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "input_dim": args.embedding_dim,
                "width_decoder": args.width_decoder,
                "output_dim": args.expand_dim,
                "lmbd": args.lmbd,
                "u": args.u,
                "v": args.v,
                "num_encoder_parameters": num_encoder_parameters,
                "num_decoder_parameters": num_decoder_parameters,
            },
            fp,
        )


def save_MLPF(args, outpath, mlpf, mlpf_model_kwargs, mode):
    """
    Saves the mlpf model in the `outpath` provided.
    Dumps the hyperparameters of the mlpf model in a json file.

    Args
        mode: choices are "ssl" or "native"
    """

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)
    print(f"Num of '{mode}-mlpf' parameters: {num_mlpf_parameters}")

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        filelist = [f for f in os.listdir(outpath) if not f.endswith(".txt")]  # don't remove the newly created logs.txt

        for f in filelist:
            try:
                shutil.rmtree(os.path.join(outpath, f))
            except Exception:
                os.remove(os.path.join(outpath, f))

    with open(f"{outpath}/mlpf_model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(mlpf_model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "data_split_mode": args.data_split_mode,
                "n_epochs": args.n_epochs_mlpf,
                "lr": args.lr,
                "batch_size_mlpf": args.batch_size_mlpf,
                "width": args.width_mlpf,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "mode": mode,
                "FineTune_VICReg": args.FineTune_VICReg,
                "num_mlpf_parameters": num_mlpf_parameters,
            },
            fp,
        )


def data_split(dataset, data_split_mode):
    """
    Depending on the data split mode chosen, the function returns different data splits.

    Choices for data_split_mode
        1. `quick`: uses only 1 datafile for quick debugging. Nothing interesting there.
        2. `domain_adaptation`: uses the following split schema.
            - Sample1: VicReg training domain, "data", e.g. 80% of QCD events
            - Sample2: supervised training domain, "MC", e.g. 80% of ttbar events
            - Sample3: validation in the supervised training domain: 20% of ttbar events
            - Sample4, validation in the other domain: 20% of QCD events
        3. `mix`: uses a mix of all samples.

    Returns (each as a list)
        data_train_VICReg, data_valid_VICReg, data_train_mlpf, data_valid_mlpf

    """
    print(f"Will use data split mode `{data_split_mode}`.")

    if data_split_mode == "quick":
        data = torch.load(f"{dataset}/p8_ee_ZH_Htautau_ecm380/processed/data_0.pt")
        data_train_VICReg = data[: round(0.8 * len(data))]
        data_valid_VICReg = data[: round(0.8 * len(data))]
        data_train_mlpf = data_train_VICReg
        data_valid_mlpf = data_valid_VICReg

    elif data_split_mode == "domain_adaptation":

        qcd_files = glob.glob(f"{dataset}/p8_ee_qcd_ecm365/processed/*")
        ttbar_files = glob.glob(f"{dataset}/p8_ee_tt_ecm365/processed/*")

        qcd_data = []
        for file in list(qcd_files):
            qcd_data += torch.load(f"{file}")

        ttbar_data = []
        for file in list(ttbar_files):
            ttbar_data += torch.load(f"{file}")

        data_train_VICReg = qcd_data[: round(0.8 * len(qcd_data))]
        data_valid_VICReg = qcd_data[round(0.8 * len(qcd_data)) :]
        data_train_mlpf = qcd_data[: round(0.8 * len(ttbar_data))]
        data_valid_mlpf = qcd_data[round(0.8 * len(ttbar_data)) :]

    elif data_split_mode == "mix":

        data = []
        for sample in ["p8_ee_qcd_ecm365", "p8_ee_tt_ecm365"]:

            files = glob.glob(f"{dataset}/{sample}/processed/*")
            data_per_sample = []
            for file in files:
                data_per_sample += torch.load(f"{file}")

            data += data_per_sample

        # shuffle datafiles belonging to different samples
        random.shuffle(data)

        data_VICReg = data[: round(0.9 * len(data))]
        data_mlpf = data[round(0.9 * len(data)) :]

        data_train_VICReg = data_VICReg[: round(0.8 * len(data_VICReg))]
        data_valid_VICReg = data_VICReg[round(0.8 * len(data_VICReg)) :]
        data_train_mlpf = data_mlpf[: round(0.8 * len(data_mlpf))]
        data_valid_mlpf = data_mlpf[round(0.8 * len(data_mlpf)) :]

    print(f"Will use {len(data_train_VICReg)} events to train VICReg")
    print(f"Will use {len(data_valid_VICReg)} events to validate VICReg")
    print(f"Will use {len(data_train_mlpf)} events to train MLPF")
    print(f"Will use {len(data_valid_mlpf)} events to validate MLPF")

    return data_train_VICReg, data_valid_VICReg, data_train_mlpf, data_valid_mlpf
