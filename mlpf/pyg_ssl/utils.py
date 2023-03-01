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

    print("Loading a previously trained model..")
    vicreg_state_dict = torch.load(f"{outpath}/VICReg_best_epoch_weights.pth", map_location=device)

    with open(f"{outpath}/encoder_model_kwargs.pkl", "rb") as f:
        encoder_model_kwargs = pkl.load(f)
    with open(f"{outpath}/decoder_model_kwargs.pkl", "rb") as f:
        decoder_model_kwargs = pkl.load(f)

    return vicreg_state_dict, encoder_model_kwargs, decoder_model_kwargs


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
                "bs_VICReg": args.bs_VICReg,
                "width_encoder": args.width_encoder,
                "embedding_dim": args.embedding_dim_VICReg,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "width_decoder": args.width_decoder,
                "output_dim": args.expand_dim,
                "lmbd": args.lmbd,
                "mu": args.mu,
                "nu": args.nu,
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
                "bs_mlpf": args.bs_mlpf,
                "width": args.width_mlpf,
                "embedding_dim": args.embedding_dim_mlpf,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "mode": mode,
                "num_mlpf_parameters": num_mlpf_parameters,
            },
            fp,
        )


def data_split(dataset, data_split_mode):
    """
    Depending on the data split mode chosen, the function returns different data splits.

    Choices for data_split_mode
        1. `quick`: uses only 1 datafile of each sample for quick debugging. Nothing interesting there.
        2. `domain_adaptation`: uses QCD samples to train/validate VICReg and TTbar samples to train/validate MLPF.
        3. `mix`: uses a mix of both QCD and TTbar samples to train/validate VICReg and MLPF.

    Returns (each as a list)
        data_VICReg_train, data_VICReg_valid, data_mlpf_train, data_mlpf_valid, data_test_qcd, data_test_ttbar

    """
    print(f"Will use data split mode `{data_split_mode}`")

    if data_split_mode == "quick":
        data_qcd = torch.load(f"{dataset}/p8_ee_qcd_ecm365/processed/data_0.pt")
        data_ttbar = torch.load(f"{dataset}/p8_ee_tt_ecm365/processed/data_0.pt")

        data_test_qcd = data_qcd[: round(0.1 * len(data_qcd))]
        data_test_ttbar = data_ttbar[: round(0.1 * len(data_ttbar))]

        # label remaining data as `rem`
        rem_qcd = data_qcd[round(0.1 * len(data_qcd)) :]
        rem_ttbar = data_ttbar[round(0.1 * len(data_qcd)) :]

        data_VICReg = rem_qcd[: round(0.8 * len(rem_qcd))] + rem_ttbar[: round(0.8 * len(rem_ttbar))]
        data_mlpf = rem_qcd[round(0.8 * len(rem_qcd)) :] + rem_ttbar[round(0.8 * len(rem_ttbar)) :]

        # shuffle the samples after mixing (not super necessary since the DataLoaders will shuffle anyway)
        random.shuffle(data_VICReg)
        random.shuffle(data_mlpf)

        data_VICReg_train = data_VICReg[: round(0.9 * len(data_VICReg))]
        data_VICReg_valid = data_VICReg[round(0.9 * len(data_VICReg)) :]

        data_mlpf_train = data_mlpf[: round(0.9 * len(data_mlpf))]
        data_mlpf_valid = data_mlpf[round(0.9 * len(data_mlpf)) :]

    else:  # actual meaningful data splits
        # load the qcd and ttbar samples seperately
        qcd_files = glob.glob(f"{dataset}/p8_ee_qcd_ecm365/processed/*")
        ttbar_files = glob.glob(f"{dataset}/p8_ee_tt_ecm365/processed/*")

        data_qcd = []
        for file in list(qcd_files):
            data_qcd += torch.load(f"{file}")

        data_ttbar = []
        for file in list(ttbar_files):
            data_ttbar += torch.load(f"{file}")

        # use 10% of each sample for testing
        frac_qcd_test = round(0.1 * len(data_qcd))
        frac_tt_test = round(0.1 * len(data_ttbar))
        data_test_qcd = data_qcd[:frac_qcd_test]
        data_test_ttbar = data_ttbar[:frac_tt_test]

        # label remaining data as `rem`
        rem_qcd = data_qcd[frac_qcd_test:]
        rem_ttbar = data_ttbar[frac_tt_test:]

        frac_qcd_train = round(0.8 * len(rem_qcd))
        frac_tt_train = round(0.8 * len(rem_ttbar))

        if data_split_mode == "domain_adaptation":
            """
            use remaining QCD samples for VICReg with an 80-20 train-val split.
            use remaining TTbar samples for MLPF with an 80-20 train-val split.
            """
            data_VICReg_train = rem_qcd[:frac_qcd_train]
            data_VICReg_valid = rem_qcd[frac_qcd_train:]

            data_mlpf_train = rem_ttbar[:frac_tt_train]
            data_mlpf_valid = rem_ttbar[frac_tt_train:]

        elif data_split_mode == "mix":
            """
            use (80% of QCD + 80% of remaining TTbar) samples for VICReg with a 90-10 train-val split.
            use (20% of QCD + 20% of remaining TTbar) samples for MLPF with a 90-10 train-val split.
            """
            data_VICReg = rem_qcd[:frac_qcd_train] + rem_ttbar[:frac_tt_train]
            data_mlpf = rem_qcd[frac_qcd_train:] + rem_ttbar[frac_tt_train:]

            # shuffle the samples after mixing (not super necessary since the DataLoaders will shuffle anyway)
            random.shuffle(data_VICReg)
            random.shuffle(data_mlpf)

            frac_VICReg_train = round(0.9 * len(data_VICReg))
            data_VICReg_train = data_VICReg[:frac_VICReg_train]
            data_VICReg_valid = data_VICReg[frac_VICReg_train:]

            frac_mlpf_train = round(0.9 * len(data_mlpf))
            data_mlpf_train = data_mlpf[:frac_mlpf_train]
            data_mlpf_valid = data_mlpf[frac_mlpf_train:]

    print(f"Will use {len(data_VICReg_train)} events to train VICReg")
    print(f"Will use {len(data_VICReg_valid)} events to validate VICReg")
    print(f"Will use {len(data_mlpf_train)} events to train MLPF")
    print(f"Will use {len(data_mlpf_valid)} events to validate MLPF")

    return data_VICReg_train, data_VICReg_valid, data_mlpf_train, data_mlpf_valid, data_test_qcd, data_test_ttbar
