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


#
def distinguish_PFelements(event):
    """
    Function that takes an event~Batch() and splits it into two Batch() objects representing the tracks/clusters.
    In case of multigpu trainings, the event will be a list of Batch() objects, and must be handled differently.
    """

    track_id = 1
    cluster_id = 2

    if isinstance(event, list):  # for multigpu instances
        # tracks, clusters = [], []
        out = []
        for ev in event:
            # tracks.append(
            #     Batch(
            #         x=ev.x[ev.x[:, 0] == track_id][:, 1:].float()[
            #             :, :TRACKS_X
            #         ],  # remove the first input feature which is not needed anymore
            #         ygen=ev.ygen[ev.x[:, 0] == track_id],
            #         ygen_id=ev.ygen_id[ev.x[:, 0] == track_id],
            #         ycand=ev.ycand[ev.x[:, 0] == track_id],
            #         ycand_id=ev.ycand_id[ev.x[:, 0] == track_id],
            #     )
            # )
            # clusters.append(
            #     Batch(
            #         x=ev.x[ev.x[:, 0] == cluster_id][:, 1:].float()[
            #             :, :CLUSTERS_X
            #         ],  # remove the first input feature which is not needed anymore
            #         ygen=ev.ygen[ev.x[:, 0] == cluster_id],
            #         ygen_id=ev.ygen_id[ev.x[:, 0] == cluster_id],
            #         ycand=ev.ycand[ev.x[:, 0] == cluster_id],
            #         ycand_id=ev.ycand_id[ev.x[:, 0] == cluster_id],
            #     )
            # )
            out.append(
                Batch(
                    tracks_x=ev.x[ev.x[:, 0] == track_id][:, 1:].float()[
                        :, :TRACKS_X
                    ],  # remove the first input feature which is not needed anymore
                    tracks_ygen=ev.ygen[ev.x[:, 0] == track_id],
                    tracks_ygen_id=ev.ygen_id[ev.x[:, 0] == track_id],
                    tracks_ycand=ev.ycand[ev.x[:, 0] == track_id],
                    tracks_ycand_id=ev.ycand_id[ev.x[:, 0] == track_id],
                    clusters_x=ev.x[ev.x[:, 0] == cluster_id][:, 1:].float()[
                        :, :CLUSTERS_X
                    ],  # remove the first input feature which is not needed anymore
                    clusters_ygen=ev.ygen[ev.x[:, 0] == cluster_id],
                    clusters_ygen_id=ev.ygen_id[ev.x[:, 0] == cluster_id],
                    clusters_ycand=ev.ycand[ev.x[:, 0] == cluster_id],
                    clusters_ycand_id=ev.ycand_id[ev.x[:, 0] == cluster_id],
                )
            )
    else:
        # tracks = Batch(
        #     x=event.x[event.x[:, 0] == track_id][:, 1:].float()[
        #         :, :TRACKS_X
        #     ],  # remove the first input feature which is not needed anymore
        #     ygen=event.ygen[event.x[:, 0] == track_id],
        #     ygen_id=event.ygen_id[event.x[:, 0] == track_id],
        #     ycand=event.ycand[event.x[:, 0] == track_id],
        #     ycand_id=event.ycand_id[event.x[:, 0] == track_id],
        #     batch=event.batch[event.x[:, 0] == track_id],
        # )
        # clusters = Batch(
        #     x=event.x[event.x[:, 0] == cluster_id][:, 1:].float()[
        #         :, :CLUSTERS_X
        #     ],  # remove the first input feature which is not needed anymore
        #     ygen=event.ygen[event.x[:, 0] == cluster_id],
        #     ygen_id=event.ygen_id[event.x[:, 0] == cluster_id],
        #     ycand=event.ycand[event.x[:, 0] == cluster_id],
        #     ycand_id=event.ycand_id[event.x[:, 0] == cluster_id],
        #     batch=event.batch[event.x[:, 0] == cluster_id],
        # )

        out = Batch(
            tracks_x=event.x[event.x[:, 0] == track_id][:, 1:].float()[
                :, :TRACKS_X
            ],  # remove the first input feature which is not needed anymore
            tracks_ygen=event.ygen[event.x[:, 0] == track_id],
            tracks_ygen_id=event.ygen_id[event.x[:, 0] == track_id],
            tracks_ycand=event.ycand[event.x[:, 0] == track_id],
            tracks_ycand_id=event.ycand_id[event.x[:, 0] == track_id],
            tracks_batch=event.batch[event.x[:, 0] == track_id],
            clusters_x=event.x[event.x[:, 0] == cluster_id][:, 1:].float()[
                :, :CLUSTERS_X
            ],  # remove the first input feature which is not needed anymore
            clusters_ygen=event.ygen[event.x[:, 0] == cluster_id],
            clusters_ygen_id=event.ygen_id[event.x[:, 0] == cluster_id],
            clusters_ycand=event.ycand[event.x[:, 0] == cluster_id],
            clusters_ycand_id=event.ycand_id[event.x[:, 0] == cluster_id],
            clusters_batch=event.batch[event.x[:, 0] == cluster_id],
        )

    return out


# conversly, function that combines the learned latent representations back into one Batch() object
def combine_PFelements(tracks, clusters):
    """
    Function that takes tracks~Batch() and clusters~Batch() and combines them into a single event~Batch().
    In case of multigpu trainings, the tracks and clusters will each be a list of Batch() objects and must be iterated over.
    """

    if isinstance(tracks, list):  # for multigpu instances
        event = []
        for i in range(len(tracks)):
            event.append(
                Batch(
                    x=torch.cat([tracks[i].x, clusters[i].x]),
                    ygen=torch.cat([tracks[i].ygen, clusters[i].ygen]),
                    ygen_id=torch.cat([tracks[i].ygen_id, clusters[i].ygen_id]),
                    ycand=torch.cat([tracks[i].ycand, clusters[i].ycand]),
                    ycand_id=torch.cat([tracks[i].ycand_id, clusters[i].ycand_id]),
                )
            )

    else:
        event = Batch(
            x=torch.cat([tracks.x, clusters.x]),
            ygen=torch.cat([tracks.ygen, clusters.ygen]),
            ygen_id=torch.cat([tracks.ygen_id, clusters.ygen_id]),
            ycand=torch.cat([tracks.ycand, clusters.ycand]),
            ycand_id=torch.cat([tracks.ycand_id, clusters.ycand_id]),
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
                "embedding_dim": args.embedding_dim_VICReg,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
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
                "embedding_dim": args.embedding_dim_mlpf,
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

        data_VICReg_train = data_test_qcd + data_test_ttbar
        data_VICReg_valid = data_test_qcd + data_test_ttbar

        data_mlpf_train = data_test_qcd + data_test_ttbar
        data_mlpf_valid = data_test_qcd + data_test_ttbar

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
        data_test_qcd = data_qcd[: round(0.1 * len(data_qcd))]
        data_test_ttbar = data_ttbar[: round(0.1 * len(data_ttbar))]

        # label remaining data as `rem`
        rem_qcd = data_qcd[round(0.1 * len(data_qcd)) :]
        rem_ttbar = data_ttbar[round(0.1 * len(data_qcd)) :]

        if data_split_mode == "domain_adaptation":
            """
            use QCD samples for VICReg with an 80-20 split.
            use TTbar samples for MLPF with an 80-20 split.
            """
            data_VICReg_train = rem_qcd[: round(0.8 * len(rem_qcd))]
            data_VICReg_valid = rem_qcd[round(0.8 * len(rem_qcd)) :]

            data_mlpf_train = rem_ttbar[: round(0.8 * len(rem_ttbar))]
            data_mlpf_valid = rem_ttbar[round(0.8 * len(rem_ttbar)) :]

        elif data_split_mode == "mix":
            """
            use (80% of QCD + 80% of TTbar) samples for VICReg with a 90-10 split.
            use (20% of QCD + 20% of TTbar) samples for MLPF with a 90-10 split.
            """
            data_VICReg = rem_qcd[: round(0.8 * len(rem_qcd))] + rem_ttbar[: round(0.8 * len(rem_ttbar))]
            data_mlpf = rem_qcd[round(0.8 * len(rem_qcd)) :] + rem_ttbar[round(0.8 * len(rem_ttbar)) :]

            # shuffle the samples after mixing (not super necessary since the DataLoaders will shuffle anyway)
            random.shuffle(data_VICReg)
            random.shuffle(data_mlpf)

            data_VICReg_train = data_VICReg[: round(0.9 * len(data_VICReg))]
            data_VICReg_valid = data_VICReg[round(0.9 * len(data_VICReg)) :]

            data_mlpf_train = data_mlpf[: round(0.9 * len(data_mlpf))]
            data_mlpf_valid = data_mlpf[round(0.9 * len(data_mlpf)) :]

    print(f"Will use {len(data_VICReg_train)} events to train VICReg")
    print(f"Will use {len(data_VICReg_valid)} events to validate VICReg")
    print(f"Will use {len(data_mlpf_train)} events to train MLPF")
    print(f"Will use {len(data_mlpf_valid)} events to validate MLPF")

    return data_VICReg_train, data_VICReg_valid, data_mlpf_train, data_mlpf_valid, data_test_qcd, data_test_ttbar
