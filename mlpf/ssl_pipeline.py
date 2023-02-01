import os.path as osp

import matplotlib
import mplhep
import numpy as np
import torch
import torch_geometric
from pyg_ssl.args import parse_args
from pyg_ssl.mlpf import MLPF
from pyg_ssl.training_mlpf import training_loop_mlpf
from pyg_ssl.training_VICReg import training_loop_VICReg
from pyg_ssl.utils import CLUSTERS_X, TRACKS_X, data_split, load_VICReg, save_MLPF, save_VICReg
from pyg_ssl.VICReg import DECODER, ENCODER

# from pyg_ssl.evaluate import evaluate, make_multiplicity_plots_both


matplotlib.use("Agg")
mplhep.style.use(mplhep.styles.CMS)

"""
Developing a PyTorch Geometric semi-supervised (VICReg-based https://arxiv.org/abs/2105.04906) pipeline
for particleflow reconstruction on CLIC datasets.

Author: Farouk Mokhtar
"""


# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device
if torch.cuda.device_count():
    device = torch.device("cuda:0")
    print(f"Will use {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print("Will use cpu")


if __name__ == "__main__":

    args = parse_args()

    world_size = torch.cuda.device_count()

    # our data size varies from batch to batch, because each set of N_batch events has a different number of particles
    torch.backends.cudnn.benchmark = False

    # torch.autograd.set_detect_anomaly(True)

    # load the clic dataset
    data_train_VICReg, data_valid_VICReg, data_train_mlpf, data_valid_mlpf = data_split(args.dataset, args.data_split_mode)

    # setup the directory path to hold all models and plots
    outpath = osp.join(args.outpath, args.prefix_VICReg)

    # load a pre-trained VICReg model
    if args.load_VICReg:
        encoder_state_dict, encoder_model_kwargs, decoder_state_dict, decoder_model_kwargs = load_VICReg(device, outpath)

        encoder = ENCODER(**encoder_model_kwargs)
        decoder = DECODER(**decoder_model_kwargs)

        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        decoder = decoder.to(device)
        encoder = encoder.to(device)

    else:
        encoder_model_kwargs = {
            "embedding_dim": args.embedding_dim_VICReg,
            "width": args.width_encoder,
            "num_convs": args.num_convs,
            "space_dim": args.space_dim,
            "propagate_dim": args.propagate_dim,
            "k": args.nearest,
        }

        decoder_model_kwargs = {
            "input_dim": args.embedding_dim_VICReg,
            "output_dim": args.expand_dim,
            "width": args.width_decoder,
        }

        encoder = ENCODER(**encoder_model_kwargs).to(device)
        decoder = DECODER(**decoder_model_kwargs).to(device)

        print("Encoder", encoder)
        print("Decoder", decoder)
        print(f"VICReg model name: {args.prefix_VICReg}")

        # save model_kwargs and hyperparameters
        save_VICReg(args, outpath, encoder, encoder_model_kwargs, decoder, decoder_model_kwargs)

        print(f"Training VICReg over {args.n_epochs_VICReg} epochs")

        train_loader = torch_geometric.loader.DataLoader(data_train_VICReg, args.batch_size_VICReg)
        valid_loader = torch_geometric.loader.DataLoader(data_valid_VICReg, args.batch_size_VICReg)

        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

        training_loop_VICReg(
            device,
            encoder,
            decoder,
            train_loader,
            valid_loader,
            args.n_epochs_VICReg,
            args.patience,
            optimizer,
            outpath,
            args.lmbd,
            args.u,
            args.v,
        )

    if args.train_mlpf:
        print("------> Progressing to MLPF trainings...")
        print(f"Will use {len(data_train_mlpf)} events for train")
        print(f"Will use {len(data_valid_mlpf)} events for valid")

        train_loader = torch_geometric.loader.DataLoader(data_train_mlpf, args.batch_size_mlpf)
        valid_loader = torch_geometric.loader.DataLoader(data_valid_mlpf, args.batch_size_mlpf)

        input_ = max(CLUSTERS_X, TRACKS_X) + 1  # max cz we pad when we concatenate them & +1 cz there's the `type` feature

        if args.ssl:

            mlpf_model_kwargs = {
                "input_dim": input_ + args.embedding_dim_VICReg,
                "embedding_dim": args.embedding_dim_mlpf,
                "width": args.width_mlpf,
                "num_convs": args.num_convs,
                "native_mlpf": False,
            }

            mlpf_ssl = MLPF(**mlpf_model_kwargs).to(device)
            print(mlpf_ssl)
            print(f"MLPF model name: {args.prefix_mlpf}_ssl")

            # make mlpf specific directory
            outpath_ssl = osp.join(f"{outpath}/MLPF/", f"{args.prefix_mlpf}_ssl")
            save_MLPF(args, outpath_ssl, mlpf_ssl, mlpf_model_kwargs, mode="ssl")

            print(f"- Training ssl based MLPF over {args.n_epochs_mlpf} epochs")

            training_loop_mlpf(
                device,
                encoder,
                mlpf_ssl,
                train_loader,
                valid_loader,
                args.n_epochs_mlpf,
                args.patience,
                args.lr,
                outpath_ssl,
                mode="ssl",
                FineTune_VICReg=args.FineTune_VICReg,
            )

            # evaluate the ssl-based mlpf on both the VICReg validation and the mlpf validation datasets
            if args.evaluate_mlpf:
                from pyg_ssl.evaluate import evaluate

                ret_ssl = evaluate(
                    device,
                    encoder,
                    decoder,
                    mlpf_ssl,
                    args.batch_size_mlpf,
                    "ssl",
                    outpath_ssl,
                    [data_valid_VICReg, data_valid_mlpf],
                    ["valid_dataset_VICReg", "valid_dataset_mlpf"],
                )

        if args.native:

            mlpf_model_kwargs = {
                "input_dim": input_,
                "embedding_dim": args.embedding_dim_mlpf,
                "width": args.width_mlpf,
                "num_convs": args.num_convs,
                "native_mlpf": True,
            }

            mlpf_native = MLPF(**mlpf_model_kwargs).to(device)
            print(mlpf_native)
            print(f"MLPF model name: {args.prefix_mlpf}_native")

            # make mlpf specific directory
            outpath_native = osp.join(f"{outpath}/MLPF/", f"{args.prefix_mlpf}_native")
            save_MLPF(args, outpath_native, mlpf_native, mlpf_model_kwargs, mode="native")

            print(f"- Training native MLPF over {args.n_epochs_mlpf} epochs")

            training_loop_mlpf(
                device,
                encoder,
                mlpf_native,
                train_loader,
                valid_loader,
                args.n_epochs_mlpf,
                args.patience,
                args.lr,
                outpath_native,
                mode="native",
                FineTune_VICReg=False,
            )

            # evaluate the native mlpf on both the VICReg validation and the mlpf validation datasets
            if args.evaluate_mlpf:
                from pyg_ssl.evaluate import evaluate

                ret_native = evaluate(
                    device,
                    encoder,
                    decoder,
                    mlpf_native,
                    args.batch_size_mlpf,
                    "native",
                    outpath_native,
                    [data_valid_VICReg, data_valid_mlpf],
                    ["valid_dataset_VICReg", "valid_dataset_mlpf"],
                )

        if args.ssl & args.native:
            # plot multiplicity plot of both at the same time
            if args.evaluate_mlpf:
                from pyg_ssl.evaluate import make_multiplicity_plots_both

                make_multiplicity_plots_both(ret_ssl, ret_native, outpath_ssl)
