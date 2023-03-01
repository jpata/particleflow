import datetime
import os.path as osp
import platform

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
from pyg_ssl.VICReg import DECODER, ENCODER, VICReg

matplotlib.use("Agg")
mplhep.style.use(mplhep.styles.CMS)

"""
Developing a PyTorch Geometric semi-supervised (VICReg-based https://arxiv.org/abs/2105.04906) pipeline
for particleflow reconstruction on CLIC datasets.

Authors: Farouk Mokhtar, Joosep Pata.
"""


# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device(s)
if torch.cuda.device_count():
    device = torch.device("cuda:0")
    print(f"Will use {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print("Will use cpu")
multi_gpu = torch.cuda.device_count() > 1

if __name__ == "__main__":

    args = parse_args()

    # our data size varies from batch to batch, because each set of N_batch events has a different number of particles
    torch.backends.cudnn.benchmark = False

    # torch.autograd.set_detect_anomaly(True)

    # load the clic dataset
    data_VICReg_train, data_VICReg_valid, data_mlpf_train, data_mlpf_valid, data_test_qcd, data_test_ttbar = data_split(
        args.dataset, args.data_split_mode
    )

    # setup the directory path to hold all models and plots
    if args.prefix_VICReg is None:
        args.prefix_VICReg = "pyg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "." + platform.node()
    outpath = osp.join(args.outpath, args.prefix_VICReg)

    # load a pre-trained VICReg model
    if args.load_VICReg:
        vicreg_state_dict, encoder_model_kwargs, decoder_model_kwargs = load_VICReg(device, outpath)

        vicreg_encoder = ENCODER(**encoder_model_kwargs)
        vicreg_decoder = DECODER(**decoder_model_kwargs)

        vicreg = VICReg(vicreg_encoder, vicreg_decoder)
        vicreg.load_state_dict(vicreg_state_dict)
        vicreg.to(device)

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

        vicreg_encoder = ENCODER(**encoder_model_kwargs)
        vicreg_decoder = DECODER(**decoder_model_kwargs)
        vicreg = VICReg(vicreg_encoder, vicreg_decoder)
        vicreg.to(device)

        # save model_kwargs and hyperparameters
        save_VICReg(args, outpath, vicreg_encoder, encoder_model_kwargs, vicreg_decoder, decoder_model_kwargs)

        if multi_gpu:
            vicreg = torch_geometric.nn.DataParallel(vicreg)
            train_loader = torch_geometric.loader.DataListLoader(data_VICReg_train, args.bs_VICReg)
            valid_loader = torch_geometric.loader.DataListLoader(data_VICReg_valid, args.bs_VICReg)
        else:
            train_loader = torch_geometric.loader.DataLoader(data_VICReg_train, args.bs_VICReg)
            valid_loader = torch_geometric.loader.DataLoader(data_VICReg_valid, args.bs_VICReg)

        optimizer = torch.optim.SGD(vicreg.parameters(), lr=args.lr)

        print(vicreg)
        print(f"VICReg model name: {args.prefix_VICReg}")
        print(f"Training VICReg over {args.n_epochs_VICReg} epochs")
        training_loop_VICReg(
            multi_gpu,
            device,
            vicreg,
            {"train": train_loader, "valid": valid_loader},
            args.n_epochs_VICReg,
            args.patience,
            optimizer,
            {"lmbd": args.lmbd, "mu": args.mu, "nu": args.nu},
            outpath,
        )

    if args.train_mlpf:
        print("------> Progressing to MLPF trainings...")
        print(f"Will use {len(data_mlpf_train)} events for train")
        print(f"Will use {len(data_mlpf_valid)} events for valid")

        train_loader = torch_geometric.loader.DataLoader(data_mlpf_train, args.bs_mlpf)
        valid_loader = torch_geometric.loader.DataLoader(data_mlpf_valid, args.bs_mlpf)

        input_ = max(CLUSTERS_X, TRACKS_X) + 1  # max cz we pad when we concatenate them & +1 cz there's the `type` feature

        if args.ssl:

            mlpf_model_kwargs = {
                "input_dim": input_,
                "embedding_dim": args.embedding_dim_mlpf,
                "width": args.width_mlpf,
                "k": args.nearest,
                "num_convs": args.num_convs_mlpf,
                "dropout": args.dropout_mlpf,
                "ssl": True,
                "VICReg_embedding_dim": args.embedding_dim_VICReg,
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
                mlpf_ssl,
                train_loader,
                valid_loader,
                args.n_epochs_mlpf,
                args.patience,
                args.lr,
                outpath_ssl,
                vicreg_encoder,
            )

            # evaluate the ssl-based mlpf on both the QCD and TTbar samples
            if args.evaluate_mlpf:
                from pyg_ssl.evaluate import evaluate

                ret_ssl = evaluate(
                    device,
                    vicreg_encoder,
                    mlpf_ssl,
                    args.bs_mlpf,
                    "ssl",
                    outpath_ssl,
                    {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
                )

        if args.native:

            mlpf_model_kwargs = {
                "input_dim": input_,
                "embedding_dim": args.embedding_dim_mlpf,
                "width": args.width_mlpf,
                "k": args.nearest,
                "num_convs": args.num_convs_mlpf,
                "dropout": args.dropout_mlpf,
                "ssl": False,
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
                mlpf_native,
                train_loader,
                valid_loader,
                args.n_epochs_mlpf,
                args.patience,
                args.lr,
                outpath_native,
            )

            # evaluate the native mlpf on both the QCD and TTbar samples
            if args.evaluate_mlpf:
                from pyg_ssl.evaluate import evaluate

                ret_native = evaluate(
                    device,
                    vicreg_encoder,
                    mlpf_native,
                    args.bs_mlpf,
                    "native",
                    outpath_native,
                    {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
                )

        # if args.ssl & args.native:
        #     # plot multiplicity plot of both at the same time
        #     if args.evaluate_mlpf:
        #         from pyg_ssl.evaluate import make_multiplicity_plots_both

        #         make_multiplicity_plots_both(ret_ssl, ret_native, outpath_ssl)
