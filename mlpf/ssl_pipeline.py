import os.path as osp

import matplotlib
import numpy as np
import torch
import torch_geometric
from pyg_ssl.args import parse_args
from pyg_ssl.evaluate import evaluate
from pyg_ssl.mlpf import MLPF
from pyg_ssl.training_mlpf import training_loop_mlpf
from pyg_ssl.training_VICReg import training_loop_VICReg
from pyg_ssl.utils import data_split, load_VICReg, save_MLPF, save_VICReg
from pyg_ssl.VICReg import DECODER, ENCODER

matplotlib.use("Agg")


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

    torch.backends.cudnn.benchmark = True

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
            "width": args.width_encoder,
            "embedding_dim": args.embedding_dim,
            "num_convs": args.num_convs,
            "space_dim": args.space_dim,
            "propagate_dim": args.propagate_dim,
            "k": args.nearest,
        }

        decoder_model_kwargs = {
            "input_dim": args.embedding_dim,
            "width": args.width_decoder,
            "output_dim": args.expand_dim,
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

        batch_size_test = 1

        if args.ssl:

            mlpf_model_kwargs = {
                "embedding_dim": encoder.conv[1].out_channels,
                "width": args.width_mlpf,
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

            # evaluate the ssl-based mlpf on the VICReg validation
            print("Testing the ssl model on the VICReg validation dataset")
            evaluate(
                device,
                encoder,
                decoder,
                mlpf_ssl,
                batch_size_test,
                "ssl",
                data_valid_VICReg,
                "valid_dataset_VICReg",
                outpath_ssl,
            )
            # evaluate the ssl-based mlpf on the mlpf validation
            print("Testing the ssl model on the mlpf validation dataset")
            evaluate(
                device,
                encoder,
                decoder,
                mlpf_ssl,
                batch_size_test,
                "ssl",
                data_valid_mlpf,
                "valid_dataset_mlpf",
                outpath_ssl,
            )

        if args.native:
            mlpf_model_kwargs = {
                "input_dim": 12,
                "width": args.width_mlpf,
                "native_mlpf": True,
                "embedding_dim": args.embedding_dim,
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

            # evaluate the native mlpf on the VICReg validation
            print("Testing the native model on the VICReg validation dataset")
            evaluate(
                device,
                encoder,
                decoder,
                mlpf_native,
                batch_size_test,
                "native",
                data_valid_VICReg,
                "valid_dataset_VICReg",
                outpath_native,
            )
            # evaluate the native mlpf on the mlpf validation
            print("Testing the native model on the mlpf validation dataset")
            evaluate(
                device,
                encoder,
                decoder,
                mlpf_native,
                batch_size_test,
                "native",
                data_valid_mlpf,
                "valid_dataset_mlpf",
                outpath_native,
            )
