import json
import os
import os.path as osp
import pickle as pkl
import sys

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_geometric
from pyg_ssl import (
    DECODER,
    ENCODER,
    MLPF,
    evaluate,
    load_VICReg,
    parse_args,
    plot_conf_matrix,
    save_MLPF,
    save_VICReg,
    training_loop_mlpf,
    training_loop_VICReg,
)

matplotlib.use("Agg")


"""
Developing a PyTorch Geometric semi-supervised pipeline (based on VICReg) for CLIC datasets.

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

    # setup the directory path to hold all models and plots
    outpath = osp.join(args.outpath, args.model_prefix_VICReg)

    # load the clic dataset
    import glob

    all_files = glob.glob(f"{args.dataset}/mix/data_*")
    data = []
    for f in all_files:
        data += torch.load(f"{f}")
    if len(data) == 0:
        print("failed to load dataset, check --dataset path")
        sys.exit(0)
    else:
        print(f"Will use {len(data)} to pretrain VICReg")

    # load a pre-trained VICReg model
    if args.load_VICReg:
        (
            encoder_state_dict,
            encoder_model_kwargs,
            decoder_state_dict,
            decoder_model_kwargs,
        ) = load_VICReg(device, outpath)

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

        encoder = ENCODER(**encoder_model_kwargs)
        decoder = DECODER(**decoder_model_kwargs)

        print("Encoder", encoder)
        print("Decoder", decoder)
        print(f"VICReg model name: {args.model_prefix_VICReg}")

        # save model_kwargs and hyperparameters
        save_VICReg(args, outpath, encoder_model_kwargs, decoder_model_kwargs)

        print("Training over {} epochs".format(args.n_epochs))

        data_train = data[: int(0.8 * len(data))]
        data_valid = data[int(0.8 * len(data)) :]

        train_loader = torch_geometric.loader.DataLoader(data_train, args.batch_size)
        valid_loader = torch_geometric.loader.DataLoader(data_valid, args.batch_size)

        decoder = decoder.to(device)
        encoder = encoder.to(device)

        optimizer = torch.optim.SGD(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1.5e-4,
        )

        training_loop_VICReg(
            device,
            encoder,
            decoder,
            train_loader,
            valid_loader,
            args.n_epochs,
            args.patience,
            optimizer,
            outpath,
        )

    if args.train_mlpf:

        data_train = data[:4000]
        data_valid = data[4000:8000]
        data_test = data[8000:]

        train_loader = torch_geometric.loader.DataLoader(data_train, args.batch_size)
        valid_loader = torch_geometric.loader.DataLoader(data_valid, args.batch_size)

        mlpf_model_kwargs = {
            "input_dim": encoder.conv[1].out_channels,
            "width": args.width_mlpf,
        }

        mlpf = MLPF(**mlpf_model_kwargs)
        mlpf = mlpf.to(device)
        print(mlpf)
        print(f"MLPF model name: {args.model_prefix_mlpf}")

        # make mlpf specific directory
        outpath = osp.join(f"{outpath}/MLPF/", args.model_prefix_mlpf)
        save_MLPF(args, outpath, mlpf_model_kwargs)

        optimizer = torch.optim.SGD(mlpf.parameters(), lr=args.lr)

        print(f"Training MLPF")

        training_loop_mlpf(
            device,
            encoder,
            mlpf,
            train_loader,
            valid_loader,
            args.n_epochs,
            args.patience,
            optimizer,
            outpath,
        )

        # test
        test_loader = torch_geometric.loader.DataLoader(data_test, args.batch_size)

        conf_matrix = evaluate(device, encoder, decoder, mlpf, test_loader)

        plot_conf_matrix(conf_matrix, "SSL based MLPF", outpath)

        with open(f"{outpath}/conf_matrix_test.pkl", "wb") as f:
            pkl.dump(conf_matrix, f)
