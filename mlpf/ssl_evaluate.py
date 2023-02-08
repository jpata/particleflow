import datetime
import os.path as osp
import pickle as pkl
import platform

import torch
from pyg_ssl.args import parse_args
from pyg_ssl.evaluate import evaluate, make_multiplicity_plots_both
from pyg_ssl.mlpf import MLPF
from pyg_ssl.utils import data_split, load_VICReg
from pyg_ssl.VICReg import DECODER, ENCODER

if __name__ == "__main__":
    import sys

    sys.path.append("")

    # define the global base device
    if torch.cuda.device_count():
        device = torch.device("cuda:0")
        print(f"Will use {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print("Will use cpu")

    args = parse_args()

    # load the clic dataset
    _, _, _, _, data_test_qcd, data_test_ttbar = data_split(args.dataset, args.data_split_mode)

    # setup the directory path to hold all models and plots
    if args.prefix_VICReg is None:
        args.prefix_VICReg = "pyg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "." + platform.node()
    outpath = osp.join(args.outpath, args.prefix_VICReg)

    # load a pre-trained VICReg model
    encoder_state_dict, encoder_model_kwargs, decoder_state_dict, decoder_model_kwargs = load_VICReg(device, outpath)

    encoder = ENCODER(**encoder_model_kwargs)
    decoder = DECODER(**decoder_model_kwargs)

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # load a pre-trained MLPF model
    if args.ssl:
        outpath_ssl = osp.join(f"{outpath}/MLPF/", f"{args.prefix_mlpf}_ssl")

        print("Loading a previously trained ssl model..")
        mlpf_ssl_state_dict = torch.load(f"{outpath_ssl}/mlpf_ssl_best_epoch_weights.pth", map_location=device)

        with open(f"{outpath_ssl}/mlpf_model_kwargs.pkl", "rb") as f:
            mlpf_model_kwargs = pkl.load(f)

        mlpf_ssl = MLPF(**mlpf_model_kwargs).to(device)

        ret_ssl = evaluate(
            device,
            encoder,
            decoder,
            mlpf_ssl,
            args.batch_size_mlpf,
            "ssl",
            outpath_ssl,
            {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
        )

    if args.native:
        outpath_native = osp.join(f"{outpath}/MLPF/", f"{args.prefix_mlpf}_native")

        print("Loading a previously trained ssl model..")
        mlpf_native_state_dict = torch.load(f"{outpath_native}/mlpf_native_best_epoch_weights.pth", map_location=device)

        with open(f"{outpath_native}/mlpf_model_kwargs.pkl", "rb") as f:
            mlpf_model_kwargs = pkl.load(f)

        mlpf_native = MLPF(**mlpf_model_kwargs).to(device)

        ret_native = evaluate(
            device,
            encoder,
            decoder,
            mlpf_native,
            args.batch_size_mlpf,
            "native",
            outpath_native,
            {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
        )

    if args.ssl & args.native:
        make_multiplicity_plots_both(ret_ssl, ret_native, outpath_ssl)
