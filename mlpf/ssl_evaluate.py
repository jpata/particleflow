import datetime
import os.path as osp
import pickle as pkl
import platform

import torch
from pyg.mlpf import MLPF
from pyg.ssl.args import parse_args
from pyg.ssl.evaluate import evaluate
from pyg.ssl.utils import data_split, load_VICReg
from pyg.ssl.VICReg import DECODER, ENCODER, VICReg

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
    _, _, _, _, data_test_qcd, data_test_ttbar = data_split(args.data_path, args.data_split_mode)

    # setup the directory path to hold all models and plots
    if args.prefix_VICReg is None:
        args.prefix_VICReg = "pyg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "." + platform.node()
    outpath = osp.join(args.outpath, args.prefix_VICReg)

    # load a pre-trained VICReg model
    vicreg_state_dict, encoder_model_kwargs, decoder_model_kwargs = load_VICReg(device, outpath)

    vicreg_encoder = ENCODER(**encoder_model_kwargs)
    vicreg_decoder = DECODER(**decoder_model_kwargs)

    vicreg = VICReg(vicreg_encoder, vicreg_decoder)

    try:
        vicreg.load_state_dict(vicreg_state_dict)
    except RuntimeError:
        # because model was saved using dataparallel
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in vicreg_state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        vicreg_state_dict = new_state_dict
        vicreg.load_state_dict(vicreg_state_dict)

    vicreg.to(device)

    # load a pre-trained MLPF model
    if args.ssl:
        outpath_ssl = osp.join(f"{outpath}/MLPF/", f"{args.prefix}_ssl")

        print("Loading a previously trained ssl model..")
        mlpf_ssl_state_dict = torch.load(f"{outpath_ssl}/best_epoch_weights.pth", map_location=device)

        with open(f"{outpath_ssl}/model_kwargs.pkl", "rb") as f:
            mlpf_model_kwargs = pkl.load(f)

        mlpf_ssl = MLPF(**mlpf_model_kwargs).to(device)
        mlpf_ssl.load_state_dict(mlpf_ssl_state_dict)

        ret_ssl = evaluate(
            device,
            vicreg_encoder,
            mlpf_ssl,
            args.bs,
            "ssl",
            outpath_ssl,
            {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
        )

    if args.native:
        outpath_native = osp.join(f"{outpath}/MLPF/", f"{args.prefix}_native")

        print("Loading a previously trained ssl model..")
        mlpf_native_state_dict = torch.load(f"{outpath_native}/best_epoch_weights.pth", map_location=device)

        with open(f"{outpath_native}/model_kwargs.pkl", "rb") as f:
            mlpf_model_kwargs = pkl.load(f)

        mlpf_native = MLPF(**mlpf_model_kwargs).to(device)
        mlpf_native.load_state_dict(mlpf_native_state_dict)

        ret_native = evaluate(
            device,
            vicreg_encoder,
            mlpf_native,
            args.bs,
            "native",
            outpath_native,
            {"QCD": data_test_qcd, "TTBar": data_test_ttbar},
        )

    # if args.ssl & args.native:
    #     from pyg.ssl.evaluate import make_multiplicity_plots_both

    #     make_multiplicity_plots_both(ret_ssl, ret_native, outpath_ssl)
