import json
import pickle as pkl

import pandas as pd
import torch
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ConstantLR
import logging

# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254

# All possible PFElement types
ELEM_TYPES = {
    "cms": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "clic": [0, 1, 2],
}

# Some element types are defined, but do not exist in the dataset at all
ELEM_TYPES_NONZERO = {
    "cms": [1, 4, 5, 6, 8, 9, 10, 11],
    "clic": [1, 2],
}

CLASS_LABELS = {
    "cms": [0, 211, 130, 1, 2, 22, 11, 13, 15],
    "clic": [0, 211, 130, 22, 11, 13],
    "clic_hits": [0, 211, 130, 22, 11, 13],
}

CLASS_NAMES_LATEX = {
    "cms": ["none", "Charged Hadron", "Neutral Hadron", "HFEM", "HFHAD", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$", r"$\tau$"],
    "clic": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    "clic_hits": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
}
CLASS_NAMES = {
    "cms": ["none", "chhad", "nhad", "HFEM", "HFHAD", "gamma", "ele", "mu", "tau"],
    "clic": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    "clic_hits": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
}
CLASS_NAMES_CAPITALIZED = {
    "cms": ["none", "Charged hadron", "Neutral hadron", "HFEM", "HFHAD", "Photon", "Electron", "Muon", "Tau"],
    "clic": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    "clic_hits": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
}

X_FEATURES = {
    "cms": [
        "typ_idx",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "e",
        "layer",
        "depth",
        "charge",
        "trajpoint",
        "eta_ecal",
        "phi_ecal",
        "eta_hcal",
        "phi_hcal",
        "muon_dt_hits",
        "muon_csc_hits",
        "muon_type",
        "px",
        "py",
        "pz",
        "deltap",
        "sigmadeltap",
        "gsf_electronseed_trkorecal",
        "gsf_electronseed_dnn1",
        "gsf_electronseed_dnn2",
        "gsf_electronseed_dnn3",
        "gsf_electronseed_dnn4",
        "gsf_electronseed_dnn5",
        "num_hits",
        "cluster_flags",
        "corr_energy",
        "corr_energy_err",
        "vx",
        "vy",
        "vz",
        "pterror",
        "etaerror",
        "phierror",
        "lambd",
        "lambdaerror",
        "theta",
        "thetaerror",
        "time",
        "timeerror",
        "etaerror1",
        "etaerror2",
        "etaerror3",
        "etaerror4",
        "phierror1",
        "phierror2",
        "phierror3",
        "phierror4",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    ],
    "clic": [
        "type",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | iTheta",
        "radiusOfInnermostHit | energy_ecal",
        "tanLambda | energy_hcal",
        "D0 | energy_other",
        "omega | num_hits",
        "Z0 | sigma_x",
        "time | sigma_y",
        "Null | sigma_z",
    ],
    "clic_hits": [
        "elemtype",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | time",
        "radiusOfInnermostHit | subdetector",
        "tanLambda | type",
        "D0 | Null",
        "omega | Null",
        "Z0 | Null",
        "time | Null",
    ],
}

Y_FEATURES = ["cls_id", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]


def unpack_target(y, model):
    ret = {}
    ret["cls_id"] = y[..., 0].long()
    ret["charge"] = torch.clamp((y[..., 1] + 1).to(dtype=torch.float32), 0, 2)  # -1, 0, 1 -> 0, 1, 2

    for i, feat in enumerate(Y_FEATURES):
        if i >= 2:  # skip the cls and charge as they are defined above
            ret[feat] = y[..., i].to(dtype=torch.float32)
    ret["phi"] = torch.atan2(ret["sin_phi"], ret["cos_phi"])

    # do some sanity checks
    # assert torch.all(ret["pt"] >= 0.0)  # pt
    # assert torch.all(torch.abs(ret["sin_phi"]) <= 1.0)  # sin_phi
    # assert torch.all(torch.abs(ret["cos_phi"]) <= 1.0)  # cos_phi
    # assert torch.all(ret["energy"] >= 0.0)  # energy

    # note ~ momentum = ["pt", "eta", "sin_phi", "cos_phi", "energy"]
    ret["momentum"] = y[..., 2:7].to(dtype=torch.float32)
    ret["p4"] = torch.cat([ret["pt"].unsqueeze(-1), ret["eta"].unsqueeze(-1), ret["phi"].unsqueeze(-1), ret["energy"].unsqueeze(-1)], axis=-1)

    ret["ispu"] = y[..., -1]

    ret["pt_bins"] = torch.clamp(torch.searchsorted(model.bins_pt, ret["pt"].contiguous()), 0, len(model.bins_pt)-1)
    ret["eta_bins"] = torch.clamp(torch.searchsorted(model.bins_eta, ret["eta"].contiguous()), 0, len(model.bins_pt)-1)
    ret["sin_phi_bins"] = torch.clamp(torch.searchsorted(model.bins_sin_phi, ret["sin_phi"].contiguous()), 0, len(model.bins_pt)-1)
    ret["cos_phi_bins"] = torch.clamp(torch.searchsorted(model.bins_cos_phi, ret["cos_phi"].contiguous()), 0, len(model.bins_pt)-1)
    ret["energy_bins"] = torch.clamp(torch.searchsorted(model.bins_energy, ret["energy"].contiguous()), 0, len(model.bins_pt)-1)

    return ret


def unpack_predictions(preds):
    ret = {}
    ret["cls_binary"], ret["cls_id_onehot"], ret["momentum"], ret_momentum_bins = preds
    preds_pt_bins, preds_eta_bins, preds_sin_phi_bins, preds_cos_phi_bins, preds_energy_bins = ret_momentum_bins
    
    # ret["cls_id_onehot"], ret["momentum"] = preds

    # ret["charge"] = torch.argmax(ret["charge"], axis=1, keepdim=True) - 1

    # unpacking
    ret["pt"] = ret["momentum"][..., 0]
    ret["eta"] = ret["momentum"][..., 1]
    ret["sin_phi"] = ret["momentum"][..., 2]
    ret["cos_phi"] = ret["momentum"][..., 3]
    ret["energy"] = ret["momentum"][..., 4]

    # first get the cases where a particle was predicted
    ret["cls_id"] = torch.argmax(ret["cls_binary"], axis=-1)
    # when a particle was predicted, get the particle ID
    ret["cls_id"][ret["cls_id"] == 1] = torch.argmax(ret["cls_id_onehot"], axis=-1)[ret["cls_id"] == 1]

    # get the predicted particle ID
    # ret["cls_id"] = torch.argmax(ret["cls_id_onehot"], axis=-1)

    # particle properties
    ret["phi"] = torch.atan2(ret["sin_phi"], ret["cos_phi"])
    ret["p4"] = torch.cat(
        [
            ret["pt"].unsqueeze(axis=-1),
            ret["eta"].unsqueeze(axis=-1),
            ret["phi"].unsqueeze(axis=-1),
            ret["energy"].unsqueeze(axis=-1),
        ],
        axis=-1,
    )
    ret["pt_bins"] = preds_pt_bins
    ret["eta_bins"] = preds_eta_bins
    ret["sin_phi_bins"] = preds_sin_phi_bins
    ret["cos_phi_bins"] = preds_cos_phi_bins
    ret["energy_bins"] = preds_energy_bins

    return ret


def save_HPs(args, mlpf, model_kwargs, outdir):
    """Simple function to store the model parameters and training hyperparameters."""

    with open(f"{outdir}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)

    with open(f"{outdir}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump({**{"Num of mlpf parameters": num_mlpf_parameters}, **vars(args)}, fp)


def get_model_state_dict(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()


def print_optimizer_stats(optimizer, stage):
    print(f"\nOptimizer statistics {stage}:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Parameter group {i}:")
        print(f"    Learning rate: {param_group['lr']}")
        print(f"    Weight decay: {param_group['weight_decay']}")
        if "momentum" in param_group:
            print(f"    Momentum: {param_group['momentum']}")
        elif "betas" in param_group:
            print(f"    Betas: {param_group['betas']}")

    if hasattr(optimizer, "state"):
        print("  Optimizer state:")
        print(f"    Number of steps: {len(optimizer.state)}")
        if len(optimizer.state) > 0:
            first_param = next(iter(optimizer.state.values()))
            for key, value in first_param.items():
                if torch.is_tensor(value):
                    print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {value}")


def load_checkpoint(checkpoint, model, optimizer=None):
    if optimizer:
        print_optimizer_stats(optimizer, "Before loading")

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Loaded optimizer state")
        print_optimizer_stats(optimizer, "After loading")
        return model, optimizer
    else:
        return model


def save_checkpoint(checkpoint_path, model, optimizer=None, extra_state=None):
    torch.save(
        {
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "extra_state": extra_state,
        },
        checkpoint_path,
    )


def load_lr_schedule(lr_schedule, checkpoint):
    "Loads the lr_schedule's state dict from checkpoint"
    if "lr_schedule_state_dict" in checkpoint["extra_state"].keys():
        lr_schedule.load_state_dict(checkpoint["extra_state"]["lr_schedule_state_dict"])
        return lr_schedule
    else:
        raise KeyError("Couldn't find LR schedule state dict in checkpoint. extra_state contains: {}".format(checkpoint["extra_state"].keys()))


def get_lr_schedule(config, opt, epochs=None, steps_per_epoch=None, last_epoch=-1):
    # we step the schedule every mini-batch so need to multiply by steps_per_epoch
    last_batch = last_epoch * steps_per_epoch - 1 if last_epoch != -1 else -1
    if config["lr_schedule"] == "constant":
        lr_schedule = ConstantLR(opt, factor=1.0, total_iters=steps_per_epoch * epochs)
    elif config["lr_schedule"] == "onecycle":
        lr_schedule = OneCycleLR(
            opt,
            max_lr=config["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            last_epoch=last_batch,
            pct_start=config["lr_schedule_config"]["onecycle"]["pct_start"] or 0.3,
        )
    elif config["lr_schedule"] == "cosinedecay":
        lr_schedule = CosineAnnealingLR(opt, T_max=steps_per_epoch * epochs, last_epoch=last_batch, eta_min=1e-5)
    else:
        raise ValueError("Supported values for lr_schedule are 'constant', 'onecycle' and 'cosinedecay'.")
    return lr_schedule


def count_parameters(model):
    column_names = ["Modules", "Trainable parameters", "Non-trainable parameters"]
    table = pd.DataFrame(columns=column_names)
    trainable_params = 0
    nontrainable_params = 0
    for ii, (name, parameter) in enumerate(model.named_parameters()):
        params = parameter.numel()
        if not parameter.requires_grad:
            table = pd.concat(
                [
                    table,
                    pd.DataFrame({column_names[0]: name, column_names[1]: 0, column_names[2]: params}, index=[ii]),
                ]
            )
            nontrainable_params += params
        else:
            table = pd.concat(
                [
                    table,
                    pd.DataFrame({column_names[0]: name, column_names[1]: params, column_names[2]: 0}, index=[ii]),
                ]
            )
            trainable_params += params
    return trainable_params, nontrainable_params, table
