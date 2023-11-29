import json
import pickle as pkl

import torch
import torch.utils.data

# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
CLASS_LABELS = {
    "cms": [0, 211, 130, 1, 2, 22, 11, 13, 15],
    "delphes": [0, 211, 130, 22, 11, 13],
    "clic": [0, 211, 130, 22, 11, 13],
}

CLASS_NAMES_LATEX = {
    "cms": ["none", "Charged Hadron", "Neutral Hadron", "HFEM", "HFHAD", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$", r"$\tau$"],
    "delphes": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    "clic": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
}
CLASS_NAMES = {
    "cms": ["none", "chhad", "nhad", "HFEM", "HFHAD", "gamma", "ele", "mu", "tau"],
    "delphes": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    "clic": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
}
CLASS_NAMES_CAPITALIZED = {
    "cms": ["none", "Charged hadron", "Neutral hadron", "HFEM", "HFHAD", "Photon", "Electron", "Muon", "Tau"],
    "delphes": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    "clic": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
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
    ],
    "delphes": [
        "Track|cluster",
        "$p_{T}|E_{T}$",
        r"$\eta$",
        r"$Sin(\phi)$",
        r"$Cos(\phi)$",
        "P|E",
        r"$\eta_\mathrm{out}|E_{em}$",
        r"$Sin(\(phi)_\mathrm{out}|E_{had}$",
        r"$Cos(\phi)_\mathrm{out}|E_{had}$",
        "charge",
        "is_gen_mu",
        "is_gen_el",
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
}

Y_FEATURES = ["cls_id", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"]


def unpack_target(y):
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
    ret["momentum"] = y[..., 2:-1].to(dtype=torch.float32)
    ret["p4"] = torch.cat(
        [ret["pt"].unsqueeze(1), ret["eta"].unsqueeze(1), ret["phi"].unsqueeze(1), ret["energy"].unsqueeze(1)], axis=1
    )

    return ret


def unpack_predictions(preds):
    ret = {}
    ret["cls_id_onehot"], ret["momentum"], ret["charge"] = preds

    # ret["charge"] = torch.argmax(ret["charge"], axis=1, keepdim=True) - 1

    # unpacking
    ret["pt"] = ret["momentum"][..., 0]
    ret["eta"] = ret["momentum"][..., 1]
    ret["sin_phi"] = ret["momentum"][..., 2]
    ret["cos_phi"] = ret["momentum"][..., 3]
    ret["energy"] = ret["momentum"][..., 4]

    # new variables
    ret["cls_id"] = torch.argmax(ret["cls_id_onehot"], axis=-1)
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


def load_checkpoint(checkpoint, model, optimizer=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
