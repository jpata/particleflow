import json
import os
import os.path as osp
import pickle as pkl

import torch

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

Y_FEATURES = {
    "cms": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "delphes": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "clic": ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"],
}


def save_mlpf(args, mlpf, model_kwargs):
    if not osp.isdir(args.model_prefix):
        os.system(f"mkdir -p {args.model_prefix}")

    else:  # if directory already exists
        assert args.overwrite, f"model {args.model_prefix} already exists, please delete it"

        print("model already exists, deleting it")
        os.system(f"rm -rf {args.model_prefix}")
        os.system(f"mkdir -p {args.model_prefix}")

    with open(f"{args.model_prefix}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)

    with open(f"{args.model_prefix}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump({**{"Num of mlpf parameters": num_mlpf_parameters}, **vars(args)}, fp)


def load_mlpf(device, outpath):
    with open(outpath + "/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(outpath + "/best_epoch_weights.pth", map_location=device)

    return state_dict, model_kwargs
