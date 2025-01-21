import glob
import json
import math

import awkward
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import sklearn
import sklearn.metrics
import tqdm
import vector
import torch

SAMPLE_LABEL_CMS = {
    "TTbar_14TeV_TuneCUETP8M1_cfi": r"$\mathrm{t}\overline{\mathrm{t}}$+PU events",
    "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi": r"$Z\rightarrow \tau \tau$+PU events",
    "QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi": r"high-$p_T$ QCD+PU events",
    "QCDForPF_14TeV_TuneCUETP8M1_cfi": r"QCD+PU events",
    "SingleElectronFlatPt1To1000_pythia8_cfi": r"single $e^\pm$ events",
    "SingleGammaFlatPt1To1000_pythia8_cfi": r"single $\gamma$ events",
    "SingleMuFlatLogPt_100MeVto2TeV_cfi": r"single $\mu^\pm$ events",
    "SingleNeutronFlatPt0p7To1000_cfi": "single neutron events",
    "SinglePi0Pt1To1000_pythia8_cfi": r"single $\pi^0$ events",
    "SinglePiMinusFlatPt0p7To1000_cfi": r"single $\pi^\pm$ events",
    "SingleProtonMinusFlatPt0p7To1000_cfi": r"single proton events",
    "SingleTauFlatPt1To1000_cfi": r"single $\tau^\pm$ events",
    "RelValQCD_FlatPt_15_3000HS_14": r"QCD $15 < p_T < 3000$ GeV + PU events",
    "RelValTTbar_14TeV": r"$\mathrm{t}\overline{\mathrm{t}}$+PU events",
}

pid_to_text = {
    211: r"ch. had.",
    130: r"n. had.",
    1: r"HF hadron (EM)",
    2: r"HF hadron (HAD)",
    11: r"$e^{\pm}$",
    13: r"$\mu^{\pm}$",
    22: r"$\gamma$",
}

ELEM_LABELS_CMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ELEM_NAMES_CMS = [
    "NONE",
    "TRACK",
    "PS1",
    "PS2",
    "ECAL",
    "HCAL",
    "GSF",
    "BREM",
    "HFEM",
    "HFHAD",
    "SC",
    "HO",
]

CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13]
CLASS_NAMES_CMS = [
    r"none",
    r"ch.had",
    r"n.had",
    r"HFHAD",
    r"HFEM",
    r"$\gamma$",
    r"$e^\pm$",
    r"$\mu^\pm$",
]

CLASS_LABELS_CLIC = [0, 211, 130, 22, 11, 13]
CLASS_NAMES_CLIC = [
    r"none",
    r"ch.had",
    r"n.had",
    r"$\gamma$",
    r"$e^\pm$",
    r"$\mu^\pm$",
]

CLASS_LABELS = {
    "cms": CLASS_LABELS_CMS,
    "clic": CLASS_LABELS_CLIC,
}

labels = {
    "met": "$p_{\mathrm{T}}^{\mathrm{miss}}$ [GeV]",
    "gen_met": "$p_{\mathrm{T,truth}}^\mathrm{miss}$ [GeV]",
    "gen_mom": "$p_{\mathrm{truth}}$ [GeV]",
    "gen_jet": "jet $p_{\mathrm{T,truth}}$ [GeV]",
    "target_jet": "jet $p_{\mathrm{T,target}}$ [GeV]",
    "gen_jet_eta": "jet $\eta_{\mathrm{truth}}$ [GeV]",
    "reco_met": "$p_{\mathrm{T,reco}}^\mathrm{miss}$ [GeV]",
    "reco_gen_met_ratio": "$p_{\mathrm{T,reco}}^\mathrm{miss} / p_{\\mathrm{T,truth}}^\mathrm{miss}$",
    "reco_gen_mom_ratio": "$p_{\mathrm{reco}} / p_{\\mathrm{truth}}$",
    "reco_gen_jet_ratio": "jet $p_{\mathrm{T,reco}} / p_{\\mathrm{T,truth}}$",
    "reco_target_jet_ratio": "jet $p_{\mathrm{T,reco}} / p_{\\mathrm{T,target}}$",
    "gen_met_range": "${} \less p_{{\mathrm{{T,truth}}}}^\mathrm{{miss}}\leq {}$",
    "gen_mom_range": "${} \less p_{{\mathrm{{truth}}}}\leq {}$",
    "gen_jet_range": "${} \less p_{{\mathrm{{T,truth}}}} \leq {}$",
    "gen_jet_range_eta": "${} \less \eta_{{\mathrm{{truth}}}} \leq {}$",
}


def get_class_names(sample_name):
    if sample_name.startswith("clic_"):
        return CLASS_NAMES_CLIC
    elif sample_name.startswith("cms_"):
        return CLASS_NAMES_CMS
    if sample_name.startswith("cld_"):
        return CLASS_NAMES_CLIC
    else:
        raise Exception("Unknown sample name: {}".format(sample_name))


EVALUATION_DATASET_NAMES = {
    "cld_edm_ttbar_pf": r"$e^+e^- \rightarrow \mathrm{t}\overline{\mathrm{t}}$",
    "clic_edm_ttbar_pf": r"$e^+e^- \rightarrow \mathrm{t}\overline{\mathrm{t}}$",
    "clic_edm_ttbar_pu10_pf": r"$e^+e^- \rightarrow \mathrm{t}\overline{\mathrm{t}}$, PU10",
    "clic_edm_ttbar_hits_pf": r"$e^+e^- \rightarrow \mathrm{t}\overline{\mathrm{t}}$",
    "clic_edm_qq_pf": r"$e^+e^- \rightarrow \gamma/\mathrm{Z}^* \rightarrow \mathrm{hadrons}$",
    "clic_edm_ww_fullhad_pf": r"$e^+e^- \rightarrow WW \rightarrow \mathrm{hadrons}$",
    "clic_edm_zh_tautau_pf": r"$e^+e^- \rightarrow ZH \rightarrow \tau \tau$",
    "cms_pf_qcd": r"QCD $p_T \in [15, 3000]\ \mathrm{GeV}$+PU",
    "cms_pf_ztt": r"$\mathrm{Z}\rightarrow \mathrm{\tau}\mathrm{\tau}$+PU",
    "cms_pf_ztt_nopu": r"$\mathrm{Z}\rightarrow \mathrm{\tau}\mathrm{\tau}$",
    "cms_pf_vbf": r"VBF+PU",
    "cms_pf_ttbar": r"$\mathrm{t}\overline{\mathrm{t}}$+PU",
    "cms_pf_ttbar_nopu": r"$\mathrm{t}\overline{\mathrm{t}}$",
    "cms_pf_qcd_nopu": r"QCD $p_T \in [15, 3000]\ \mathrm{GeV}$",
    "cms_pf_vbf_nopu": r"VBF",
    "cms_pf_multi_particle_gun": r"multi particle gun events",
    "cms_pf_single_electron": r"single electron particle gun events",
    "cms_pf_single_gamma": r"single photon gun events",
    "cms_pf_single_mu": r"single muon particle gun events",
    "cms_pf_single_pi": r"single pion particle gun events",
    "cms_pf_single_pi0": r"single neutral pion particle gun events",
    "cms_pf_single_proton": r"single proton particle gun events",
    "cms_pf_single_tau": r"single tau particle gun events",
    "cms_pf_single_k0": r"single K0 particle gun events",
    "cms_pf_sms_t1tttt": r"sms t1tttt events",
}


def load_loss_history(path, min_epoch=None, max_epoch=None):
    ret = {}
    for fi in glob.glob(path):
        data = json.load(open(fi))
        epoch = int(fi.split("_")[-1].split(".")[0])
        ret[epoch] = data

    if not max_epoch:
        max_epoch = max(ret.keys())
    if not min_epoch:
        min_epoch = min(ret.keys())

    ret2 = []
    for i in range(min_epoch, max_epoch + 1):
        ret2.append(ret[i])
    return pandas.DataFrame(ret2)


def loss_plot(train, test, fname, margin=0.05, smoothing=False, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    plt.figure()

    alpha = 0.2 if smoothing else 1.0
    l0 = None if smoothing else "train"
    l1 = None if smoothing else "test"
    p0 = plt.plot(train, alpha=alpha, label=l0)
    p1 = plt.plot(test, alpha=alpha, label=l1)

    if smoothing:
        train_smooth = np.convolve(train, np.ones(5) / 5, mode="valid")
        plt.plot(train_smooth, color=p0[0].get_color(), lw=2, label="train")
        test_smooth = np.convolve(test, np.ones(5) / 5, mode="valid")
        plt.plot(test_smooth, color=p1[0].get_color(), lw=2, label="test")

    plt.ylim(test[-1] * (1.0 - margin), test[-1] * (1.0 + margin))
    plt.legend(loc=3, frameon=False)
    plt.xlabel("epoch")

    if title:
        plt.title(title)

    save_img(
        fname,
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def format_dataset_name(dataset):
    return EVALUATION_DATASET_NAMES[dataset]


def med_iqr(arr):
    if len(arr) > 0:
        p25 = np.percentile(arr, 25)
        p50 = np.percentile(arr, 50)
        p75 = np.percentile(arr, 75)
    else:
        p25 = 0.0
        p50 = 0.0
        p75 = 0.0
    return p50, p75 - p25


def get_eff(df, pid):
    v0 = np.sum(df == pid)
    return v0 / len(df), np.sqrt(v0) / len(df)


def get_fake(df, pid):
    v0 = np.sum(df != pid)
    return v0 / len(df), np.sqrt(v0) / len(df)


def experiment_label(ax, experiment="CMS", tag1="Simulation Preliminary", tag2="Run 3 (14 TeV)", x0=0.01, x1=0.17, x2=0.98, y=1.01):
    plt.figtext(
        x0,
        y,
        experiment,
        fontweight="bold",
        wrap=True,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.figtext(
        x1,
        y,
        tag1,
        style="italic",
        wrap=True,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.figtext(
        x2,
        y,
        tag2,
        wrap=False,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )


def cms_label(ax):
    return experiment_label(ax, experiment="CMS", tag1="Simulation (Private Work)", tag2="Run 3 (14 TeV)", x1=0.13)


def clic_label(ax):
    return experiment_label(ax, experiment="Key4HEP-CLICdp", tag1="Sim.", tag2="ee (380 GeV)", x1=0.42)


def cld_label(ax):
    return experiment_label(ax, experiment="Key4HEP-CLD", tag1="Sim.", tag2="ee (365 GeV)", x1=0.35)


EXPERIMENT_LABELS = {
    "cms": cms_label,
    "clic": clic_label,
    "cld": cld_label,
}


def sample_label(ax, sample, additional_text="", x=0.03, y=0.97):
    text = EVALUATION_DATASET_NAMES[sample]
    plt.text(x, y, text + additional_text, ha="left", va="top", transform=ax.transAxes)


def particle_label(ax, pid):
    plt.text(
        0.03,
        0.92,
        pid_to_text[pid],
        va="top",
        ha="left",
        size=10,
        transform=ax.transAxes,
    )


def load_eval_data(path, max_files=None):
    yvals = []
    filenames = []
    print("path", path)

    filelist = list(glob.glob(path))

    if max_files is not None:
        filelist = filelist[:max_files]

    for fi in tqdm.tqdm(filelist):
        dd = awkward.from_parquet(fi)
        yvals.append(dd)
        filenames.append(fi)

    data = awkward.concatenate(yvals, axis=0)
    X = data["inputs"]

    yvals = {}
    for typ in ["target", "cand", "pred"]:
        for k in data["particles"][typ].fields:
            yvals["{}_{}".format(typ, k)] = data["particles"][typ][k]

    for typ in ["target", "cand", "pred"]:
        # Compute phi, px, py, pz
        yvals[typ + "_phi"] = np.arctan2(yvals[typ + "_sin_phi"], yvals[typ + "_cos_phi"])
        yvals[typ + "_px"] = yvals[typ + "_pt"] * yvals[typ + "_cos_phi"]
        yvals[typ + "_py"] = yvals[typ + "_pt"] * yvals[typ + "_sin_phi"]
        yvals[typ + "_pz"] = yvals[typ + "_pt"] * np.sinh(yvals[typ + "_eta"])

    for typ in ["gen", "cand", "pred", "target"]:
        # Get the jet vectors
        jetvec = vector.awk(data["jets"][typ])
        jetvec = awkward.Array(jetvec, with_name="Momentum4D")
        for k in ["pt", "eta", "phi", "energy"]:
            yvals["jets_{}_{}".format(typ, k)] = getattr(jetvec, k)

    for typ in ["target", "cand", "pred"]:
        for val in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
            yvals["{}_{}".format(typ, val)] = yvals["{}_{}".format(typ, val)] * (yvals["{}_cls_id".format(typ)] != 0)

    yvals.update(compute_jet_ratio(data, yvals))
    yvals["gen_met"] = data["genmet"]
    return yvals, X, filenames


def compute_jet_ratio(data, yvals):
    ret = {}
    # flatten across event dimension

    for match1, match2 in [("gen", "pred"), ("gen", "cand"), ("gen", "target"), ("target", "pred"), ("target", "cand")]:
        for val in ["pt", "eta"]:
            ret[f"jet_{match1}_to_{match2}_{match1}{val}"] = awkward.to_numpy(
                awkward.flatten(
                    yvals[f"jets_{match1}_{val}"][data["matched_jets"][f"{match1}_to_{match2}"][match1]],
                    axis=1,
                )
            )
            ret[f"jet_{match1}_to_{match2}_{match2}{val}"] = awkward.to_numpy(
                awkward.flatten(
                    yvals[f"jets_{match2}_{val}"][data["matched_jets"][f"{match1}_to_{match2}"][match2]],
                    axis=1,
                )
            )
            ret[f"jet_ratio_{match1}_to_{match2}_{val}"] = (
                ret[f"jet_{match1}_to_{match2}_{match2}{val}"] / ret[f"jet_{match1}_to_{match2}_{match1}{val}"]
            )
    return ret


def compute_met_and_ratio(yvals):
    msk_target = yvals["target_cls_id"] != 0
    target_px = yvals["target_px"][msk_target]
    target_py = yvals["target_py"][msk_target]

    msk_pred = yvals["pred_cls_id"] != 0
    pred_px = yvals["pred_px"][msk_pred]
    pred_py = yvals["pred_py"][msk_pred]

    msk_cand = yvals["cand_cls_id"] != 0
    cand_px = yvals["cand_px"][msk_cand]
    cand_py = yvals["cand_py"][msk_cand]

    gen_met = yvals["gen_met"]

    target_met = awkward.to_numpy(np.sqrt(np.sum(target_px, axis=1) ** 2 + np.sum(target_py, axis=1) ** 2))
    pred_met = awkward.to_numpy(np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2))
    cand_met = awkward.to_numpy(np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2))

    met_ratio_target = awkward.to_numpy(target_met / gen_met)
    met_ratio_pred = awkward.to_numpy(pred_met / gen_met)
    met_ratio_cand = awkward.to_numpy(cand_met / gen_met)

    return {
        "gen_met": gen_met,
        "target_met": target_met,
        "pred_met": pred_met,
        "cand_met": cand_met,
        "ratio_target": met_ratio_target,
        "ratio_pred": met_ratio_pred,
        "ratio_cand": met_ratio_cand,
    }


def compute_3dmomentum_and_ratio(yvals):
    msk_target = yvals["target_cls_id"] != 0
    target_px = yvals["target_px"][msk_target]
    target_py = yvals["target_py"][msk_target]
    target_pz = yvals["target_pz"][msk_target]

    msk_pred = yvals["pred_cls_id"] != 0
    pred_px = yvals["pred_px"][msk_pred]
    pred_py = yvals["pred_py"][msk_pred]
    pred_pz = yvals["pred_pz"][msk_pred]

    msk_cand = yvals["cand_cls_id"] != 0
    cand_px = yvals["cand_px"][msk_cand]
    cand_py = yvals["cand_py"][msk_cand]
    cand_pz = yvals["cand_pz"][msk_cand]

    target_mom = awkward.to_numpy(np.sqrt(np.sum(target_px, axis=1) ** 2 + np.sum(target_py, axis=1) ** 2 + np.sum(target_pz, axis=1) ** 2))
    pred_mom = awkward.to_numpy(np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2 + np.sum(pred_pz, axis=1) ** 2))
    cand_mom = awkward.to_numpy(np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2 + np.sum(cand_pz, axis=1) ** 2))

    mom_ratio_pred = awkward.to_numpy(pred_mom / target_mom)
    mom_ratio_cand = awkward.to_numpy(cand_mom / target_mom)

    return {
        "target_mom": target_mom,
        "pred_mom": pred_mom,
        "cand_mom": cand_mom,
        "ratio_pred": mom_ratio_pred,
        "ratio_cand": mom_ratio_cand,
    }


def save_img(outfile, epoch=None, cp_dir=None, comet_experiment=None):
    if cp_dir:
        image_path = str(cp_dir / outfile)
        plt.savefig(image_path, dpi=100, bbox_inches="tight")
        plt.savefig(image_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.clf()
        if comet_experiment:
            comet_experiment.log_image(image_path, step=epoch - 1)


def plot_jets(yvals, epoch=None, cp_dir=None, comet_experiment=None, sample=None, dataset=None):

    plt.figure()
    b = np.logspace(1, 3, 100)
    pt = awkward.to_numpy(awkward.flatten(yvals["jets_target_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_cand_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_pred_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_gen_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Truth",
    )

    plt.xscale("log")
    plt.xlabel("jet $p_T$")
    plt.ylabel("Jets / bin")
    plt.legend(loc="best")

    plt.yscale("log")
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 10 * ylim[1])

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_pt_log.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    b = np.linspace(0, 1000, 100)

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_target_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_cand_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_pred_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_gen_pt"]))
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Truth",
    )

    plt.xlabel("jet $p_T$")
    plt.ylabel("Jets / bin")
    plt.yscale("log")
    plt.legend(loc="best")
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 10 * ylim[1])

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_pt.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    b = np.linspace(-5, 5, 100)
    eta = awkward.to_numpy(awkward.flatten(yvals["jets_target_eta"]))
    plt.hist(
        eta,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )

    eta = awkward.to_numpy(awkward.flatten(yvals["jets_cand_eta"]))
    plt.hist(
        eta,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )

    eta = awkward.to_numpy(awkward.flatten(yvals["jets_pred_eta"]))
    plt.hist(
        eta,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )

    eta = awkward.to_numpy(awkward.flatten(yvals["jets_gen_eta"]))
    plt.hist(
        eta,
        bins=b,
        histtype="step",
        lw=2,
        label="Truth",
    )

    plt.xlabel("jet $\eta$")
    plt.ylabel("Jets / bin")
    plt.yscale("log")
    plt.legend(loc="best")
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 10 * ylim[1])

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_jet_ratio(
    yvals,
    epoch=None,
    cp_dir=None,
    comet_experiment=None,
    title=None,
    bins=None,
    file_modifier="",
    logy=False,
    dataset=None,
    sample=None,
):
    plt.figure()
    ax = plt.axes()

    if bins is None:
        bins = np.linspace(0, 5, 500)

    ret_dict = {}
    p = med_iqr(yvals["jet_ratio_gen_to_target_pt"])
    ret_dict["jet_ratio_gen_to_target_pt"] = {
        "med": p[0],
        "iqr": p[1],
        "match_frac": awkward.count(yvals["jet_ratio_gen_to_target_pt"]) / awkward.count(yvals["jets_gen_pt"]),
    }
    plt.hist(
        yvals["jet_ratio_gen_to_target_pt"],
        bins=bins,
        histtype="step",
        lw=2,
        label="target $({:.2f}\pm{:.2f})$".format(p[0], p[1]),
    )

    p = med_iqr(yvals["jet_ratio_gen_to_cand_pt"])
    ret_dict["jet_ratio_gen_to_cand_pt"] = {
        "med": p[0],
        "iqr": p[1],
        "match_frac": awkward.count(yvals["jet_ratio_gen_to_cand_pt"]) / awkward.count(yvals["jets_gen_pt"]),
    }
    plt.hist(
        yvals["jet_ratio_gen_to_cand_pt"],
        bins=bins,
        histtype="step",
        lw=2,
        label="PF $({:.2f}\pm{:.2f})$".format(p[0], p[1]),
    )

    p = med_iqr(yvals["jet_ratio_gen_to_pred_pt"])
    ret_dict["jet_ratio_gen_to_pred_pt"] = {
        "med": p[0],
        "iqr": p[1],
        "match_frac": awkward.count(yvals["jet_ratio_gen_to_pred_pt"]) / awkward.count(yvals["jets_gen_pt"]),
    }
    plt.hist(
        yvals["jet_ratio_gen_to_pred_pt"],
        bins=bins,
        histtype="step",
        lw=2,
        label="MLPF $({:.2f}\pm{:.2f})$".format(p[0], p[1]),
    )

    plt.xlabel(labels["reco_gen_jet_ratio"])
    plt.ylabel("Matched jets / bin")
    plt.legend(loc="best")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1.2 * ylim[1])

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(10, 10 * ylim[1])

    save_img(
        "jet_res{}.png".format(file_modifier),
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    ax = plt.axes()

    p = med_iqr(yvals["jet_ratio_target_to_cand_pt"])
    ret_dict["jet_ratio_target_to_cand_pt"] = {
        "med": p[0],
        "iqr": p[1],
        "match_frac": awkward.count(yvals["jet_ratio_target_to_cand_pt"]) / awkward.count(yvals["jets_target_pt"]),
    }
    plt.plot([], [])
    plt.hist(
        yvals["jet_ratio_target_to_cand_pt"],
        bins=bins,
        histtype="step",
        lw=2,
        label="PF $({:.2f}\pm{:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(yvals["jet_ratio_target_to_pred_pt"])
    ret_dict["jet_ratio_target_to_pred_pt"] = {
        "med": p[0],
        "iqr": p[1],
        "match_frac": awkward.count(yvals["jet_ratio_target_to_pred_pt"]) / awkward.count(yvals["jets_target_pt"]),
    }
    plt.hist(
        yvals["jet_ratio_target_to_pred_pt"],
        bins=bins,
        histtype="step",
        lw=2,
        label="MLPF $({:.2f}\pm{:.2f})$".format(p[0], p[1]),
    )
    plt.xlabel(labels["reco_target_jet_ratio"])
    plt.ylabel("Matched jets / bin")
    plt.legend(loc="best")
    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_res{}_vstarget.png".format(file_modifier),
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )
    return ret_dict


def plot_met(met_ratio, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    maxval = max(
        [
            np.max(met_ratio["target_met"]),
            np.max(met_ratio["cand_met"]),
            np.max(met_ratio["pred_met"]),
        ]
    )
    minval = min(
        [
            np.min(met_ratio["target_met"]),
            np.min(met_ratio["cand_met"]),
            np.min(met_ratio["pred_met"]),
        ]
    )
    maxval = math.ceil(np.log10(maxval))
    minval = math.floor(np.log10(max(minval, 1e-2)))

    plt.figure()
    b = np.logspace(minval, maxval, 500)
    plt.hist(
        met_ratio["target_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )
    plt.hist(
        met_ratio["cand_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )
    plt.hist(
        met_ratio["pred_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )
    plt.hist(
        met_ratio["gen_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="Truth",
    )
    plt.xlabel(labels["met"])
    plt.ylabel("Events / bin")
    plt.legend(loc="best", title=title)
    plt.xscale("log")
    plt.yscale("log")

    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 10 * ylim[1])

    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)

    save_img("met_log.png", epoch, cp_dir=cp_dir, comet_experiment=comet_experiment)

    b = np.linspace(0, 300, 500)
    plt.hist(
        met_ratio["target_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )
    plt.hist(
        met_ratio["cand_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )
    plt.hist(
        met_ratio["pred_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )
    plt.hist(
        met_ratio["gen_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="Truth",
    )
    plt.xlabel(labels["met"])
    plt.ylabel("Events / bin")
    plt.legend(loc="best", title=title)
    plt.yscale("log")
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 10 * ylim[1])
    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)

    save_img("met.png", epoch, cp_dir=cp_dir, comet_experiment=comet_experiment)


def plot_met_ratio(
    met_ratio,
    epoch=None,
    cp_dir=None,
    comet_experiment=None,
    title=None,
    bins=None,
    file_modifier="",
    logy=False,
    sample=None,
    dataset=None,
):
    plt.figure()
    ax = plt.axes()
    if bins is None:
        bins = np.linspace(0, 20, 500)

    mask = met_ratio["gen_met"] > 5

    p = med_iqr(met_ratio["ratio_target"][mask])
    plt.hist(
        met_ratio["ratio_target"][mask],
        bins=bins,
        histtype="step",
        lw=2,
        label="target $({:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )

    p = med_iqr(met_ratio["ratio_cand"][mask])
    plt.hist(
        met_ratio["ratio_cand"][mask],
        bins=bins,
        histtype="step",
        lw=2,
        label="PF $({:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )

    p = med_iqr(met_ratio["ratio_pred"][mask])
    plt.hist(
        met_ratio["ratio_pred"][mask],
        bins=bins,
        histtype="step",
        lw=2,
        label="MLPF $({:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )

    plt.xlabel(labels["reco_gen_met_ratio"])
    plt.ylabel("Events / bin")
    plt.legend(loc="best", title=title)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1.2 * ylim[1])

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(10, 10 * ylim[1])

    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)

    save_img(
        "met_res{}.png".format(file_modifier),
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_3dmomentum_ratio(mom_ratio, epoch=None, cp_dir=None, comet_experiment=None, title=None, bins=None, file_modifier="", logy=False):
    plt.figure()
    ax = plt.axes()
    if bins is None:
        bins = np.linspace(0, 20, 100)

    p = med_iqr(mom_ratio["ratio_cand"])
    plt.hist(
        mom_ratio["ratio_cand"],
        bins=bins,
        histtype="step",
        lw=2,
        label="PF $({:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(mom_ratio["ratio_pred"])
    plt.hist(
        mom_ratio["ratio_pred"],
        bins=bins,
        histtype="step",
        lw=2,
        label="MLPF $({:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    plt.xlabel(labels["reco_gen_mom_ratio"])
    plt.ylabel("Events / bin")
    plt.legend(loc="best", title=title)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1.2 * ylim[1])

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(10, 10 * ylim[1])

    save_img(
        "mom_res{}.png".format(file_modifier),
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def compute_distances(distribution_1, distribution_2, ratio):
    if len(distribution_1) > 0 and len(distribution_2) > 0:
        wd = scipy.stats.wasserstein_distance(distribution_1, distribution_2)
        p25 = np.percentile(ratio, 25)
        p50 = np.percentile(ratio, 50)
        p75 = np.percentile(ratio, 75)
    else:
        wd = 0.0
        p25 = 0.0
        p50 = 0.0
        p75 = 0.0
    iqr = p75 - p25
    return {"wd": wd, "p25": p25, "p50": p50, "p75": p75, "iqr": iqr}


def plot_rocs(yvals, class_names, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    ncls = len(yvals["gen_cls"][0, 0])
    plt.figure()
    for icls in range(ncls):
        predvals = awkward.to_numpy(awkward.flatten(yvals["pred_cls"][:, :, icls]))
        truevals = awkward.to_numpy(awkward.flatten(yvals["gen_cls_id"] == icls))
        fpr, tpr, _ = sklearn.metrics.roc_curve(truevals, predvals)
        plt.plot(fpr, tpr, label=class_names[icls])
    plt.xlim(1e-7, 1)
    plt.ylim(1e-7, 1)
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    if title:
        plt.title(title)
    plt.yscale("log")
    plt.xscale("log")
    save_img(
        "roc.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_num_elements(X, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    # compute the number of unpadded elements per event
    num_Xelems = awkward.sum(X[:, :, 0] != 0, axis=-1)
    maxval = np.max(num_Xelems)

    plt.figure()
    plt.hist(num_Xelems, bins=np.linspace(0, int(1.2 * maxval), 100))
    plt.xlabel("PFElements / event")
    plt.ylabel("Events / bin")
    if title:
        plt.title(title)
    save_img(
        "num_elements.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_sum_energy(yvals, class_names, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    cls_ids = np.unique(awkward.values_astype(awkward.flatten(yvals["gen_cls_id"]), np.int64))

    for cls_id in cls_ids:
        if cls_id == 0:
            msk = yvals["gen_cls_id"] != 0
            clname = "all particles"
        else:
            msk = yvals["gen_cls_id"] == cls_id
            clname = class_names[cls_id]

        sum_gen_energy = awkward.to_numpy(awkward.sum(yvals["gen_energy"][msk], axis=1))
        sum_cand_energy = awkward.to_numpy(awkward.sum(yvals["cand_energy"][msk], axis=1))
        sum_pred_energy = awkward.to_numpy(awkward.sum(yvals["pred_energy"][msk], axis=1))

        mean = np.mean(sum_gen_energy)
        std = np.std(sum_gen_energy)
        max_e = mean + 2 * std
        min_e = max(mean - 2 * std, 0)

        # 1D hist of sum energy
        b = np.linspace(min_e, max_e, 100)
        plt.figure()
        plt.hist(sum_cand_energy, bins=b, label="PF", histtype="step", lw=2)
        plt.hist(sum_pred_energy, bins=b, label="MLPF", histtype="step", lw=2)
        plt.hist(sum_gen_energy, bins=b, label="Truth", histtype="step", lw=2)
        plt.xlabel("total energy / event [GeV]")
        plt.ylabel("events / bin")
        if title:
            plt.title(title + ", " + clname)
        plt.legend(loc="best")
        save_img(
            "sum_energy_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

        # 2D hist of gen vs. PF energy
        plt.figure()
        plt.hist2d(sum_gen_energy, sum_cand_energy, bins=(b, b), cmap="hot_r")
        plt.plot([min_e, max_e], [min_e, max_e], color="black", ls="--")
        plt.xlabel("total true energy / event [GeV]")
        plt.ylabel("total PF energy / event [GeV]")
        if title:
            plt.title(title + ", " + clname)
        save_img(
            "sum_gen_cand_energy_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

        # 2D hist of gen vs. MLPF energy
        plt.figure()
        plt.hist2d(sum_gen_energy, sum_pred_energy, bins=(b, b), cmap="hot_r")
        plt.plot([min_e, max_e], [min_e, max_e], color="black", ls="--")
        plt.xlabel("total true energy / event [GeV]")
        plt.ylabel("total MLPF energy / event [GeV]")
        if title:
            plt.title(title + ", " + clname)
        save_img(
            "sum_gen_pred_energy_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

        min_e = np.log10(max(min_e, 1e-2))
        max_e = np.log10(max_e) + 1

        b = np.logspace(min_e, max_e, 100)
        plt.figure()
        plt.hist2d(sum_gen_energy, sum_cand_energy, bins=(b, b), cmap="hot_r")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(
            [10**min_e, 10**max_e],
            [10**min_e, 10**max_e],
            color="black",
            ls="--",
        )
        plt.xlabel("total true energy / event [GeV]")
        plt.ylabel("total reconstructed energy / event [GeV]")
        if title:
            plt.title(title + ", " + clname + ", PF")
        save_img(
            "sum_gen_cand_energy_log_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

        b = np.logspace(min_e, max_e, 100)
        plt.figure()
        plt.hist2d(sum_gen_energy, sum_pred_energy, bins=(b, b), cmap="hot_r")
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(
            [10**min_e, 10**max_e],
            [10**min_e, 10**max_e],
            color="black",
            ls="--",
        )
        plt.xlabel("total true energy / event [GeV]")
        plt.ylabel("total reconstructed energy / event [GeV]")
        if title:
            plt.title(title + ", " + clname + ", MLPF")
        save_img(
            "sum_gen_pred_energy_log_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )


def plot_particle_multiplicity(X, yvals, class_names, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    cls_ids = np.unique(awkward.values_astype(awkward.flatten(yvals["gen_cls_id"]), np.int64))

    for cls_id in cls_ids:
        if cls_id == 0:
            continue

        clname = class_names[cls_id]

        plt.figure()
        gen_vals = awkward.sum(yvals["gen_cls_id"][X[:, :, 0] != 0] == cls_id, axis=1)
        cand_vals = awkward.sum(yvals["cand_cls_id"][X[:, :, 0] != 0] == cls_id, axis=1)
        pred_vals = awkward.sum(yvals["pred_cls_id"][X[:, :, 0] != 0] == cls_id, axis=1)

        plt.scatter(gen_vals, cand_vals, alpha=0.5)
        plt.scatter(gen_vals, pred_vals, alpha=0.5)
        max_val = 1.2 * np.max(gen_vals)
        plt.plot([0, max_val], [0, max_val], color="black")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        if title:
            plt.title(title + ", " + clname)

        save_img(
            "particle_multiplicity_{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )


def plot_particle_response(X, yvals, class_names, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    msk_cand = yvals["cand_cls_id"] != 0
    msk_pred = yvals["pred_cls_id"] != 0
    msk_target = yvals["target_cls_id"] != 0

    typ_ids = np.unique(awkward.values_astype(awkward.flatten(X[:, :, 0]), np.int64))

    for typ in typ_ids:
        if typ == 0:
            continue
        msk_typ = X[:, :, 0] == typ

        for idx_elem_feature, val in [(1, "pt"), (2, "eta"), (3, "sin_phi"), (4, "cos_phi"), (5, "energy")]:
            elem_x = awkward.to_numpy(awkward.flatten(X[:, :, idx_elem_feature][msk_target & msk_typ]))
            target_x = awkward.to_numpy(awkward.flatten(yvals["target_" + val][msk_target & msk_typ]))
            ratio_elem_target = elem_x / target_x

            # cases where targets and PF candidates exist
            target_x = awkward.to_numpy(awkward.flatten(yvals["target_" + val][msk_target & msk_cand & msk_typ]))
            cand_x = awkward.to_numpy(awkward.flatten(yvals["cand_" + val][msk_target & msk_cand & msk_typ]))
            ratio_cand_target = cand_x / target_x

            # cases where targets and MLPF candidates exist
            target_x = awkward.to_numpy(awkward.flatten(yvals["target_" + val][msk_target & msk_pred & msk_typ]))
            pred_x = awkward.to_numpy(awkward.flatten(yvals["pred_" + val][msk_target & msk_pred & msk_typ]))
            ratio_pred_target = pred_x / target_x

            plt.figure()
            ax = plt.axes()
            b = np.linspace(0, 5, 100)
            plt.hist(ratio_elem_target, bins=b, label="PFElements", histtype="step", lw=2)
            plt.hist(ratio_cand_target, bins=b, label="PF", histtype="step", lw=2)
            plt.hist(ratio_pred_target, bins=b, label="MLPF", histtype="step", lw=2)
            plt.legend(loc="best")
            EXPERIMENT_LABELS[dataset](ax)
            sample_label(ax, sample)
            ax.text(0.03, 0.88, "PFELement {}".format(typ), transform=ax.transAxes)
            plt.xlabel("Reconstructed / target particle " + val)
            save_img(
                "particle_{}_ratio_typ{}.png".format(val, typ),
                epoch,
                cp_dir=cp_dir,
                comet_experiment=comet_experiment,
            )
            plt.clf()


def plot_particle_ratio(yvals, class_names, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    msk_cand = yvals["cand_cls_id"] != 0
    msk_pred = yvals["pred_cls_id"] != 0
    msk_target = yvals["target_cls_id"] != 0

    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_pt"][msk_target & msk_cand]))
    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_pt"][msk_target & msk_pred]))
    target_cand_pt = awkward.to_numpy(awkward.flatten(yvals["target_pt"][msk_target & msk_cand]))
    target_pred_pt = awkward.to_numpy(awkward.flatten(yvals["target_pt"][msk_target & msk_pred]))
    ratio_cand_pt = cand_pt / target_cand_pt
    ratio_pred_pt = pred_pt / target_pred_pt

    cand_e = awkward.to_numpy(awkward.flatten(yvals["cand_energy"][msk_target & msk_cand]))
    pred_e = awkward.to_numpy(awkward.flatten(yvals["pred_energy"][msk_target & msk_pred]))
    target_cand_e = awkward.to_numpy(awkward.flatten(yvals["target_energy"][msk_target & msk_cand]))
    target_pred_e = awkward.to_numpy(awkward.flatten(yvals["target_energy"][msk_target & msk_pred]))
    ratio_cand_e = cand_e / target_cand_e
    ratio_pred_e = pred_e / target_pred_e

    target_cls_id = awkward.flatten(yvals["target_cls_id"][msk_target])
    target_cls_id1 = awkward.flatten(yvals["target_cls_id"][msk_target & msk_cand])
    target_cls_id2 = awkward.flatten(yvals["target_cls_id"][msk_target & msk_pred])
    cls_ids = np.unique(awkward.values_astype(target_cls_id, np.int64))
    for cls_id in cls_ids:
        if cls_id == 0:
            continue
        particle_label_nice = pid_to_text[CLASS_LABELS[dataset][cls_id]]
        plt.figure()
        ax = plt.axes()
        b = np.linspace(0, 3, 100)
        plt.plot([], [])
        plt.hist(ratio_cand_pt[target_cls_id1 == cls_id], bins=b, label="PF", histtype="step")
        plt.hist(ratio_pred_pt[target_cls_id2 == cls_id], bins=b, label="MLPF", histtype="step")
        plt.legend(loc="best")

        EXPERIMENT_LABELS[dataset](ax)
        sample_label(ax, sample)
        ax.text(0.03, 0.88, particle_label_nice, transform=ax.transAxes)
        plt.xlabel("Reconstructed / target $p_T$")
        save_img(
            "particle_pt_ratio_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )
        plt.clf()

        plt.figure()
        ax = plt.axes()
        b = np.linspace(0, 3, 100)
        plt.plot([], [])
        plt.hist(ratio_cand_e[target_cls_id1 == cls_id], bins=b, label="PF", histtype="step")
        plt.hist(ratio_pred_e[target_cls_id2 == cls_id], bins=b, label="MLPF", histtype="step")
        plt.legend(loc="best")
        EXPERIMENT_LABELS[dataset](ax)
        sample_label(ax, sample)
        ax.text(0.03, 0.88, particle_label_nice, transform=ax.transAxes)
        plt.xlabel("Reconstructed / target $E$")
        save_img(
            "particle_e_ratio_cls{}.png".format(cls_id),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )
        plt.clf()


def plot_elements(X, yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    uniq_elems = np.unique(awkward.flatten(X[:, :, 0]))
    for elem in uniq_elems:
        elem = int(elem)
        msk_elem = (X[:, :, 0] == elem) & (yvals["target_cls_id"] != 0) & (yvals["pred_cls_id"] != 0)
        ratio_target = np.log(awkward.to_numpy(awkward.flatten((yvals["target_pt"] / X[:, :, 1])[msk_elem])))
        ratio_pred = np.log(awkward.to_numpy(awkward.flatten((yvals["pred_pt"] / X[:, :, 1])[msk_elem])))
        ratio_target[np.isnan(ratio_target)] = 0
        ratio_pred[np.isnan(ratio_pred)] = 0
        ratio_target[np.isinf(ratio_target)] = 0
        ratio_pred[np.isinf(ratio_pred)] = 0

        mean_ratio = np.mean(ratio_target)
        std_ratio = np.std(ratio_target)
        bins = np.linspace(mean_ratio - std_ratio, mean_ratio + std_ratio, 100)

        plt.figure()
        ax = plt.axes()
        plt.hist2d(ratio_target, ratio_pred, bins=bins, cmap="hot_r")
        plt.xlabel("log [target pT / elem pT]")
        plt.ylabel("log [reco pT / elem pT]")
        plt.plot(
            [mean_ratio - 2 * std_ratio, mean_ratio + 2 * std_ratio], [mean_ratio - 2 * std_ratio, mean_ratio + 2 * std_ratio], color="black", ls="--"
        )
        EXPERIMENT_LABELS[dataset](ax)
        sample_label(ax, sample)
        save_img(
            "elem_{}_pt_mlpf.png".format(elem),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )
        plt.clf()

        plt.figure()
        ax = plt.axes()
        plt.hist(ratio_target, bins=bins, label="Target", histtype="step")
        plt.hist(ratio_pred, bins=bins, label="MLPF", histtype="step")
        plt.xlabel("log [pT / elem pT]")
        plt.legend(loc="best")
        EXPERIMENT_LABELS[dataset](ax)
        sample_label(ax, sample)
        save_img(
            "elem_{}_pt.png".format(elem),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )
        plt.clf()


def plot_particles(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    msk_cand = yvals["cand_cls_id"] != 0
    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_pt"][msk_cand], axis=1))

    msk_pred = yvals["pred_cls_id"] != 0
    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_pt"][msk_pred], axis=1))

    msk_target = yvals["target_cls_id"] != 0
    target_pt = awkward.to_numpy(awkward.flatten(yvals["target_pt"][msk_target], axis=1))

    b = np.logspace(-2, 4, 100)
    plt.figure()
    ax = plt.gca()
    plt.hist(
        target_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )
    plt.hist(
        cand_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )
    plt.hist(
        pred_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Particle $p_T$ [GeV]")
    plt.ylabel("Number of particles / bin")
    plt.legend(loc="best")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "particle_pt_log.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    b = np.linspace(0, 200, 100)
    plt.figure()
    ax = plt.gca()
    plt.hist(
        target_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )
    plt.hist(
        cand_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )
    plt.hist(
        pred_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )
    plt.yscale("log")
    plt.xlabel("Particle $p_T$ [GeV]")
    plt.ylabel("Number of particles / bin")
    plt.legend(loc="best")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)

    save_img(
        "particle_pt.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    msk_cand = yvals["cand_cls_id"] != 0
    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_eta"][msk_cand], axis=1))

    msk_pred = yvals["pred_cls_id"] != 0
    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_eta"][msk_pred], axis=1))

    msk_target = yvals["target_cls_id"] != 0
    target_pt = awkward.to_numpy(awkward.flatten(yvals["target_eta"][msk_target], axis=1))

    b = np.linspace(-8, 8, 100)
    plt.figure()
    ax = plt.gca()
    plt.hist(
        target_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Target",
    )
    plt.hist(
        cand_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF",
    )
    plt.hist(
        pred_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF",
    )
    plt.xlabel(r"Particle $\eta$")
    plt.ylabel("Number of particles / bin")
    plt.legend(loc="best")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "particle_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    msk_cand = yvals["cand_cls_id"] != 0
    msk_pred = yvals["pred_cls_id"] != 0
    msk_target = yvals["target_cls_id"] != 0

    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_pt"][msk_cand & msk_target], axis=1))
    target_pt = awkward.to_numpy(awkward.flatten(yvals["target_pt"][msk_cand & msk_target], axis=1))
    b = np.logspace(-1, 2, 100)
    plt.figure()
    ax = plt.axes()
    plt.hist2d(target_pt, cand_pt, bins=(b, b), cmap="hot_r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Target particle $p_T$ [GeV]")
    plt.ylabel("Reconstructed particle $p_T$ [GeV]")
    plt.plot([10**-1, 10**2], [10**-1, 10**2], color="black", ls="--")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample, additional_text="\nPF")
    save_img(
        "particle_pt_target_vs_pf.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_pt"][msk_pred & msk_target], axis=1))
    target_pt = awkward.to_numpy(awkward.flatten(yvals["target_pt"][msk_pred & msk_target], axis=1))
    b = np.logspace(-1, 2, 100)
    plt.figure()
    ax = plt.axes()
    plt.hist2d(target_pt, pred_pt, bins=(b, b), cmap="hot_r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Target particle $p_T$ [GeV]")
    plt.ylabel("Reconstructed particle $p_T$ [GeV]")
    plt.plot([10**-1, 10**2], [10**-1, 10**2], color="black", ls="--")

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample, additional_text="\nMLPF")
    save_img(
        "particle_pt_target_vs_mlpf.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_jet_response_binned_vstarget(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    pf_genjet_pt = yvals["jet_target_to_cand_targetpt"]
    mlpf_genjet_pt = yvals["jet_target_to_pred_targetpt"]

    pf_response = yvals["jet_ratio_target_to_cand_pt"]
    mlpf_response = yvals["jet_ratio_target_to_pred_pt"]

    genjet_bins = [10, 20, 40, 60, 80, 100, 200, 400, 800]

    x_vals = []
    pf_vals = []
    mlpf_vals = []

    b = np.linspace(0.5, 1.5, 500)
    for ibin in range(len(genjet_bins) - 1):
        plt.figure()
        ax = plt.axes()
        lim_low = genjet_bins[ibin]
        lim_hi = genjet_bins[ibin + 1]
        x_vals.append(np.mean([lim_low, lim_hi]))

        mask_genjet = (pf_genjet_pt > lim_low) & (pf_genjet_pt <= lim_hi)
        pf_subsample = pf_response[mask_genjet]
        if len(pf_subsample) > 0:
            pf_p25 = np.percentile(pf_subsample, 25)
            pf_p50 = np.percentile(pf_subsample, 50)
            pf_p75 = np.percentile(pf_subsample, 75)
        else:
            pf_p25 = 0
            pf_p50 = 0
            pf_p75 = 0
        pf_vals.append([pf_p25, pf_p50, pf_p75])

        mask_genjet = (mlpf_genjet_pt > lim_low) & (mlpf_genjet_pt <= lim_hi)
        mlpf_subsample = mlpf_response[mask_genjet]
        if len(mlpf_subsample) > 0:
            mlpf_p25 = np.percentile(mlpf_subsample, 25)
            mlpf_p50 = np.percentile(mlpf_subsample, 50)
            mlpf_p75 = np.percentile(mlpf_subsample, 75)
        else:
            mlpf_p25 = 0
            mlpf_p50 = 0
            mlpf_p75 = 0
        mlpf_vals.append([mlpf_p25, mlpf_p50, mlpf_p75])

        plt.plot([], [])
        plt.hist(pf_subsample, bins=b, histtype="step", lw=1, label="PF ({:.4f}, {:.4f}, {:.4f})".format(pf_p25, pf_p50, pf_p75))
        plt.hist(mlpf_subsample, bins=b, histtype="step", lw=1, label="MLPF ({:.4f}, {:.4f}, {:.4f})".format(mlpf_p25, mlpf_p50, mlpf_p75))

        plt.ylabel("Matched jets / bin")
        plt.xlabel(labels["reco_target_jet_ratio"])
        plt.axvline(1.0, ymax=0.7, color="black", ls="--")
        plt.legend(loc=1, fontsize=16)
        plt.title(labels["gen_jet_range"].format(lim_low, lim_hi))
        plt.yscale("log")
        save_img(
            "jet_response_vstarget_binned_pt{}.png".format(lim_low),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

    x_vals = np.array(x_vals)
    pf_vals = np.array(pf_vals)
    mlpf_vals = np.array(mlpf_vals)

    # Plot median and IQR as a function of gen pt
    plt.figure()
    ax = plt.gca()
    plt.plot([], [])
    plt.plot(x_vals, pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.ylabel("Response median")
    plt.xlabel(labels["target_jet"])
    plt.tight_layout()
    plt.axhline(1.0, color="black", ls="--", lw=0.5)

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_response_med_pt_vstarget.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    ax = plt.gca()
    plt.plot([], [])
    plt.plot(x_vals, (pf_vals[:, 2] - pf_vals[:, 0]) / pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, (mlpf_vals[:, 2] - mlpf_vals[:, 0]) / mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.ylabel("Response IQR / median")
    plt.xlabel(labels["target_jet"])
    plt.ylim(bottom=0)
    plt.tight_layout()

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_response_iqr_over_med_pt_vstarget.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_jet_response_binned(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    target_genjet_pt = yvals["jet_gen_to_target_genpt"]
    pf_genjet_pt = yvals["jet_gen_to_cand_genpt"]
    mlpf_genjet_pt = yvals["jet_gen_to_pred_genpt"]

    target_response = yvals["jet_ratio_gen_to_target_pt"]
    pf_response = yvals["jet_ratio_gen_to_cand_pt"]
    mlpf_response = yvals["jet_ratio_gen_to_pred_pt"]

    genjet_bins = [10, 20, 40, 60, 80, 100, 200, 400, 800]

    x_vals = []
    target_vals = []
    pf_vals = []
    mlpf_vals = []

    b = np.linspace(0.5, 1.5, 500)
    for ibin in range(len(genjet_bins) - 1):
        plt.figure()
        ax = plt.axes()
        lim_low = genjet_bins[ibin]
        lim_hi = genjet_bins[ibin + 1]
        x_vals.append(np.mean([lim_low, lim_hi]))

        mask_genjet = (target_genjet_pt > lim_low) & (target_genjet_pt <= lim_hi)
        target_subsample = target_response[mask_genjet]
        if len(target_subsample) > 0:
            target_p25 = np.percentile(target_subsample, 25)
            target_p50 = np.percentile(target_subsample, 50)
            target_p75 = np.percentile(target_subsample, 75)
        else:
            target_p25 = 0
            target_p50 = 0
            target_p75 = 0
        target_vals.append([target_p25, target_p50, target_p75])

        mask_genjet = (pf_genjet_pt > lim_low) & (pf_genjet_pt <= lim_hi)
        pf_subsample = pf_response[mask_genjet]
        if len(pf_subsample) > 0:
            pf_p25 = np.percentile(pf_subsample, 25)
            pf_p50 = np.percentile(pf_subsample, 50)
            pf_p75 = np.percentile(pf_subsample, 75)
        else:
            pf_p25 = 0
            pf_p50 = 0
            pf_p75 = 0
        pf_vals.append([pf_p25, pf_p50, pf_p75])

        mask_genjet = (mlpf_genjet_pt > lim_low) & (mlpf_genjet_pt <= lim_hi)
        mlpf_subsample = mlpf_response[mask_genjet]
        if len(mlpf_subsample) > 0:
            mlpf_p25 = np.percentile(mlpf_subsample, 25)
            mlpf_p50 = np.percentile(mlpf_subsample, 50)
            mlpf_p75 = np.percentile(mlpf_subsample, 75)
        else:
            mlpf_p25 = 0
            mlpf_p50 = 0
            mlpf_p75 = 0
        mlpf_vals.append([mlpf_p25, mlpf_p50, mlpf_p75])

        plt.hist(target_subsample, bins=b, histtype="step", lw=1, label="Target ({:.4f}, {:.4f}, {:.4f})".format(target_p25, target_p50, target_p75))
        plt.hist(pf_subsample, bins=b, histtype="step", lw=1, label="PF ({:.4f}, {:.4f}, {:.4f})".format(pf_p25, pf_p50, pf_p75))
        plt.hist(mlpf_subsample, bins=b, histtype="step", lw=1, label="MLPF ({:.4f}, {:.4f}, {:.4f})".format(mlpf_p25, mlpf_p50, mlpf_p75))

        plt.ylabel("Matched jets / bin")
        plt.xlabel(labels["reco_gen_jet_ratio"])
        plt.axvline(1.0, ymax=0.7, color="black", ls="--")
        plt.legend(loc=1, fontsize=16)
        plt.title(labels["gen_jet_range"].format(lim_low, lim_hi))
        plt.yscale("log")
        save_img(
            "jet_response_binned_{}.png".format(lim_low),
            epoch,
            cp_dir=cp_dir,
            comet_experiment=comet_experiment,
        )

    x_vals = np.array(x_vals)
    target_vals = np.array(target_vals)
    pf_vals = np.array(pf_vals)
    mlpf_vals = np.array(mlpf_vals)

    # Plot median and IQR as a function of gen pt
    plt.figure()
    ax = plt.gca()
    plt.plot(x_vals, target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.ylabel("Response median")
    plt.xlabel(labels["gen_jet"])
    plt.tight_layout()
    plt.axhline(1.0, color="black", ls="--", lw=0.5)

    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_response_med_pt.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    ax = plt.gca()
    plt.plot(x_vals, (target_vals[:, 2] - target_vals[:, 0]) / target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, (pf_vals[:, 2] - pf_vals[:, 0]) / pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, (mlpf_vals[:, 2] - mlpf_vals[:, 0]) / mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.ylabel("Response IQR / median")
    plt.xlabel(labels["gen_jet"])
    plt.tight_layout()
    plt.ylim(bottom=0)
    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "jet_response_iqr_over_med_pt.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_jet_response_binned_eta(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    target_genjet_eta = yvals["jet_gen_to_target_geneta"]
    pf_genjet_eta = yvals["jet_gen_to_cand_geneta"]
    mlpf_genjet_eta = yvals["jet_gen_to_pred_geneta"]

    target_response = yvals["jet_ratio_gen_to_target_pt"]
    pf_response = yvals["jet_ratio_gen_to_cand_pt"]
    mlpf_response = yvals["jet_ratio_gen_to_pred_pt"]

    genjet_bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    x_vals = []
    target_vals = []
    pf_vals = []
    mlpf_vals = []
    b = np.linspace(0, 5, 100)

    fig, axs = plt.subplots(3, 3, figsize=(3 * 5, 3 * 5))
    axs = axs.flatten()
    for ibin in range(len(genjet_bins) - 1):
        lim_low = genjet_bins[ibin]
        lim_hi = genjet_bins[ibin + 1]
        x_vals.append(np.mean([lim_low, lim_hi]))

        mask_genjet = (target_genjet_eta > lim_low) & (target_genjet_eta <= lim_hi)
        target_subsample = target_response[mask_genjet]
        if len(target_subsample) > 0:
            target_p25 = np.percentile(target_subsample, 25)
            target_p50 = np.percentile(target_subsample, 50)
            target_p75 = np.percentile(target_subsample, 75)
        else:
            target_p25 = 0
            target_p50 = 0
            target_p75 = 0
        target_vals.append([target_p25, target_p50, target_p75])

        mask_genjet = (pf_genjet_eta > lim_low) & (pf_genjet_eta <= lim_hi)
        pf_subsample = pf_response[mask_genjet]
        if len(pf_subsample) > 0:
            pf_p25 = np.percentile(pf_subsample, 25)
            pf_p50 = np.percentile(pf_subsample, 50)
            pf_p75 = np.percentile(pf_subsample, 75)
        else:
            pf_p25 = 0
            pf_p50 = 0
            pf_p75 = 0
        pf_vals.append([pf_p25, pf_p50, pf_p75])

        mask_genjet = (mlpf_genjet_eta > lim_low) & (mlpf_genjet_eta <= lim_hi)
        mlpf_subsample = mlpf_response[mask_genjet]

        if len(mlpf_subsample) > 0:
            mlpf_p25 = np.percentile(mlpf_subsample, 25)
            mlpf_p50 = np.percentile(mlpf_subsample, 50)
            mlpf_p75 = np.percentile(mlpf_subsample, 75)
        else:
            mlpf_p25 = 0
            mlpf_p50 = 0
            mlpf_p75 = 0
        mlpf_vals.append([mlpf_p25, mlpf_p50, mlpf_p75])

        plt.sca(axs[ibin])
        plt.hist(target_subsample, bins=b, histtype="step", lw=2, label="Target")
        plt.hist(pf_subsample, bins=b, histtype="step", lw=2, label="PF")
        plt.hist(mlpf_subsample, bins=b, histtype="step", lw=2, label="MLPF")
        plt.ylabel("Matched jets / bin")
        plt.xlabel(labels["reco_gen_jet_ratio"])
        plt.legend(loc=1, fontsize=16)
        plt.title(labels["gen_jet_range_eta"].format(lim_low, lim_hi))
        plt.yscale("log")

    plt.tight_layout()
    save_img(
        "jet_response_binned_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    x_vals = np.array(x_vals)
    target_vals = np.array(target_vals)
    pf_vals = np.array(pf_vals)
    mlpf_vals = np.array(mlpf_vals)

    # Plot median and IQR as a function of gen eta
    plt.figure()
    ax = plt.gca()
    plt.plot(x_vals, target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.ylabel("Response median")
    plt.xlabel(labels["gen_jet_eta"])
    plt.tight_layout()
    plt.legend(loc=1, fontsize=16, title=title)
    plt.axhline(1.0, color="black", ls="--", lw=0.5)
    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)
    save_img(
        "jet_response_med_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    ax = plt.gca()
    plt.plot(x_vals, (target_vals[:, 2] - target_vals[:, 0]) / target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, (pf_vals[:, 2] - pf_vals[:, 0]) / pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, (mlpf_vals[:, 2] - mlpf_vals[:, 0]) / mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.ylabel("Response IQR / median")
    plt.xlabel(labels["gen_jet_eta"])
    plt.tight_layout()
    plt.legend(loc=1, fontsize=16, title=title)
    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)
    save_img(
        "jet_response_iqr_over_med_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_met_response_binned(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None, sample=None, dataset=None):
    genmet = yvals["gen_met"]

    target_response = yvals["ratio_target"]
    pf_response = yvals["ratio_cand"]
    mlpf_response = yvals["ratio_pred"]

    genmet_bins = [10, 20, 40, 60, 80, 100, 200]

    x_vals = []
    target_vals = []
    pf_vals = []
    mlpf_vals = []
    b = np.linspace(0, 5, 100)

    fig, axs = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
    axs = axs.flatten()
    for ibin in range(len(genmet_bins) - 1):
        lim_low = genmet_bins[ibin]
        lim_hi = genmet_bins[ibin + 1]
        x_vals.append(np.mean([lim_low, lim_hi]))

        mask_gen = (genmet > lim_low) & (genmet <= lim_hi)
        target_subsample = target_response[mask_gen]
        if len(target_subsample) > 0:
            target_p25 = np.percentile(target_subsample, 25)
            target_p50 = np.percentile(target_subsample, 50)
            target_p75 = np.percentile(target_subsample, 75)
        else:
            target_p25 = 0.0
            target_p50 = 0.0
            target_p75 = 0.0
        target_vals.append([target_p25, target_p50, target_p75])

        pf_subsample = pf_response[mask_gen]
        if len(pf_subsample) > 0:
            pf_p25 = np.percentile(pf_subsample, 25)
            pf_p50 = np.percentile(pf_subsample, 50)
            pf_p75 = np.percentile(pf_subsample, 75)
        else:
            pf_p25 = 0.0
            pf_p50 = 0.0
            pf_p75 = 0.0
        pf_vals.append([pf_p25, pf_p50, pf_p75])

        mlpf_subsample = mlpf_response[mask_gen]
        if len(pf_subsample) > 0:
            mlpf_p25 = np.percentile(mlpf_subsample, 25)
            mlpf_p50 = np.percentile(mlpf_subsample, 50)
            mlpf_p75 = np.percentile(mlpf_subsample, 75)
        else:
            mlpf_p25 = 0.0
            mlpf_p50 = 0.0
            mlpf_p75 = 0.0
        mlpf_vals.append([mlpf_p25, mlpf_p50, mlpf_p75])

        plt.sca(axs[ibin])
        plt.hist(target_subsample, bins=b, histtype="step", lw=2, label="Target")
        plt.hist(pf_subsample, bins=b, histtype="step", lw=2, label="PF")
        plt.hist(mlpf_subsample, bins=b, histtype="step", lw=2, label="MLPF")
        plt.ylabel("Events / bin")
        plt.xlabel(labels["reco_gen_met_ratio"])
        plt.axvline(1.0, ymax=0.7, color="black", ls="--")
        plt.legend(loc=1, fontsize=16)
        plt.title(labels["gen_met_range"].format(lim_low, lim_hi))
        plt.yscale("log")

    plt.tight_layout()
    save_img(
        "met_response_binned.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    x_vals = np.array(x_vals)
    target_vals = np.array(target_vals)
    pf_vals = np.array(pf_vals)
    mlpf_vals = np.array(mlpf_vals)

    # Plot median and IQR as a function of gen met
    plt.figure()
    plt.plot(x_vals, target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.ylabel("Response median")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.xlabel(labels["gen_met"])
    plt.tight_layout()
    ax = plt.gca()
    plt.axhline(1.0, color="black", ls="--", lw=0.5)
    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)
    save_img(
        "met_response_med.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    plt.plot(x_vals, (target_vals[:, 2] - target_vals[:, 0]) / target_vals[:, 1], marker="o", label="Target")
    plt.plot(x_vals, (pf_vals[:, 2] - pf_vals[:, 0]) / pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, (mlpf_vals[:, 2] - mlpf_vals[:, 0]) / mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.ylabel("Response IQR / median")
    plt.legend(loc=1, fontsize=16, title=title)
    plt.xlabel(labels["gen_met"])
    plt.tight_layout()
    plt.ylim(bottom=0)
    ax = plt.gca()
    EXPERIMENT_LABELS[dataset](ax)
    sample_label(ax, sample)
    save_img(
        "met_response_iqr_over_med.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_3dmomentum_response_binned(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    genmet = yvals["gen_mom"]

    pf_response = yvals["ratio_cand"]
    mlpf_response = yvals["ratio_pred"]

    genmet_bins = [10, 20, 40, 60, 80, 100, 200]

    x_vals = []
    pf_vals = []
    mlpf_vals = []
    b = np.linspace(0, 2, 100)

    fig, axs = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
    axs = axs.flatten()
    for ibin in range(len(genmet_bins) - 1):
        lim_low = genmet_bins[ibin]
        lim_hi = genmet_bins[ibin + 1]
        x_vals.append(np.mean([lim_low, lim_hi]))

        mask_gen = (genmet > lim_low) & (genmet <= lim_hi)
        pf_subsample = pf_response[mask_gen]
        if len(pf_subsample) > 0:
            pf_p25 = np.percentile(pf_subsample, 25)
            pf_p50 = np.percentile(pf_subsample, 50)
            pf_p75 = np.percentile(pf_subsample, 75)
        else:
            pf_p25 = 0.0
            pf_p50 = 0.0
            pf_p75 = 0.0
        pf_vals.append([pf_p25, pf_p50, pf_p75])

        mlpf_subsample = mlpf_response[mask_gen]
        if len(pf_subsample) > 0:
            mlpf_p25 = np.percentile(mlpf_subsample, 25)
            mlpf_p50 = np.percentile(mlpf_subsample, 50)
            mlpf_p75 = np.percentile(mlpf_subsample, 75)
        else:
            mlpf_p25 = 0.0
            mlpf_p50 = 0.0
            mlpf_p75 = 0.0
        mlpf_vals.append([mlpf_p25, mlpf_p50, mlpf_p75])

        plt.sca(axs[ibin])
        plt.hist(pf_subsample, bins=b, histtype="step", lw=2, label="PF")
        plt.hist(mlpf_subsample, bins=b, histtype="step", lw=2, label="MLPF")
        plt.xlim(0, 2)
        plt.xticks([0, 0.5, 1, 1.5, 2])
        plt.ylabel("Events / bin")
        plt.xlabel(labels["reco_gen_mom_ratio"])
        plt.axvline(1.0, ymax=0.7, color="black", ls="--")
        plt.legend(loc=1, fontsize=16)
        plt.title(labels["gen_mom_range"].format(lim_low, lim_hi))
        plt.yscale("log")

    plt.tight_layout()
    save_img(
        "mom_response_binned.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    x_vals = np.array(x_vals)
    pf_vals = np.array(pf_vals)
    mlpf_vals = np.array(mlpf_vals)

    # Plot median and IQR as a function of gen pt
    plt.figure()
    plt.plot(x_vals, (pf_vals[:, 2] - pf_vals[:, 0]) / pf_vals[:, 1], marker="o", label="PF")
    plt.plot(x_vals, (mlpf_vals[:, 2] - mlpf_vals[:, 0]) / mlpf_vals[:, 1], marker="o", label="MLPF")
    plt.ylabel("Response IQR / median")
    plt.xlabel(labels["gen_mom"])
    plt.legend(loc=1, fontsize=16, title=title)
    plt.tight_layout()
    plt.ylim(bottom=0)
    save_img(
        "mom_response_med_iqr.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_pu_fraction(yvals, epoch=None, cp_dir=None, dataset=None, sample=None, comet_experiment=None):
    plt.figure()
    ax = plt.axes()
    bins = np.linspace(0, 1, 100)
    target_ispu = awkward.flatten(yvals["target_ispu"])
    pred_ispu = torch.sigmoid(torch.Tensor(awkward.flatten(yvals["pred_ispu"][:, :, 0]))).numpy()
    plt.hist(target_ispu, bins=bins, label="target", histtype="step")
    plt.hist(pred_ispu, bins=bins, label="MLPF", histtype="step")
    plt.legend(loc=1, fontsize=16)
    plt.xlabel("PU fraction")
    plt.yscale("log")
    if dataset:
        EXPERIMENT_LABELS[dataset](ax)
    if sample:
        sample_label(ax, sample)
    save_img(
        "pu_frac.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )
