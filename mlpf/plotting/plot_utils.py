import glob
import math

import awkward
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
import vector

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
    211: r"charged hadrons ($\pi^\pm$, ...)",
    130: r"neutral hadrons (K, ...)",
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

EVALUATION_DATASET_NAMES = {
    "clic_ttbar_pf": r"CLIC $ee \rightarrow \mathrm{t}\overline{\mathrm{t}}$",
    "delphes_pf": r"Delphes-CMS $pp \rightarrow \mathrm{QCD}$",
    "cms_pf_qcd_high_pt": r"CMS high-$p_T$ QCD+PU events",
    "cms_pf_ttbar": r"CMS $\mathrm{t}\overline{\mathrm{t}}$+PU events",
    "cms_pf_single_neutron": r"CMS single neutron particle gun events",
    "clic_edm_ttbar_pf": r"CLIC $ee \rightarrow \mathrm{t}\overline{\mathrm{t}}$",
    "clic_edm_qcd_pf": r"CLIC $ee \rightarrow \gamma/\mathrm{Z}^* \rightarrow \mathrm{hadrons}$",
    "clic_edm_zz_fullhad_pf": r"CLIC $ee \rightarrow \mathrm{ZZ} \rightarrow \mathrm{hadrons}$",
}


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


def cms_label(ax, x0=0.01, x1=0.15, x2=0.98, y=0.94):
    plt.figtext(
        x0,
        y,
        "CMS",
        fontweight="bold",
        wrap=True,
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    plt.figtext(
        x1,
        y,
        "Simulation Preliminary",
        style="italic",
        wrap=True,
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    plt.figtext(
        x2,
        y,
        "Run 3 (14 TeV)",
        wrap=False,
        horizontalalignment="right",
        transform=ax.transAxes,
    )


def sample_label(ax, sample, additional_text="", x=0.01, y=0.87):
    text = SAMPLE_LABEL_CMS[sample]
    plt.text(x, y, text + additional_text, ha="left", transform=ax.transAxes)


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
    for typ in ["gen", "cand", "pred"]:
        for k in data["particles"][typ].fields:
            yvals["{}_{}".format(typ, k)] = data["particles"][typ][k]

    # Get the classification output as a class ID
    yvals["gen_cls_id"] = np.argmax(yvals["gen_cls"], axis=-1)
    yvals["cand_cls_id"] = np.argmax(yvals["cand_cls"], axis=-1)
    yvals["pred_cls_id"] = np.argmax(yvals["pred_cls"], axis=-1)

    for typ in ["gen", "cand", "pred"]:

        # Compute phi, px, py
        yvals[typ + "_phi"] = np.arctan2(yvals[typ + "_sin_phi"], yvals[typ + "_cos_phi"])
        yvals[typ + "_px"] = yvals[typ + "_pt"] * yvals[typ + "_cos_phi"]
        yvals[typ + "_py"] = yvals[typ + "_pt"] * yvals[typ + "_sin_phi"]

        # Get the jet vectors
        jetvec = vector.awk(data["jets"][typ])
        for k in ["pt", "eta", "phi", "energy"]:
            yvals["jets_{}_{}".format(typ, k)] = getattr(jetvec, k)

    for typ in ["gen", "cand", "pred"]:
        for val in ["pt", "eta", "sin_phi", "cos_phi", "charge", "energy"]:
            yvals["{}_{}".format(typ, val)] = yvals["{}_{}".format(typ, val)] * (yvals["{}_cls_id".format(typ)] != 0)

    yvals.update(compute_jet_ratio(data, yvals))

    return yvals, X, filenames


def compute_jet_ratio(data, yvals):
    ret = {}
    # flatten across event dimension
    ret["jet_gen_to_pred_genpt"] = awkward.to_numpy(
        awkward.flatten(
            vector.awk(data["jets"]["gen"][data["matched_jets"]["gen_to_pred"]["gen"]]).pt,
            axis=1,
        )
    )
    ret["jet_gen_to_pred_predpt"] = awkward.to_numpy(
        awkward.flatten(
            vector.awk(data["jets"]["pred"][data["matched_jets"]["gen_to_pred"]["pred"]]).pt,
            axis=1,
        )
    )
    ret["jet_gen_to_cand_genpt"] = awkward.to_numpy(
        awkward.flatten(
            vector.awk(data["jets"]["gen"][data["matched_jets"]["gen_to_cand"]["gen"]]).pt,
            axis=1,
        )
    )
    ret["jet_gen_to_cand_candpt"] = awkward.to_numpy(
        awkward.flatten(
            vector.awk(data["jets"]["cand"][data["matched_jets"]["gen_to_cand"]["cand"]]).pt,
            axis=1,
        )
    )

    ret["jet_ratio_pred"] = ret["jet_gen_to_pred_predpt"] / ret["jet_gen_to_pred_genpt"]
    ret["jet_ratio_cand"] = ret["jet_gen_to_cand_candpt"] / ret["jet_gen_to_cand_genpt"]
    return ret


def compute_met_and_ratio(yvals):
    msk_gen = yvals["gen_cls_id"] != 0
    gen_px = yvals["gen_px"][msk_gen]
    gen_py = yvals["gen_py"][msk_gen]

    msk_pred = yvals["pred_cls_id"] != 0
    pred_px = yvals["pred_px"][msk_pred]
    pred_py = yvals["pred_py"][msk_pred]

    msk_cand = yvals["cand_cls_id"] != 0
    cand_px = yvals["cand_px"][msk_cand]
    cand_py = yvals["cand_py"][msk_cand]

    gen_met = awkward.to_numpy(np.sqrt(np.sum(gen_px, axis=1) ** 2 + np.sum(gen_py, axis=1) ** 2))
    pred_met = awkward.to_numpy(np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2))
    cand_met = awkward.to_numpy(np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2))

    met_ratio_pred = awkward.to_numpy(pred_met / gen_met)
    met_ratio_cand = awkward.to_numpy(cand_met / gen_met)

    return {
        "gen_met": gen_met,
        "pred_met": pred_met,
        "cand_met": cand_met,
        "ratio_pred": met_ratio_pred,
        "ratio_cand": met_ratio_cand,
    }


def save_img(outfile, epoch, cp_dir=None, comet_experiment=None):
    if cp_dir:
        image_path = str(cp_dir / outfile)
        plt.savefig(image_path, bbox_inches="tight", dpi=100)
        plt.clf()
        if comet_experiment:
            comet_experiment.log_image(image_path, step=epoch - 1)


def plot_jets(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    plt.figure()
    b = np.logspace(0, 3, 100)

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_cand_pt"]))
    p = med_iqr(pt)
    n = len(pt)
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f}, N={})$".format(p[0], p[1], n),
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_pred_pt"]))
    p = med_iqr(pt)
    n = len(pt)
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f}, N={})$".format(p[0], p[1], n),
    )

    pt = awkward.to_numpy(awkward.flatten(yvals["jets_gen_pt"]))
    p = med_iqr(pt)
    n = len(pt)
    plt.hist(
        pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Gen $(M={:.2f}, IQR={:.2f}, N={})$".format(p[0], p[1], n),
    )

    plt.xscale("log")
    plt.xlabel("jet $p_T$")
    plt.ylabel("number of jets / bin")
    plt.legend(loc="best")
    if title:
        plt.title(title)
    save_img(
        "jet_pt.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_jet_ratio(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    plt.figure()
    b = np.linspace(0, 5, 100)

    p = med_iqr(yvals["jet_ratio_cand"])
    n_matched = len(yvals["jet_ratio_cand"])
    plt.hist(
        yvals["jet_ratio_cand"],
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f}, N={})$".format(p[0], p[1], n_matched),
    )
    p = med_iqr(yvals["jet_ratio_pred"])
    n_matched = len(yvals["jet_ratio_pred"])
    plt.hist(
        yvals["jet_ratio_pred"],
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f}, N={})$".format(p[0], p[1], n_matched),
    )
    plt.xlabel("jet $p_T$ reco/gen")
    plt.ylabel("number of matched jets")
    plt.legend(loc="best")
    if title:
        plt.title(title)
    save_img(
        "jet_res.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_met_and_ratio(met_ratio, epoch=None, cp_dir=None, comet_experiment=None, title=None):

    # MET
    plt.figure()
    maxval = max(
        [
            np.max(met_ratio["gen_met"]),
            np.max(met_ratio["cand_met"]),
            np.max(met_ratio["pred_met"]),
        ]
    )
    minval = min(
        [
            np.min(met_ratio["gen_met"]),
            np.min(met_ratio["cand_met"]),
            np.min(met_ratio["pred_met"]),
        ]
    )
    maxval = math.ceil(np.log10(maxval))
    minval = math.floor(np.log10(max(minval, 1e-2)))

    b = np.logspace(minval, maxval, 100)
    p = med_iqr(met_ratio["cand_met"])
    plt.hist(
        met_ratio["cand_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(met_ratio["pred_met"])
    plt.hist(
        met_ratio["pred_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(met_ratio["gen_met"])
    plt.hist(
        met_ratio["gen_met"],
        bins=b,
        histtype="step",
        lw=2,
        label="Truth $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    plt.xlabel("MET [GeV]")
    plt.ylabel("Number of events / bin")
    plt.legend(loc="best")
    plt.xscale("log")
    if title:
        plt.title(title)
    save_img("met.png", epoch, cp_dir=cp_dir, comet_experiment=comet_experiment)

    # Ratio
    plt.figure()
    b = np.linspace(0, 20, 100)

    p = med_iqr(met_ratio["ratio_cand"])
    plt.hist(
        met_ratio["ratio_cand"],
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(met_ratio["ratio_pred"])
    plt.hist(
        met_ratio["ratio_pred"],
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    plt.xlabel("MET reco/gen")
    plt.ylabel("number of events")
    plt.legend(loc="best")
    if title:
        plt.title(title)
    save_img(
        "met_res.png",
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


def plot_num_elements(X, epoch=None, cp_dir=None, comet_experiment=None, title=None):

    # compute the number of unpadded elements per event
    num_Xelems = awkward.sum(X[:, :, 0] != 0, axis=-1)
    maxval = np.max(num_Xelems)

    plt.figure()
    plt.hist(num_Xelems, bins=np.linspace(0, int(1.2 * maxval), 100))
    plt.xlabel("Number of PFElements / event")
    plt.ylabel("Number of events / bin")
    if title:
        plt.title(title)
    save_img(
        "num_elements.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_sum_energy(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None):

    sum_gen_energy = awkward.to_numpy(awkward.sum(yvals["gen_energy"], axis=1))
    sum_cand_energy = awkward.to_numpy(awkward.sum(yvals["cand_energy"], axis=1))
    sum_pred_energy = awkward.to_numpy(awkward.sum(yvals["pred_energy"], axis=1))

    max_e = max(
        [
            np.max(sum_gen_energy),
            np.max(sum_cand_energy),
            np.max(sum_pred_energy),
        ]
    )
    min_e = min(
        [
            np.min(sum_gen_energy),
            np.min(sum_cand_energy),
            np.min(sum_pred_energy),
        ]
    )

    max_e = int(1.2 * max_e)
    min_e = int(0.8 * min_e)

    b = np.linspace(min_e, max_e, 100)
    plt.figure()
    plt.hist2d(sum_gen_energy, sum_cand_energy, bins=(b, b), cmap="hot_r")
    plt.plot([min_e, max_e], [min_e, max_e], color="black", ls="--")
    plt.xlabel("total true energy / event [GeV]")
    plt.ylabel("total PF energy / event [GeV]")
    if title:
        plt.title(title)
    save_img(
        "sum_gen_cand_energy.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    plt.figure()
    plt.hist2d(sum_gen_energy, sum_pred_energy, bins=(b, b), cmap="hot_r")
    plt.plot([min_e, max_e], [min_e, max_e], color="black", ls="--")
    plt.xlabel("total true energy / event [GeV]")
    plt.ylabel("total MLPF energy / event [GeV]")
    if title:
        plt.title(title)
    save_img(
        "sum_gen_pred_energy.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    max_e = max(
        [
            np.max(sum_gen_energy),
            np.max(sum_cand_energy),
            np.max(sum_pred_energy),
        ]
    )
    min_e = min(
        [
            np.min(sum_gen_energy),
            np.min(sum_cand_energy),
            np.min(sum_pred_energy),
        ]
    )
    max_e = math.ceil(np.log10(max_e))
    min_e = math.floor(np.log10(max(min_e, 1e-2)))

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
        plt.title(title + ", PF")
    save_img(
        "sum_gen_cand_energy_log.png",
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
        plt.title(title + ", MLPF")
    save_img(
        "sum_gen_pred_energy_log.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )


def plot_particles(yvals, epoch=None, cp_dir=None, comet_experiment=None, title=None):
    msk_cand = yvals["cand_cls_id"] != 0
    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_pt"][msk_cand], axis=1))

    msk_pred = yvals["pred_cls_id"] != 0
    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_pt"][msk_pred], axis=1))

    msk_gen = yvals["gen_cls_id"] != 0
    gen_pt = awkward.to_numpy(awkward.flatten(yvals["gen_pt"][msk_gen], axis=1))

    b = np.logspace(-1, 4, 100)
    plt.figure()
    p = med_iqr(cand_pt)
    plt.hist(
        cand_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(pred_pt)
    plt.hist(
        pred_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(gen_pt)
    plt.hist(
        gen_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Truth $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    plt.xscale("log")
    plt.xlabel("Particle $p_T$ [GeV]")
    plt.ylabel("Number of particles / bin")
    if title:
        plt.title(title)
    plt.legend(loc="best")
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

    msk_gen = yvals["gen_cls_id"] != 0
    gen_pt = awkward.to_numpy(awkward.flatten(yvals["gen_eta"][msk_gen], axis=1))

    b = np.linspace(-8, 8, 100)
    plt.figure()
    p = med_iqr(cand_pt)
    plt.hist(
        cand_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="PF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(pred_pt)
    plt.hist(
        pred_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="MLPF $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    p = med_iqr(gen_pt)
    plt.hist(
        gen_pt,
        bins=b,
        histtype="step",
        lw=2,
        label="Truth $(M={:.2f}, IQR={:.2f})$".format(p[0], p[1]),
    )
    plt.xlabel(r"Particle $\eta$")
    plt.ylabel("Number of particles / bin")
    if title:
        plt.title(title)
    plt.legend(loc="best")
    save_img(
        "particle_eta.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    msk_cand = yvals["cand_cls_id"] != 0
    msk_pred = yvals["pred_cls_id"] != 0
    msk_gen = yvals["gen_cls_id"] != 0

    cand_pt = awkward.to_numpy(awkward.flatten(yvals["cand_pt"][msk_cand & msk_gen], axis=1))
    gen_pt = awkward.to_numpy(awkward.flatten(yvals["gen_pt"][msk_cand & msk_gen], axis=1))
    b = np.logspace(-1, 4, 100)
    plt.figure()
    plt.hist2d(gen_pt, cand_pt, bins=(b, b), cmap="hot_r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True particle $p_T$ [GeV]")
    plt.ylabel("Reconstructed particle $p_T$ [GeV]")
    plt.plot([10**-1, 10**4], [10**-1, 10**4], color="black", ls="--")
    if title:
        plt.title(title + ", PF")
    save_img(
        "particle_pt_gen_vs_pf.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )

    pred_pt = awkward.to_numpy(awkward.flatten(yvals["pred_pt"][msk_pred & msk_gen], axis=1))
    gen_pt = awkward.to_numpy(awkward.flatten(yvals["gen_pt"][msk_pred & msk_gen], axis=1))
    b = np.logspace(-1, 4, 100)
    plt.figure()
    plt.hist2d(gen_pt, pred_pt, bins=(b, b), cmap="hot_r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True particle $p_T$ [GeV]")
    plt.ylabel("Reconstructed particle $p_T$ [GeV]")
    plt.plot([10**-1, 10**4], [10**-1, 10**4], color="black", ls="--")
    if title:
        plt.title(title + ", MLPF")
    save_img(
        "particle_pt_gen_vs_mlpf.png",
        epoch,
        cp_dir=cp_dir,
        comet_experiment=comet_experiment,
    )
