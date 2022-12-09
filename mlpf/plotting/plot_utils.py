import glob

import awkward
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
ELEM_NAMES_CMS = ["NONE", "TRACK", "PS1", "PS2", "ECAL", "HCAL", "GSF", "BREM", "HFEM", "HFHAD", "SC", "HO"]

CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13]
CLASS_NAMES_CMS = ["none", "ch.had", "n.had", "HFHAD", "HFEM", "$\gamma$", "$e^\pm$", "$\mu^\pm$"]

bins = {
    211: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 61),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    -211: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    130: {
        "E_val": np.linspace(0, 5, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    22: {
        "E_val": np.linspace(0, 2, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-2, 2, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    11: {
        "E_val": np.linspace(0, 10, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    13: {
        "E_val": np.linspace(0, 10, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-4, 4, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    1: {
        "E_val": np.linspace(0, 100, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-6, 6, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
    2: {
        "E_val": np.linspace(0, 50, 61),
        "E_res": np.linspace(-1, 1, 61),
        "eta_val": np.linspace(-6, 6, 61),
        "eta_res": np.linspace(-0.5, 0.5, 41),
        "E_xlabel": "Energy [GeV]",
        "eta_xlabel": "$\eta$",
        "phi_val": np.linspace(-4, 4, 61),
        "phi_res": np.linspace(-0.5, 0.5, 41),
        "phi_xlabel": "$\phi$",
        "true_val": "reco PF",
        "pred_val": "ML-PF",
    },
}


def get_eff(df, pid):
    v0 = np.sum(df == pid)
    return v0 / len(df), np.sqrt(v0) / len(df)


def get_fake(df, pid):
    v0 = np.sum(df != pid)
    return v0 / len(df), np.sqrt(v0) / len(df)


def cms_label(ax, x0=0.01, x1=0.15, x2=0.98, y=0.94):
    plt.figtext(x0, y, "CMS", fontweight="bold", wrap=True, horizontalalignment="left", transform=ax.transAxes)
    plt.figtext(
        x1, y, "Simulation Preliminary", style="italic", wrap=True, horizontalalignment="left", transform=ax.transAxes
    )
    plt.figtext(x2, y, "Run 3 (14 TeV)", wrap=False, horizontalalignment="right", transform=ax.transAxes)


def sample_label(ax, sample, additional_text="", x=0.01, y=0.87):
    text = SAMPLE_LABEL_CMS[sample]
    plt.text(x, y, text + additional_text, ha="left", transform=ax.transAxes)


def particle_label(ax, pid):
    plt.text(0.03, 0.92, pid_to_text[pid], va="top", ha="left", size=10, transform=ax.transAxes)


def load_eval_data(path):
    yvals = []
    filenames = []
    for fi in glob.glob(path):
        dd = awkward.from_parquet(fi)
        yvals.append(dd)
        filenames.append(fi)

    yvals_awk = awkward.concatenate(yvals, axis=0)
    particles = {k: yvals_awk["particles"][k] for k in yvals_awk["particles"].fields}

    return yvals_awk, particles, filenames


def compute_met_and_ratio(particles):
    msk_gen = np.argmax(particles["gen"]["cls"], axis=-1) != 0
    gen_px = particles["gen"]["pt"][msk_gen] * particles["gen"]["cos_phi"][msk_gen]
    gen_py = particles["gen"]["pt"][msk_gen] * particles["gen"]["sin_phi"][msk_gen]

    msk_pred = np.argmax(particles["pred"]["cls"], axis=-1) != 0
    pred_px = particles["pred"]["pt"][msk_pred] * particles["pred"]["cos_phi"][msk_pred]
    pred_py = particles["pred"]["pt"][msk_pred] * particles["pred"]["sin_phi"][msk_pred]

    msk_cand = np.argmax(particles["cand"]["cls"], axis=-1) != 0
    cand_px = particles["cand"]["pt"][msk_cand] * particles["cand"]["cos_phi"][msk_cand]
    cand_py = particles["cand"]["pt"][msk_cand] * particles["cand"]["sin_phi"][msk_cand]

    gen_met = np.sqrt(np.sum(gen_px, axis=1) ** 2 + np.sum(gen_py, axis=1) ** 2)
    pred_met = np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2)
    cand_met = np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2)

    met_ratio_pred = awkward.to_numpy((pred_met - gen_met) / gen_met)
    met_ratio_cand = awkward.to_numpy((cand_met - gen_met) / gen_met)

    return {
        "gen_met": gen_met,
        "pred_met": pred_met,
        "cand_met": cand_met,
        "ratio_pred": met_ratio_pred,
        "ratio_cand": met_ratio_cand,
    }


def compute_jet_ratio(yvals_awk):
    # flatten across event dimension
    gen_to_pred_genpt = awkward.flatten(
        vector.arr(yvals_awk["jets"]["gen"][yvals_awk["matched_jets"]["gen_to_pred"]["gen"]]).pt, axis=1
    )
    gen_to_pred_predpt = awkward.flatten(
        vector.arr(yvals_awk["jets"]["pred"][yvals_awk["matched_jets"]["gen_to_pred"]["pred"]]).pt, axis=1
    )
    gen_to_cand_genpt = awkward.flatten(
        vector.arr(yvals_awk["jets"]["gen"][yvals_awk["matched_jets"]["gen_to_cand"]["gen"]]).pt, axis=1
    )
    gen_to_cand_candpt = awkward.flatten(
        vector.arr(yvals_awk["jets"]["cand"][yvals_awk["matched_jets"]["gen_to_cand"]["cand"]]).pt, axis=1
    )

    jet_ratio_pred = (gen_to_pred_predpt - gen_to_pred_genpt) / gen_to_pred_genpt
    jet_ratio_cand = (gen_to_cand_candpt - gen_to_cand_genpt) / gen_to_cand_genpt

    return {
        "gen_to_pred_genpt": gen_to_pred_genpt,
        "gen_to_pred_predpt": gen_to_pred_predpt,
        "gen_to_cand_genpt": gen_to_cand_genpt,
        "gen_to_cand_candpt": gen_to_cand_candpt,
        "ratio_pred": jet_ratio_pred,
        "ratio_cand": jet_ratio_cand,
    }


def plot_jet_ratio(jet_ratio, epoch, cp_dir=None, comet_experiment=None):
    plt.figure()
    b = np.linspace(-2, 5, 100)
    plt.hist(jet_ratio["ratio_cand"], bins=b, histtype="step", lw=2, label="PF")
    plt.hist(jet_ratio["ratio_pred"], bins=b, histtype="step", lw=2, label="MLPF")
    plt.xlabel("jet pT (reco-gen)/gen")
    plt.ylabel("number of matched jets")
    plt.legend(loc="best")
    if cp_dir:
        image_path = str(cp_dir / "jet_res.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=100)
        plt.clf()
    if comet_experiment:
        comet_experiment.log_image(image_path, step=epoch - 1)


def plot_met_ratio(met_ratio, epoch, cp_dir=None, comet_experiment=None):
    plt.figure()
    b = np.linspace(-1, 20, 100)
    plt.hist(met_ratio["ratio_cand"], bins=b, histtype="step", lw=2, label="PF")
    plt.hist(met_ratio["ratio_pred"], bins=b, histtype="step", lw=2, label="MLPF")
    plt.xlabel("MET (reco-gen)/gen")
    plt.ylabel("number of events")
    plt.legend(loc="best")
    if cp_dir:
        image_path = str(cp_dir / "met_res.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=100)
        plt.clf()
    if comet_experiment:
        comet_experiment.log_image(image_path, step=epoch - 1)


def compute_distances(distribution_1, distribution_2, ratio):
    wd = scipy.stats.wasserstein_distance(distribution_1, distribution_2)
    p25 = np.percentile(ratio, 25)
    p50 = np.percentile(ratio, 50)
    p75 = np.percentile(ratio, 75)
    iqr = p75 - p25
    return {"wd": wd, "p25": p25, "p50": p50, "p75": p75, "iqr": iqr}
