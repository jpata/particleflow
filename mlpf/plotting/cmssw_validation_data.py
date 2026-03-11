import click
import uproot
import awkward
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import glob
import os
import json
import pandas
import tqdm
from scipy.interpolate import RegularGridInterpolator
from mlpf.plotting.utils import to_bh
import mlpf.plotting.plot_utils as plot_utils

# Standard CMS style
mplhep.style.use("CMS")
matplotlib.rcParams["axes.labelsize"] = 35

# Plotting constants from notebook
JET_LABEL_AK4 = r"AK4 jets, $p_{T}$ > 20 GeV, $|\eta|$ < 2.5"
SELECTION_LABEL = "di-jets"
SAMPLE_LABEL_COORDS = 0.02, 0.96
SAMPLE_LABEL_FONTSIZE = 30
LEGEND_FONTSIZE = 30
PF_COLOR = "#f3a041"
MLPF_COLOR = "#d23b3d"
PF_LINESTYLE = "-."
MLPF_LINESTYLE = "-"


def is_in_golden_json(run, luminosity_block, golden_json_data):
    run_str = str(run)
    if run_str not in golden_json_data:
        return False
    lumi_ranges = golden_json_data[run_str]
    for lumi_range in lumi_ranges:
        if len(lumi_range) == 2:
            lumi_first, lumi_last = lumi_range
            if lumi_first <= luminosity_block <= lumi_last:
                return True
    return False


def apply_filters(data):
    filt = (
        data["Flag_HBHENoiseFilter"]
        & data["Flag_CSCTightHaloFilter"]
        & data["Flag_hcalLaserEventFilter"]
        & data["Flag_goodVertices"]
        & data["Flag_eeBadScFilter"]
        & data["Flag_ecalLaserCorrFilter"]
        & data["Flag_trkPOGFilters"]
        & data["HLT_VBF_DiPFJet125_45_Mjj1200"]
    )
    return filt


def load_nano(files):
    ret = []
    branches = [
        "PFMET_pt",
        "PuppiMET_pt",
        "Jet_rawFactor",
        "Jet_pt",
        "Jet_nConstituents",
        "Jet_chMultiplicity",
        "Jet_neMultiplicity",
        "Jet_nElectrons",
        "Jet_nMuons",
        "Jet_chEmEF",
        "Jet_chHEF",
        "Jet_hfEmEF",
        "Jet_hfHEF",
        "Jet_muEF",
        "Jet_neEmEF",
        "Jet_neHEF",
        "Electron_pt",
        "Muon_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
        "HLT_AK8PFJet500",
        "HLT_PFJet500",
        "HLT_VBF_DiPFJet125_45_Mjj1200",
        "PV_npvsGood",
        "run",
        "luminosityBlock",
        "event",
        "Flag_HBHENoiseFilter",
        "Flag_CSCTightHaloFilter",
        "Flag_hcalLaserEventFilter",
        "Flag_goodVertices",
        "Flag_eeBadScFilter",
        "Flag_ecalLaserCorrFilter",
        "Flag_trkPOGFilters",
    ]
    for fn in tqdm.tqdm(files, desc="Loading files"):
        with uproot.open(fn) as f:
            tt = f["Events"]
            # Only read existing branches
            available_branches = [b for b in branches if b in tt.keys()]
            data = tt.arrays(available_branches)
            ret.append(data)
    return awkward.concatenate(ret)


def compute_jetid(data):
    jetid = (
        (np.abs(data["Jet_pt_corr"]) >= 20)
        & (np.abs(data["Jet_eta"]) <= 2.6)
        & (np.abs(data["Jet_neHEF"]) < 0.9)
        & (np.abs(data["Jet_neEmEF"]) < 0.9)
        & (np.abs(data["Jet_nConstituents"]) > 1)
        & (np.abs(data["Jet_muEF"]) < 0.8)
        & (np.abs(data["Jet_chHEF"]) > 0.01)
        & (np.abs(data["Jet_chMultiplicity"]) > 0)
        & (np.abs(data["Jet_chEmEF"]) < 0.8)
    )
    return jetid


def get_jet_pt(data):
    jetid = compute_jetid(data)
    high_pt_jets = (data["Jet_pt_corr"] > 60) & (np.abs(data["Jet_eta"]) < 2.5) & jetid
    two_good_jets = awkward.sum(high_pt_jets, axis=1) > 1

    low_pt_jets = (data["Jet_pt"] > 5) & jetid
    n_low_pt_jets = awkward.sum(low_pt_jets, axis=1)
    exactly_two_jets_mask = n_low_pt_jets >= 2

    mask_2_jets = two_good_jets & exactly_two_jets_mask
    evs_2_jets = data[mask_2_jets]
    njets = np.arange(len(evs_2_jets["Jet_pt_corr"]))

    jet_indices = awkward.argsort(evs_2_jets["Jet_pt_corr"], axis=1, ascending=False)
    leading_jet = jet_indices[:, 0]
    subleading_jet = jet_indices[:, 1]

    leading_jet_pt = evs_2_jets["Jet_pt_corr"][njets, leading_jet]
    subleading_jet_pt = evs_2_jets["Jet_pt_corr"][njets, subleading_jet]
    leading_jet_eta = evs_2_jets["Jet_eta"][njets, leading_jet]
    subleading_jet_eta = evs_2_jets["Jet_eta"][njets, subleading_jet]
    leading_jet_phi = evs_2_jets["Jet_phi"][njets, leading_jet]
    subleading_jet_phi = evs_2_jets["Jet_phi"][njets, subleading_jet]
    leading_jet_mass = evs_2_jets["Jet_mass"][njets, leading_jet]
    subleading_jet_mass = evs_2_jets["Jet_mass"][njets, subleading_jet]

    delta_phi = abs(leading_jet_phi - subleading_jet_phi)
    delta_phi = np.minimum(delta_phi, 2 * np.pi - delta_phi)
    back_to_back_mask = delta_phi > 2.7

    # Apply back-to-back mask to all variables
    leading_jet_pt = leading_jet_pt[back_to_back_mask]
    subleading_jet_pt = subleading_jet_pt[back_to_back_mask]
    leading_jet_eta = leading_jet_eta[back_to_back_mask]
    subleading_jet_eta = subleading_jet_eta[back_to_back_mask]
    leading_jet_phi = leading_jet_phi[back_to_back_mask]
    subleading_jet_phi = subleading_jet_phi[back_to_back_mask]
    leading_jet_mass = leading_jet_mass[back_to_back_mask]
    subleading_jet_mass = subleading_jet_mass[back_to_back_mask]

    # Calculate four-momentum components
    leading_energy = np.sqrt(leading_jet_pt**2 * np.cosh(leading_jet_eta) ** 2 + leading_jet_mass**2)
    subleading_energy = np.sqrt(subleading_jet_pt**2 * np.cosh(subleading_jet_eta) ** 2 + subleading_jet_mass**2)

    leading_px = leading_jet_pt * np.cos(leading_jet_phi)
    leading_py = leading_jet_pt * np.sin(leading_jet_phi)
    leading_pz = leading_jet_pt * np.sinh(leading_jet_eta)

    subleading_px = subleading_jet_pt * np.cos(subleading_jet_phi)
    subleading_py = subleading_jet_pt * np.sin(subleading_jet_phi)
    subleading_pz = subleading_jet_pt * np.sinh(subleading_jet_eta)

    dijet_mass = np.sqrt(
        (leading_energy + subleading_energy) ** 2
        - (leading_px + subleading_px) ** 2
        - (leading_py + subleading_py) ** 2
        - (leading_pz + subleading_pz) ** 2
    )

    return mask_2_jets, leading_jet_pt, subleading_jet_pt, dijet_mass


@click.command()
@click.option("--input-pf", required=True, help="Input PF NanoAOD files (glob pattern)")
@click.option("--input-mlpf", required=True, help="Input MLPF NanoAOD files (glob pattern)")
@click.option("--golden-json", required=True, help="Path to Golden JSON file")
@click.option("--jec-file", required=True, help="Path to JEC .npz file")
@click.option("--lumi-csv", required=True, help="Path to brilcalc lumi CSV")
@click.option("--output-dir", required=True, help="Output directory for plots")
@click.option("--sample-label", default="", help="Optional dataset label (e.g. Run2024C)")
def main(input_pf, input_mlpf, golden_json, jec_file, lumi_csv, output_dir, sample_label):
    os.makedirs(output_dir, exist_ok=True)

    pf_files = glob.glob(input_pf)
    mlpf_files = glob.glob(input_mlpf)

    pf_files_d = {os.path.basename(fn): fn for fn in pf_files}
    mlpf_files_d = {os.path.basename(fn): fn for fn in mlpf_files}
    common_files = sorted(list(set(pf_files_d.keys()).intersection(set(mlpf_files_d.keys()))))

    print(f"Found {len(common_files)} common files")

    data_pf = load_nano([pf_files_d[fn] for fn in common_files])
    data_mlpf = load_nano([mlpf_files_d[fn] for fn in common_files])

    # Filter by Golden JSON and standard flags
    with open(golden_json, "r") as f:
        golden_data = json.load(f)

    is_golden_pf = np.array([is_in_golden_json(run, lumi, golden_data) for run, lumi in zip(data_pf["run"], data_pf["luminosityBlock"])])
    is_golden_mlpf = np.array([is_in_golden_json(run, lumi, golden_data) for run, lumi in zip(data_mlpf["run"], data_mlpf["luminosityBlock"])])

    data_pf = data_pf[is_golden_pf & apply_filters(data_pf)]
    data_mlpf = data_mlpf[is_golden_mlpf & apply_filters(data_mlpf)]

    print(f"Events after filtering: PF={len(data_pf)}, MLPF={len(data_mlpf)}")

    # Apply JECs
    data_pf["Jet_pt_raw"] = data_pf["Jet_pt"] * (1.0 - data_pf["Jet_rawFactor"])
    data_mlpf["Jet_pt_raw"] = data_mlpf["Jet_pt"] * (1.0 - data_mlpf["Jet_rawFactor"])

    eta_reco_bins = [-6, -5.191, -4.0, -2.964, -1.392, 0, 1.392, 2.964, 4.0, 5.191, 6]
    pt_gen_bins = [10, 20, 30, 40, 50, 80, 120, 200, 500, 3000]

    fi = np.load(jec_file)
    reciprocal_med_pf = fi["corr_map_pf"]
    reciprocal_med_mlpf = fi["corr_map_mlpf"]

    def midpoints(x):
        return (x[1:] + x[:-1]) / 2

    interp_pf = RegularGridInterpolator(
        (midpoints(np.array(eta_reco_bins)), midpoints(np.array(pt_gen_bins))),
        reciprocal_med_pf,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    interp_mlpf = RegularGridInterpolator(
        (midpoints(np.array(eta_reco_bins)), midpoints(np.array(pt_gen_bins))),
        reciprocal_med_mlpf,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def apply_jec(data, interp):
        eta = awkward.to_numpy(awkward.flatten(data["Jet_eta"]))
        pt = awkward.to_numpy(awkward.flatten(data["Jet_pt_raw"]))
        corr = interp(np.stack([eta, pt]).T)
        return awkward.unflatten(corr, awkward.count(data["Jet_eta"], axis=1))

    data_pf["Jet_pt_corr"] = apply_jec(data_pf, interp_pf) * data_pf["Jet_pt_raw"]
    data_mlpf["Jet_pt_corr"] = apply_jec(data_mlpf, interp_mlpf) * data_mlpf["Jet_pt_raw"]

    # Luminosity
    runs_used = np.unique(data_pf["run"])
    lumi_df = pandas.read_csv(lumi_csv)
    lumi_df["run"] = [int(r.split(":")[0]) for r in lumi_df["#run:fill"]]
    int_lumi = lumi_df[lumi_df["run"].isin(runs_used)]["recorded(/fb)"].sum() * 1000
    lumi_text = f"{int_lumi:.0f} pb$^{{-1}}$ (13.6 TeV)"

    # Plotting
    plot_configs = [
        ("nmu", lambda d: awkward.num(d["Muon_pt"], axis=1), np.linspace(0, 10, 11), "Number of muons", False, "PF", "MLPF", 1e6),
        ("nele", lambda d: awkward.num(d["Electron_pt"], axis=1), np.linspace(0, 20, 21), "Number of electrons", False, "PF", "MLPF", 1e6),
        (
            "njet",
            lambda d: awkward.num(d["Jet_pt_corr"][compute_jetid(d)], axis=1),
            np.linspace(0, 20, 21),
            "Number of jets",
            True,
            "PF",
            "MLPF",
            1e6,
        ),
        (
            "jet_pt",
            lambda d: awkward.flatten(d["Jet_pt_corr"][compute_jetid(d)]),
            np.linspace(0, 2000, 201),
            plot_utils.labels["pt"],
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_nConstituents",
            lambda d: awkward.flatten(d["Jet_nConstituents"]),
            np.linspace(0, 100, 101),
            "Jet constituents (Jet_nConstituents)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_muEF",
            lambda d: awkward.flatten(d["Jet_muEF"]),
            np.linspace(0, 1, 101),
            "Jet muon frac (Jet_muEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_chEmEF",
            lambda d: awkward.flatten(d["Jet_chEmEF"]),
            np.linspace(0, 1, 101),
            "Jet charged EM frac (Jet_chEmEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_chHEF",
            lambda d: awkward.flatten(d["Jet_chHEF"]),
            np.linspace(0, 1, 101),
            "Jet charged hadronic frac (Jet_chHEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_neEmEF",
            lambda d: awkward.flatten(d["Jet_neEmEF"]),
            np.linspace(0, 1, 101),
            "Jet neutral EM frac (Jet_neEmEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_neHEF",
            lambda d: awkward.flatten(d["Jet_neHEF"]),
            np.linspace(0, 1, 101),
            "Jet neutral hadronic frac (Jet_neHEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_hfEmEF",
            lambda d: awkward.flatten(d["Jet_hfEmEF"]),
            np.linspace(0, 2, 101),
            "Jet HF EM frac (hfEmEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_hfHEF",
            lambda d: awkward.flatten(d["Jet_hfHEF"]),
            np.linspace(0, 1, 101),
            "Jet HF hadronic frac (Jet_hfHEF)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_nElectrons",
            lambda d: awkward.flatten(d["Jet_nElectrons"]),
            np.linspace(0, 10, 11),
            "Jet num. electrons (nElectrons)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_nMuons",
            lambda d: awkward.flatten(d["Jet_nMuons"]),
            np.linspace(0, 10, 11),
            "Jet num. muons (nMuon)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_ch_mult",
            lambda d: awkward.flatten(d["Jet_chMultiplicity"]),
            np.linspace(0, 60, 61),
            "Jet charged multiplicity (chMultiplicity)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        (
            "jet_ne_mult",
            lambda d: awkward.flatten(d["Jet_neMultiplicity"]),
            np.linspace(0, 60, 61),
            "Jet neutral multiplicity (neMultiplicity)",
            True,
            "PF-PUPPI",
            "MLPF-PUPPI",
            1e7,
        ),
        ("npvs", lambda d: d["PV_npvsGood"], np.linspace(0, 100, 101), "Number of good PVs", True, "PF", "MLPF", 1e6),
    ]

    for name, func, bins, xlabel, add_jet_label, label_pf, label_mlpf, ylim_max in plot_configs:
        plt.figure()
        ax = plt.axes()
        plt.hist(func(data_pf), bins=bins, histtype="step", label=label_pf, ls=PF_LINESTYLE, lw=2, color=PF_COLOR)
        plt.hist(func(data_mlpf), bins=bins, histtype="step", label=label_mlpf, ls=MLPF_LINESTYLE, lw=2, color=MLPF_COLOR)
        plt.yscale("log")
        mplhep.cms.label("", data=True, rlabel=lumi_text)
        text = SELECTION_LABEL
        if sample_label:
            text = f"{sample_label}\n{SELECTION_LABEL}"
        if add_jet_label:
            text += "\n" + JET_LABEL_AK4
        ax.text(SAMPLE_LABEL_COORDS[0], SAMPLE_LABEL_COORDS[1], text, va="top", transform=ax.transAxes, fontsize=SAMPLE_LABEL_FONTSIZE)
        plt.ylim(1, ylim_max)
        plt.legend(loc="best", fontsize=LEGEND_FONTSIZE)
        plt.xlabel(xlabel)
        plt.ylabel("Counts")
        plt.savefig(f"{output_dir}/jetmet0_{name}.pdf")
        plt.close()

    # Dijet and MET plots
    evmask_pf, lj_pt_pf, slj_pt_pf, dijet_mass_pf = get_jet_pt(data_pf)
    evmask_mlpf, lj_pt_mlpf, slj_pt_mlpf, dijet_mass_mlpf = get_jet_pt(data_mlpf)

    event_label = SELECTION_LABEL
    if sample_label:
        event_label = f"{sample_label}\n{SELECTION_LABEL}"
    event_label += "\n$\\geq 2$ jets, $p_T>60$ GeV, $|\\eta|<2.5$"

    # MET plot with ratio
    def plot_with_ratio(h_pf, h_mlpf, bins, xlabel, filename, ylim_max=1e5, legend_loc=(0.5, 0.65)):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        plt.sca(a0)
        x0 = mplhep.histplot(h_pf, histtype="step", label="PF-PUPPI", ls=PF_LINESTYLE, lw=2, color=PF_COLOR)
        x1 = mplhep.histplot(h_mlpf, histtype="step", label="MLPF-PUPPI", ls=MLPF_LINESTYLE, lw=2, color=MLPF_COLOR)
        plt.yscale("log")
        mplhep.cms.label("", data=True, rlabel=lumi_text)
        a0.text(SAMPLE_LABEL_COORDS[0], SAMPLE_LABEL_COORDS[1], event_label, va="top", transform=a0.transAxes, fontsize=SAMPLE_LABEL_FONTSIZE)

        handles, labels = a0.get_legend_handles_labels()
        handles = [x0[0].stairs, x1[0].stairs]
        a0.legend(handles, labels, loc=legend_loc, fontsize=LEGEND_FONTSIZE)
        plt.ylabel("Counts")
        plt.ylim(1, ylim_max)

        plt.sca(a1)
        pf_counts = h_pf.counts()
        sigma_ratio = np.sqrt(np.where(pf_counts > 0, 2 / pf_counts, 0))

        mplhep.histplot(h_pf / h_pf, color=PF_COLOR, ls=PF_LINESTYLE, lw=2)
        # band for PF error
        mplhep.histplot(h_pf / h_pf, yerr=sigma_ratio, edgecolor=PF_COLOR, ls=PF_LINESTYLE, lw=2, histtype="band", facecolor=PF_COLOR, alpha=0.3)
        mplhep.histplot(h_mlpf / h_pf, color=MLPF_COLOR, ls=MLPF_LINESTYLE, lw=2)

        plt.ylim(0, 2)
        plt.ylabel("MLPF / PF")
        plt.xlim(bins[0], bins[-1])
        plt.xlabel(xlabel)
        plt.savefig(f"{output_dir}/{filename}")
        plt.close()

    met_bins = np.linspace(0, 300, 61)
    h_met_pf = to_bh(data_pf["PuppiMET_pt"][evmask_pf], met_bins)
    h_met_mlpf = to_bh(data_mlpf["PuppiMET_pt"][evmask_mlpf], met_bins)
    plot_with_ratio(h_met_pf, h_met_mlpf, met_bins, plot_utils.labels["met"], "jetmet0_met.pdf")

    pt_bins = np.linspace(60, 1500, 61)
    h_lj_pf = to_bh(lj_pt_pf, pt_bins)
    h_lj_mlpf = to_bh(lj_pt_mlpf, pt_bins)
    plot_with_ratio(h_lj_pf, h_lj_mlpf, pt_bins, "Leading jet " + plot_utils.labels["pt"], "jetmet0_leading_jet_pt.pdf")

    # Subleading jet pt without ratio as in notebook
    plt.figure()
    ax = plt.axes()
    plt.hist(slj_pt_pf, bins=np.linspace(0, 1500, 101), histtype="step", label="PF-PUPPI", ls=PF_LINESTYLE, lw=2, color=PF_COLOR)
    plt.hist(slj_pt_mlpf, bins=np.linspace(0, 1500, 101), histtype="step", label="MLPF-PUPPI", ls=MLPF_LINESTYLE, lw=2, color=MLPF_COLOR)
    plt.yscale("log")
    mplhep.cms.label("", data=True, rlabel=lumi_text)
    ax.text(SAMPLE_LABEL_COORDS[0], SAMPLE_LABEL_COORDS[1], event_label, va="top", transform=ax.transAxes, fontsize=SAMPLE_LABEL_FONTSIZE)
    plt.ylim(1, 1e5)
    plt.legend(loc=(0.50, 0.55), fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Subleading jet " + plot_utils.labels["pt"])
    plt.ylabel("Counts")
    plt.savefig(f"{output_dir}/jetmet0_subleading_jet_pt.pdf")
    plt.close()

    mass_bins = np.linspace(0, 5000, 41)
    h_mass_pf = to_bh(dijet_mass_pf, mass_bins)
    h_mass_mlpf = to_bh(dijet_mass_mlpf, mass_bins)
    plot_with_ratio(h_mass_pf, h_mass_mlpf, mass_bins, "Dijet mass (GeV)", "jetmet0_dijet_mass.pdf")

    asym_bins = np.linspace(0, 0.6, 61)
    asym_pf = (lj_pt_pf - slj_pt_pf) / (lj_pt_pf + slj_pt_pf)
    asym_mlpf = (lj_pt_mlpf - slj_pt_mlpf) / (lj_pt_mlpf + slj_pt_mlpf)
    h_asym_pf = to_bh(asym_pf, asym_bins)
    h_asym_mlpf = to_bh(asym_mlpf, asym_bins)
    plot_with_ratio(
        h_asym_pf,
        h_asym_mlpf,
        asym_bins,
        r"Dijet asymmetry $(p^{1}_{T}-p^{2}_{T})/(p^{1}_{T}+p^{2}_{T})$",
        "jetmet0_dijet_asymmetry.pdf",
        ylim_max=1e4,
    )


if __name__ == "__main__":
    main()
