import click
import os
import awkward
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import mplhep
import boost_histogram as bh
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from mlpf.plotting.utils import compute_response
from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, med_iqr, sample_label

def midpoints(x):
    return (x[1:] + x[:-1]) / 2

@click.command()
@click.option('--input-pf-parquet', required=True, type=str)
@click.option('--input-mlpf-parquet', required=True, type=str)
@click.option('--corrections-file', required=True, type=str, help="NPZ file with correction maps.")
@click.option('--output-dir', required=True, type=str)
@click.option('--jet-type', default='ak4', type=click.Choice(['ak4', 'ak8']))
@click.option('--sample-name', required=True, type=str, help='Sample name (e.g., QCD_PU_13p6)')
def make_plots(input_pf_parquet, input_mlpf_parquet, corrections_file, output_dir, jet_type, sample_name):
    """Applies corrections and generates validation plots."""

    os.makedirs(output_dir, exist_ok=True)
    mplhep.style.use("CMS")

    def sample_name_to_process(sample_name):
        if "QCD" in sample_name:
            key = "cms_pf_qcd"
        elif "TTbar" in sample_name:
            key = "cms_pf_ttbar"
        elif "PhotonJet" in sample_name:
            key = "cms_pf_photonjet"
        elif "ZTT" in sample_name:
            key = "cms_pf_ztt"
        else:
            return sample_name
        if "PU" not in sample_name:
            key += "_nopu"
        return key

    process_name = sample_name_to_process(sample_name)
    plot_sample_name = EVALUATION_DATASET_NAMES.get(process_name, sample_name)

    # plotting style variables
    legend_fontsize = 30
    sample_label_fontsize = 30
    addtext_fontsize = 25
    jet_label_coords = 0.02, 0.86
    jet_label_coords_single = 0.02, 0.88
    sample_label_coords = 0.02, 0.96
    default_cycler = plt.rcParams['axes.prop_cycle']
    pf_color = list(default_cycler)[1]["color"]
    mlpf_color = list(default_cycler)[2]["color"]
    pf_linestyle = "-."
    mlpf_linestyle = "-"
    mlpf_label = "MLPF-PUPPI"

    def varbins(*args):
        newlist = []
        for arg in args[:-1]:
            newlist.append(arg[:-1])
        newlist.append(args[-1])
        return np.concatenate(newlist)
    
    eta_bins_for_response = np.linspace(-5, 5, 81)
    eta_bins_for_kinematics = np.linspace(-5, 5, 51)

    if sample_name.startswith("QCD_"):
        pt_bins_for_response = varbins(np.linspace(20, 100, 21), np.linspace(100, 200, 5), np.linspace(200, 1000, 5))
        pt_bins_for_kinematics = varbins(np.linspace(20, 100, 21), np.linspace(100, 200, 5), np.linspace(200, 1000, 5))
        fatjet_bins = varbins(np.linspace(200, 1000, 5))
        met_bins = varbins(np.linspace(0, 150, 21), np.linspace(150, 500, 5))
        pt_bins_for_pu = [(0, 30), (30, 60), (60, 100), (100, 200), (200,5000)]
    elif sample_name.startswith("TTbar_"):
        pt_bins_for_response = varbins(np.linspace(20, 100, 21), np.linspace(100, 250, 5))
        pt_bins_for_kinematics = varbins(np.linspace(20, 100, 21), np.linspace(100, 250, 5))
        fatjet_bins = varbins(np.linspace(100, 400, 5))
        met_bins = varbins(np.linspace(0, 150, 21), np.linspace(150, 250, 5))
        pt_bins_for_pu = [(0, 30), (30, 60), (60, 100), (100, 200), (200,5000)]
    elif sample_name.startswith("PhotonJet_"):
        pt_bins_for_response = varbins(np.linspace(20, 60, 21), np.linspace(60, 120, 2))
        pt_bins_for_kinematics = varbins(np.linspace(20, 60, 21), np.linspace(60, 120, 2))
        fatjet_bins = varbins(np.linspace(0, 1000, 2))
        met_bins = varbins(np.linspace(0, 200, 41))
        pt_bins_for_pu = [(0, 30), (30, 60), (60, 100)]

    def to_bh(data, bins):
        h1 = bh.Histogram(bh.axis.Variable(bins))
        h1.fill(data)
        return h1

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    def compute_scale_res(response):
        h0 = to_bh(response, np.linspace(0, 2, 100))
        if h0.values().sum() > 0:
            try:
                parameters1, _ = curve_fit(
                    Gauss,
                    h0.axes[0].centers,
                    h0.values() / h0.values().sum(),
                    p0=[1.0, 1.0, 1.0],
                    maxfev=1000000,
                    method="dogbox",
                    bounds=[(-np.inf, 0.5, 0.0), (np.inf, 1.5, 2.0)],
                )
                norm = parameters1[0] * h0.values().sum()
                mean = parameters1[1]
                sigma = parameters1[2]
                return norm, mean, sigma
            except RuntimeError:
                return 0, 0, 0
        else:
            return 0, 0, 0

    def plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        variable_gen, variable_reco, bins,
        xlabel, filename,
        logy=True, raw_or_corr="raw",
        eta_cut=None, jet_label=""
    ):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        jet_pt_var = f"{jet_prefix}_pt_{raw_or_corr}"
        jet_eta_var = f"{jet_prefix}_eta"
        genjet_pt_var = f"{genjet_prefix}_pt"
        genjet_eta_var = f"{genjet_prefix}_eta"

        min_jet_pt = 5 if jet_prefix == "Jet" else 150

        # Gen jets
        gen_jet_mask = (data_pf[genjet_pt_var] > min_jet_pt)
        if eta_cut:
            gen_jet_mask = gen_jet_mask & (np.abs(data_pf[genjet_eta_var]) < eta_cut)
        
        h0 = to_bh(awkward.flatten(data_pf[f"{genjet_prefix}_{variable_gen}"][gen_jet_mask]), bins)

        # PF jets
        pf_jet_mask = (data_pf[jet_pt_var] > min_jet_pt)
        if jet_prefix == "Jet":
            pf_jet_mask = pf_jet_mask & ((data_pf["Jet_neMultiplicity"] > 1) | (np.abs(data_pf["Jet_eta"]) < 3))
        if eta_cut:
            pf_jet_mask = pf_jet_mask & (np.abs(data_pf[jet_eta_var]) < eta_cut)
        
        h1 = to_bh(awkward.flatten(data_pf[f"{jet_prefix}_{variable_reco}"][pf_jet_mask]), bins)

        # MLPF jets
        mlpf_jet_mask = (data_mlpf[jet_pt_var] > min_jet_pt)
        if jet_prefix == "Jet":
            mlpf_jet_mask = mlpf_jet_mask & ((data_mlpf["Jet_neMultiplicity"] > 1) | (np.abs(data_mlpf["Jet_eta"]) < 3))
        if eta_cut:
            mlpf_jet_mask = mlpf_jet_mask & (np.abs(data_mlpf[jet_eta_var]) < eta_cut)

        h2 = to_bh(awkward.flatten(data_mlpf[f"{jet_prefix}_{variable_reco}"][mlpf_jet_mask]), bins)

        plt.sca(a0)
        x0 = mplhep.histplot(h0, histtype="step", lw=2, label="Gen.", binwnorm=1.0, ls="--")
        x1 = mplhep.histplot(h1, histtype="step", lw=2, label="PF-PUPPI", binwnorm=1.0, ls=pf_linestyle)
        x2 = mplhep.histplot(h2, histtype="step", lw=2, label=mlpf_label, binwnorm=1.0, ls=mlpf_linestyle)

        if logy:
            plt.yscale("log")
            a0.set_ylim(top=a0.get_ylim()[1]*100)
        
        mplhep.cms.label("", data=False, com=13.6, year='Run 3', ax=a0)
        a0.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=a0.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
        
        jet_label_text = jet_label
        if eta_cut:
            jet_label_text += f", |$\eta$| < {eta_cut}"
        a0.text(jet_label_coords[0], jet_label_coords[1], jet_label_text, transform=a0.transAxes, fontsize=addtext_fontsize, ha="left", va="top")
        
        handles, labels = a0.get_legend_handles_labels()
        handles = [x0[0].stairs, x1[0].stairs, x2[0].stairs]
        a0.legend(handles, labels, loc=(0.5, 0.45), fontsize=legend_fontsize)
        plt.ylabel("Count")

        plt.sca(a1)
        mplhep.histplot(h0 / h0, histtype="step", lw=2, ls="--")
        mplhep.histplot(h1 / h0, histtype="step", lw=2, ls=pf_linestyle)
        mplhep.histplot(h2 / h0, histtype="step", lw=2, ls=mlpf_linestyle)
        plt.ylim(0.8, 1.2)
        plt.ylabel("Reco / Gen")
        plt.xlabel(xlabel)

        if variable_reco.startswith("pt"):
            plt.xscale("log")
        
        plt.xlim(min(bins), max(bins))
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def jet_response_plot(
        resp_pf, resp_mlpf,
        data_baseline, data_mlpf,
        response="response",
        genjet_min_pt=0,
        genjet_max_pt=5000,
        genjet_min_eta=-6,
        genjet_max_eta=6,
        jet_label="",
        additional_label="",
        jet_pt="Jet_pt_corr",
        genjet_pt="GenJet_pt",
        genjet_eta="GenJet_eta",
        additional_cut=lambda data: data["Pileup_nTrueInt"] >= 0,
        filename="jet_response.pdf"
    ):
        plt.figure()
        ax = plt.axes()
        b = np.linspace(0, 2, 101)

        mplhep.cms.label("", data=False, com=13.6, year='Run 3', ax=ax)
        ax.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
        ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label + additional_label, transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top")

        add_cut_pf = additional_cut(data_baseline)
        add_cut_mlpf = additional_cut(data_mlpf)

        jet_response_pf = awkward.flatten(
            resp_pf[response][(resp_pf[genjet_pt] >= genjet_min_pt)
            & (resp_pf[genjet_pt] < genjet_max_pt)
            & (resp_pf[genjet_eta] >= genjet_min_eta)
            & (resp_pf[genjet_eta] < genjet_max_eta)
            & add_cut_pf]
        )
        jet_response_mlpf = awkward.flatten(
            resp_mlpf[response][(resp_mlpf[genjet_pt] >= genjet_min_pt)
            & (resp_mlpf[genjet_pt] < genjet_max_pt)
            & (resp_mlpf[genjet_eta] >= genjet_min_eta)
            & (resp_mlpf[genjet_eta] < genjet_max_eta)
            & add_cut_mlpf]
        )

        h0 = to_bh(jet_response_pf, b)
        h1 = to_bh(jet_response_mlpf, b)

        norm_pf, mean_pf, std_pf = compute_scale_res(jet_response_pf)
        norm_mlpf, mean_mlpf, std_mlpf = compute_scale_res(jet_response_mlpf)
        med_pf, iqr_pf = med_iqr(jet_response_pf)
        med_mlpf, iqr_mlpf = med_iqr(jet_response_mlpf)

        x0 = mplhep.histplot(h0, histtype="step", lw=2, label="PF-PUPPI\nmean: {:.2f} std: {:.2f}\nmed: {:.2f} IQR: {:.2f}".format(mean_pf, std_pf, med_pf, iqr_pf), ls=pf_linestyle, color=pf_color)
        x1 = mplhep.histplot(h1, histtype="step", lw=2, label="{}\nmean: {:.2f} std: {:.2f}\nmed: {:.2f} IQR: {:.2f}".format(mlpf_label, mean_mlpf, std_mlpf, med_mlpf, iqr_mlpf), ls=mlpf_linestyle, color=mlpf_color)
        if norm_pf > 0:
            plt.plot(h0.axes[0].centers, Gauss(h0.axes[0].centers, norm_pf, mean_pf, std_pf), color=pf_color)
        if norm_mlpf > 0:
            plt.plot(h1.axes[0].centers, Gauss(h1.axes[0].centers, norm_mlpf, mean_mlpf, std_mlpf), color=mlpf_color)

        handles, labels = ax.get_legend_handles_labels()
        handles = [x0[0].stairs, x1[0].stairs]
        ax.legend(handles, labels, loc=(0.50, 0.65), fontsize=20)
        
        jet_label_corr = "Corr. jet "
        jet_label_raw = "Raw jet "
        jl = jet_label_corr if jet_pt.endswith("_corr") else jet_label_raw
        plt.xlabel(jl + "pT response")
        plt.ylabel("Count")

        ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_x(-0.01)
        ax.yaxis.get_offset_text().set_ha("right")

        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        return ((med_pf, iqr_pf, mean_pf, std_pf), (med_mlpf, iqr_mlpf, mean_mlpf, std_mlpf))

    def get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins, variable_name,
        response_type="response",
        jet_prefix="Jet", genjet_prefix="GenJet",
        jet_label="",
    ):
        med_vals_pf, iqr_vals_pf, mean_vals_pf, sigma_vals_pf = [], [], [], []
        med_vals_mlpf, iqr_vals_mlpf, mean_vals_mlpf, sigma_vals_mlpf = [], [], [], []

        for ibin in range(len(variable_bins) - 1):
            min_bin_val, max_bin_val = variable_bins[ibin], variable_bins[ibin + 1]

            stats_pf, stats_mlpf = jet_response_plot(
                resp_pf, resp_mlpf, data_pf, data_mlpf,
                response=response_type,
                genjet_min_pt=min_bin_val if "pt" in variable_name else 0,
                genjet_max_pt=max_bin_val if "pt" in variable_name else 5000,
                genjet_min_eta=min_bin_val if "eta" in variable_name else -6,
                genjet_max_eta=max_bin_val if "eta" in variable_name else 6,
                jet_label=jet_label,
                additional_label=f", {min_bin_val:.2f}<{variable_name.replace(genjet_prefix+'_', '')}<{max_bin_val:.2f}",
                jet_pt=f"{jet_prefix}_pt_corr" if response_type == "response" else f"{jet_prefix}_pt_raw",
                genjet_pt=f"{genjet_prefix}_pt",
                genjet_eta=f"{genjet_prefix}_eta",
                filename=f"{jet_prefix}_response_{response_type}_bin_{variable_name.replace(genjet_prefix+'_', '')}_{ibin}.pdf"
            )
            med_vals_pf.append(stats_pf[0])
            iqr_vals_pf.append(stats_pf[1])
            mean_vals_pf.append(stats_pf[2])
            sigma_vals_pf.append(stats_pf[3])
            med_vals_mlpf.append(stats_mlpf[0])
            iqr_vals_mlpf.append(stats_mlpf[1])
            mean_vals_mlpf.append(stats_mlpf[2])
            sigma_vals_mlpf.append(stats_mlpf[3])

        return np.array(med_vals_pf), np.array(iqr_vals_pf), np.array(mean_vals_pf), np.array(sigma_vals_pf), \
               np.array(med_vals_mlpf), np.array(iqr_vals_mlpf), np.array(mean_vals_mlpf), np.array(sigma_vals_mlpf)

    data_pf = awkward.from_parquet(input_pf_parquet)
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)

    corrections = np.load(corrections_file)
    corr_map_pf = corrections['corr_map_pf']
    corr_map_mlpf = corrections['corr_map_mlpf']
    eta_bins = corrections['eta_bins']
    pt_bins = corrections['pt_bins']

    interp_pf = RegularGridInterpolator(
        (midpoints(np.array(eta_bins)), midpoints(np.array(pt_bins))),
        corr_map_pf, method='linear', bounds_error=False, fill_value=None
    )
    interp_mlpf = RegularGridInterpolator(
        (midpoints(np.array(eta_bins)), midpoints(np.array(pt_bins))),
        corr_map_mlpf, method='linear', bounds_error=False, fill_value=None
    )

    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    corr_pf_interp = interp_pf(np.stack([
        awkward.to_numpy(awkward.flatten(data_pf[jet_prefix + "_eta"])),
        awkward.to_numpy(awkward.flatten(data_pf[jet_prefix + "_pt_raw"]))
    ]).T)
    corr_pf_interp = awkward.unflatten(corr_pf_interp, awkward.count(data_pf[jet_prefix + "_eta"], axis=1))
    data_pf[jet_prefix + "_pt_corr"] = data_pf[jet_prefix + "_pt_raw"] * corr_pf_interp

    corr_mlpf_interp = interp_mlpf(np.stack([
        awkward.to_numpy(awkward.flatten(data_mlpf[jet_prefix + "_eta"])),
        awkward.to_numpy(awkward.flatten(data_mlpf[jet_prefix + "_pt_raw"]))
    ]).T)
    corr_mlpf_interp = awkward.unflatten(corr_mlpf_interp, awkward.count(data_mlpf[jet_prefix + "_eta"], axis=1))
    data_mlpf[jet_prefix + "_pt_corr"] = data_mlpf[jet_prefix + "_pt_raw"] * corr_mlpf_interp

    if jet_type == 'ak4':
        deltar_cut = 0.2
        jet_label = "AK4 jets, $p_T$ > 5 GeV"
    else:  # ak8
        deltar_cut = 0.4
        jet_label = "AK8 jets, $p_T$ > 150 GeV"

    # Plot kinematic distributions
    plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        "pt", "pt_raw", pt_bins_for_kinematics,
        f"{jet_prefix} pT [GeV]", f"{jet_type}_pt_raw.pdf",
        raw_or_corr="raw", jet_label=jet_label
    )
    plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        "pt", "pt_corr", pt_bins_for_kinematics,
        f"Corrected {jet_prefix} pT [GeV]", f"{jet_type}_pt_corr.pdf",
        raw_or_corr="corr", jet_label=jet_label
    )
    plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        "eta", "eta", eta_bins_for_kinematics,
        f"{jet_prefix} $\eta$", f"{jet_type}_eta.pdf",
        logy=True, jet_label=jet_label
    )
    if jet_type == 'ak4':
        plot_kinematic_distribution(
            data_pf, data_mlpf,
            jet_prefix, genjet_prefix,
            "pt", "pt_raw", pt_bins_for_kinematics,
            f"{jet_prefix} pT [GeV]", f"{jet_type}_pt_raw_barrel.pdf",
            raw_or_corr="raw", eta_cut=2.5, jet_label=jet_label
        )
        plot_kinematic_distribution(
            data_pf, data_mlpf,
            jet_prefix, genjet_prefix,
            "pt", "pt_corr", pt_bins_for_kinematics,
            f"Corrected {jet_prefix} pT [GeV]", f"{jet_type}_pt_corr_barrel.pdf",
            raw_or_corr="corr", eta_cut=2.5, jet_label=jet_label
        )

    resp_pf = compute_response(data_pf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_mlpf = compute_response(data_mlpf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)

    # Generate plots
    jet_response_plot(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        response="response_raw", jet_pt=f"{jet_prefix}_pt_raw",
        jet_label=jet_label,
        filename=f"{jet_type}_jet_pt_ratio_raw.pdf"
    )
    jet_response_plot(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        response="response", jet_pt=f"{jet_prefix}_pt_corr",
        jet_label=jet_label,
        filename=f"{jet_type}_jet_pt_ratio_corr.pdf"
    )

    med_pf_vs_pt, iqr_pf_vs_pt, mean_pf_vs_pt, sigma_pf_vs_pt, med_mlpf_vs_pt, iqr_mlpf_vs_pt, mean_mlpf_vs_pt, sigma_mlpf_vs_pt = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=pt_bins_for_response, variable_name=f"{genjet_prefix}_pt",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )
    med_pf_vs_pt_raw, iqr_pf_vs_pt_raw, mean_pf_vs_pt_raw, sigma_pf_vs_pt_raw, med_mlpf_vs_pt_raw, iqr_mlpf_vs_pt_raw, mean_mlpf_vs_pt_raw, sigma_mlpf_vs_pt_raw = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=pt_bins_for_response, variable_name=f"{genjet_prefix}_pt",
        response_type="response_raw",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )

    med_pf_vs_eta, iqr_pf_vs_eta, mean_pf_vs_eta, sigma_pf_vs_eta, med_mlpf_vs_eta, iqr_mlpf_vs_eta, mean_mlpf_vs_eta, sigma_mlpf_vs_eta = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=eta_bins_for_response, variable_name=f"{genjet_prefix}_eta",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )
    med_pf_vs_eta_raw, iqr_pf_vs_eta_raw, mean_pf_vs_eta_raw, sigma_pf_vs_eta_raw, med_mlpf_vs_eta_raw, iqr_mlpf_vs_eta_raw, mean_mlpf_vs_eta_raw, sigma_mlpf_vs_eta_raw = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=eta_bins_for_response, variable_name=f"{genjet_prefix}_eta",
        response_type="response_raw",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )

    # Plot scale vs pt
    fig, ax = plt.subplots()
    ax.plot(midpoints(pt_bins_for_response), mean_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), mean_mlpf_vs_pt, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), mean_pf_vs_pt_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(pt_bins_for_response), mean_mlpf_vs_pt_raw, label=mlpf_label + " raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("GenJet $p_T$ [GeV]")
    ax.set_ylabel("Mean response")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.5)
    plt.axhline(1.0, color="black", ls="--")
    mplhep.cms.label(ax=ax, data=False, com=13.6, year='Run 3')
    ax.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
    ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label, transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top")
    fig.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_pt.pdf"))
    plt.close(fig)

    # Plot resolution vs pt
    fig, ax = plt.subplots()
    ax.plot(midpoints(pt_bins_for_response), sigma_pf_vs_pt / mean_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), sigma_mlpf_vs_pt / mean_mlpf_vs_pt, label=f"{mlpf_label}", color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), sigma_pf_vs_pt_raw / mean_pf_vs_pt_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(pt_bins_for_response), sigma_mlpf_vs_pt_raw / mean_mlpf_vs_pt_raw, label=f"{mlpf_label} raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("GenJet $p_T$ [GeV]")
    ax.set_ylabel("Response resolution")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year='Run 3')
    ax.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
    ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label, transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top")
    fig.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_pt.pdf"))
    plt.close(fig)

    # Plot scale vs eta
    fig, ax = plt.subplots()
    ax.plot(midpoints(eta_bins_for_response), mean_pf_vs_eta, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), mean_mlpf_vs_eta, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), mean_pf_vs_eta_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(eta_bins_for_response), mean_mlpf_vs_eta_raw, label=mlpf_label + " raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("GenJet $\eta$")
    ax.set_ylabel("Mean response")
    ax.legend()
    ax.set_ylim(0.5, 1.5)
    plt.axhline(1.0, color="black", ls="--")
    mplhep.cms.label(ax=ax, data=False, com=13.6, year='Run 3')
    ax.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
    ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label, transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top")
    fig.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_eta.pdf"))
    plt.close(fig)

    # Plot resolution vs eta
    fig, ax = plt.subplots()
    ax.plot(midpoints(eta_bins_for_response), sigma_pf_vs_eta / mean_pf_vs_eta, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), sigma_mlpf_vs_eta / mean_mlpf_vs_eta, label=f"{mlpf_label}", color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), sigma_pf_vs_eta_raw / mean_pf_vs_eta_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(eta_bins_for_response), sigma_mlpf_vs_eta_raw / mean_mlpf_vs_eta_raw, label=f"{mlpf_label} raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("GenJet $\eta$")
    ax.set_ylabel("Response resolution")
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year='Run 3')
    ax.text(sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top")
    ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label, transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top")
    fig.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_eta.pdf"))
    plt.close(fig)

    print(f"Generated plots in {output_dir}")


if __name__ == '__main__':
    make_plots()
