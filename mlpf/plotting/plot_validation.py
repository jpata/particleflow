import click
import os
import awkward
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D
from mlpf.plotting.utils import compute_response, Gauss, to_bh, compute_scale_res
from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, med_iqr, sample_name_to_process, midpoints


@click.command()
@click.option("--input-pf-parquet", required=True, type=str)
@click.option("--input-mlpf-parquet", required=True, type=str)
@click.option("--corrections-file", required=True, type=str, help="NPZ file with correction maps.")
@click.option("--output-dir", required=True, type=str)
@click.option("--jet-type", default="ak4", type=click.Choice(["ak4", "ak8"]))
@click.option("--sample-name", required=True, type=str, help="Sample name (e.g., QCD_PU_13p6)")
@click.option("--fiducial-cuts", default="inclusive", type=click.Choice(["inclusive", "eta_less_2p5"]))
def make_plots(input_pf_parquet, input_mlpf_parquet, corrections_file, output_dir, jet_type, sample_name, fiducial_cuts):
    """Applies corrections and generates validation plots."""

    output_dir = str(Path(output_dir, sample_name, jet_type, fiducial_cuts))

    os.makedirs(output_dir, exist_ok=True)
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    process_name = sample_name_to_process(sample_name)
    plot_sample_name = EVALUATION_DATASET_NAMES.get(process_name, sample_name)

    # plotting style variables
    legend_fontsize = 30
    legend_fontsize_jet_response = 20
    legend_loc = (0.5, 0.45)
    legend_loc_effpur = (0.5, 0.68)
    legend_loc_scalereso = (0.40, 0.50)
    legend_loc_jet_response = (0.05, 0.55)
    sample_label_fontsize = 30
    addtext_fontsize = 25
    jet_label_coords = 0.02, 0.86
    jet_label_coords_single = 0.02, 0.88
    sample_label_coords = 0.02, 0.96
    default_cycler = plt.rcParams["axes.prop_cycle"]
    pf_color = list(default_cycler)[1]["color"]
    mlpf_color = list(default_cycler)[2]["color"]
    pf_linestyle = "-."
    mlpf_linestyle = "-"
    mlpf_label = "MLPF-PUPPI"

    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    # fiducial cut for kinematic distributions only
    min_jet_pt = 0
    if jet_prefix == "Jet":
        min_jet_pt = 10
    elif jet_prefix == "FatJet":
        min_jet_pt = 150

    def varbins(*args):
        newlist = []
        for arg in args[:-1]:
            newlist.append(arg[:-1])
        newlist.append(args[-1])
        return np.concatenate(newlist)

    eta_bins_for_response = np.linspace(-5, 5, 81)
    eta_bins_for_kinematics = np.linspace(-5, 5, 51)
    eta_bins_for_pureff = np.linspace(-5, 5, 51)

    if sample_name.startswith("QCD_"):
        if jet_type == "ak4":
            pt_bins_for_response = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000])
            pt_bins_for_kinematics = varbins(np.linspace(20, 100, 21), np.linspace(100, 200, 5), np.linspace(200, 1000, 5))
            pt_bins_for_pureff = varbins(np.linspace(1, 20, 5), np.linspace(20, 100, 21), np.linspace(100, 200, 5), np.linspace(200, 1000, 5))
            pt_bins_for_pu = [(0, 30), (30, 60), (60, 100), (100, 200), (200, 5000)]
        elif jet_type == "ak8":
            pt_bins_for_response = varbins(np.linspace(20, 1000, 5))
            pt_bins_for_kinematics = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]
            pt_bins_for_pureff = varbins(np.linspace(1, 20, 5), np.linspace(20, 100, 5), np.linspace(100, 1000, 5))
            pt_bins_for_pu = [(100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500)]
    elif sample_name.startswith("TTbar_"):
        if jet_type == "ak4":
            pt_bins_for_response = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 250, 500])
            pt_bins_for_kinematics = varbins(np.linspace(20, 100, 21), np.linspace(100, 250, 5))
            pt_bins_for_pureff = varbins(np.linspace(1, 20, 5), np.linspace(20, 100, 21), np.linspace(100, 250, 5))
        elif jet_type == "ak8":
            pt_bins_for_response = varbins(np.linspace(10, 400, 5))
            pt_bins_for_kinematics = varbins(np.linspace(10, 400, 5))
            pt_bins_for_pureff = varbins(np.linspace(1, 400, 5))
        pt_bins_for_pu = [(0, 30), (30, 60), (60, 100), (100, 200), (200, 5000)]
    elif sample_name.startswith("PhotonJet_"):
        if jet_type == "ak4":
            pt_bins_for_response = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300])
            pt_bins_for_kinematics = varbins(np.linspace(20, 60, 21), np.linspace(60, 120, 2))
            pt_bins_for_pureff = varbins(np.linspace(5, 20, 5), np.linspace(20, 60, 21), np.linspace(60, 120, 2))
        elif jet_type == "ak8":
            pt_bins_for_response = varbins(np.linspace(1, 1000, 5))
            pt_bins_for_kinematics = varbins(np.linspace(1, 1000, 5))
            pt_bins_for_pureff = varbins(np.linspace(1, 1000, 5))
        pt_bins_for_pu = [(0, 30), (30, 60), (60, 100)]

    def plot_kinematic_distribution(
        data_pf, data_mlpf, jet_prefix, genjet_prefix, variable_gen, variable_reco, bins, xlabel, filename, logy=True, raw_or_corr="raw", jet_label=""
    ):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        jet_pt_var = f"{jet_prefix}_pt_{raw_or_corr}"
        genjet_pt_var = f"{genjet_prefix}_pt"

        print(f"Jet kinematic plots with pT > {min_jet_pt}")

        # Gen jets
        gen_jet_mask = data_pf[genjet_pt_var] > min_jet_pt

        h0 = to_bh(awkward.flatten(data_pf[f"{genjet_prefix}_{variable_gen}"][gen_jet_mask]), bins)

        # PF jets
        pf_jet_mask = data_pf[jet_pt_var] > min_jet_pt
        if jet_prefix == "Jet":
            pf_jet_mask = pf_jet_mask & ((data_pf["Jet_neMultiplicity"] > 1) | (np.abs(data_pf["Jet_eta"]) < 3))

        h1 = to_bh(awkward.flatten(data_pf[f"{jet_prefix}_{variable_reco}"][pf_jet_mask]), bins)

        # MLPF jets
        mlpf_jet_mask = data_mlpf[jet_pt_var] > min_jet_pt
        if jet_prefix == "Jet":
            mlpf_jet_mask = mlpf_jet_mask & ((data_mlpf["Jet_neMultiplicity"] > 1) | (np.abs(data_mlpf["Jet_eta"]) < 3))

        h2 = to_bh(awkward.flatten(data_mlpf[f"{jet_prefix}_{variable_reco}"][mlpf_jet_mask]), bins)

        plt.sca(a0)
        x0 = mplhep.histplot(h0, histtype="step", lw=2, label="Gen.", binwnorm=1.0, ls="--")
        x1 = mplhep.histplot(h1, histtype="step", lw=2, label="PF-PUPPI", binwnorm=1.0, ls=pf_linestyle)
        x2 = mplhep.histplot(h2, histtype="step", lw=2, label=mlpf_label, binwnorm=1.0, ls=mlpf_linestyle)

        if logy:
            plt.yscale("log")
            # ensure legend fits on eta plot
            mult = 100
            if variable_gen == "eta":
                mult = 1000
            a0.set_ylim(bottom=100, top=a0.get_ylim()[1] * mult)

        mplhep.cms.label("", data=False, com=13.6, year="Run 3", ax=a0)
        a0.text(
            sample_label_coords[0],
            sample_label_coords[1],
            plot_sample_name,
            transform=a0.transAxes,
            fontsize=sample_label_fontsize,
            ha="left",
            va="top",
        )

        jet_label_text = jet_label
        a0.text(jet_label_coords[0], jet_label_coords[1], jet_label_text, transform=a0.transAxes, fontsize=addtext_fontsize, ha="left", va="top")

        handles, labels = a0.get_legend_handles_labels()
        handles = [x0[0].stairs, x1[0].stairs, x2[0].stairs]
        a0.legend(handles, labels, loc=legend_loc, fontsize=legend_fontsize)
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
        resp_pf,
        resp_mlpf,
        data_baseline,
        data_mlpf,
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
        filename="jet_response.pdf",
        save_figure=False,
    ):
        plt.figure()
        ax = plt.axes()
        b = np.linspace(0, 2, 101)

        mplhep.cms.label("", data=False, com=13.6, year="Run 3", ax=ax)
        ax.text(
            sample_label_coords[0],
            sample_label_coords[1],
            plot_sample_name,
            transform=ax.transAxes,
            fontsize=sample_label_fontsize,
            ha="left",
            va="top",
        )
        ax.text(
            jet_label_coords_single[0],
            jet_label_coords_single[1],
            jet_label + additional_label,
            transform=ax.transAxes,
            fontsize=addtext_fontsize,
            ha="left",
            va="top",
        )

        add_cut_pf = additional_cut(data_baseline)
        add_cut_mlpf = additional_cut(data_mlpf)

        jet_response_pf = awkward.flatten(
            resp_pf[response][
                (resp_pf[genjet_pt] >= genjet_min_pt)
                & (resp_pf[genjet_pt] < genjet_max_pt)
                & (resp_pf[genjet_eta] >= genjet_min_eta)
                & (resp_pf[genjet_eta] < genjet_max_eta)
                & add_cut_pf
            ]
        )
        jet_response_mlpf = awkward.flatten(
            resp_mlpf[response][
                (resp_mlpf[genjet_pt] >= genjet_min_pt)
                & (resp_mlpf[genjet_pt] < genjet_max_pt)
                & (resp_mlpf[genjet_eta] >= genjet_min_eta)
                & (resp_mlpf[genjet_eta] < genjet_max_eta)
                & add_cut_mlpf
            ]
        )

        h0 = to_bh(jet_response_pf, b)
        h1 = to_bh(jet_response_mlpf, b)

        norm_pf, mean_pf, std_pf = compute_scale_res(jet_response_pf)
        norm_mlpf, mean_mlpf, std_mlpf = compute_scale_res(jet_response_mlpf)
        med_pf, iqr_pf = med_iqr(jet_response_pf)
        med_mlpf, iqr_mlpf = med_iqr(jet_response_mlpf)

        x0 = mplhep.histplot(
            h0,
            histtype="step",
            lw=2,
            label="PF-PUPPI\nmean: {:.2f} std: {:.2f}\nmed: {:.2f} IQR: {:.2f}".format(mean_pf, std_pf, med_pf, iqr_pf),
            ls=pf_linestyle,
            color=pf_color,
        )
        x1 = mplhep.histplot(
            h1,
            histtype="step",
            lw=2,
            label="{}\nmean: {:.2f} std: {:.2f}\nmed: {:.2f} IQR: {:.2f}".format(mlpf_label, mean_mlpf, std_mlpf, med_mlpf, iqr_mlpf),
            ls=mlpf_linestyle,
            color=mlpf_color,
        )
        if norm_pf > 0:
            plt.plot(h0.axes[0].centers, Gauss(h0.axes[0].centers, norm_pf, mean_pf, std_pf), color=pf_color)
        if norm_mlpf > 0:
            plt.plot(h1.axes[0].centers, Gauss(h1.axes[0].centers, norm_mlpf, mean_mlpf, std_mlpf), color=mlpf_color)

        handles, labels = ax.get_legend_handles_labels()
        handles = [x0[0].stairs, x1[0].stairs]
        ax.legend(handles, labels, loc=legend_loc_jet_response, fontsize=legend_fontsize_jet_response)

        jet_label_corr = "Corr. jet "
        jet_label_raw = "Raw jet "
        jl = jet_label_corr if jet_pt.endswith("_corr") else jet_label_raw
        plt.xlabel(jl + "$p_{T}/p_{T,ptcl}$ response")
        plt.ylabel("Count")

        ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_x(-0.01)
        ax.yaxis.get_offset_text().set_ha("right")

        if save_figure:
            plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        return ((med_pf, iqr_pf, mean_pf, std_pf), (med_mlpf, iqr_mlpf, mean_mlpf, std_mlpf))

    def get_response_in_bins(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        variable_bins,
        variable_name,
        response_type="response",
        jet_prefix="Jet",
        genjet_prefix="GenJet",
        jet_label="",
    ):
        med_vals_pf, iqr_vals_pf, mean_vals_pf, sigma_vals_pf = [], [], [], []
        med_vals_mlpf, iqr_vals_mlpf, mean_vals_mlpf, sigma_vals_mlpf = [], [], [], []

        varname_pretty = ""
        if variable_name == "GenJet_eta" or variable_name == "GenJetAK8_eta":
            varname_pretty = "$\eta_{ptcl}$"
        elif variable_name == "GenJet_pt" or variable_name == "GenJetAK8_pt":
            varname_pretty = "$p_{T,ptcl}$"
        else:
            raise Exception(f"Unknown variable name {variable_name}")

        for ibin in range(len(variable_bins) - 1):
            min_bin_val, max_bin_val = variable_bins[ibin], variable_bins[ibin + 1]

            stats_pf, stats_mlpf = jet_response_plot(
                resp_pf,
                resp_mlpf,
                data_pf,
                data_mlpf,
                response=response_type,
                genjet_min_pt=min_bin_val if "pt" in variable_name else 0,
                genjet_max_pt=max_bin_val if "pt" in variable_name else 5000,
                genjet_min_eta=min_bin_val if "eta" in variable_name else -6,
                genjet_max_eta=max_bin_val if "eta" in variable_name else 6,
                jet_label=jet_label,
                additional_label=f", {min_bin_val:.2f}<{varname_pretty}<{max_bin_val:.2f}",
                jet_pt=f"{jet_prefix}_pt_corr" if response_type == "response" else f"{jet_prefix}_pt_raw",
                genjet_pt=f"{genjet_prefix}_pt",
                genjet_eta=f"{genjet_prefix}_eta",
                filename=f"{jet_prefix}_response_{response_type}_bin_{variable_name.replace(genjet_prefix+'_', '')}_{ibin}.pdf",
            )
            med_vals_pf.append(stats_pf[0])
            iqr_vals_pf.append(stats_pf[1])
            mean_vals_pf.append(stats_pf[2])
            sigma_vals_pf.append(stats_pf[3])
            med_vals_mlpf.append(stats_mlpf[0])
            iqr_vals_mlpf.append(stats_mlpf[1])
            mean_vals_mlpf.append(stats_mlpf[2])
            sigma_vals_mlpf.append(stats_mlpf[3])

        return (
            np.array(med_vals_pf),
            np.array(iqr_vals_pf),
            np.array(mean_vals_pf),
            np.array(sigma_vals_pf),
            np.array(med_vals_mlpf),
            np.array(iqr_vals_mlpf),
            np.array(mean_vals_mlpf),
            np.array(sigma_vals_mlpf),
        )

    def plot_efficiency_vs_kin(
        h_total_gen,
        h_matched_pf,
        h_matched_mlpf,
        bins,
        xlabel,
        output_dir,
        filename,
        jet_label,
    ):
        fig, ax = plt.subplots()

        eff_pf = h_matched_pf / h_total_gen
        eff_mlpf = h_matched_mlpf / h_total_gen

        mplhep.histplot(eff_pf, ax=ax, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
        mplhep.histplot(eff_mlpf, ax=ax, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Jet efficiency")
        ax.legend(fontsize=legend_fontsize, loc=legend_loc_effpur)
        if "p_T" in xlabel:
            ax.set_xscale("log")
        ax.set_ylim(0.0, 1.5)
        ax.set_xlim(np.min(h_total_gen.axes[0]), np.max(h_total_gen.axes[0]))
        plt.axhline(1.0, color="black", ls="--")
        mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
        ax.text(
            sample_label_coords[0],
            sample_label_coords[1],
            plot_sample_name,
            transform=ax.transAxes,
            fontsize=sample_label_fontsize,
            ha="left",
            va="top",
        )
        ax.text(
            jet_label_coords_single[0],
            jet_label_coords_single[1],
            jet_label,
            transform=ax.transAxes,
            fontsize=addtext_fontsize,
            ha="left",
            va="top",
        )
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

    def plot_purity_vs_kin(
        h_total_pf,
        h_total_mlpf,
        h_matched_pf,
        h_matched_mlpf,
        bins,
        xlabel,
        output_dir,
        filename,
        jet_label,
    ):
        fig, ax = plt.subplots()

        pur_pf = h_matched_pf / h_total_pf
        pur_mlpf = h_matched_mlpf / h_total_mlpf

        mplhep.histplot(pur_pf, ax=ax, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
        mplhep.histplot(pur_mlpf, ax=ax, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Jet purity")
        ax.legend(fontsize=legend_fontsize, loc=legend_loc_effpur)
        if "p_T" in xlabel:
            ax.set_xscale("log")
        ax.set_ylim(0.0, 1.5)
        ax.set_xlim(np.min(h_total_pf.axes[0]), np.max(h_total_pf.axes[0]))
        plt.axhline(1.0, color="black", ls="--")
        mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
        ax.text(
            sample_label_coords[0],
            sample_label_coords[1],
            plot_sample_name,
            transform=ax.transAxes,
            fontsize=sample_label_fontsize,
            ha="left",
            va="top",
        )
        ax.text(
            jet_label_coords_single[0],
            jet_label_coords_single[1],
            jet_label,
            transform=ax.transAxes,
            fontsize=addtext_fontsize,
            ha="left",
            va="top",
        )
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

    data_pf = awkward.from_parquet(input_pf_parquet)
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)

    jet_label = ""
    if jet_type == "ak4":
        deltar_cut = 0.2
        jet_label = f"AK4 jets, $p_T$ > {min_jet_pt} GeV"
    elif jet_type == "ak8":
        deltar_cut = 0.4
        jet_label = f"AK8 jets, $p_T$ > {min_jet_pt} GeV"

    if jet_type == "ak4":
        jet_label_inclusive = "AK4 jets"
    elif jet_type == "ak8":
        jet_label_inclusive = "AK8 jets"

    if fiducial_cuts == "eta_less_2p5":
        eta_label = ", $|η|$ < 2.5"
        jet_label += eta_label
        jet_label_inclusive += eta_label
        for data in [data_pf, data_mlpf]:
            msk_rj_eta = np.abs(data["Jet_eta"]) < 2.5
            for k in data.fields:
                if k.startswith("Jet_"):
                    data[k] = data[k][msk_rj_eta]
            msk_gj_eta = np.abs(data["GenJet_eta"]) < 2.5
            for k in data.fields:
                if k.startswith("GenJet_"):
                    data[k] = data[k][msk_gj_eta]

    corrections = np.load(corrections_file)
    corr_map_pf = corrections["corr_map_pf"]
    corr_map_mlpf = corrections["corr_map_mlpf"]
    eta_bins = corrections["eta_bins"]
    pt_bins = corrections["pt_bins"]

    interp_pf = RegularGridInterpolator(
        (midpoints(np.array(eta_bins)), midpoints(np.array(pt_bins))), corr_map_pf, method="linear", bounds_error=False, fill_value=None
    )
    interp_mlpf = RegularGridInterpolator(
        (midpoints(np.array(eta_bins)), midpoints(np.array(pt_bins))), corr_map_mlpf, method="linear", bounds_error=False, fill_value=None
    )

    corr_pf_interp = interp_pf(
        np.stack(
            [awkward.to_numpy(awkward.flatten(data_pf[jet_prefix + "_eta"])), awkward.to_numpy(awkward.flatten(data_pf[jet_prefix + "_pt_raw"]))]
        ).T
    )
    corr_pf_interp = awkward.unflatten(corr_pf_interp, awkward.count(data_pf[jet_prefix + "_eta"], axis=1))
    data_pf[jet_prefix + "_pt_corr"] = data_pf[jet_prefix + "_pt_raw"] * corr_pf_interp

    corr_mlpf_interp = interp_mlpf(
        np.stack(
            [awkward.to_numpy(awkward.flatten(data_mlpf[jet_prefix + "_eta"])), awkward.to_numpy(awkward.flatten(data_mlpf[jet_prefix + "_pt_raw"]))]
        ).T
    )
    corr_mlpf_interp = awkward.unflatten(corr_mlpf_interp, awkward.count(data_mlpf[jet_prefix + "_eta"], axis=1))
    data_mlpf[jet_prefix + "_pt_corr"] = data_mlpf[jet_prefix + "_pt_raw"] * corr_mlpf_interp

    # Plot kinematic distributions
    plot_kinematic_distribution(
        data_pf,
        data_mlpf,
        jet_prefix,
        genjet_prefix,
        "pt",
        "pt_raw",
        pt_bins_for_kinematics,
        "Raw $p_T$ (GeV)",
        f"{jet_type}_pt_raw.pdf",
        raw_or_corr="raw",
        jet_label=jet_label,
    )
    plot_kinematic_distribution(
        data_pf,
        data_mlpf,
        jet_prefix,
        genjet_prefix,
        "pt",
        "pt_corr",
        pt_bins_for_kinematics,
        "Corrected $p_T$ (GeV)",
        f"{jet_type}_pt_corr.pdf",
        raw_or_corr="corr",
        jet_label=jet_label,
    )
    plot_kinematic_distribution(
        data_pf,
        data_mlpf,
        jet_prefix,
        genjet_prefix,
        "eta",
        "eta",
        eta_bins_for_kinematics,
        "$\eta$",
        f"{jet_type}_eta.pdf",
        logy=True,
        jet_label=jet_label,
    )
    resp_pf = compute_response(data_pf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_mlpf = compute_response(data_mlpf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)

    # Efficiency: fraction of gen jets that are matched to reco
    gen_jet_mask = data_pf[f"{genjet_prefix}_pt"] > 0
    h_total_gen_pt = to_bh(awkward.flatten(data_pf[f"{genjet_prefix}_pt"][gen_jet_mask]), pt_bins_for_pureff)
    h_total_gen_eta = to_bh(awkward.flatten(data_pf[f"{genjet_prefix}_eta"][gen_jet_mask]), eta_bins_for_pureff)

    h_matched_pf_gen_pt = to_bh(awkward.flatten(resp_pf[f"{genjet_prefix}_pt_unfiltered"]), pt_bins_for_pureff)
    h_matched_pf_gen_eta = to_bh(awkward.flatten(resp_pf[f"{genjet_prefix}_eta_unfiltered"]), eta_bins_for_pureff)

    h_matched_mlpf_gen_pt = to_bh(awkward.flatten(resp_mlpf[f"{genjet_prefix}_pt_unfiltered"]), pt_bins_for_pureff)
    h_matched_mlpf_gen_eta = to_bh(awkward.flatten(resp_mlpf[f"{genjet_prefix}_eta_unfiltered"]), eta_bins_for_pureff)

    plot_efficiency_vs_kin(
        h_total_gen_pt,
        h_matched_pf_gen_pt,
        h_matched_mlpf_gen_pt,
        pt_bins_for_pureff,
        "$p_{T,ptcl}$ (GeV)",
        output_dir,
        f"{jet_type}_efficiency_vs_pt.pdf",
        jet_label_inclusive,
    )
    plot_efficiency_vs_kin(
        h_total_gen_eta,
        h_matched_pf_gen_eta,
        h_matched_mlpf_gen_eta,
        eta_bins_for_pureff,
        "$η_{ptcl}$",
        output_dir,
        f"{jet_type}_efficiency_vs_eta.pdf",
        jet_label_inclusive,
    )

    # Purity: fraction of reco jets matched to gen
    pf_jet_mask = data_pf[f"{jet_prefix}_pt_raw"] > 0
    mlpf_jet_mask = data_mlpf[f"{jet_prefix}_pt_raw"] > 0
    if jet_prefix == "Jet":
        pf_jet_mask = pf_jet_mask & ((data_pf["Jet_neMultiplicity"] > 1) | (np.abs(data_pf["Jet_eta"]) < 3))
        mlpf_jet_mask = mlpf_jet_mask & ((data_mlpf["Jet_neMultiplicity"] > 1) | (np.abs(data_mlpf["Jet_eta"]) < 3))

    h_total_pf_pt = to_bh(awkward.flatten(data_pf[f"{jet_prefix}_pt_raw"][pf_jet_mask]), pt_bins_for_pureff)
    h_total_pf_eta = to_bh(awkward.flatten(data_pf[f"{jet_prefix}_eta"][pf_jet_mask]), eta_bins_for_pureff)
    h_total_mlpf_pt = to_bh(awkward.flatten(data_mlpf[f"{jet_prefix}_pt_raw"][mlpf_jet_mask]), pt_bins_for_pureff)
    h_total_mlpf_eta = to_bh(awkward.flatten(data_mlpf[f"{jet_prefix}_eta"][mlpf_jet_mask]), eta_bins_for_pureff)

    h_matched_pf_reco_pt = to_bh(awkward.flatten(resp_pf[f"{jet_prefix}_pt_raw_unfiltered"]), pt_bins_for_pureff)
    h_matched_pf_reco_eta = to_bh(awkward.flatten(resp_pf[f"{jet_prefix}_eta_unfiltered"]), eta_bins_for_pureff)

    h_matched_mlpf_reco_pt = to_bh(awkward.flatten(resp_mlpf[f"{jet_prefix}_pt_raw_unfiltered"]), pt_bins_for_pureff)
    h_matched_mlpf_reco_eta = to_bh(awkward.flatten(resp_mlpf[f"{jet_prefix}_eta_unfiltered"]), eta_bins_for_pureff)

    plot_purity_vs_kin(
        h_total_pf_pt,
        h_total_mlpf_pt,
        h_matched_pf_reco_pt,
        h_matched_mlpf_reco_pt,
        pt_bins_for_pureff,
        "$p_T$ (GeV)",
        output_dir,
        f"{jet_type}_purity_vs_pt.pdf",
        jet_label_inclusive,
    )
    plot_purity_vs_kin(
        h_total_pf_eta,
        h_total_mlpf_eta,
        h_matched_pf_reco_eta,
        h_matched_mlpf_reco_eta,
        eta_bins_for_pureff,
        "$η$",
        output_dir,
        f"{jet_type}_purity_vs_eta.pdf",
        jet_label_inclusive,
    )

    # overall jet response ratio plots
    jet_response_plot(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        response="response_raw",
        jet_pt=f"{jet_prefix}_pt_raw",
        jet_label=jet_label,
        genjet_pt=f"{genjet_prefix}_pt",
        genjet_eta=f"{genjet_prefix}_eta",
        filename=f"{jet_type}_jet_pt_ratio_raw.pdf",
        save_figure=True,
    )
    jet_response_plot(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        response="response",
        jet_pt=f"{jet_prefix}_pt_corr",
        jet_label=jet_label_inclusive,
        genjet_pt=f"{genjet_prefix}_pt",
        genjet_eta=f"{genjet_prefix}_eta",
        filename=f"{jet_type}_jet_pt_ratio_corr.pdf",
        save_figure=True,
    )

    med_pf_vs_pt, iqr_pf_vs_pt, mean_pf_vs_pt, sigma_pf_vs_pt, med_mlpf_vs_pt, iqr_mlpf_vs_pt, mean_mlpf_vs_pt, sigma_mlpf_vs_pt = (
        get_response_in_bins(
            resp_pf,
            resp_mlpf,
            data_pf,
            data_mlpf,
            variable_bins=pt_bins_for_response,
            variable_name=f"{genjet_prefix}_pt",
            jet_prefix=jet_prefix,
            genjet_prefix=genjet_prefix,
            jet_label=jet_label_inclusive,
        )
    )
    (
        med_pf_vs_pt_raw,
        iqr_pf_vs_pt_raw,
        mean_pf_vs_pt_raw,
        sigma_pf_vs_pt_raw,
        med_mlpf_vs_pt_raw,
        iqr_mlpf_vs_pt_raw,
        mean_mlpf_vs_pt_raw,
        sigma_mlpf_vs_pt_raw,
    ) = get_response_in_bins(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        variable_bins=pt_bins_for_response,
        variable_name=f"{genjet_prefix}_pt",
        response_type="response_raw",
        jet_prefix=jet_prefix,
        genjet_prefix=genjet_prefix,
        jet_label=jet_label_inclusive,
    )

    med_pf_vs_eta, iqr_pf_vs_eta, mean_pf_vs_eta, sigma_pf_vs_eta, med_mlpf_vs_eta, iqr_mlpf_vs_eta, mean_mlpf_vs_eta, sigma_mlpf_vs_eta = (
        get_response_in_bins(
            resp_pf,
            resp_mlpf,
            data_pf,
            data_mlpf,
            variable_bins=eta_bins_for_response,
            variable_name=f"{genjet_prefix}_eta",
            jet_prefix=jet_prefix,
            genjet_prefix=genjet_prefix,
            jet_label=jet_label_inclusive,
        )
    )
    (
        med_pf_vs_eta_raw,
        iqr_pf_vs_eta_raw,
        mean_pf_vs_eta_raw,
        sigma_pf_vs_eta_raw,
        med_mlpf_vs_eta_raw,
        iqr_mlpf_vs_eta_raw,
        mean_mlpf_vs_eta_raw,
        sigma_mlpf_vs_eta_raw,
    ) = get_response_in_bins(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        variable_bins=eta_bins_for_response,
        variable_name=f"{genjet_prefix}_eta",
        response_type="response_raw",
        jet_prefix=jet_prefix,
        genjet_prefix=genjet_prefix,
        jet_label=jet_label_inclusive,
    )

    # Plot scale vs pt
    fig, ax = plt.subplots()
    ax.plot(midpoints(pt_bins_for_response), mean_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), mean_mlpf_vs_pt, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(pt_bins_for_response), mean_pf_vs_pt_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(pt_bins_for_response), mean_mlpf_vs_pt_raw, label=f"{mlpf_label} raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("GenJet $p_T$ (GeV)")
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.5)
    plt.axhline(1.0, color="black", ls="--")
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label_inclusive,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )
    fig.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_pt.pdf"))
    plt.close(fig)

    # Plot resolution vs pt
    fig, ax = plt.subplots()
    ax.plot(midpoints(pt_bins_for_response), sigma_pf_vs_pt / mean_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(
        midpoints(pt_bins_for_response),
        sigma_mlpf_vs_pt / mean_mlpf_vs_pt,
        label=f"{mlpf_label}",
        color=mlpf_color,
        linestyle=mlpf_linestyle,
        lw=3,
    )
    # ax.plot(
    #     midpoints(pt_bins_for_response), sigma_pf_vs_pt_raw / mean_pf_vs_pt_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5
    # )
    # ax.plot(
    #     midpoints(pt_bins_for_response),
    #     sigma_mlpf_vs_pt_raw / mean_mlpf_vs_pt_raw,
    #     label=f"{mlpf_label} raw",
    #     color=mlpf_color,
    #     linestyle=mlpf_linestyle,
    #     lw=0.5,
    # )
    ax.set_xlabel("$p_{T,ptcl}$ (GeV)")
    ax.set_ylabel("Response resolution")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label_inclusive,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )
    fig.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_pt.pdf"))
    plt.close(fig)

    # Plot scale vs eta
    fig, ax = plt.subplots()
    ax.plot(midpoints(eta_bins_for_response), mean_pf_vs_eta, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), mean_mlpf_vs_eta, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.plot(midpoints(eta_bins_for_response), mean_pf_vs_eta_raw, label="PF-PUPPI raw", color=pf_color, linestyle=pf_linestyle, lw=0.5)
    ax.plot(midpoints(eta_bins_for_response), mean_mlpf_vs_eta_raw, label=f"{mlpf_label} raw", color=mlpf_color, linestyle=mlpf_linestyle, lw=0.5)
    ax.set_xlabel("$η_{ptcl}$")
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_ylim(0.5, 1.5)
    plt.axhline(1.0, color="black", ls="--")
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label_inclusive,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )
    fig.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_eta.pdf"))
    plt.close(fig)

    # Plot resolution vs eta
    fig, ax = plt.subplots()
    ax.plot(midpoints(eta_bins_for_response), sigma_pf_vs_eta / mean_pf_vs_eta, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(
        midpoints(eta_bins_for_response),
        sigma_mlpf_vs_eta / mean_mlpf_vs_eta,
        label=f"{mlpf_label}",
        color=mlpf_color,
        linestyle=mlpf_linestyle,
        lw=3,
    )
    # ax.plot(
    #     midpoints(eta_bins_for_response),
    #     sigma_pf_vs_eta_raw / mean_pf_vs_eta_raw,
    #     label="PF-PUPPI raw",
    #     color=pf_color,
    #     linestyle=pf_linestyle,
    #     lw=0.5,
    # )
    # ax.plot(
    #     midpoints(eta_bins_for_response),
    #     sigma_mlpf_vs_eta_raw / mean_mlpf_vs_eta_raw,
    #     label=f"{mlpf_label} raw",
    #     color=mlpf_color,
    #     linestyle=mlpf_linestyle,
    #     lw=0.5,
    # )
    ax.set_xlabel("$\eta_{ptcl}$")
    ax.set_ylabel("Response resolution")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_ylim(0.0, 1.0)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label_inclusive,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )
    fig.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_eta.pdf"))
    plt.close(fig)

    # Plot PU-dependent jet response
    if "PU" in sample_name:

        def plot_response_vs_pu(resp_pf, resp_mlpf, data_pf, data_mlpf):
            stats_pu_pf = []
            stats_pu_mlpf = []

            # Define PU bins from notebook
            pu_bins = [(55, 60), (60, 65), (65, 70), (70, 75)]
            if not pu_bins:
                return

            for pt_range in pt_bins_for_pu:
                pt_min, pt_max = pt_range
                row_stats_pf = []
                row_stats_mlpf = []
                for pu_min, pu_max in pu_bins:
                    filename = f"{jet_type}_jet_pt_ratio_corr_pt{pt_min}to{pt_max}_pu{pu_min}to{pu_max}.pdf"

                    s_pf, s_mlpf = jet_response_plot(
                        resp_pf,
                        resp_mlpf,
                        data_pf,
                        data_mlpf,
                        response="response",
                        genjet_min_pt=pt_min,
                        genjet_max_pt=pt_max,
                        jet_label=jet_label_inclusive,
                        additional_label=f", {pt_min}<$p_{{T,ptcl}}$<{pt_max}, {pu_min}≤$N_{{PV}}$<{pu_max}",
                        jet_pt=f"{jet_prefix}_pt_corr",
                        genjet_pt=f"{genjet_prefix}_pt",
                        genjet_eta=f"{genjet_prefix}_eta",
                        additional_cut=lambda data: (data["Pileup_nTrueInt"] >= pu_min) & (data["Pileup_nTrueInt"] < pu_max),
                        filename=filename,
                    )
                    row_stats_pf.append(s_pf)
                    row_stats_mlpf.append(s_mlpf)
                stats_pu_pf.append(row_stats_pf)
                stats_pu_mlpf.append(row_stats_mlpf)

            stats_pu_pf = np.array(stats_pu_pf)
            stats_pu_mlpf = np.array(stats_pu_mlpf)

            with np.errstate(divide="ignore", invalid="ignore"):
                # resolution = sigma / median
                resolution_pf = np.nan_to_num(stats_pu_pf[:, :, 3] / stats_pu_pf[:, :, 0])
                resolution_mlpf = np.nan_to_num(stats_pu_mlpf[:, :, 3] / stats_pu_mlpf[:, :, 0])

            markers = ["o", "v", "^", "x", "s"]
            fig, ax = plt.subplots()

            pu_bin_centers = [b[0] + (b[1] - b[0]) / 2 for b in pu_bins]

            for i, pt_range in enumerate(pt_bins_for_pu):
                label = f"{pt_range[0]}-{pt_range[1]} GeV"
                ax.plot(
                    pu_bin_centers,
                    resolution_pf[i],
                    label=label,
                    marker=markers[i % len(markers)],
                    color=pf_color,
                    linestyle=pf_linestyle,
                )
                ax.plot(
                    pu_bin_centers,
                    resolution_mlpf[i],
                    marker=markers[i % len(markers)],
                    color=mlpf_color,
                    linestyle=mlpf_linestyle,
                )

            handles, labels = ax.get_legend_handles_labels()

            pt_handles = []
            for i, pt_range in enumerate(pt_bins_for_pu):
                label = f"{pt_range[0]}-{pt_range[1]}"
                if pt_range[1] > 1000:
                    label = f">{pt_range[0]}"
                pt_handles.append(Line2D([0], [0], color="black", marker=markers[i % len(markers)], linestyle="None", label=label))

            leg1 = ax.legend(handles=pt_handles, title=r"$p_{T,ptcl}$ (GeV)", loc=(0.6, 0.43))
            ax.add_artist(leg1)

            algo_handles = [
                Line2D([0], [0], color=pf_color, lw=2, label="PF-PUPPI", ls=pf_linestyle),
                Line2D([0], [0], color=mlpf_color, lw=2, label=mlpf_label, ls=mlpf_linestyle),
            ]
            ax.legend(handles=algo_handles, title="Algorithm", loc=(0.3, 0.55))

            ax.set_xlabel("True $N_{PV}$")
            ax.set_ylabel("Response resolution (σ/median)")
            ax.set_ylim(0, 1.5)
            mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
            ax.text(
                sample_label_coords[0],
                sample_label_coords[1],
                plot_sample_name,
                transform=ax.transAxes,
                fontsize=sample_label_fontsize,
                ha="left",
                va="top",
            )
            ax.text(
                jet_label_coords_single[0],
                jet_label_coords_single[1],
                jet_label_inclusive,
                transform=ax.transAxes,
                fontsize=addtext_fontsize,
                ha="left",
                va="top",
            )

            fig.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_npv.pdf"))
            plt.close(fig)

        plot_response_vs_pu(resp_pf, resp_mlpf, data_pf, data_mlpf)

    print(f"Generated plots in {output_dir}")


if __name__ == "__main__":
    make_plots()
