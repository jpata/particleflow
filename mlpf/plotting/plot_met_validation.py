import click
import os
import awkward
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
import boost_histogram as bh
from pathlib import Path
from matplotlib.lines import Line2D
from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, med_iqr, sample_name_to_process
from mlpf.plotting.plot_utils import labels as XLABELS


def midpoints(x):
    return (x[1:] + x[:-1]) / 2


def to_bh(data, bins):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    return h1


def plot_met_distribution(
    data_pf,
    data_mlpf,
    bins,
    xlabel,
    filename,
    logy,
    sample_name,
    mlpf_label,
    legend_loc,
    legend_fontsize,
    sample_label_coords,
    sample_label_fontsize,
    ratio_ylim,
):
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Gen MET
    h0 = to_bh(data_pf["GenMET_pt"], bins)

    # PF MET (Puppi)
    h1 = to_bh(data_pf["PuppiMET_pt"], bins)

    # MLPF MET
    h2 = to_bh(data_mlpf["PuppiMET_pt"], bins)

    plt.sca(a0)
    x0 = mplhep.histplot(h0, histtype="step", lw=2, label="Gen.", binwnorm=1.0, ls="--")
    x1 = mplhep.histplot(h1, histtype="step", lw=2, label="PF-PUPPI", binwnorm=1.0, ls="-.")
    x2 = mplhep.histplot(h2, histtype="step", lw=2, label=mlpf_label, binwnorm=1.0, ls="-")

    if logy:
        plt.yscale("log")
        a0.set_ylim(bottom=1, top=a0.get_ylim()[1] * 1000)

    mplhep.cms.label("", data=False, com=13.6, year="Run 3", ax=a0)
    a0.text(
        sample_label_coords[0],
        sample_label_coords[1],
        sample_name,
        transform=a0.transAxes,
        fontsize=sample_label_fontsize,
        ha="left",
        va="top",
    )

    handles, labels = a0.get_legend_handles_labels()
    handles = [x0[0].stairs, x1[0].stairs, x2[0].stairs]
    a0.legend(handles, labels, loc=legend_loc, fontsize=legend_fontsize)
    plt.ylabel("Count")

    plt.sca(a1)
    plt.plot([], [])
    mplhep.histplot(h1 / h1, histtype="step", lw=2, ls="-.")
    mplhep.histplot(h2 / h1, histtype="step", lw=2, ls="-")
    plt.ylim(0.0, ratio_ylim)
    plt.ylabel("MLPF / PF")
    plt.xlabel(xlabel)

    if "pT" in xlabel or "MET" in xlabel:
        plt.xscale("log")

    plt.xlim(min(bins), max(bins))
    plt.savefig(filename)
    plt.close()


def met_response_plot(
    resp_pf,
    resp_mlpf,
    data_pf,
    data_mlpf,
    genmet_min_pt,
    genmet_max_pt,
    additional_label,
    additional_cut,
    filename,
    sample_name,
    mlpf_label,
    legend_loc,
    legend_fontsize,
    sample_label_coords,
    sample_label_fontsize,
    addtext_fontsize,
    jet_label_coords_single,
    pf_color,
    mlpf_color,
    pf_linestyle,
    mlpf_linestyle,
):
    plt.figure()
    ax = plt.axes()
    b = np.linspace(0, 5, 101)

    mplhep.cms.label("", data=False, com=13.6, year="Run 3", ax=ax)
    ax.text(
        sample_label_coords[0],
        sample_label_coords[1],
        sample_name,
        transform=ax.transAxes,
        fontsize=sample_label_fontsize,
        ha="left",
        va="top",
    )
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        "PUPPI MET" + additional_label,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )

    add_cut_pf = additional_cut(data_pf)
    add_cut_mlpf = additional_cut(data_mlpf)

    mask_pf = (resp_pf["GenMET_pt"] >= genmet_min_pt) & (resp_pf["GenMET_pt"] < genmet_max_pt) & add_cut_pf
    met_response_pf = resp_pf["response"][mask_pf]

    mask_mlpf = (resp_mlpf["GenMET_pt"] >= genmet_min_pt) & (resp_mlpf["GenMET_pt"] < genmet_max_pt) & add_cut_mlpf
    met_response_mlpf = resp_mlpf["response"][mask_mlpf]

    h0 = to_bh(met_response_pf, b)
    h1 = to_bh(met_response_mlpf, b)

    med_pf, iqr_pf = med_iqr(met_response_pf)
    med_mlpf, iqr_mlpf = med_iqr(met_response_mlpf)

    x0 = mplhep.histplot(
        h0,
        histtype="step",
        lw=2,
        label="PF-PUPPI\nmed: {:.2f} IQR: {:.2f}".format(med_pf, iqr_pf),
        ls=pf_linestyle,
        color=pf_color,
    )
    x1 = mplhep.histplot(
        h1,
        histtype="step",
        lw=2,
        label="{}\nmed: {:.2f} IQR: {:.2f}".format(mlpf_label, med_mlpf, iqr_mlpf),
        ls=mlpf_linestyle,
        color=mlpf_color,
    )

    handles, labels = ax.get_legend_handles_labels()
    handles = [x0[0].stairs, x1[0].stairs]
    ax.legend(handles, labels, loc=legend_loc, fontsize=legend_fontsize)

    plt.xlabel("MET response")
    plt.ylabel("Count")

    ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_x(-0.01)
    ax.yaxis.get_offset_text().set_ha("right")

    # plt.savefig(filename)
    plt.close()
    return ((med_pf, iqr_pf), (med_mlpf, iqr_mlpf))


def get_met_response_in_bins(
    resp_pf,
    resp_mlpf,
    data_pf,
    data_mlpf,
    variable_bins,
    output_dir,
    sample_name,
    mlpf_label,
    **kwargs,
):
    med_vals_pf, iqr_vals_pf = [], []
    med_vals_mlpf, iqr_vals_mlpf = [], []

    for ibin in range(len(variable_bins) - 1):
        min_bin_val, max_bin_val = variable_bins[ibin], variable_bins[ibin + 1]

        stats_pf, stats_mlpf = met_response_plot(
            resp_pf,
            resp_mlpf,
            data_pf,
            data_mlpf,
            genmet_min_pt=min_bin_val,
            genmet_max_pt=max_bin_val,
            additional_label=f", {min_bin_val:.0f} < GenMET pT < {max_bin_val:.0f} GeV",
            additional_cut=lambda data: data["Pileup_nTrueInt"] >= 0,
            filename=os.path.join(output_dir, f"met_response_bin_genmet_pt_{ibin}.pdf"),
            sample_name=sample_name,
            mlpf_label=mlpf_label,
            **kwargs,
        )
        med_vals_pf.append(stats_pf[0])
        iqr_vals_pf.append(stats_pf[1])
        med_vals_mlpf.append(stats_mlpf[0])
        iqr_vals_mlpf.append(stats_mlpf[1])

    return (
        np.array(med_vals_pf),
        np.array(iqr_vals_pf),
        np.array(med_vals_mlpf),
        np.array(iqr_vals_mlpf),
    )


def plot_met_response_vs_pu(resp_pf, resp_mlpf, data_pf, data_mlpf, output_dir, sample_name, mlpf_label, **kwargs):
    stats_pu_pf = []
    stats_pu_mlpf = []

    pu_bins = [(55, 60), (60, 65), (65, 70), (70, 75)]
    met_bins_for_pu = [(0, 30), (30, 60), (60, 100), (100, 200), (200, 5000)]

    for pt_range in met_bins_for_pu:
        pt_min, pt_max = pt_range
        row_stats_pf = []
        row_stats_mlpf = []
        for pu_min, pu_max in pu_bins:
            filename = os.path.join(output_dir, f"met_response_pt{pt_min}to{pt_max}_pu{pu_min}to{pu_max}.pdf")

            s_pf, s_mlpf = met_response_plot(
                resp_pf,
                resp_mlpf,
                data_pf,
                data_mlpf,
                genmet_min_pt=pt_min,
                genmet_max_pt=pt_max,
                additional_label=f", {pt_min}<$p_{{T,gen}}$<{pt_max}, {pu_min}≤$N_{{PV}}$<{pu_max}",
                additional_cut=lambda data: (data["Pileup_nTrueInt"] >= pu_min) & (data["Pileup_nTrueInt"] < pu_max),
                filename=filename,
                sample_name=sample_name,
                mlpf_label=mlpf_label,
                **kwargs,
            )
            row_stats_pf.append(s_pf)
            row_stats_mlpf.append(s_mlpf)
        stats_pu_pf.append(row_stats_pf)
        stats_pu_mlpf.append(row_stats_mlpf)

    stats_pu_pf = np.array(stats_pu_pf)
    stats_pu_mlpf = np.array(stats_pu_mlpf)

    with np.errstate(divide="ignore", invalid="ignore"):
        resolution_pf = np.nan_to_num(stats_pu_pf[:, :, 1] / stats_pu_pf[:, :, 0])
        resolution_mlpf = np.nan_to_num(stats_pu_mlpf[:, :, 1] / stats_pu_mlpf[:, :, 0])

    markers = ["o", "v", "^", "x", "s"]
    fig, ax = plt.subplots()

    pu_bin_centers = [b[0] + (b[1] - b[0]) / 2 for b in pu_bins]

    for i, pt_range in enumerate(met_bins_for_pu):
        label = f"{pt_range[0]}-{pt_range[1]} GeV"
        ax.plot(
            pu_bin_centers,
            resolution_pf[i],
            label=label,
            marker=markers[i % len(markers)],
            color=kwargs["pf_color"],
            linestyle=kwargs["pf_linestyle"],
        )
        ax.plot(
            pu_bin_centers,
            resolution_mlpf[i],
            marker=markers[i % len(markers)],
            color=kwargs["mlpf_color"],
            linestyle=kwargs["mlpf_linestyle"],
        )

    pt_handles = []
    for i, pt_range in enumerate(met_bins_for_pu):
        label = f"{pt_range[0]}-{pt_range[1]}"
        if pt_range[1] > 1000:
            label = f">{pt_range[0]}"
        pt_handles.append(Line2D([0], [0], color="black", marker=markers[i % len(markers)], linestyle="None", label=label))

    leg1 = ax.legend(handles=pt_handles, title=XLABELS["gen_met"], loc=(0.6, 0.43))
    ax.add_artist(leg1)

    algo_handles = [
        Line2D([0], [0], color=kwargs["pf_color"], lw=2, label="PF-PUPPI", ls=kwargs["pf_linestyle"]),
        Line2D([0], [0], color=kwargs["mlpf_color"], lw=2, label=mlpf_label, ls=kwargs["mlpf_linestyle"]),
    ]
    ax.legend(handles=algo_handles, title="Algorithm", loc=(0.3, 0.55))

    ax.set_xlabel("True $N_{PV}$")
    ax.set_ylabel("MET response resolution (IQR/med.)")
    ax.set_ylim(0, 1.5)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        kwargs["sample_label_coords"][0],
        kwargs["sample_label_coords"][1],
        sample_name,
        transform=ax.transAxes,
        fontsize=kwargs["sample_label_fontsize"],
        ha="left",
        va="top",
    )
    ax.text(
        kwargs["jet_label_coords_single"][0],
        kwargs["jet_label_coords_single"][1],
        "PUPPI MET",
        transform=ax.transAxes,
        fontsize=kwargs["addtext_fontsize"],
        ha="left",
        va="top",
    )

    fig.savefig(os.path.join(output_dir, "met_resolution_vs_npv.pdf"))
    plt.close(fig)


@click.command()
@click.option("--input-pf-parquet", required=True, type=str)
@click.option("--input-mlpf-parquet", required=True, type=str)
@click.option("--output-dir", required=True, type=str)
@click.option("--sample-name", required=True, type=str, help="Sample name (e.g., QCD_PU_13p6)")
def make_plots(input_pf_parquet, input_mlpf_parquet, output_dir, sample_name):
    """Generates MET validation plots."""

    output_dir = str(Path(output_dir, sample_name, "met"))
    os.makedirs(output_dir, exist_ok=True)
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    process_name = sample_name_to_process(sample_name)
    plot_sample_name = EVALUATION_DATASET_NAMES.get(process_name, sample_name)

    # plotting style variables
    legend_fontsize = 30
    legend_loc = (0.5, 0.45)
    legend_loc_scalereso = (0.50, 0.65)
    legend_loc_met_response = (0.3, 0.45)
    sample_label_fontsize = 30
    addtext_fontsize = 25
    jet_label_coords_single = 0.02, 0.88
    sample_label_coords = 0.02, 0.96
    default_cycler = plt.rcParams["axes.prop_cycle"]
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

    if sample_name.startswith("QCD_"):
        met_bins = varbins(np.linspace(0, 150, 21), np.linspace(150, 500, 5))
        met_bins_for_response = np.array([5, 20, 40, 60, 80, 100, 150, 200, 300, 500])
        ratio_ylim = 2
    elif sample_name.startswith("TTbar_"):
        met_bins = varbins(np.linspace(0, 150, 21), np.linspace(150, 250, 5))
        met_bins_for_response = np.array([5, 20, 40, 60, 80, 100, 150, 250])
        ratio_ylim = 2
    elif sample_name.startswith("PhotonJet_"):
        met_bins = varbins(np.linspace(0, 200, 41))
        met_bins_for_response = np.array([5, 20, 40, 60, 80, 100, 150, 200])
        ratio_ylim = 2
    else:
        met_bins = np.linspace(0, 500, 51)
        met_bins_for_response = np.linspace(1, 500, 26)
        ratio_ylim = 2

    data_pf = awkward.from_parquet(input_pf_parquet)
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)

    plot_style_kwargs = {
        "legend_loc": legend_loc,
        "legend_fontsize": legend_fontsize,
        "sample_label_coords": sample_label_coords,
        "sample_label_fontsize": sample_label_fontsize,
        "addtext_fontsize": addtext_fontsize,
        "jet_label_coords_single": jet_label_coords_single,
        "pf_color": pf_color,
        "mlpf_color": mlpf_color,
        "pf_linestyle": pf_linestyle,
        "mlpf_linestyle": mlpf_linestyle,
    }

    # MET distribution
    plot_met_distribution(
        data_pf,
        data_mlpf,
        met_bins,
        XLABELS["met"],
        os.path.join(output_dir, "met_pt.pdf"),
        logy=True,
        sample_name=plot_sample_name,
        mlpf_label=mlpf_label,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        sample_label_coords=sample_label_coords,
        sample_label_fontsize=sample_label_fontsize,
        ratio_ylim=ratio_ylim,
    )

    # Define MET response, avoiding division by zero
    genmet_pt_pf = data_pf["GenMET_pt"]
    genmet_pt_mlpf = data_mlpf["GenMET_pt"]

    msk = genmet_pt_pf != 0
    resp_pf = {
        "response": np.divide(data_pf["PuppiMET_pt"][msk], genmet_pt_pf[msk]),
        "GenMET_pt": genmet_pt_pf,
    }
    msk = genmet_pt_mlpf != 0
    resp_mlpf = {
        "response": np.divide(data_mlpf["PuppiMET_pt"][msk], genmet_pt_mlpf[msk]),
        "GenMET_pt": genmet_pt_mlpf,
    }

    # overall MET response plot
    plot_style_kwargs_response = plot_style_kwargs.copy()
    plot_style_kwargs_response["legend_loc"] = legend_loc_met_response
    met_response_plot(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        genmet_min_pt=5,
        genmet_max_pt=10000,
        additional_label=", $\mathrm{MET}_{ptcl} > 5$",
        additional_cut=lambda data: data["Pileup_nTrueInt"] >= 0,
        filename=os.path.join(output_dir, "met_response.pdf"),
        sample_name=plot_sample_name,
        mlpf_label=mlpf_label,
        **plot_style_kwargs_response,
    )

    # get response in bins of GenMET pt
    (
        med_pf_vs_pt,
        iqr_pf_vs_pt,
        med_mlpf_vs_pt,
        iqr_mlpf_vs_pt,
    ) = get_met_response_in_bins(
        resp_pf,
        resp_mlpf,
        data_pf,
        data_mlpf,
        variable_bins=met_bins_for_response,
        output_dir=output_dir,
        sample_name=plot_sample_name,
        mlpf_label=mlpf_label,
        **plot_style_kwargs,
    )

    # Plot scale vs pt
    fig, ax = plt.subplots()
    ax.plot(midpoints(met_bins_for_response), med_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(met_bins_for_response), med_mlpf_vs_pt, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.set_xlabel(XLABELS["gen_met"])
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 4.0)
    plt.axhline(1.0, color="black", ls="--")
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0], jet_label_coords_single[1], "PUPPI MET", transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top"
    )
    fig.savefig(os.path.join(output_dir, "met_scale_vs_pt.pdf"))
    plt.close(fig)

    # Plot resolution vs pt
    fig, ax = plt.subplots()
    with np.errstate(divide="ignore", invalid="ignore"):
        res_pf = np.nan_to_num(iqr_pf_vs_pt / med_pf_vs_pt)
        res_mlpf = np.nan_to_num(iqr_mlpf_vs_pt / med_mlpf_vs_pt)
    ax.plot(midpoints(met_bins_for_response), res_pf, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle, lw=3)
    ax.plot(midpoints(met_bins_for_response), res_mlpf, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle, lw=3)
    ax.set_xlabel(XLABELS["gen_met"])
    ax.set_ylabel("Response resolution")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc_scalereso)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 2.0)
    mplhep.cms.label(ax=ax, data=False, com=13.6, year="Run 3")
    ax.text(
        sample_label_coords[0], sample_label_coords[1], plot_sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize, ha="left", va="top"
    )
    ax.text(
        jet_label_coords_single[0], jet_label_coords_single[1], "PUPPI MET", transform=ax.transAxes, fontsize=addtext_fontsize, ha="left", va="top"
    )
    fig.savefig(os.path.join(output_dir, "met_resolution_vs_pt.pdf"))
    plt.close(fig)

    # Plot PU-dependent MET response
    if "PU" in sample_name:
        plot_met_response_vs_pu(resp_pf, resp_mlpf, data_pf, data_mlpf, output_dir, plot_sample_name, mlpf_label, **plot_style_kwargs)

    print(f"Generated plots in {output_dir}")


if __name__ == "__main__":
    make_plots()
