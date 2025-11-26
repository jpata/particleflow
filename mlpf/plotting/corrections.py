import click
import awkward
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from mlpf.plotting.utils import compute_response
import matplotlib.pyplot as plt
import mplhep
from mlpf.plotting.plot_utils import sample_label, cms_label, med_iqr, sample_name_to_process, midpoints
from mlpf.plotting.utils import Gauss, to_bh, compute_scale_res

# settings from notebook
mplhep.style.use("CMS")
np.seterr(divide="ignore")
legend_fontsize = 20
sample_label_fontsize = 40
addtext_fontsize = 25
jet_label_coords_single = 0.02, 0.86
sample_label_coords = 0.02, 0.96
jet_label_corr = "Corr. jet "
jet_label_raw = "Raw jet "
pf_linestyle = "-."
mlpf_linestyle = "-"


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
    additional_cut=lambda data: data["Jet_pt"] > 0,  # dummy
    mlpf_label="MLPF",
    physics_process="unknown",
):

    plt.figure()
    ax = plt.axes()
    b = np.linspace(0, 2, 101)

    cms_label(ax)
    sample_label(ax, physics_process, x=sample_label_coords[0], y=sample_label_coords[1], fontsize=sample_label_fontsize)
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label + additional_label,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
    )

    # the additional cut is not available in this context, so we disable it
    # add_cut_pf = additional_cut(data_baseline)
    # add_cut_mlpf = additional_cut(data_mlpf)
    add_cut_pf = awkward.ones_like(resp_pf[response], dtype=bool)
    add_cut_mlpf = awkward.ones_like(resp_mlpf[response], dtype=bool)

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

    default_cycler = plt.rcParams["axes.prop_cycle"]
    pf_color = list(default_cycler)[1]["color"]
    mlpf_color = list(default_cycler)[2]["color"]

    plt.plot([], [])
    x0 = mplhep.histplot(
        h0,
        histtype="step",
        lw=2,
        label="PF\nmean: {:.2f} std: {:.2f}\nmed: {:.2f} IQR: {:.2f}".format(mean_pf, std_pf, med_pf, iqr_pf),
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
    ax.legend(handles, labels, loc=(0.50, 0.50), fontsize=legend_fontsize)
    jl = jet_label_corr if jet_pt.endswith("_corr") else jet_label_raw
    plt.xlabel(jl + "pT response")
    plt.ylabel("Count")

    ax.set_ylim(0, 1.5 * ax.get_ylim()[1])

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_x(-0.01)
    ax.yaxis.get_offset_text().set_ha("right")
    return (
        (med_pf, np.std(jet_response_pf), len(jet_response_pf), std_pf),
        (med_mlpf, np.std(jet_response_mlpf), len(jet_response_mlpf), std_mlpf),
    )


def save_jet_response_plots(
    eta_reco_bins,
    pt_gen_bins,
    resp_pf,
    resp_mlpf,
    data_pf,
    data_mlpf,
    outpath,
    jet_type="ak4",
    sample="QCD",
    response_type="response_raw",
    suffix="raw",
):
    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]
    process_name = sample_name_to_process(sample)

    for ibin_eta in range(len(eta_reco_bins) - 1):
        for ibin_pt in range(len(pt_gen_bins) - 1):
            jet_response_plot(
                resp_pf,
                resp_mlpf,
                data_pf,
                data_mlpf,
                response=response_type,
                jet_pt=f"{jet_prefix}_pt_{suffix}",
                genjet_eta=jet_prefix + "_eta",  # here we intentionally cut on recojet instead of genjet
                genjet_pt=genjet_prefix + "_pt",
                genjet_min_eta=eta_reco_bins[ibin_eta],
                genjet_max_eta=eta_reco_bins[ibin_eta + 1],
                genjet_min_pt=pt_gen_bins[ibin_pt],
                genjet_max_pt=pt_gen_bins[ibin_pt + 1],
                jet_label=f"{jet_type} jets",
                physics_process=process_name,
                additional_label=", %.0f<$p_{{T,ptcl}}$<%.0f, %.2f<$\eta_{{reco}}$<%.2f"
                % (pt_gen_bins[ibin_pt], pt_gen_bins[ibin_pt + 1], eta_reco_bins[ibin_eta], eta_reco_bins[ibin_eta + 1]),
            )
            plt.savefig(f"{outpath}/{jet_type}_jet_pt_ratio_{suffix}_etabin{ibin_eta}_ptbin{ibin_pt}.pdf")
            plt.close()


def save_jet_correction_heatmaps(corr_map_pf, corr_map_mlpf, eta_reco_bins, pt_gen_bins, outpath, jet_type, sample_name):
    """Saves a 2D heatmap of jet corrections."""
    import matplotlib.colors

    for cmap, name in [(corr_map_pf, "pf"), (corr_map_mlpf, "mlpf")]:
        plt.figure()
        ax = plt.axes()
        plt.imshow(cmap, norm=matplotlib.colors.Normalize(vmin=1.0, vmax=2.5))
        plt.colorbar(shrink=0.5, label="1/median")
        plt.xticks(range(len(pt_gen_bins) - 1), ["{:.0f}".format(x) for x in midpoints(np.array(pt_gen_bins))])
        plt.yticks(range(len(eta_reco_bins) - 1), ["{:.2f}".format(x) for x in midpoints(np.array(eta_reco_bins))])
        plt.xlabel("$p_{T,ptcl}$ (GeV)")
        plt.ylabel("$\eta_{reco}$")
        ax.text(
            jet_label_coords_single[0],
            jet_label_coords_single[1],
            f"{name.upper()} {jet_type} jets",
            transform=ax.transAxes,
            fontsize=addtext_fontsize,
        )
        plt.savefig(f"{outpath}/{jet_type}_correction_map_{name}.pdf")
        plt.close()


def fill_nan(reciprocal):
    mask = np.isnan(reciprocal) | np.isinf(reciprocal)
    if not np.any(mask):
        return reciprocal

    valid_coords = np.where(~mask)
    if len(valid_coords[0]) == 0:  # all are nan
        return np.ones_like(reciprocal)

    nan_coords = np.where(mask)
    valid_values = reciprocal[~mask]

    imputed_values = griddata(np.vstack(valid_coords).T, valid_values, np.vstack(nan_coords).T, method="nearest")
    reciprocal = reciprocal.copy()
    reciprocal[nan_coords] = imputed_values
    return reciprocal


def calculate_correction_map(resp_data, eta_bins, pt_bins, jet_type="ak4", response_type="response_raw"):
    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    resp_stats_med = np.zeros((len(eta_bins) - 1, len(pt_bins) - 1))

    for ibin_eta in range(len(eta_bins) - 1):
        for ibin_pt in range(len(pt_bins) - 1):

            mask = (
                (resp_data[genjet_prefix + "_pt"] >= pt_bins[ibin_pt])
                & (resp_data[genjet_prefix + "_pt"] < pt_bins[ibin_pt + 1])
                & (resp_data[jet_prefix + "_eta"] >= eta_bins[ibin_eta])
                & (resp_data[jet_prefix + "_eta"] < eta_bins[ibin_eta + 1])
            )

            response_raw = awkward.flatten(resp_data[response_type][mask])
            median, _ = med_iqr(response_raw)
            print(f"Response eta_bin={ibin_eta} pt_bin={ibin_pt} resp={len(response_raw)} median={median}")
            resp_stats_med[ibin_eta, ibin_pt] = median

    reciprocal_med = 1.0 / resp_stats_med
    reciprocal_med = fill_nan(reciprocal_med)

    return reciprocal_med


@click.command()
@click.option("--input-pf-parquet", required=True, type=str)
@click.option("--input-mlpf-parquet", required=True, type=str)
@click.option("--corrections-file", required=True, type=str)
@click.option("--jet-type", default="ak4", type=click.Choice(["ak4", "ak8"]))
@click.option("--output-dir", default=None, type=str, help="Directory to save validation plots.")
@click.option("--sample-name", default="QCD", type=str, help="Sample name for plots.")
def make_corrections(input_pf_parquet, input_mlpf_parquet, corrections_file, jet_type, output_dir, sample_name):
    """Generates jet energy correction maps."""

    data_pf = awkward.from_parquet(input_pf_parquet)
    print("loaded {}: {}".format(input_pf_parquet, len(data_pf["Jet_pt"])))
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)
    print("loaded {}: {}".format(input_mlpf_parquet, len(data_mlpf["Jet_pt"])))

    if jet_type == "ak4":
        jet_coll = "Jet"
        genjet_coll = "GenJet"
        deltar_cut = 0.2
        eta_reco_bins = [-6, -5.191, -4.0, -2.964, -1.392, 0, 1.392, 2.964, 4.0, 5.191, 6]
        pt_gen_bins = [10, 20, 30, 40, 50, 80, 120, 200, 500, 3000]
    else:  # ak8
        jet_coll = "FatJet"
        genjet_coll = "GenJetAK8"
        deltar_cut = 0.4
        eta_reco_bins = [-6, -5.191, -4.0, -2.964, -1.392, 0, 1.392, 2.964, 4.0, 5.191, 6]
        pt_gen_bins = [200, 300, 400, 500, 3000]

    print("computing PF response")
    resp_pf = compute_response(data_pf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)
    print("computing MLPF response")
    resp_mlpf = compute_response(data_mlpf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)

    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        save_jet_response_plots(eta_reco_bins, pt_gen_bins, resp_pf, resp_mlpf, data_pf, data_mlpf, output_dir, jet_type, sample_name)

    corr_map_pf = calculate_correction_map(resp_pf, eta_reco_bins, pt_gen_bins, jet_type=jet_type)
    corr_map_mlpf = calculate_correction_map(resp_mlpf, eta_reco_bins, pt_gen_bins, jet_type=jet_type)
    if output_dir:
        save_jet_correction_heatmaps(corr_map_pf, corr_map_mlpf, eta_reco_bins, pt_gen_bins, output_dir, jet_type, sample_name)

        interp_pf = RegularGridInterpolator(
            (midpoints(np.array(eta_reco_bins)), midpoints(np.array(pt_gen_bins))),
            corr_map_pf,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        interp_mlpf = RegularGridInterpolator(
            (midpoints(np.array(eta_reco_bins)), midpoints(np.array(pt_gen_bins))),
            corr_map_mlpf,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        corr_pf_interp = interp_pf(
            np.stack(
                [
                    awkward.to_numpy(awkward.flatten(data_pf[jet_coll + "_eta"])),
                    awkward.to_numpy(awkward.flatten(data_pf[jet_coll + "_pt_raw"])),
                ]
            ).T
        )
        corr_pf_interp = awkward.unflatten(corr_pf_interp, awkward.count(data_pf[jet_coll + "_eta"], axis=1))
        data_pf[jet_coll + "_pt_corr"] = data_pf[jet_coll + "_pt_raw"] * corr_pf_interp

        corr_mlpf_interp = interp_mlpf(
            np.stack(
                [
                    awkward.to_numpy(awkward.flatten(data_mlpf[jet_coll + "_eta"])),
                    awkward.to_numpy(awkward.flatten(data_mlpf[jet_coll + "_pt_raw"])),
                ]
            ).T
        )
        corr_mlpf_interp = awkward.unflatten(corr_mlpf_interp, awkward.count(data_mlpf[jet_coll + "_eta"], axis=1))
        data_mlpf[jet_coll + "_pt_corr"] = data_mlpf[jet_coll + "_pt_raw"] * corr_mlpf_interp

        print("computing corrected PF response")
        resp_pf_corr = compute_response(data_pf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)
        print("computing corrected MLPF response")
        resp_mlpf_corr = compute_response(data_mlpf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)

        # get the per-bin medians after corrections
        corr_map_pf2 = calculate_correction_map(resp_pf_corr, eta_reco_bins, pt_gen_bins, jet_type=jet_type, response_type="response")
        corr_map_mlpf2 = calculate_correction_map(resp_mlpf_corr, eta_reco_bins, pt_gen_bins, jet_type=jet_type, response_type="response")
        save_jet_correction_heatmaps(corr_map_pf2, corr_map_mlpf2, eta_reco_bins, pt_gen_bins, output_dir, jet_type + "corr", sample_name)

        save_jet_response_plots(
            eta_reco_bins,
            pt_gen_bins,
            resp_pf_corr,
            resp_mlpf_corr,
            data_pf,
            data_mlpf,
            output_dir,
            jet_type,
            sample_name,
            response_type="response",
            suffix="corr",
        )

    np.savez(corrections_file, corr_map_pf=corr_map_pf, corr_map_mlpf=corr_map_mlpf, eta_bins=eta_reco_bins, pt_bins=pt_gen_bins)

    print(f"Saved correction maps to {corrections_file}")


if __name__ == "__main__":
    make_corrections()
