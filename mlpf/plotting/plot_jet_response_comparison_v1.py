import click
import awkward
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
import boost_histogram as bh
from pathlib import Path

from mlpf.plotting.utils import compute_response
from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, sample_name_to_process

default_cycler = plt.rcParams["axes.prop_cycle"]
pf_color = list(default_cycler)[0]["color"]
mlpf_color = list(default_cycler)[1]["color"]
pf_linestyle = "-."
mlpf_linestyle = "-"
legend_loc_met = (0.45, 0.65)
legend_loc_jet_response = (0.05, 0.65)
pf_label = "PF-PUPPI"
mlpf_label = "MLPF-PUPPI"


def to_bh(data, bins):
    """Converts numpy array to boost_histogram object."""
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    return h1


def apply_dz(data):
    """Applies a cut on the z-coordinate of the primary vertex."""
    abs_dz = np.abs(data["GenVtx_z"] - data["PV_z"])
    mask_dz = abs_dz < 0.2
    return data[mask_dz]


def varbins(*args):
    """Helper function to define variable width bins."""
    newlist = []
    for arg in args[:-1]:
        newlist.append(arg[:-1])
    newlist.append(args[-1])
    return np.concatenate(newlist)


def plot_met_comparison(data_pf, data_mlpf, output_dir, sample_name, tev):
    """Plots the MET comparison between PF and MLPF"""
    plt.figure()
    ax = plt.axes()

    bins = np.linspace(0, 500, 51)

    sample_label_coords = 0.02, 0.96
    sample_label_fontsize = 30
    legend_fontsize = 30

    h_pf = to_bh(data_pf["PuppiMET_pt"], bins=bins)
    h_mlpf = to_bh(data_mlpf["PuppiMET_pt"], bins=bins)
    mplhep.histplot(h_pf, histtype="step", lw=2, density=True, color=pf_color, ls=pf_linestyle)
    mplhep.histplot(h_mlpf, histtype="step", lw=2, density=True, color=mlpf_color, ls=mlpf_linestyle)

    plt.xlabel("$p_{T,miss}$ (GeV)")
    plt.ylabel("Normalized count")
    plt.ylim(1e-3, 1e-1)
    plt.yscale("log")

    process_name = sample_name_to_process(sample_name)
    plot_sample_name = EVALUATION_DATASET_NAMES.get(process_name, sample_name)

    ax.text(
        sample_label_coords[0],
        sample_label_coords[1],
        plot_sample_name,
        transform=ax.transAxes,
        fontsize=sample_label_fontsize,
        ha="left",
        va="top",
    )

    mplhep.cms.label("", data=False, rlabel=f"Run 3 ({tev} TeV)")

    ax.set_ylim(bottom=1e-5)
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color=pf_color, ls=pf_linestyle, lw=2, label=pf_label),
        matplotlib.lines.Line2D([0], [0], color=mlpf_color, ls=mlpf_linestyle, lw=2, label=mlpf_label),
    ]
    ax.legend(handles=legend_elements, fontsize=legend_fontsize, loc=legend_loc_met)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"{sample_name}_met_dist_comp.pdf")
    plt.close()


def plot_jet_response_comparison(resp_pf, resp_mlpf, output_dir, sample_name, jet_type, jet_label, tev):
    """Plots the jet response comparison for PF and MLPF"""
    plt.figure()
    ax = plt.axes()
    b = np.linspace(0, 2, 101)

    sample_label_coords = 0.02, 0.96
    jet_label_coords_single = 0.02, 0.88
    sample_label_fontsize = 30
    addtext_fontsize = 25
    legend_fontsize = 30

    response_pf = awkward.flatten(resp_pf["response_raw"])
    response_mlpf = awkward.flatten(resp_mlpf["response_raw"])

    h_pf = to_bh(response_pf, bins=b)
    h_mlpf = to_bh(response_mlpf, bins=b)

    mplhep.histplot(h_pf, histtype="step", lw=2, density=True, color=pf_color, ls=pf_linestyle)
    mplhep.histplot(h_mlpf, histtype="step", lw=2, density=True, color=mlpf_color, ls=mlpf_linestyle)

    plt.xlabel("Raw jet $p_T / p_{T,ptcl}$ response")
    plt.ylabel("Normalized count")

    process_name = sample_name_to_process(sample_name)
    plot_sample_name = EVALUATION_DATASET_NAMES.get(process_name, sample_name)

    ax.text(
        sample_label_coords[0],
        sample_label_coords[1],
        plot_sample_name,
        transform=ax.transAxes,
        fontsize=sample_label_fontsize,
        ha="left",
        va="top",
    )

    mplhep.cms.label("", data=False, rlabel=f"Run 3 ({tev} TeV)")
    ax.text(
        jet_label_coords_single[0],
        jet_label_coords_single[1],
        jet_label,
        transform=ax.transAxes,
        fontsize=addtext_fontsize,
        ha="left",
        va="top",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color=pf_color, ls=pf_linestyle, lw=2, label=pf_label),
        matplotlib.lines.Line2D([0], [0], color=mlpf_color, ls=mlpf_linestyle, lw=2, label=mlpf_label),
    ]
    ax.legend(handles=legend_elements, fontsize=legend_fontsize, loc=legend_loc_jet_response)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"{sample_name}_{jet_type}_jet_pt_ratio_comp.pdf")
    plt.close()


@click.command()
@click.option("--input-pf-parquet", required=True, type=str, help="Input parquet file for PF sample.")
@click.option("--input-mlpf-parquet", required=True, type=str, help="Input parquet file for MLPF sample.")
@click.option("--output-dir", required=True, type=str, help="Output directory for plots.")
@click.option("--jet-type", default="ak4", type=click.Choice(["ak4", "ak8"]), help="Jet type to plot.")
@click.option("--sample-name", required=True, type=str, help="Sample name for plot labels (e.g., QCD_PU).")
@click.option("--tev", required=True, type=str, help="Center of mass energy")
def main(input_pf_parquet, input_mlpf_parquet, output_dir, jet_type, sample_name, tev):
    """
    Generates a comparison plot of the jet response distribution
    """
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    data_pf = awkward.from_parquet(input_pf_parquet)
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)

    jet_label = f"AK{jet_type[2:]} jets"

    # Fiducial cuts
    max_jet_abs_eta = 2.5
    fiducial_cuts = "eta_less_2p5"

    if fiducial_cuts == "eta_less_2p5":
        eta_label = f", $|η|$ < {max_jet_abs_eta}"
        jet_label += eta_label
        for data in [data_pf, data_mlpf]:
            msk_rj_eta = np.abs(data["Jet_eta"]) < max_jet_abs_eta
            for k in data.fields:
                if k.startswith("Jet_"):
                    data[k] = data[k][msk_rj_eta]
            msk_gj_eta = np.abs(data["GenJet_eta"]) < max_jet_abs_eta
            for k in data.fields:
                if k.startswith("GenJet_"):
                    data[k] = data[k][msk_gj_eta]

    data_pf = apply_dz(data_pf)
    data_mlpf = apply_dz(data_mlpf)

    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    data_pf[f"{jet_prefix}_pt_raw"] = data_pf[f"{jet_prefix}_pt"] * (1.0 - data_pf[f"{jet_prefix}_rawFactor"])
    data_mlpf[f"{jet_prefix}_pt_raw"] = data_mlpf[f"{jet_prefix}_pt"] * (1.0 - data_mlpf[f"{jet_prefix}_rawFactor"])

    # Placeholder for corrected pt
    data_pf[f"{jet_prefix}_pt_corr"] = data_pf[f"{jet_prefix}_pt_raw"]
    data_mlpf[f"{jet_prefix}_pt_corr"] = data_mlpf[f"{jet_prefix}_pt_raw"]

    deltar_cut = 0.2 if jet_type == "ak4" else 0.4
    resp_pf = compute_response(data_pf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_mlpf = compute_response(data_mlpf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)

    plot_jet_response_comparison(resp_pf, resp_mlpf, output_dir, sample_name, jet_type, jet_label, tev)
    plot_met_comparison(data_pf, data_mlpf, output_dir, sample_name, tev)

    print(f"Generated plots in {output_dir}")


if __name__ == "__main__":
    main()
