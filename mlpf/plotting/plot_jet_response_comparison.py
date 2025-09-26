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
legend_loc_jet_response = (0.45, 0.65)

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


def plot_met_comparison(data_13p6_pf, data_13p6_mlpf, data_14_pf, data_14_mlpf, output_dir, sample_name):
    """Plots the MET comparison between 13.6 TeV and 14 TeV samples, for PF and MLPF."""
    plt.figure()
    ax = plt.axes()

    bins = np.linspace(0, 500, 51)

    sample_label_coords = 0.02, 0.96
    sample_label_fontsize = 30
    legend_fontsize = 30

    h_13p6_pf = to_bh(data_13p6_pf["PuppiMET_pt"], bins=bins)
    h_13p6_mlpf = to_bh(data_13p6_mlpf["PuppiMET_pt"], bins=bins)
    h_14_pf = to_bh(data_14_pf["PuppiMET_pt"], bins=bins)
    h_14_mlpf = to_bh(data_14_mlpf["PuppiMET_pt"], bins=bins)
    mplhep.histplot(h_14_pf, histtype="step", lw=1, density=True, label="14 TeV, PF", ls="--", color=pf_color)
    mplhep.histplot(h_13p6_pf, histtype="step", lw=1, density=True, label="13.6 TeV, PF", color=pf_color)
    mplhep.histplot(h_14_mlpf, histtype="step", lw=2, density=True, label="14 TeV, MLPF", ls="--", color=mlpf_color)
    mplhep.histplot(h_13p6_mlpf, histtype="step", lw=2, density=True, label="13.6 TeV, MLPF", color=mlpf_color)

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

    mplhep.cms.label("", data=False, rlabel="Run 3 configuration")

    ax.set_ylim(bottom=1e-5)
    plt.legend(fontsize=legend_fontsize, loc=legend_loc_jet_response)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"{sample_name}_met_dist_14vs13p6.pdf")
    plt.close()


def plot_jet_response_comparison(resp_13p6_pf, resp_13p6_mlpf, resp_14_pf, resp_14_mlpf, output_dir, sample_name, jet_type, jet_label):
    """Plots the jet response comparison between 13.6 TeV and 14 TeV samples, for PF and MLPF."""
    plt.figure()
    ax = plt.axes()
    b = np.linspace(0, 2, 101)

    sample_label_coords = 0.02, 0.96
    jet_label_coords_single = 0.02, 0.88
    sample_label_fontsize = 30
    addtext_fontsize = 25
    legend_fontsize = 30

    response_13p6_pf = awkward.flatten(resp_13p6_pf["response_raw"])
    response_13p6_mlpf = awkward.flatten(resp_13p6_mlpf["response_raw"])
    response_14_pf = awkward.flatten(resp_14_pf["response_raw"])
    response_14_mlpf = awkward.flatten(resp_14_mlpf["response_raw"])

    h_13p6_pf = to_bh(response_13p6_pf, bins=b)
    h_13p6_mlpf = to_bh(response_13p6_mlpf, bins=b)
    h_14_pf = to_bh(response_14_pf, bins=b)
    h_14_mlpf = to_bh(response_14_mlpf, bins=b)

    mplhep.histplot(h_14_pf, histtype="step", lw=1, density=True, label="14 TeV, PF", ls="--", color=pf_color)
    mplhep.histplot(h_13p6_pf, histtype="step", lw=1, density=True, label="13.6 TeV, PF", color=pf_color)
    mplhep.histplot(h_14_mlpf, histtype="step", lw=2, density=True, label="14 TeV, MLPF", ls="--", color=mlpf_color)
    mplhep.histplot(h_13p6_mlpf, histtype="step", lw=2, density=True, label="13.6 TeV, MLPF", color=mlpf_color)

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

    mplhep.cms.label("", data=False, rlabel="Run 3 configuration")
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
    plt.legend(fontsize=legend_fontsize, loc=legend_loc_jet_response)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"{sample_name}_{jet_type}_jet_pt_ratio_14vs13p6.pdf")
    plt.close()


@click.command()
@click.option("--input-13p6-tev-pf-parquet", required=True, type=str, help="Input parquet file for 13.6 TeV PF sample.")
@click.option("--input-13p6-tev-mlpf-parquet", required=True, type=str, help="Input parquet file for 13.6 TeV MLPF sample.")
@click.option("--input-14-tev-pf-parquet", required=True, type=str, help="Input parquet file for 14 TeV PF sample.")
@click.option("--input-14-tev-mlpf-parquet", required=True, type=str, help="Input parquet file for 14 TeV MLPF sample.")
@click.option("--output-dir", required=True, type=str, help="Output directory for plots.")
@click.option("--jet-type", default="ak4", type=click.Choice(["ak4", "ak8"]), help="Jet type to plot.")
@click.option("--sample-name", required=True, type=str, help="Sample name for plot labels (e.g., QCD_PU).")
def main(
    input_13p6_tev_pf_parquet, input_13p6_tev_mlpf_parquet, input_14_tev_pf_parquet, input_14_tev_mlpf_parquet, output_dir, jet_type, sample_name
):
    """
    Generates a comparison plot of the jet response distribution
    for 14 TeV and 13.6 TeV samples.
    """
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    data_13p6_pf = awkward.from_parquet(input_13p6_tev_pf_parquet)
    data_13p6_mlpf = awkward.from_parquet(input_13p6_tev_mlpf_parquet)
    data_14_pf = awkward.from_parquet(input_14_tev_pf_parquet)
    data_14_mlpf = awkward.from_parquet(input_14_tev_mlpf_parquet)

    jet_label = f"AK{jet_type[2:]} jets"

    # Fiducial cuts
    max_jet_abs_eta = 2.5
    fiducial_cuts = "eta_less_2p5"

    if fiducial_cuts == "eta_less_2p5":
        eta_label = f", 0 < $|η|$ < {max_jet_abs_eta}"
        jet_label += eta_label
        for data in [data_13p6_pf, data_13p6_mlpf, data_14_pf, data_14_mlpf]:
            msk_rj_eta = np.abs(data["Jet_eta"]) < max_jet_abs_eta
            for k in data.fields:
                if k.startswith("Jet_"):
                    data[k] = data[k][msk_rj_eta]
            msk_gj_eta = np.abs(data["GenJet_eta"]) < max_jet_abs_eta
            for k in data.fields:
                if k.startswith("GenJet_"):
                    data[k] = data[k][msk_gj_eta]

    data_13p6_pf = apply_dz(data_13p6_pf)
    data_13p6_mlpf = apply_dz(data_13p6_mlpf)
    data_14_pf = apply_dz(data_14_pf)
    data_14_mlpf = apply_dz(data_14_mlpf)

    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    data_13p6_pf[f"{jet_prefix}_pt_raw"] = data_13p6_pf[f"{jet_prefix}_pt"] * (1.0 - data_13p6_pf[f"{jet_prefix}_rawFactor"])
    data_13p6_mlpf[f"{jet_prefix}_pt_raw"] = data_13p6_mlpf[f"{jet_prefix}_pt"] * (1.0 - data_13p6_mlpf[f"{jet_prefix}_rawFactor"])
    data_14_pf[f"{jet_prefix}_pt_raw"] = data_14_pf[f"{jet_prefix}_pt"] * (1.0 - data_14_pf[f"{jet_prefix}_rawFactor"])
    data_14_mlpf[f"{jet_prefix}_pt_raw"] = data_14_mlpf[f"{jet_prefix}_pt"] * (1.0 - data_14_mlpf[f"{jet_prefix}_rawFactor"])

    # Placeholder for corrected pt
    data_13p6_pf[f"{jet_prefix}_pt_corr"] = data_13p6_pf[f"{jet_prefix}_pt_raw"]
    data_13p6_mlpf[f"{jet_prefix}_pt_corr"] = data_13p6_mlpf[f"{jet_prefix}_pt_raw"]
    data_14_pf[f"{jet_prefix}_pt_corr"] = data_14_pf[f"{jet_prefix}_pt_raw"]
    data_14_mlpf[f"{jet_prefix}_pt_corr"] = data_14_mlpf[f"{jet_prefix}_pt_raw"]

    deltar_cut = 0.2 if jet_type == "ak4" else 0.4
    resp_13p6_pf = compute_response(data_13p6_pf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_13p6_mlpf = compute_response(data_13p6_mlpf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_14_pf = compute_response(data_14_pf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_14_mlpf = compute_response(data_14_mlpf, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)

    plot_jet_response_comparison(resp_13p6_pf, resp_13p6_mlpf, resp_14_pf, resp_14_mlpf, output_dir, sample_name, jet_type, jet_label)
    plot_met_comparison(data_13p6_pf, data_13p6_mlpf, data_14_pf, data_14_mlpf, output_dir, sample_name)

    print(f"Generated plots in {output_dir}")


if __name__ == "__main__":
    main()
