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


def plot_jet_response_comparison(resp_13p6, resp_14, output_dir, sample_name, jet_type, jet_label):
    """Plots the jet response comparison between 13.6 TeV and 14 TeV samples."""
    plt.figure()
    ax = plt.axes()
    b = np.linspace(0, 2, 101)

    sample_label_coords = 0.02, 0.96
    jet_label_coords_single = 0.02, 0.88
    sample_label_fontsize = 30
    addtext_fontsize = 25
    legend_fontsize = 30

    response_13p6 = awkward.flatten(resp_13p6["response_raw"])
    response_14 = awkward.flatten(resp_14["response_raw"])

    h_13p6 = to_bh(response_13p6, bins=b)
    h_14 = to_bh(response_14, bins=b)

    mplhep.histplot(h_14, histtype="step", lw=2, density=True, label="14 TeV", ls="--")
    mplhep.histplot(h_13p6, histtype="step", lw=2, density=True, label="13.6 TeV")

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
    plt.legend(fontsize=legend_fontsize)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"{sample_name}_{jet_type}_jet_pt_ratio_14vs13p6.pdf")
    plt.close()


@click.command()
@click.option("--input-13p6-tev-parquet", required=True, type=str, help="Input parquet file for 13.6 TeV sample.")
@click.option("--input-14-tev-parquet", required=True, type=str, help="Input parquet file for 14 TeV sample.")
@click.option("--output-dir", required=True, type=str, help="Output directory for plots.")
@click.option("--jet-type", default="ak4", type=click.Choice(["ak4", "ak8"]), help="Jet type to plot.")
@click.option("--sample-name", required=True, type=str, help="Sample name for plot labels (e.g., QCD_PU).")
def main(input_13p6_tev_parquet, input_14_tev_parquet, output_dir, jet_type, sample_name):
    """
    Generates a comparison plot of the jet response distribution
    for 14 TeV and 13.6 TeV samples.
    """
    mplhep.style.use("CMS")
    matplotlib.rcParams["axes.labelsize"] = 35

    data_13p6 = awkward.from_parquet(input_13p6_tev_parquet)
    data_14 = awkward.from_parquet(input_14_tev_parquet)

    data_13p6 = apply_dz(data_13p6)
    data_14 = apply_dz(data_14)

    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    data_13p6[f"{jet_prefix}_pt_raw"] = data_13p6[f"{jet_prefix}_pt"] * (1.0 - data_13p6[f"{jet_prefix}_rawFactor"])
    data_14[f"{jet_prefix}_pt_raw"] = data_14[f"{jet_prefix}_pt"] * (1.0 - data_14[f"{jet_prefix}_rawFactor"])

    # Placeholder for corrected pt
    data_13p6[f"{jet_prefix}_pt_corr"] = data_13p6[f"{jet_prefix}_pt_raw"]
    data_14[f"{jet_prefix}_pt_corr"] = data_14[f"{jet_prefix}_pt_raw"]

    deltar_cut = 0.2 if jet_type == "ak4" else 0.4
    resp_13p6 = compute_response(data_13p6, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)
    resp_14 = compute_response(data_14, jet_coll=jet_prefix, genjet_coll=genjet_prefix, deltar_cut=deltar_cut)

    jet_label = f"AK{jet_type[2:]} jets"

    plot_jet_response_comparison(resp_13p6, resp_14, output_dir, sample_name, jet_type, jet_label)

    print(f"Generated plot in {output_dir}")


if __name__ == "__main__":
    main()
