import click
import glob
import os
import awkward
import uproot
import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
import boost_histogram as bh
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import numba
from scipy.interpolate import griddata

def load_nano(fn):
    """Loads events from a nanoaod file."""
    tt = uproot.open(fn).get("Events")
    ret = {}
    for k in [
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_genJetIdx",
        "Jet_rawFactor",
        "Jet_chMultiplicity",
        "Jet_neMultiplicity",
        "Jet_chEmEF",
        "Jet_chHEF",
        "Jet_neEmEF",
        "Jet_neHEF",
        "Jet_neHadMultiplicity",
        "FatJet_pt",
        "FatJet_eta",
        "FatJet_phi",
        "FatJet_genJetAK8Idx",
        "FatJet_rawFactor",
        "GenJet_pt",
        "GenJet_eta",
        "GenJet_phi",
        "GenJet_partonFlavour",
        "GenJetAK8_pt",
        "GenJetAK8_eta",
        "GenJetAK8_phi",
        "GenMET_pt",
        "GenMET_phi",
        "PFMET_pt", "PFMET_phi",
        "PuppiMET_pt", "PuppiMET_phi",
        "RawPFMET_pt", "RawPFMET_phi",
        "Pileup_nPU", "Pileup_nTrueInt",
        "GenVtx_z",
        "PV_z",
    ]:
        ret[k] = tt.arrays(k)[k]
    return [ret, ]

def load_multiprocess(files, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(load_nano, files), total=len(files)))
    successful_results = [r for r in results if r is not None]
    return awkward.concatenate(successful_results)

@numba.njit
def deltaphi_nb(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))

@numba.njit
def deltar_nb(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = deltaphi_nb(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

@numba.njit
def match_jets_nb(j1_eta, j1_phi, j2_eta, j2_phi, deltar_cut):
    assert(len(j1_eta)==len(j2_eta))
    assert(len(j1_phi)==len(j2_phi))
    assert(len(j1_eta)==len(j1_phi))
    iev = len(j1_eta)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    drs_ev = []
    for ev in range(iev):
        jet_inds_1 = []
        jet_inds_2 = []
        drs = []

        while True:
            if len(j1_eta[ev])==0 or len(j2_eta[ev])==0:
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break

            drs_jets = 999*np.ones((len(j1_eta[ev]), len(j2_eta[ev])), dtype=np.float64)

            for ij1 in range(len(j1_eta[ev])):
                if ij1 in jet_inds_1:
                    continue
                for ij2 in range(len(j2_eta[ev])):
                    if ij2 in jet_inds_2:
                        continue

                    eta1 = j1_eta[ev][ij1]
                    eta2 = j2_eta[ev][ij2]
                    phi1 = j1_phi[ev][ij1]
                    phi2 = j2_phi[ev][ij2]
                    dr = deltar_nb(eta1, phi1, eta2, phi2)
                    drs_jets[ij1, ij2] = dr

            if np.all(drs_jets==999):
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break

            flat_index = np.argmin(drs_jets)
            num_rows, num_cols = drs_jets.shape
            ij1_min = flat_index // num_cols
            ij2_min = flat_index % num_cols

            jet_inds_1.append(ij1_min)
            jet_inds_2.append(ij2_min)
            drs.append(drs_jets[ij1_min, ij2_min])
            
            if len(jet_inds_1) == len(j1_eta[ev]) or len(jet_inds_2) == len(j2_eta[ev]):
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break
    return jet_inds_1_ev, jet_inds_2_ev, drs_ev

def compute_response(data, jet_coll="Jet", genjet_coll="GenJet", deltar_cut=0.2):
    rj_idx, gj_idx, drs = match_jets_nb(data[jet_coll+"_eta"], data[jet_coll+"_phi"], data[genjet_coll+"_eta"], data[genjet_coll+"_phi"], deltar_cut)
    
    rj_idx = awkward.Array(rj_idx)
    gj_idx = awkward.Array(gj_idx)
    drs = awkward.Array(drs)
    
    pair_sort = awkward.argsort(data[genjet_coll+"_pt"][gj_idx], axis=1, ascending=False)[:, :3]

    gj_pt = data[genjet_coll+"_pt"][gj_idx][pair_sort]
    gj_eta = data[genjet_coll+"_eta"][gj_idx][pair_sort]
    
    if jet_coll+"_pt_corr" not in data.fields:
        data[jet_coll+"_pt_corr"] = data[jet_coll+"_pt_raw"]

    rj_pt_corr = data[jet_coll+"_pt_corr"][rj_idx][pair_sort]
    rj_pt_raw = data[jet_coll+"_pt_raw"][rj_idx][pair_sort]
    rj_eta = data[jet_coll+"_eta"][rj_idx][pair_sort]
    dr = drs[pair_sort]

    mask_top3 = (dr<deltar_cut)
    
    if jet_coll == "Jet":
        mask_top3 = mask_top3 & (
            (
                (data["Jet_neMultiplicity"][rj_idx][pair_sort]>1)
            ) | 
            (np.abs(data["Jet_eta"][rj_idx][pair_sort])<3)
        )

    response_corr = (rj_pt_corr/gj_pt)
    response_raw = (rj_pt_raw/gj_pt)
    
    # For efficiency and purity
    mask = (drs < deltar_cut)
    if jet_coll == "Jet":
        mask = mask & (
            ((data["Jet_neMultiplicity"][rj_idx] > 1)) |
            (np.abs(data["Jet_eta"][rj_idx]) < 3)
        )
    
    gj_pt_unfiltered = data[genjet_coll+"_pt"][gj_idx]
    gj_eta_unfiltered = data[genjet_coll+"_eta"][gj_idx]
    rj_pt_raw_unfiltered = data[jet_coll+"_pt_raw"][rj_idx]
    rj_pt_corr_unfiltered = data[jet_coll+"_pt_corr"][rj_idx]
    rj_eta_unfiltered = data[jet_coll+"_eta"][rj_idx]

    return {
        "response": response_corr[mask_top3],
        "response_raw": response_raw[mask_top3],
        "dr": dr[mask_top3],
        jet_coll+"_pt_corr": rj_pt_corr[mask_top3],
        jet_coll+"_pt_raw": rj_pt_raw[mask_top3],
        jet_coll+"_eta": rj_eta[mask_top3],
        genjet_coll+"_pt": gj_pt[mask_top3],
        genjet_coll+"_eta": gj_eta[mask_top3],
        # Unfiltered collections for efficiency/purity
        f"{jet_coll}_pt_corr_unfiltered": rj_pt_corr_unfiltered[mask],
        f"{jet_coll}_pt_raw_unfiltered": rj_pt_raw_unfiltered[mask],
        f"{jet_coll}_eta_unfiltered": rj_eta_unfiltered[mask],
        f"{genjet_coll}_pt_unfiltered": gj_pt_unfiltered[mask],
        f"{genjet_coll}_eta_unfiltered": gj_eta_unfiltered[mask],
    }

def med_iqr(arr):
    if len(arr) == 0:
        return 0, 0
    q = np.quantile(arr, [0.25, 0.5, 0.75])
    return q[1], q[2] - q[0]

def fill_nan(reciprocal):
    mask = (np.isnan(reciprocal) | np.isinf(reciprocal))
    if not np.any(mask):
        return reciprocal
    
    valid_coords = np.where(~mask)
    if len(valid_coords[0]) == 0: # all are nan
        return np.ones_like(reciprocal)

    nan_coords = np.where(mask)
    valid_values = reciprocal[~mask]
    
    imputed_values = griddata(
        np.vstack(valid_coords).T,
        valid_values,
        np.vstack(nan_coords).T,
        method='nearest'
    )
    reciprocal = reciprocal.copy()
    reciprocal[nan_coords] = imputed_values
    return reciprocal

def calculate_correction_map(resp_data, eta_bins, pt_bins, jet_type="ak4"):
    jet_prefixes = {"ak4": "Jet", "ak8": "FatJet"}
    jet_prefix = jet_prefixes[jet_type]
    genjet_prefixes = {"ak4": "GenJet", "ak8": "GenJetAK8"}
    genjet_prefix = genjet_prefixes[jet_type]

    resp_stats_med = np.zeros((len(eta_bins)-1, len(pt_bins)-1))

    for ibin_eta in range(len(eta_bins)-1):
        for ibin_pt in range(len(pt_bins)-1):
            
            mask = (
                (resp_data[genjet_prefix + "_pt"] >= pt_bins[ibin_pt]) &
                (resp_data[genjet_prefix + "_pt"] < pt_bins[ibin_pt+1]) &
                (resp_data[jet_prefix + "_eta"] >= eta_bins[ibin_eta]) &
                (resp_data[jet_prefix + "_eta"] < eta_bins[ibin_eta+1])
            )
            
            response_raw = awkward.flatten(resp_data["response_raw"][mask])
            
            median, _ = med_iqr(response_raw)
            resp_stats_med[ibin_eta, ibin_pt] = median

    reciprocal_med = 1.0 / resp_stats_med
    reciprocal_med = fill_nan(reciprocal_med)
    
    return reciprocal_med

@click.group()
def cli():
    """Validation script for MLPF."""
    pass

@cli.command()
@click.option('--input-dir', required=True, type=str, help='Input directory with ROOT files')
@click.option('--sample', required=True, type=str, help='Sample name (e.g., QCD_PU_13p6)')
@click.option('--output-dir', default=".", type=str, help='Output directory for parquet files')
@click.option('--max-files', default=-1, type=int, help='Maximum number of files to process')
@click.option('--max-workers', default=8, type=int, help='Number of worker processes')
def prepare_data(input_dir, sample, output_dir, max_files, max_workers):
    """Loads ROOT files, processes them, and saves to Parquet format."""
    
    os.makedirs(output_dir, exist_ok=True)

    pf_files = glob.glob(f"{input_dir}/{sample}_pf/step4_NANO_jme_*.root")
    mlpf_files = glob.glob(f"{input_dir}/{sample}_mlpf/step4_NANO_jme_*.root")

    pf_files_d = {os.path.basename(fn): fn for fn in pf_files}
    mlpf_files_d = {os.path.basename(fn): fn for fn in mlpf_files}

    common_files = list(set(pf_files_d.keys()).intersection(set(mlpf_files_d.keys())))
    if max_files != -1:
        common_files = common_files[:max_files]

    print(f"Found {len(common_files)} common files.")

    for mlpf_or_pf in ["mlpf", "pf"]:
        output_file = f"{output_dir}/{sample}_{mlpf_or_pf}.parquet"

        if mlpf_or_pf == "pf":
            files_to_process = [pf_files_d[fn] for fn in common_files]
        else:
            files_to_process = [mlpf_files_d[fn] for fn in common_files]
        
        print(f"Processing {len(files_to_process)} files for {sample}_{mlpf_or_pf}")
        
        data = load_multiprocess(files_to_process, max_workers=max_workers)
        
        if data is None:
            print(f"No data loaded for {sample}_{mlpf_or_pf}")
            continue

        data = awkward.Array({k: awkward.flatten(data[k], axis=1) for k in data.fields})
        
        if "Jet_pt" in data.fields:
            data["Jet_pt_raw"] = data["Jet_pt"] * (1.0 - data["Jet_rawFactor"])
        if "FatJet_pt" in data.fields:
            data["FatJet_pt_raw"] = data["FatJet_pt"] * (1.0 - data["FatJet_rawFactor"])

        if "GenVtx_z" in data.fields and "PV_z" in data.fields:
            abs_dz = np.abs(data["GenVtx_z"] - data["PV_z"])
            mask_dz = abs_dz < 0.2
            data = data[mask_dz]

        awkward.to_parquet(data, output_file)
        print(f"Saved data to {output_file}")

@cli.command()
@click.option('--input-pf-parquet', required=True, type=str)
@click.option('--input-mlpf-parquet', required=True, type=str)
@click.option('--output-file', required=True, type=str)
@click.option('--jet-type', default='ak4', type=click.Choice(['ak4', 'ak8']))
def make_corrections(input_pf_parquet, input_mlpf_parquet, output_file, jet_type):
    """Generates jet energy correction maps."""
    
    data_pf = awkward.from_parquet(input_pf_parquet)
    data_mlpf = awkward.from_parquet(input_mlpf_parquet)

    if jet_type == 'ak4':
        jet_coll = "Jet"
        genjet_coll = "GenJet"
        deltar_cut = 0.2
        eta_reco_bins = [-5.191, -2.964, -1.392, 0, 1.392, 2.964, 5.191]
        pt_gen_bins = [10, 20, 30, 40, 50, 80, 120, 200, 500, 3000]
    else: # ak8
        jet_coll = "FatJet"
        genjet_coll = "GenJetAK8"
        deltar_cut = 0.4
        eta_reco_bins = [-5.191, -2.964, -1.392, 0, 1.392, 2.964, 5.191]
        pt_gen_bins = [200, 300, 400, 500, 3000]

    resp_pf = compute_response(data_pf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)
    resp_mlpf = compute_response(data_mlpf, jet_coll=jet_coll, genjet_coll=genjet_coll, deltar_cut=deltar_cut)

    corr_map_pf = calculate_correction_map(resp_pf, eta_reco_bins, pt_gen_bins, jet_type=jet_type)
    corr_map_mlpf = calculate_correction_map(resp_mlpf, eta_reco_bins, pt_gen_bins, jet_type=jet_type)

    np.savez(output_file,
             corr_map_pf=corr_map_pf,
             corr_map_mlpf=corr_map_mlpf,
             eta_bins=eta_reco_bins,
             pt_bins=pt_gen_bins)
    
    print(f"Saved correction maps to {output_file}")

@cli.command()
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

    # plotting style variables
    legend_fontsize = 30
    sample_label_fontsize = 30
    addtext_fontsize = 25
    jet_label_coords_single = 0.02, 0.86
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
        variable, bins,
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
        
        h0 = to_bh(awkward.flatten(data_pf[f"{genjet_prefix}_{variable}"][gen_jet_mask]), bins)

        # PF jets
        pf_jet_mask = (data_pf[jet_pt_var] > min_jet_pt)
        if jet_prefix == "Jet":
            pf_jet_mask = pf_jet_mask & ((data_pf["Jet_neMultiplicity"] > 1) | (np.abs(data_pf["Jet_eta"]) < 3))
        if eta_cut:
            pf_jet_mask = pf_jet_mask & (np.abs(data_pf[jet_eta_var]) < eta_cut)
        
        h1 = to_bh(awkward.flatten(data_pf[f"{jet_prefix}_{variable}"][pf_jet_mask]), bins)

        # MLPF jets
        mlpf_jet_mask = (data_mlpf[jet_pt_var] > min_jet_pt)
        if jet_prefix == "Jet":
            mlpf_jet_mask = mlpf_jet_mask & ((data_mlpf["Jet_neMultiplicity"] > 1) | (np.abs(data_mlpf["Jet_eta"]) < 3))
        if eta_cut:
            mlpf_jet_mask = mlpf_jet_mask & (np.abs(data_mlpf[jet_eta_var]) < eta_cut)

        h2 = to_bh(awkward.flatten(data_mlpf[f"{jet_prefix}_{variable}"][mlpf_jet_mask]), bins)

        plt.sca(a0)
        x0 = mplhep.histplot(h0, histtype="step", lw=2, label="Gen.", binwnorm=1.0, ls="--")
        x1 = mplhep.histplot(h1, histtype="step", lw=2, label="PF-PUPPI", binwnorm=1.0, ls=pf_linestyle)
        x2 = mplhep.histplot(h2, histtype="step", lw=2, label=mlpf_label, binwnorm=1.0, ls=mlpf_linestyle)

        if logy:
            plt.yscale("log")
        
        mplhep.cms.label("", data=False, com=13.6, year='Run 3', ax=a0)
        a0.text(sample_label_coords[0], sample_label_coords[1], sample_name, transform=a0.transAxes, fontsize=sample_label_fontsize)
        
        jet_label_text = jet_label
        if eta_cut:
            jet_label_text += f", |$\eta$| < {eta_cut}"
        a0.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label_text, transform=a0.transAxes, fontsize=addtext_fontsize)
        
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

        if variable == "pt":
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
        ax.text(sample_label_coords[0], sample_label_coords[1], sample_name, transform=ax.transAxes, fontsize=sample_label_fontsize)
        ax.text(jet_label_coords_single[0], jet_label_coords_single[1], jet_label + additional_label, transform=ax.transAxes, fontsize=addtext_fontsize)

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
                jet_pt=f"{jet_prefix}_pt_corr",
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

    def midpoints(x):
        return (x[1:] + x[:-1]) / 2

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
        "pt", pt_bins_for_kinematics,
        f"{jet_prefix} pT [GeV]", f"{jet_type}_pt_raw.pdf",
        raw_or_corr="raw", jet_label=jet_label
    )
    plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        "pt", pt_bins_for_kinematics,
        f"Corrected {jet_prefix} pT [GeV]", f"{jet_type}_pt_corr.pdf",
        raw_or_corr="corr", jet_label=jet_label
    )
    plot_kinematic_distribution(
        data_pf, data_mlpf,
        jet_prefix, genjet_prefix,
        "eta", eta_bins_for_kinematics,
        f"{jet_prefix} eta", f"{jet_type}_eta.pdf",
        logy=False, jet_label=jet_label
    )
    if jet_type == 'ak4':
        plot_kinematic_distribution(
            data_pf, data_mlpf,
            jet_prefix, genjet_prefix,
            "pt", pt_bins_for_kinematics,
            f"{jet_prefix} pT [GeV]", f"{jet_type}_pt_raw_barrel.pdf",
            raw_or_corr="raw", eta_cut=2.5, jet_label=jet_label
        )
        plot_kinematic_distribution(
            data_pf, data_mlpf,
            jet_prefix, genjet_prefix,
            "pt", pt_bins_for_kinematics,
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

    med_pf_vs_pt, iqr_pf_vs_pt, _, sigma_pf_vs_pt, med_mlpf_vs_pt, iqr_mlpf_vs_pt, _, sigma_mlpf_vs_pt = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=pt_bins_for_response, variable_name=f"{genjet_prefix}_pt",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )

    med_pf_vs_eta, iqr_pf_vs_eta, _, sigma_pf_vs_eta, med_mlpf_vs_eta, iqr_mlpf_vs_eta, _, sigma_mlpf_vs_eta = get_response_in_bins(
        resp_pf, resp_mlpf, data_pf, data_mlpf,
        variable_bins=eta_bins_for_response, variable_name=f"{genjet_prefix}_eta",
        jet_prefix=jet_prefix, genjet_prefix=genjet_prefix, jet_label=jet_label,
    )

    # Plot scale vs pt
    plt.figure()
    plt.plot(midpoints(pt_bins_for_response), med_pf_vs_pt, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle)
    plt.plot(midpoints(pt_bins_for_response), med_mlpf_vs_pt, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle)
    plt.xlabel("GenJet $p_T$ [GeV]")
    plt.ylabel("Median response")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_pt.pdf"))
    plt.close()

    # Plot resolution vs pt
    plt.figure()
    plt.plot(midpoints(pt_bins_for_response), iqr_pf_vs_pt / med_pf_vs_pt, label="PF-PUPPI (IQR/median)", color=pf_color, linestyle=pf_linestyle)
    plt.plot(midpoints(pt_bins_for_response), iqr_mlpf_vs_pt / med_mlpf_vs_pt, label="MLPF-PUPPI (IQR/median)", color=mlpf_color, linestyle=mlpf_linestyle)
    plt.xlabel("GenJet $p_T$ [GeV]")
    plt.ylabel("Response resolution")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_pt.pdf"))
    plt.close()

    # Plot scale vs eta
    plt.figure()
    plt.plot(midpoints(eta_bins_for_response), med_pf_vs_eta, label="PF-PUPPI", color=pf_color, linestyle=pf_linestyle)
    plt.plot(midpoints(eta_bins_for_response), med_mlpf_vs_eta, label=mlpf_label, color=mlpf_color, linestyle=mlpf_linestyle)
    plt.xlabel("GenJet $\eta$")
    plt.ylabel("Median response")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{jet_type}_scale_vs_eta.pdf"))
    plt.close()

    # Plot resolution vs eta
    plt.figure()
    plt.plot(midpoints(eta_bins_for_response), iqr_pf_vs_eta / med_pf_vs_eta, label="PF-PUPPI (IQR/median)", color=pf_color, linestyle=pf_linestyle)
    plt.plot(midpoints(eta_bins_for_response), iqr_mlpf_vs_eta / med_mlpf_vs_eta, label="MLPF-PUPPI (IQR/median)", color=mlpf_color, linestyle=mlpf_linestyle)
    plt.xlabel("GenJet $\eta$")
    plt.ylabel("Response resolution")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{jet_type}_resolution_vs_eta.pdf"))
    plt.close()

    print(f"Generated plots in {output_dir}")


if __name__ == '__main__':
    cli()
