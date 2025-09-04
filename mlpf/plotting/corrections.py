import click
import awkward
import numpy as np
from scipy.interpolate import griddata
from mlpf.plotting.utils import compute_response, med_iqr

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

@click.command()
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

if __name__ == '__main__':
    make_corrections()
