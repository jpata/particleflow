import click
import glob
import os
import awkward
import uproot
import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np


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
        "PFMET_pt",
        "PFMET_phi",
        "PuppiMET_pt",
        "PuppiMET_phi",
        "RawPFMET_pt",
        "RawPFMET_phi",
        "Pileup_nPU",
        "Pileup_nTrueInt",
        "GenVtx_z",
        "PV_z",
    ]:
        ret[k] = tt.arrays(k)[k]
    return [
        ret,
    ]


def load_multiprocess(files, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(load_nano, files), total=len(files)))
    successful_results = [r for r in results if r is not None]
    return awkward.concatenate(successful_results)


@click.command()
@click.option("--input-dir", required=True, type=str, help="Input directory with ROOT files")
@click.option("--sample", required=True, type=str, help="Sample name (e.g., QCD_PU_13p6)")
@click.option("--output-dir", default=".", type=str, help="Output directory for parquet files")
@click.option("--max-files", default=-1, type=int, help="Maximum number of files to process")
@click.option("--max-workers", default=8, type=int, help="Number of worker processes")
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


if __name__ == "__main__":
    prepare_data()
