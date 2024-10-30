import boost_histogram as bh
import glob
import pickle
import uproot
import awkward as ak
import numpy as np
import vector
import bz2
import pandas
import os
import fastjet
import random
import time
import tqdm

from mlpf import jet_utils


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_tree(ttree):
    particles_pythia = ttree.arrays(["gen_pt", "gen_eta", "gen_phi", "gen_energy", "gen_pdgid", "gen_status", "gen_daughters"])
    particles_cp = ttree.arrays(["caloparticle_pt", "caloparticle_eta", "caloparticle_phi", "caloparticle_energy", "caloparticle_pid"])
    genjet = ttree.arrays(["genjet_pt", "genjet_eta", "genjet_phi", "genjet_energy"])
    genmet = ttree.arrays(["genmet_pt"])
    return ak.Array({"pythia": particles_pythia, "cp": particles_cp, "genjet": genjet, "genmet": genmet})


def sum_overflow_into_last_bin(all_values):
    values = all_values[1:-1]
    values[-1] = values[-1] + all_values[-1]
    values[0] = values[0] + all_values[0]
    return values


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    h1[:] = sum_overflow_into_last_bin(h1.values(flow=True)[:])
    return h1


def compute_met(pt, phi, mask=None):
    if mask is None:
        mask = np.ones_like(pt, dtype=bool)
    met = np.sqrt(ak.sum(pt[mask] * np.sin(phi[mask]), axis=1) ** 2 + ak.sum(pt[mask] * np.cos(phi[mask]), axis=1) ** 2)
    return met


def process_files(sample_folder, rootfiles, pklfiles, outfile):
    pause = random.randint(100, 1000) / 1000.0
    time.sleep(pause)

    # check that the root and pkl file lists correspond to each other
    if len(rootfiles) > 0:
        assert len(rootfiles) == len(pklfiles)
        for fn1, fn2 in zip(rootfiles, pklfiles):
            assert os.path.basename(fn1).split(".")[0] == os.path.basename(fn2).split(".")[0]

        # load root files
        tts = [load_tree(uproot.open(fn)["pfana/pftree"]) for fn in rootfiles]
        tts = ak.concatenate(tts, axis=0)
        particles_pythia = tts["pythia"]
        particles_cp = tts["cp"]

    # load pkl files
    pickle_data = sum(
        [pickle.load(bz2.BZ2File(fn)) for fn in pklfiles],
        [],
    )

    for i in range(len(pickle_data)):
        for coll in ["ytarget", "ycand"]:
            pickle_data[i][coll] = pandas.DataFrame(pickle_data[i][coll])
            pickle_data[i][coll]["phi"] = np.arctan2(pickle_data[i][coll]["sin_phi"], pickle_data[i][coll]["cos_phi"])

    # get awkward and flat arrays from the data
    arrs_awk = {}
    arrs_flat = {}

    # tracks and clusters
    for coll in ["Xelem"]:
        arrs_awk[coll] = {}
        arrs_flat[coll] = {}
        for feat in ["typ", "pt", "eta", "phi", "energy"]:
            arr = [np.array(p[coll][feat][p[coll]["typ"] != 0]) for p in pickle_data]
            arrs_awk[coll][feat] = ak.unflatten(ak.concatenate(arr), [len(a) for a in arr])
            arr = [np.array(p[coll][feat]) for p in pickle_data]
            arrs_flat[coll][feat] = ak.unflatten(ak.concatenate(arr), [len(a) for a in arr])

    # MLPF targets and PF reco
    for coll in ["ytarget", "ycand"]:
        arrs_awk[coll] = {}
        arrs_flat[coll] = {}
        for feat in ["pid", "pt", "eta", "phi", "energy", "ispu"]:
            arr = [np.array(p[coll][feat][p[coll]["pid"] != 0]) for p in pickle_data]
            arrs_awk[coll][feat] = ak.unflatten(ak.concatenate(arr), [len(a) for a in arr])
            arr = [np.array(p[coll][feat]) for p in pickle_data]
            arrs_flat[coll][feat] = ak.unflatten(ak.concatenate(arr), [len(a) for a in arr])

    # pythia generator level particles
    arrs_awk["pythia"] = {}
    arrs_awk["pythia"]["pid"] = ak.from_regular([np.array(p["pythia"][:, 0]) for p in pickle_data])
    arrs_awk["pythia"]["pt"] = ak.from_regular([np.array(p["pythia"][:, 1]) for p in pickle_data])
    arrs_awk["pythia"]["eta"] = ak.from_regular([np.array(p["pythia"][:, 2]) for p in pickle_data])
    arrs_awk["pythia"]["phi"] = ak.from_regular([np.array(p["pythia"][:, 3]) for p in pickle_data])
    arrs_awk["pythia"]["energy"] = ak.from_regular([np.array(p["pythia"][:, 4]) for p in pickle_data])

    pu_mask = arrs_awk["ytarget"]["ispu"] < 0.5
    if len(rootfiles) > 0:
        abs_pid = np.abs(particles_pythia["gen_pdgid"])
        mask_pythia_nonu = (
            (particles_pythia["gen_status"] == 1)
            & (abs_pid != 12)
            & (abs_pid != 14)
            & (abs_pid != 16)  # |
            # ((particles_pythia["gen_status"]==2) & (ak.num(particles_pythia["gen_daughters"], axis=2) == 0))
        )
        mask_cp = np.abs(particles_cp["caloparticle_eta"]) < 5

        # MET from MLPF targets and from PF particles
        # ypythia_met = compute_met(particles_pythia["gen_pt"], particles_pythia["gen_phi"], mask_pythia_nonu)
        # ycaloparticle_met = compute_met(particles_cp["caloparticle_pt"], particles_cp["caloparticle_phi"], mask_cp)

    # dummy mask
    ytarget_met = compute_met(arrs_awk["ytarget"]["pt"], arrs_awk["ytarget"]["phi"])
    ytarget_nopu_met = compute_met(arrs_awk["ytarget"]["pt"], arrs_awk["ytarget"]["phi"], pu_mask)
    ycand_met = compute_met(arrs_awk["ycand"]["pt"], arrs_awk["ycand"]["phi"])

    # cluster jets
    jets_coll = {}
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    # genMet, genJets from CMSSW (should be the same as computed from Pythia)
    # genmet_cmssw = np.array([pickle_data[i]["genmet"][0, 0] for i in range(len(pickle_data))])
    genjet_cmssw = ak.from_regular([pickle_data[i]["genjet"] for i in range(len(pickle_data))])
    genmet_cmssw = ak.from_regular([pickle_data[i]["genmet"] for i in range(len(pickle_data))])
    genmet_cmssw = genmet_cmssw[:, 0, 0]
    genjet_cmssw = vector.awk(
        ak.zip(
            {
                "pt": genjet_cmssw[:, :, 0],
                "eta": genjet_cmssw[:, :, 1],
                "phi": genjet_cmssw[:, :, 2],
                "energy": genjet_cmssw[:, :, 3],
            }
        )
    )
    jets_coll["genjet"] = genjet_cmssw

    if len(rootfiles) > 0:
        vec = vector.awk(
            ak.zip(
                {
                    "pt": particles_pythia[mask_pythia_nonu]["gen_pt"],
                    "eta": particles_pythia[mask_pythia_nonu]["gen_eta"],
                    "phi": particles_pythia[mask_pythia_nonu]["gen_phi"],
                    "energy": particles_pythia[mask_pythia_nonu]["gen_energy"],
                }
            )
        )
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets_coll["pythia_nonu"] = cluster.inclusive_jets(min_pt=3)

        vec = vector.awk(
            ak.zip(
                {
                    "pt": particles_cp[mask_cp]["caloparticle_pt"],
                    "eta": particles_cp[mask_cp]["caloparticle_eta"],
                    "phi": particles_cp[mask_cp]["caloparticle_phi"],
                    "energy": particles_cp[mask_cp]["caloparticle_energy"],
                }
            )
        )
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets_coll["cp"] = cluster.inclusive_jets(min_pt=3)

    for coll in ["ytarget", "ycand"]:
        vec = vector.awk(
            ak.zip(
                {
                    "pt": arrs_awk[coll]["pt"],
                    "eta": arrs_awk[coll]["eta"],
                    "phi": arrs_awk[coll]["phi"],
                    "energy": arrs_awk[coll]["energy"],
                }
            )
        )
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets_coll[coll] = cluster.inclusive_jets(min_pt=3)

    vec = vector.awk(
        ak.zip(
            {
                "pt": arrs_awk["ytarget"]["pt"][pu_mask],
                "eta": arrs_awk["ytarget"]["eta"][pu_mask],
                "phi": arrs_awk["ytarget"]["phi"][pu_mask],
                "energy": arrs_awk["ytarget"]["energy"][pu_mask],
            }
        )
    )
    cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
    jets_coll["ytarget_nopu"] = cluster.inclusive_jets(min_pt=3)

    genjet_to_ytarget = jet_utils.match_two_jet_collections(jets_coll, "genjet", "ytarget", 0.1)
    genjet_to_ytarget_nopu = jet_utils.match_two_jet_collections(jets_coll, "genjet", "ytarget_nopu", 0.1)
    genjet_to_ycand = jet_utils.match_two_jet_collections(jets_coll, "genjet", "ycand", 0.1)

    ret = {}

    # particle distributions
    b = np.logspace(-4, 4, 100)
    if len(rootfiles) > 0:
        ret[f"{sample_folder}/particles_pt_pythia"] = to_bh(ak.flatten(particles_pythia[mask_pythia_nonu]["gen_pt"]), bins=b)
        ret[f"{sample_folder}/particles_pt_caloparticle"] = to_bh(ak.flatten(particles_cp[mask_cp]["caloparticle_pt"]), bins=b)
    ret[f"{sample_folder}/particles_pt_target"] = to_bh(ak.flatten(arrs_awk["ytarget"]["pt"]), bins=b)
    ret[f"{sample_folder}/particles_pt_target_pumask"] = to_bh(ak.flatten(arrs_awk["ytarget"]["pt"][pu_mask]), bins=b)
    ret[f"{sample_folder}/particles_pt_cand"] = to_bh(ak.flatten(arrs_awk["ycand"]["pt"]), bins=b)

    # per-pid particle distribution
    pidset = np.unique(np.abs(ak.flatten(arrs_awk["ytarget"]["pid"])))
    for pid in pidset:
        pid = int(pid)
        if len(rootfiles) > 0:
            pidmask_pythia = np.abs(particles_pythia[mask_pythia_nonu]["gen_pdgid"]) == pid
            ret[f"{sample_folder}/particle_{pid}_pt_pythia"] = to_bh(ak.flatten(particles_pythia[mask_pythia_nonu]["gen_pt"][pidmask_pythia]), bins=b)
            pidmask_cp = np.abs(particles_cp[mask_cp]["caloparticle_pid"]) == pid
            ret[f"{sample_folder}/particle_{pid}_pt_caloparticle"] = to_bh(ak.flatten(particles_cp[mask_cp]["caloparticle_pt"][pidmask_cp]), bins=b)
        pidmask_ytarget = np.abs(arrs_awk["ytarget"]["pid"]) == pid
        ret[f"{sample_folder}/particle_{pid}_pt_target"] = to_bh(ak.flatten(arrs_awk["ytarget"]["pt"][pidmask_ytarget]), bins=b)
        ret[f"{sample_folder}/particle_{pid}_pt_target_pumask"] = to_bh(ak.flatten(arrs_awk["ytarget"]["pt"][pu_mask & pidmask_ytarget]), bins=b)
        pidmask_ytarget = np.abs(arrs_awk["ycand"]["pid"]) == pid
        ret[f"{sample_folder}/particle_{pid}_pt_cand"] = to_bh(ak.flatten(arrs_awk["ycand"]["pt"]), bins=b)

    b = np.linspace(-5, 5, 100)
    if len(rootfiles) > 0:
        ret[f"{sample_folder}/particles_eta_pythia"] = to_bh(ak.flatten(particles_pythia[mask_pythia_nonu]["gen_eta"]), bins=b)
        ret[f"{sample_folder}/particles_eta_caloparticle"] = to_bh(ak.flatten(particles_cp[mask_cp]["caloparticle_eta"]), bins=b)
    ret[f"{sample_folder}/particles_eta_target"] = to_bh(ak.flatten(arrs_awk["ytarget"]["eta"]), bins=b)
    ret[f"{sample_folder}/particles_eta_target_pumask"] = to_bh(ak.flatten(arrs_awk["ytarget"]["eta"][pu_mask]), bins=b)
    ret[f"{sample_folder}/particles_eta_cand"] = to_bh(ak.flatten(arrs_awk["ycand"]["eta"]), bins=b)

    # jet pt distribution
    b = np.logspace(0, 4, 100)
    ret[f"{sample_folder}/jets_pt_genjet"] = to_bh(ak.flatten(jets_coll["genjet"].pt), bins=b)
    # ret[f"{sample_folder}/jets_pt_caloparticle"] = to_bh(ak.flatten(jets_coll["cp"].pt), bins=b)
    ret[f"{sample_folder}/jets_pt_target"] = to_bh(ak.flatten(jets_coll["ytarget"].pt), bins=b)
    ret[f"{sample_folder}/jets_pt_target_pumask"] = to_bh(ak.flatten(jets_coll["ytarget_nopu"].pt), bins=b)
    ret[f"{sample_folder}/jets_pt_cand"] = to_bh(ak.flatten(jets_coll["ycand"].pt), bins=b)

    b = np.linspace(-5, 5, 100)
    ret[f"{sample_folder}/jets_eta_genjet"] = to_bh(ak.flatten(jets_coll["genjet"].eta), bins=b)
    # ret[f"{sample_folder}/jets_eta_caloparticle"] = to_bh(ak.flatten(jets_coll["cp"].eta), bins=b)
    ret[f"{sample_folder}/jets_eta_target"] = to_bh(ak.flatten(jets_coll["ytarget"].eta), bins=b)
    ret[f"{sample_folder}/jets_eta_target_pumask"] = to_bh(ak.flatten(jets_coll["ytarget_nopu"].eta), bins=b)
    ret[f"{sample_folder}/jets_eta_cand"] = to_bh(ak.flatten(jets_coll["ycand"].eta), bins=b)

    # jet pt ratio
    b = np.linspace(0, 5, 1000)
    # ratio = ak.flatten((jets_coll["cp"][genjet_to_cp["cp"]].pt / jets_coll["genjet"][genjet_to_cp["genjet"]].pt))
    # ret[f"{sample_folder}/jets_pt_ratio_caloparticle"] = to_bh(ratio, bins=b)
    ratio = ak.flatten((jets_coll["ytarget"][genjet_to_ytarget["ytarget"]].pt / jets_coll["genjet"][genjet_to_ytarget["genjet"]].pt))
    ret[f"{sample_folder}/jets_pt_ratio_target"] = to_bh(ratio, bins=b)
    ratio = ak.flatten(
        (jets_coll["ytarget_nopu"][genjet_to_ytarget_nopu["ytarget_nopu"]].pt / jets_coll["genjet"][genjet_to_ytarget_nopu["genjet"]].pt)
    )
    ret[f"{sample_folder}/jets_pt_ratio_target_pumask"] = to_bh(ratio, bins=b)
    ratio = ak.flatten((jets_coll["ycand"][genjet_to_ycand["ycand"]].pt / jets_coll["genjet"][genjet_to_ycand["genjet"]].pt))
    ret[f"{sample_folder}/jets_pt_ratio_cand"] = to_bh(ratio, bins=b)

    b = np.linspace(0.5, 1.5, 1000)
    # ratio = ak.flatten((jets_coll["cp"][genjet_to_cp["cp"]].pt / jets_coll["genjet"][genjet_to_cp["genjet"]].pt))
    # ret[f"{sample_folder}/jets_pt_ratio2_caloparticle"] = to_bh(ratio, bins=b)
    ratio = ak.flatten((jets_coll["ytarget"][genjet_to_ytarget["ytarget"]].pt / jets_coll["genjet"][genjet_to_ytarget["genjet"]].pt))
    ret[f"{sample_folder}/jets_pt_ratio2_target"] = to_bh(ratio, bins=b)
    ratio = ak.flatten(
        (jets_coll["ytarget_nopu"][genjet_to_ytarget_nopu["ytarget_nopu"]].pt / jets_coll["genjet"][genjet_to_ytarget_nopu["genjet"]].pt)
    )
    ret[f"{sample_folder}/jets_pt_ratio2_target_pumask"] = to_bh(ratio, bins=b)
    ratio = ak.flatten((jets_coll["ycand"][genjet_to_ycand["ycand"]].pt / jets_coll["genjet"][genjet_to_ycand["genjet"]].pt))
    ret[f"{sample_folder}/jets_pt_ratio2_cand"] = to_bh(ratio, bins=b)

    b = np.logspace(-1, 3, 100)
    ret[f"{sample_folder}/met_pythia"] = to_bh(genmet_cmssw, bins=b)
    ret[f"{sample_folder}/met_target"] = to_bh(ytarget_met, bins=b)
    ret[f"{sample_folder}/met_target_pumask"] = to_bh(ytarget_nopu_met, bins=b)
    ret[f"{sample_folder}/met_cand"] = to_bh(ycand_met, bins=b)

    # print output
    # for k in sorted(ret.keys()):
    #     print(k, ret[k].__class__.__name__)

    # save output
    with open(outfile, "wb") as handle:
        pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    perjob = 50
    numjobs = 16
    is_test = False

    args = []
    ijob = 0

    # process pkl and ROOT files together
    maxfiles = 500
    path = "/local/joosep/mlpf/cms/"
    for pu_config in ["nopu", "pu55to75"]:
        for sample_folder in ["QCDForPF_14TeV_TuneCUETP8M1_cfi", "TTbar_14TeV_TuneCUETP8M1_cfi", "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi"]:
            rootfiles = sorted(glob.glob(f"{path}/20240823_simcluster/{pu_config}/{sample_folder}/root/pfntuple_*.root"))
            pklfiles = sorted(glob.glob(f"{path}/20240823_simcluster/{pu_config}/{sample_folder}/raw/pfntuple_*.pkl.bz2"))

            rootfiles_d = {fn.split("/")[-1].split(".")[0]: fn for fn in rootfiles}
            pklfiles_d = {fn.split("/")[-1].split(".")[0]: fn for fn in pklfiles}

            # find the set of common filenames betweek the root and pkl files
            common_keys = sorted(list(set(set(rootfiles_d.keys()).intersection(set(pklfiles_d.keys())))))[:maxfiles]

            # prepare chunked arguments for process_files
            for ck in chunks(common_keys, perjob):
                args.append(
                    (f"combined/{pu_config}/{sample_folder}", [rootfiles_d[c] for c in ck], [pklfiles_d[c] for c in ck], "out{}.pkl".format(ijob))
                )
                ijob += 1

    # process only pkl files
    maxfiles = -1
    path = "/scratch/persistent/joosep/"
    for pu_config in ["nopu", "pu55to75"]:
        for sample_folder in ["QCDForPF_14TeV_TuneCUETP8M1_cfi", "TTbar_14TeV_TuneCUETP8M1_cfi", "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi"]:
            pklfiles = sorted(glob.glob(f"{path}/20240823_simcluster/{pu_config}/{sample_folder}/raw/pfntuple_*.pkl.bz2"))[:maxfiles]
            for ck in chunks(pklfiles, perjob):
                args.append((f"{pu_config}/{sample_folder}", [], ck, "out{}.pkl".format(ijob)))
                ijob += 1

    if is_test:
        process_files(*args[0])
    else:
        import multiprocessing

        pool = multiprocessing.Pool(numjobs)
        pool.starmap(process_files, tqdm.tqdm(args, total=len(args)))
        pool.close()
