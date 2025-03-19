import numpy as np
import awkward
import uproot
import vector
import glob
import os
import multiprocessing
import tqdm
import argparse
from scipy.sparse import coo_matrix

from postprocessing import map_pdgid_to_candid, map_charged_to_neutral, map_neutral_to_charged, sanitize

from postprocessing import track_coll, mc_coll, particle_feature_order, track_feature_order, compute_met, compute_jets

from postprocessing import (
    get_genparticles_and_adjacencies,
    assign_to_recoobj,
    get_reco_properties,
    get_particle_feature_matrix,
    get_feature_matrix,
)

hit_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
]


def build_dummy_array(num, dtype=np.int64):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            awkward.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def assign_genparticles_to_obj(gpdata):

    n_gp = awkward.count(gpdata.gen_features["PDG"])
    n_track = awkward.count(gpdata.track_features["type"])
    n_hit = awkward.count(gpdata.hit_features["type"])

    gp_to_track = np.array(
        coo_matrix(
            (gpdata.genparticle_to_track[2], (gpdata.genparticle_to_track[0], gpdata.genparticle_to_track[1])),
            shape=(n_gp, n_track),
        ).todense()
    )

    gp_to_calohit = np.array(
        coo_matrix((gpdata.genparticle_to_hit[2], (gpdata.genparticle_to_hit[0], gpdata.genparticle_to_hit[1])), shape=(n_gp, n_hit)).todense()
    )

    # map each genparticle to a track or calohit
    gp_to_obj = -1 * np.ones((n_gp, 2), dtype=np.int32)
    set_used_tracks = set([])
    set_used_calohits = set([])
    gps_sorted_energy = sorted(range(n_gp), key=lambda x: gpdata.gen_features["energy"][x], reverse=True)

    for igp in gps_sorted_energy:

        # first check if we can match the genparticle to a track
        matched_tracks = gp_to_track[igp]
        trks = np.where(matched_tracks)[0]
        trks = sorted(trks, key=lambda x: matched_tracks[x], reverse=True)
        for trk in trks:
            # if the track was not already used for something else
            if trk not in set_used_tracks:
                gp_to_obj[igp, 0] = trk
                set_used_tracks.add(trk)
                break

        # if there was no matched track, try a calohit
        if gp_to_obj[igp, 0] == -1:
            matched_calohits = np.where(gp_to_calohit[igp])[0]
            calohits = sorted(matched_calohits, key=lambda x: gp_to_calohit[igp, x], reverse=True)
            for calohit in calohits:
                if calohit not in set_used_calohits:
                    gp_to_obj[igp, 1] = calohit
                    set_used_calohits.add(calohit)
                    break

    # unmatched = (gp_to_obj[:, 0] != -1) & (gp_to_obj[:, 1] != -1)
    return gp_to_obj, gp_to_track, gp_to_calohit


def get_recoptcl_to_obj(n_rps, reco_arr, gpdata, idx_rp_to_track, idx_rp_to_cluster):
    track_to_rp = {}
    calohit_to_rp = {}
    for irp in range(n_rps):
        assigned = False

        # get the tracks of the reco particle
        trks_begin = reco_arr["tracks_begin"][irp]
        trks_end = reco_arr["tracks_end"][irp]
        for itrk in range(trks_begin, trks_end):

            # get the index of the track
            itrk_real = idx_rp_to_track[itrk]
            assert itrk_real not in track_to_rp
            track_to_rp[itrk_real] = irp
            assigned = True

        # only look for calohits if tracks were not found
        if not assigned:

            # loop over clusters of the reco particle
            cls_begin = reco_arr["clusters_begin"][irp]
            cls_end = reco_arr["clusters_end"][irp]
            for icls in range(cls_begin, cls_end):

                # get the index of the cluster
                icls_real = idx_rp_to_cluster[icls]

                # find hits of the cluster
                calohit_inds = gpdata.hit_to_cluster[0][gpdata.hit_to_cluster[1] == icls_real]

                # get the highest-energy hit
                calohits_e_ascending = np.argsort(gpdata.hit_features["energy"][calohit_inds])
                highest_e_hit = calohit_inds[calohits_e_ascending[-1]]
                assert highest_e_hit not in calohit_to_rp
                calohit_to_rp[highest_e_hit] = irp
                assigned = True
                break
    return track_to_rp, calohit_to_rp


# permute rows of the track/hit association matrix to order the gps as in tfds format
def permute_association_matrix(old_mat, used_gps):
    i = 0
    temp_mat = list()
    new_mat = np.zeros((old_mat.shape))
    for used_gps_idx in range(len(used_gps)):
        if used_gps[used_gps_idx] == 1:
            new_mat[i] = old_mat[used_gps_idx]
            i += 1
        else:
            temp_mat.append(old_mat[used_gps_idx])
    new_mat[i:, :] = np.array(temp_mat)
    return new_mat


def process_one_file(fn, ofn, dataset, store_matrix=True):

    # output exists, do not recreate
    if os.path.isfile(ofn):
        return
    print(fn)

    fi = uproot.open(fn)

    arrs = fi["events"]

    if dataset == "clic":
        collectionIDs = {
            k: v
            for k, v in zip(
                fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_names"][0],
                fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_collectionIDs"][0],
            )
        }

        prop_data = arrs.arrays(
            [
                "MCParticles.PDG",
                "MCParticles.momentum.x",
                "MCParticles.momentum.y",
                "MCParticles.momentum.z",
                "MCParticles.mass",
                "MCParticles.charge",
                "MCParticles.generatorStatus",
                "MCParticles.simulatorStatus",
                "MCParticles.daughters_begin",
                "MCParticles.daughters_end",
                "MCParticles#1.index",
                track_coll,
                "SiTracks_1",
                "PandoraClusters",
                "PandoraClusters#1",
                "PandoraClusters#0",
                "MergedRecoParticles",
            ]
        )

        calohit_links = arrs.arrays(
            [
                "CalohitMCTruthLink.weight",
                "CalohitMCTruthLink#0.index",
                "CalohitMCTruthLink#0.collectionID",
                "CalohitMCTruthLink#1.index",
                "CalohitMCTruthLink#1.collectionID",
            ]
        )
        sitrack_links = arrs.arrays(
            [
                "SiTracksMCTruthLink.weight",
                "SiTracksMCTruthLink#0.index",
                "SiTracksMCTruthLink#0.collectionID",
                "SiTracksMCTruthLink#1.index",
                "SiTracksMCTruthLink#1.collectionID",
            ]
        )

        # maps the recoparticle track/cluster index (in tracks_begin,end and clusters_begin,end)
        # to the index in the track/cluster collection
        idx_rp_to_cluster = arrs["MergedRecoParticles#0/MergedRecoParticles#0.index"].array()
        idx_rp_to_track = arrs["MergedRecoParticles#1/MergedRecoParticles#1.index"].array()

        hit_data = {
            "ECALBarrel": arrs["ECALBarrel"].array(),
            "ECALEndcap": arrs["ECALEndcap"].array(),
            "ECALOther": arrs["ECALOther"].array(),
            "HCALBarrel": arrs["HCALBarrel"].array(),
            "HCALEndcap": arrs["HCALEndcap"].array(),
            "HCALOther": arrs["HCALOther"].array(),
            "MUON": arrs["MUON"].array(),
        }
    elif dataset == "fcc":
        collectionIDs = {
            k: v
            for k, v in zip(
                fi.get("podio_metadata").arrays("events___idTable/m_names")["events___idTable/m_names"][0],
                fi.get("podio_metadata").arrays("events___idTable/m_collectionIDs")["events___idTable/m_collectionIDs"][0],
            )
        }
        prop_data = arrs.arrays(
            [
                mc_coll,
                "MCParticles.PDG",
                "MCParticles.momentum.x",
                "MCParticles.momentum.y",
                "MCParticles.momentum.z",
                "MCParticles.mass",
                "MCParticles.charge",
                "MCParticles.generatorStatus",
                "MCParticles.simulatorStatus",
                "MCParticles.daughters_begin",
                "MCParticles.daughters_end",
                "_MCParticles_daughters/_MCParticles_daughters.index",  # similar to "MCParticles#1.index" in clic
                track_coll,
                "_SiTracks_Refitted_trackStates",
                "PandoraClusters",
                "_PandoraClusters_hits/_PandoraClusters_hits.index",
                "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
                "PandoraPFOs",
                "SiTracks_Refitted_dQdx",
            ]
        )
        calohit_links = arrs.arrays(
            [
                "CalohitMCTruthLink.weight",
                "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID",
                "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index",
                "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID",
                "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index",
            ]
        )
        sitrack_links = arrs.arrays(
            [
                "SiTracksMCTruthLink.weight",
                "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.collectionID",
                "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index",
                "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.collectionID",
                "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index",
            ]
        )

        # maps the recoparticle track/cluster index (in tracks_begin,end and clusters_begin,end)
        # to the index in the track/cluster collection
        idx_rp_to_cluster = arrs["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
        idx_rp_to_track = arrs["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()

        hit_data = {
            "ECALBarrel": arrs["ECALBarrel"].array(),
            "ECALEndcap": arrs["ECALEndcap"].array(),
            "HCALBarrel": arrs["HCALBarrel"].array(),
            "HCALEndcap": arrs["HCALEndcap"].array(),
            "HCALOther": arrs["HCALOther"].array(),
            "MUON": arrs["MUON"].array(),
        }
    else:
        raise Exception("--dataset provided is not supported. Only 'fcc' or 'clic' are supported atm.")

    # Compute truth MET and jets from status=1 pythia particles
    mc_pdg = np.abs(prop_data["MCParticles.PDG"])
    mc_st1_mask = (prop_data["MCParticles.generatorStatus"] == 1) & (mc_pdg != 12) & (mc_pdg != 14) & (mc_pdg != 16)
    mc_st1_p4 = vector.awk(
        awkward.zip(
            {
                "px": prop_data["MCParticles.momentum.x"][mc_st1_mask],
                "py": prop_data["MCParticles.momentum.y"][mc_st1_mask],
                "pz": prop_data["MCParticles.momentum.z"][mc_st1_mask],
                "mass": prop_data["MCParticles.mass"][mc_st1_mask],
            }
        )
    )

    met_st1 = compute_met(mc_st1_p4)
    genjets_st1 = compute_jets(mc_st1_p4)

    ret = []
    for iev in tqdm.tqdm(range(arrs.num_entries), total=arrs.num_entries):

        # get the reco particles
        reco_arr = get_reco_properties(dataset, prop_data, iev)

        if dataset == "clic":
            reco_type = np.abs(reco_arr["type"])
        elif dataset == "fcc":
            reco_type = np.abs(reco_arr["PDG"])
        else:
            raise Exception("--dataset provided is not supported. Only 'fcc' or 'clic' are supported atm.")

        n_rps = len(reco_type)
        reco_features = awkward.Record(
            {
                "PDG": np.abs(reco_type),
                "charge": reco_arr["charge"],
                "pt": reco_arr["pt"],
                "eta": reco_arr["eta"],
                "sin_phi": np.sin(reco_arr["phi"]),
                "cos_phi": np.cos(reco_arr["phi"]),
                "energy": reco_arr["energy"],
                "ispu": np.zeros(len(reco_type)),
                "generatorStatus": np.zeros(len(reco_type)),
                "simulatorStatus": np.zeros(len(reco_type)),
                "gp_to_track": np.zeros(len(reco_type)),
                "gp_to_cluster": np.zeros(len(reco_type)),
                "jet_idx": np.zeros(len(reco_type)),
            }
        )

        # get the genparticles and the links between genparticles and tracks/clusters
        gpdata = get_genparticles_and_adjacencies(dataset, prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs)

        # find the reconstructable genparticles and associate them to the best track/cluster
        gp_to_obj, gp_to_track, gp_to_calohit = assign_genparticles_to_obj(gpdata)

        n_tracks = len(gpdata.track_features["type"])
        n_hits = len(gpdata.hit_features["type"])
        n_gps = len(gpdata.gen_features["PDG"])
        # print("hits={} tracks={} gps={}".format(n_hits, n_tracks, n_gps))

        assert len(gp_to_obj) == len(gpdata.gen_features["PDG"])
        assert gp_to_obj.shape[1] == 2

        # for each reco particle, find the tracks and clusters associated with it
        # construct track/cluster -> recoparticle maps
        track_to_rp, hit_to_rp = get_recoptcl_to_obj(n_rps, reco_arr, gpdata, idx_rp_to_track[iev], idx_rp_to_cluster[iev])

        # get the track/cluster -> genparticle map
        track_to_gp = {itrk: igp for igp, itrk in enumerate(gp_to_obj[:, 0]) if itrk != -1}
        hit_to_gp = {ihit: igp for igp, ihit in enumerate(gp_to_obj[:, 1]) if ihit != -1}

        # keep track if all genparticles were used
        used_gps = np.zeros(n_gps, dtype=np.int64)

        # assign all track-associated genparticles to a track
        track_to_gp_all = assign_to_recoobj(n_tracks, track_to_gp, used_gps)

        gp_to_track = permute_association_matrix(gp_to_track, used_gps)
        gp_to_calohit = permute_association_matrix(gp_to_calohit, used_gps)

        # assign all calohit-associated genparticles to a calohit
        hit_to_gp_all = assign_to_recoobj(n_hits, hit_to_gp, used_gps)
        if not np.all(used_gps == 1):
            print("unmatched gen", gpdata.gen_features["energy"][used_gps == 0])

        used_rps = np.zeros(n_rps, dtype=np.int64)
        track_to_rp_all = assign_to_recoobj(n_tracks, track_to_rp, used_rps)
        hit_to_rp_all = assign_to_recoobj(n_hits, hit_to_rp, used_rps)
        if not np.all(used_rps == 1):
            print("unmatched reco", reco_features["energy"][used_rps == 0])

        gps_track = get_particle_feature_matrix(track_to_gp_all, gpdata.gen_features, particle_feature_order)
        gps_track[:, 0] = np.array([map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(gps_track[:, 0], gps_track[:, 1])])
        gps_hit = get_particle_feature_matrix(hit_to_gp_all, gpdata.gen_features, particle_feature_order)
        gps_hit[:, 0] = np.array([map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(gps_hit[:, 0], gps_hit[:, 1])])
        gps_hit[:, 1] = 0

        rps_track = get_particle_feature_matrix(track_to_rp_all, reco_features, particle_feature_order)
        rps_track[:, 0] = np.array([map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(rps_track[:, 0], rps_track[:, 1])])
        rps_hit = get_particle_feature_matrix(hit_to_rp_all, reco_features, particle_feature_order)
        rps_hit[:, 0] = np.array([map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(rps_hit[:, 0], rps_hit[:, 1])])
        rps_hit[:, 1] = 0

        # we don't want to try to reconstruct charged particles from primary clusters, make sure the charge is 0
        assert np.all(gps_hit[:, 1] == 0)
        assert np.all(rps_hit[:, 1] == 0)

        X_track = get_feature_matrix(gpdata.track_features, track_feature_order)
        X_hit = get_feature_matrix(gpdata.hit_features, hit_feature_order)
        ytarget_track = gps_track
        ytarget_hit = gps_hit
        ycand_track = rps_track
        ycand_hit = rps_hit

        sanitize(X_track)
        sanitize(X_hit)
        sanitize(ytarget_track)
        sanitize(ytarget_hit)
        sanitize(ycand_track)
        sanitize(ycand_hit)

        # cluster target particles to jets, save per-particle jet index
        ytarget = np.concatenate([ytarget_track, ytarget_hit], axis=0)
        ytarget_constituents = -1 * np.ones(len(ytarget), dtype=np.int64)
        valid = ytarget[:, 0] != 0
        # save mapping of index after masking -> index before masking as numpy array
        # inspired from:
        # https://stackoverflow.com/questions/432112/1044443#comment54747416_1044443
        cumsum = np.cumsum(valid) - 1
        _, index_mapping = np.unique(cumsum, return_index=True)
        ytarget = ytarget[valid]
        ytarget_p4 = ytarget[:, 2:7]
        ytarget_p4 = vector.awk(
            awkward.zip(
                {
                    "pt": ytarget_p4[:, 0],
                    "eta": ytarget_p4[:, 1],
                    "phi": np.arctan2(ytarget_p4[:, 2], ytarget_p4[:, 3]),
                    "energy": ytarget_p4[:, 4],
                }
            )
        )
        target_jets, target_jets_indices = compute_jets(ytarget_p4, with_indices=True)
        sorted_jet_idx = awkward.argsort(target_jets.pt, axis=-1, ascending=False).to_list()
        target_jets_indices = target_jets_indices.to_list()
        for jet_idx in sorted_jet_idx:
            jet_constituents = [index_mapping[idx] for idx in target_jets_indices[jet_idx]]  # map back to constituent index *before* masking
            ytarget_constituents[jet_constituents] = jet_idx
        ytarget_track_constituents = ytarget_constituents[: len(ytarget_track)]
        ytarget_cluster_constituents = ytarget_constituents[len(ytarget_track) :]
        ytarget_track[:, particle_feature_order.index("jet_idx")] = ytarget_track_constituents
        ytarget_hit[:, particle_feature_order.index("jet_idx")] = ytarget_cluster_constituents

        if store_matrix:
            this_ev = {
                "X_track": X_track,
                "X_hit": X_hit,
                "ytarget_track": ytarget_track,
                "ytarget_hit": ytarget_hit,
                "ycand_track": ycand_track,
                "ycand_hit": ycand_hit,
                "genjet": get_feature_matrix(genjets_st1[iev], ["pt", "eta", "phi", "energy"]),
                "genmet": met_st1[iev],
                "targetjet": get_feature_matrix(target_jets, ["pt", "eta", "phi", "energy"]),
                "gp_to_track": gp_to_track,
                "gp_to_calohit": gp_to_calohit,
            }
        else:
            this_ev = {
                "X_track": X_track,
                "X_hit": X_hit,
                "ytarget_track": ytarget_track,
                "ytarget_hit": ytarget_hit,
                "ycand_track": ycand_track,
                "ycand_hit": ycand_hit,
                "genjet": get_feature_matrix(genjets_st1[iev], ["pt", "eta", "phi", "energy"]),
                "genmet": met_st1[iev],
                "targetjet": get_feature_matrix(target_jets, ["pt", "eta", "phi", "energy"]),
                "gp_to_track": None,
                "gp_to_calohit": None,
            }
        this_ev = awkward.Record(this_ev)
        ret.append(this_ev)

    ret = awkward.Record({k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields})
    if not store_matrix:
        ret["gp_to_track"] = None
        ret["gp_to_calohit"] = None
    awkward.to_parquet(ret, ofn)


def process_sample(samp, config):
    inp = "/local/joosep/clic_edm4hep/"
    outp = "/local/joosep/mlpf_hits/clic_edm4hep/"

    pool = multiprocessing.Pool(8)

    inpath_samp = inp + samp
    outpath_samp = outp + samp
    infiles = list(glob.glob(inpath_samp + "/*.root"))
    if not os.path.isdir(outpath_samp):
        os.makedirs(outpath_samp)

    args = []
    for inf in infiles:
        of = inf.replace(inpath_samp, outpath_samp).replace(".root", ".parquet")
        args.append((inf, of, config.dataset, config.store_matrix))
    pool.starmap(process_one_file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, default=None, help="input file (root)")
    parser.add_argument("--ofn", type=str, default=None, help="output file (parquet)")
    parser.add_argument("--samples", type=str, default=None, help="sample name to specify many files")
    parser.add_argument("--dataset", type=str, help="Which detector dataset?", required=True, choices=["clic", "fcc"])

    parser.add_argument("--store-matrix", action="store_true", help="store track and hit association matrices")

    args = parser.parse_args()

    if args.samples is not None:
        process_sample(args.samples, args)
    else:
        if os.path.isdir(args.fn) is True:
            print("Will process all files in " + args.fn)

            flist = glob.glob(args.fn + "/*.root")
            for infile in flist:
                outfile = os.path.join(args.ofn, os.path.basename(infile).split(".")[0] + ".parquet")
                process_one_file(infile, outfile, args.dataset, args.store_matrix)
        else:
            infile = args.fn
            outfile = os.path.join(args.ofn, os.path.basename(infile).split(".")[0] + ".parquet")
            process_one_file(infile, outfile, args.dataset, args.store_matrix)
