import numpy as np
import awkward
import uproot
import glob
import os
import sys
import multiprocessing
import tqdm
from scipy.sparse import coo_matrix

from postprocessing import map_pdgid_to_candid, map_charged_to_neutral, map_neutral_to_charged, sanitize

from postprocessing import track_coll, mc_coll, particle_feature_order

track_feature_order = [
    "elemtype",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "p",
    "chi2",
    "ndf",
    "dEdx",
    "dEdxError",
    "radiusOfInnermostHit",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
]
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

from postprocessing import (
    get_genparticles_and_adjacencies,
    assign_to_recoobj,
    get_reco_properties,
    get_particle_feature_matrix,
    get_feature_matrix,
)


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
    return gp_to_obj


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

def process_one_file(fn, ofn):

    # output exists, do not recreate
    if os.path.isfile(ofn):
        return
    print(fn)

    fi = uproot.open(fn)

    arrs = fi["events"]

    collectionIDs = {
        k: v
        for k, v in zip(
            fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_names"][0],
            fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_collectionIDs"][0],
        )
    }

    prop_data = arrs.arrays(
        [
            mc_coll,
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

    ret = []
    for iev in tqdm.tqdm(range(arrs.num_entries), total=arrs.num_entries):

        # get the reco particles
        reco_arr = get_reco_properties(prop_data, iev)
        reco_type = np.abs(reco_arr["type"])
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
            }
        )

        # get the genparticles and the links between genparticles and tracks/clusters
        gpdata = get_genparticles_and_adjacencies(prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs)

        # find the reconstructable genparticles and associate them to the best track/cluster
        gp_to_obj = assign_genparticles_to_obj(gpdata)

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
        ygen_track = gps_track
        ygen_hit = gps_hit
        ycand_track = rps_track
        ycand_hit = rps_hit

        sanitize(X_track)
        sanitize(X_hit)
        sanitize(ygen_track)
        sanitize(ygen_hit)
        sanitize(ycand_track)
        sanitize(ycand_hit)

        this_ev = {
            "X_track": X_track,
            "X_hit": X_hit,
            "ygen_track": ygen_track,
            "ygen_hit": ygen_hit,
            "ycand_track": ycand_track,
            "ycand_hit": ycand_hit,
        }
        this_ev = awkward.Record(this_ev)
        ret.append(this_ev)

    ret = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}
    for k in ret.keys():
        if len(awkward.flatten(ret[k])) == 0:
            ret[k] = build_dummy_array(len(ret[k]), np.float32)
    ret = awkward.Record(ret)
    awkward.to_parquet(ret, ofn)


def process_sample(samp):
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
        args.append((inf, of))
    pool.starmap(process_one_file, args)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        process_sample(sys.argv[1])
    if len(sys.argv) == 3:
        process_one_file(sys.argv[1], sys.argv[2])
