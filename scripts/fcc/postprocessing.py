import numpy as np
import awkward
import uproot
import vector
import glob
import os
import sys
import multiprocessing
from scipy.sparse import coo_matrix

track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

# the feature matrices will be saved in this order
particle_feature_order = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]

# arrange track and cluster features such that pt (et), eta, phi, p (energy) are in the same spot
# so we can easily use them in skip connections
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
cluster_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "iTheta",
    "energy_ecal",
    "energy_hcal",
    "energy_other",
    "num_hits",
    "sigma_x",
    "sigma_y",
    "sigma_z",
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


def track_pt(omega):
    a = 3 * 10**-4
    b = 4  # B-field in tesla, from clicRec_e4h_input

    return a * np.abs(b / omega)


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    # photon, electron, muon
    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def map_charged_to_neutral(pdg):
    if pdg == 0:
        return 0
    if pdg == 11 or pdg == 22:
        return 22
    return 130


def map_neutral_to_charged(pdg):
    if pdg == 130 or pdg == 22:
        return 211
    return pdg


def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


class EventData:
    def __init__(
        self,
        gen_features,
        hit_features,
        cluster_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster,
        gp_merges,
    ):
        self.gen_features = gen_features  # feature matrix of the genparticles
        self.hit_features = hit_features  # feature matrix of the calo hits
        self.cluster_features = cluster_features  # feature matrix of the calo clusters
        self.track_features = track_features  # feature matrix of the tracks
        self.genparticle_to_hit = genparticle_to_hit  # sparse COO matrix of genparticles to hits (idx_gp, idx_hit, weight)
        self.genparticle_to_track = (
            genparticle_to_track  # sparse COO matrix of genparticles to tracks (idx_gp, idx_track, weight)
        )
        self.hit_to_cluster = hit_to_cluster  # sparse COO matrix of hits to clusters (idx_hit, idx_cluster, weight)
        self.gp_merges = gp_merges  # sparse COO matrix of any merged genparticles

        self.genparticle_to_hit = (
            np.array(self.genparticle_to_hit[0]),
            np.array(self.genparticle_to_hit[1]),
            np.array(self.genparticle_to_hit[2]),
        )
        self.genparticle_to_track = (
            np.array(self.genparticle_to_track[0]),
            np.array(self.genparticle_to_track[1]),
            np.array(self.genparticle_to_track[2]),
        )
        self.hit_to_cluster = (
            np.array(self.hit_to_cluster[0]),
            np.array(self.hit_to_cluster[1]),
            np.array(self.hit_to_cluster[2]),
        )
        self.gp_merges = np.array(self.gp_merges[0]), np.array(self.gp_merges[1])


def hits_to_features(hit_data, iev, coll, feats):
    feat_arr = {f: hit_data[coll + "." + f][iev] for f in feats}

    # set the subdetector type
    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(len(feat_arr["type"]), dtype=np.int32)
    if coll.startswith("ECAL"):
        feat_arr[sdcoll][:] = 0
    elif coll.startswith("HCAL"):
        feat_arr[sdcoll][:] = 1
    else:
        feat_arr[sdcoll][:] = 2

    # hit elemtype is always 2
    feat_arr["elemtype"] = 2 * np.ones(len(feat_arr["type"]), dtype=np.int32)

    # precompute some approximate et, eta, phi
    pos_mag = np.sqrt(feat_arr["position.x"] ** 2 + feat_arr["position.y"] ** 2 + feat_arr["position.z"] ** 2)
    px = (feat_arr["position.x"] / pos_mag) * feat_arr["energy"]
    py = (feat_arr["position.y"] / pos_mag) * feat_arr["energy"]
    pz = (feat_arr["position.z"] / pos_mag) * feat_arr["energy"]
    feat_arr["et"] = np.sqrt(px**2 + py**2)
    feat_arr["eta"] = 0.5 * np.log((feat_arr["energy"] + pz) / (feat_arr["energy"] - pz))
    feat_arr["sin_phi"] = py / feat_arr["energy"]
    feat_arr["cos_phi"] = px / feat_arr["energy"]

    return awkward.Record(feat_arr)


def get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs):
    feats = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]

    hit_idx_global = 0
    hit_idx_global_to_local = {}
    hit_feature_matrix = []
    for col in sorted(hit_data.keys()):
        icol = collectionIDs[col]
        hit_features = hits_to_features(hit_data[col], iev, col, feats)
        hit_feature_matrix.append(hit_features)
        for ihit in range(len(hit_data[col][col + ".energy"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1
    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = awkward.Record(
        {
            k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))])
            for k in hit_feature_matrix[0].fields
        }
    )

    # add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links["CalohitMCTruthLink"]["CalohitMCTruthLink.weight"][iev]
    calohit_to_gen_calo_colid = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.index"][iev]
    calohit_to_gen_gen_idx = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.index"][iev]
    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, w in zip(
        calohit_to_gen_calo_colid,
        calohit_to_gen_calo_idx,
        calohit_to_gen_gen_colid,
        calohit_to_gen_gen_idx,
        calohit_to_gen_weight,
    ):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(w)

    return (
        hit_feature_matrix,
        (genparticle_to_hit_matrix_coo0, genparticle_to_hit_matrix_coo1, genparticle_to_hit_matrix_w),
        hit_idx_local_to_global,
    )


def hit_cluster_adj(prop_data, hit_idx_local_to_global, iev):
    coll_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.collectionID"][iev]
    idx_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.index"][iev]
    hits_begin = prop_data["PandoraClusters"]["PandoraClusters.hits_begin"][iev]
    hits_end = prop_data["PandoraClusters"]["PandoraClusters.hits_end"][iev]

    # index in the array of all hits
    hit_to_cluster_matrix_coo0 = []
    # index in the cluster array
    hit_to_cluster_matrix_coo1 = []

    # weight
    hit_to_cluster_matrix_w = []

    # loop over all clusters
    for icluster in range(len(hits_begin)):

        # get the slice in the hit array corresponding to this cluster
        hbeg = hits_begin[icluster]
        hend = hits_end[icluster]
        idx_range = idx_arr[hbeg:hend]
        coll_range = coll_arr[hbeg:hend]

        # add edges from hit to cluster
        for icol, idx in zip(coll_range, idx_range):
            hit_to_cluster_matrix_coo0.append(hit_idx_local_to_global[(icol, idx)])
            hit_to_cluster_matrix_coo1.append(icluster)
            hit_to_cluster_matrix_w.append(1.0)
    return hit_to_cluster_matrix_coo0, hit_to_cluster_matrix_coo1, hit_to_cluster_matrix_w


def gen_to_features(prop_data, iev):
    gen_arr = prop_data[mc_coll][iev]
    gen_arr = {k.replace(mc_coll + ".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(
        awkward.zip(
            {"mass": gen_arr["mass"], "x": gen_arr["momentum.x"], "y": gen_arr["momentum.y"], "z": gen_arr["momentum.z"]}
        )
    )
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy
    gen_arr["sin_phi"] = np.sin(gen_arr["phi"])
    gen_arr["cos_phi"] = np.cos(gen_arr["phi"])

    return awkward.Record(
        {
            "PDG": gen_arr["PDG"],
            "generatorStatus": gen_arr["generatorStatus"],
            "charge": gen_arr["charge"],
            "pt": gen_arr["pt"],
            "eta": gen_arr["eta"],
            "phi": gen_arr["phi"],
            "sin_phi": gen_arr["sin_phi"],
            "cos_phi": gen_arr["cos_phi"],
            "energy": gen_arr["energy"],
        }
    )


def genparticle_track_adj(sitrack_links, iev):
    trk_to_gen_trkidx = sitrack_links["SiTracksMCTruthLink#0"]["SiTracksMCTruthLink#0.index"][iev]
    trk_to_gen_genidx = sitrack_links["SiTracksMCTruthLink#1"]["SiTracksMCTruthLink#1.index"][iev]
    trk_to_gen_w = sitrack_links["SiTracksMCTruthLink"]["SiTracksMCTruthLink.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)

    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w


def cluster_to_features(prop_data, hit_features, hit_to_cluster, iev):
    cluster_arr = prop_data["PandoraClusters"][iev]
    feats = ["type", "position.x", "position.y", "position.z", "iTheta", "phi", "energy"]
    ret = {feat: cluster_arr["PandoraClusters." + feat] for feat in feats}

    hit_idx = np.array(hit_to_cluster[0])
    cluster_idx = np.array(hit_to_cluster[1])
    cl_energy_ecal = []
    cl_energy_hcal = []
    cl_energy_other = []
    num_hits = []
    cl_sigma_x = []
    cl_sigma_y = []
    cl_sigma_z = []

    n_cl = len(ret["energy"])
    for cl in range(n_cl):
        msk_cl = cluster_idx == cl
        hits = hit_idx[msk_cl]

        num_hits.append(len(hits))

        subdets = hit_features["subdetector"][hits]

        hits_energy = hit_features["energy"][hits]

        hits_posx = hit_features["position.x"][hits]
        hits_posy = hit_features["position.y"][hits]
        hits_posz = hit_features["position.z"][hits]

        energy_ecal = np.sum(hits_energy[subdets == 0])
        energy_hcal = np.sum(hits_energy[subdets == 1])
        energy_other = np.sum(hits_energy[subdets == 2])

        cl_energy_ecal.append(energy_ecal)
        cl_energy_hcal.append(energy_hcal)
        cl_energy_other.append(energy_other)

        cl_sigma_x.append(np.std(hits_posx))
        cl_sigma_y.append(np.std(hits_posy))
        cl_sigma_z.append(np.std(hits_posz))

    ret["energy_ecal"] = np.array(cl_energy_ecal)
    ret["energy_hcal"] = np.array(cl_energy_hcal)
    ret["energy_other"] = np.array(cl_energy_other)
    ret["num_hits"] = np.array(num_hits)
    ret["sigma_x"] = np.array(cl_sigma_x)
    ret["sigma_y"] = np.array(cl_sigma_y)
    ret["sigma_z"] = np.array(cl_sigma_z)

    tt = awkward.to_numpy(np.tan(ret["iTheta"] / 2.0))
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    costheta = np.cos(ret["iTheta"])
    ez = ret["energy"] * costheta
    ret["et"] = np.sqrt(ret["energy"] ** 2 - ez**2)

    # cluster is always type 2
    ret["elemtype"] = 2 * np.ones(n_cl, dtype=np.float32)

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    return awkward.Record(ret)


def track_to_features(prop_data, iev):
    track_arr = prop_data[track_coll][iev]
    feats_from_track = ["type", "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit"]
    ret = {feat: track_arr[track_coll + "." + feat] for feat in feats_from_track}
    n_tr = len(ret["type"])

    # get the index of the first track state
    trackstate_idx = prop_data[track_coll][track_coll + ".trackStates_begin"][iev]
    # get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time"]:
        ret[k] = awkward.to_numpy(prop_data["SiTracks_1"]["SiTracks_1." + k][iev][trackstate_idx])

    ret["pt"] = awkward.to_numpy(track_pt(ret["omega"]))
    ret["px"] = awkward.to_numpy(np.cos(ret["phi"])) * ret["pt"]
    ret["py"] = awkward.to_numpy(np.sin(ret["phi"])) * ret["pt"]
    ret["pz"] = awkward.to_numpy(ret["tanLambda"]) * ret["pt"]
    ret["p"] = np.sqrt(ret["px"] ** 2 + ret["py"] ** 2 + ret["pz"] ** 2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"] > 0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    # track is always type 1
    ret["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

    return awkward.Record(ret)


def filter_adj(adj, all_to_filtered):
    i0s_new = []
    i1s_new = []
    ws_new = []
    for i0, i1, w in zip(*adj):
        if i0 in all_to_filtered:
            i0_new = all_to_filtered[i0]
            i0s_new.append(i0_new)
            i1s_new.append(i1)
            ws_new.append(w)
    return np.array(i0s_new), np.array(i1s_new), np.array(ws_new)


def get_genparticles_and_adjacencies(prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs):
    gen_features = gen_to_features(prop_data, iev)
    hit_features, genparticle_to_hit, hit_idx_local_to_global = get_calohit_matrix_and_genadj(
        hit_data, calohit_links, iev, collectionIDs
    )
    hit_to_cluster = hit_cluster_adj(prop_data, hit_idx_local_to_global, iev)
    cluster_features = cluster_to_features(prop_data, hit_features, hit_to_cluster, iev)
    track_features = track_to_features(prop_data, iev)
    genparticle_to_track = genparticle_track_adj(sitrack_links, iev)

    n_gp = awkward.count(gen_features["PDG"])
    n_track = awkward.count(track_features["type"])
    n_hit = awkward.count(hit_features["type"])
    n_cluster = awkward.count(cluster_features["type"])

    if len(genparticle_to_track[0]) > 0:
        gp_to_track = (
            coo_matrix((genparticle_to_track[2], (genparticle_to_track[0], genparticle_to_track[1])), shape=(n_gp, n_track))
            .max(axis=1)
            .todense()
        )
    else:
        gp_to_track = np.zeros((n_gp, 1))

    gp_to_calohit = coo_matrix((genparticle_to_hit[2], (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    calohit_to_cluster = coo_matrix((hit_to_cluster[2], (hit_to_cluster[0], hit_to_cluster[1])), shape=(n_hit, n_cluster))
    gp_to_cluster = (gp_to_calohit * calohit_to_cluster).sum(axis=1)

    # 60% of the hits of a track must come from the genparticle
    gp_in_tracker = np.array(gp_to_track >= 0.6)[:, 0]

    # at least 10% of the energy of the genparticle should be matched to a calorimeter cluster
    gp_in_calo = (np.array(gp_to_cluster)[:, 0] / gen_features["energy"]) > 0.1

    gp_interacted_with_detector = gp_in_tracker | gp_in_calo

    mask_visible = (gen_features["energy"] > 0.01) & gp_interacted_with_detector
    print("gps total={} visible={}".format(n_gp, np.sum(mask_visible)))
    idx_all_masked = np.where(mask_visible)[0]
    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}

    gen_features = awkward.Record({feat: gen_features[feat][mask_visible] for feat in gen_features.fields})

    genparticle_to_hit = filter_adj(genparticle_to_hit, genpart_idx_all_to_filtered)
    genparticle_to_track = filter_adj(genparticle_to_track, genpart_idx_all_to_filtered)

    return EventData(
        gen_features,
        hit_features,
        cluster_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster,
        ([], []),
    )


def assign_genparticles_to_obj_and_merge(gpdata):

    n_gp = awkward.count(gpdata.gen_features["PDG"])
    n_track = awkward.count(gpdata.track_features["type"])
    n_hit = awkward.count(gpdata.hit_features["type"])
    n_cluster = awkward.count(gpdata.cluster_features["type"])

    gp_to_track = np.array(
        coo_matrix(
            (gpdata.genparticle_to_track[2], (gpdata.genparticle_to_track[0], gpdata.genparticle_to_track[1])),
            shape=(n_gp, n_track),
        ).todense()
    )

    gp_to_calohit = coo_matrix(
        (gpdata.genparticle_to_hit[2], (gpdata.genparticle_to_hit[0], gpdata.genparticle_to_hit[1])), shape=(n_gp, n_hit)
    )
    calohit_to_cluster = coo_matrix(
        (gpdata.hit_to_cluster[2], (gpdata.hit_to_cluster[0], gpdata.hit_to_cluster[1])), shape=(n_hit, n_cluster)
    )

    gp_to_cluster = np.array((gp_to_calohit * calohit_to_cluster).todense())

    # map each genparticle to a track or a cluster
    gp_to_obj = -1 * np.ones((n_gp, 2), dtype=np.int32)
    set_used_tracks = set([])
    set_used_clusters = set([])
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

        # if there was no matched track, try a cluster
        if gp_to_obj[igp, 0] == -1:
            matched_clusters = gp_to_cluster[igp]
            clusters = np.where(matched_clusters)[0]
            clusters = sorted(clusters, key=lambda x: matched_clusters[x], reverse=True)
            for cl in clusters:
                if cl not in set_used_clusters:
                    gp_to_obj[igp, 1] = cl
                    set_used_clusters.add(cl)
                    break

    # the genparticles that could not be matched to a track or cluster are merged to the closest genparticle
    unmatched = np.where((gp_to_obj[:, 0] == -1) & (gp_to_obj[:, 1] == -1))[0]
    mask_gp_unmatched = np.ones(n_gp, dtype=bool)

    pt_arr = np.array(awkward.to_numpy(gpdata.gen_features["pt"]))
    eta_arr = np.array(awkward.to_numpy(gpdata.gen_features["eta"]))
    phi_arr = np.array(awkward.to_numpy(gpdata.gen_features["phi"]))
    energy_arr = np.array(awkward.to_numpy(gpdata.gen_features["energy"]))

    # now merge unmatched genparticles to their closest genparticle
    gp_merges_gp0 = []
    gp_merges_gp1 = []
    for igp_unmatched in unmatched:
        mask_gp_unmatched[igp_unmatched] = False
        idx_best_cluster = np.argmax(gp_to_cluster[igp_unmatched])
        idx_gp_bestcluster = np.where(gp_to_obj[:, 1] == idx_best_cluster)[0]

        # if the genparticle is not matched to any cluster, then it left a few hits to some other track
        # this is rare, happens only for low-pT particles and we don't want to try to reconstruct it
        if len(idx_gp_bestcluster) != 1:
            print("unmatched pt=", pt_arr[igp_unmatched])
            continue

        idx_gp_bestcluster = idx_gp_bestcluster[0]

        gp_merges_gp0.append(idx_gp_bestcluster)
        gp_merges_gp1.append(igp_unmatched)

        vec0 = vector.obj(
            pt=gpdata.gen_features["pt"][igp_unmatched],
            eta=gpdata.gen_features["eta"][igp_unmatched],
            phi=gpdata.gen_features["phi"][igp_unmatched],
            e=gpdata.gen_features["energy"][igp_unmatched],
        )
        vec1 = vector.obj(
            pt=gpdata.gen_features["pt"][idx_gp_bestcluster],
            eta=gpdata.gen_features["eta"][idx_gp_bestcluster],
            phi=gpdata.gen_features["phi"][idx_gp_bestcluster],
            e=gpdata.gen_features["energy"][idx_gp_bestcluster],
        )
        vec = vec0 + vec1
        pt_arr[idx_gp_bestcluster] = vec.pt
        eta_arr[idx_gp_bestcluster] = vec.eta
        phi_arr[idx_gp_bestcluster] = vec.phi
        energy_arr[idx_gp_bestcluster] = vec.energy

    gen_features_new = {
        "PDG": np.abs(gpdata.gen_features["PDG"][mask_gp_unmatched]),
        "charge": gpdata.gen_features["charge"][mask_gp_unmatched],
        "pt": pt_arr[mask_gp_unmatched],
        "eta": eta_arr[mask_gp_unmatched],
        "sin_phi": np.sin(phi_arr[mask_gp_unmatched]),
        "cos_phi": np.cos(phi_arr[mask_gp_unmatched]),
        "energy": energy_arr[mask_gp_unmatched],
    }
    assert (np.sum(gen_features_new["energy"]) - np.sum(gpdata.gen_features["energy"])) < 1e-2

    idx_all_masked = np.where(mask_gp_unmatched)[0]
    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}
    genparticle_to_hit = filter_adj(gpdata.genparticle_to_hit, genpart_idx_all_to_filtered)
    genparticle_to_track = filter_adj(gpdata.genparticle_to_track, genpart_idx_all_to_filtered)
    gp_to_obj = gp_to_obj[mask_gp_unmatched]

    return (
        EventData(
            gen_features_new,
            gpdata.hit_features,
            gpdata.cluster_features,
            gpdata.track_features,
            genparticle_to_hit,
            genparticle_to_track,
            gpdata.hit_to_cluster,
            (gp_merges_gp0, gp_merges_gp1),
        ),
        gp_to_obj,
    )


# for each PF element (track, cluster), get the index of the best-matched particle (gen or reco)
# if the PF element has no best-matched particle, returns -1
def assign_to_recoobj(n_obj, obj_to_ptcl, used_particles):
    obj_to_ptcl_all = -1 * np.ones(n_obj, dtype=np.int64)
    for iobj in range(n_obj):
        if iobj in obj_to_ptcl:
            iptcl = obj_to_ptcl[iobj]
            obj_to_ptcl_all[iobj] = iptcl
            assert used_particles[iptcl] == 0
            used_particles[iptcl] = 1
    return obj_to_ptcl_all


def get_recoptcl_to_obj(n_rps, reco_arr, idx_rp_to_track, idx_rp_to_cluster):
    track_to_rp = {}
    cluster_to_rp = {}

    # loop over the reco particles
    for irp in range(n_rps):
        assigned = False

        # find and loop over tracks associated to the reco particle
        trks_begin = reco_arr["tracks_begin"][irp]
        trks_end = reco_arr["tracks_end"][irp]
        for itrk in range(trks_begin, trks_end):

            # get the index of the track in the track collection
            itrk_real = idx_rp_to_track[itrk]
            assert itrk_real not in track_to_rp
            track_to_rp[itrk_real] = irp
            assigned = True

        # only look for clusters if tracks were not found
        if not assigned:

            # find and loop over clusters associated to the reco particle
            cls_begin = reco_arr["clusters_begin"][irp]
            cls_end = reco_arr["clusters_end"][irp]
            for icls in range(cls_begin, cls_end):

                # get the index of the cluster in the cluster collection
                icls_real = idx_rp_to_cluster[icls]
                assert icls_real not in cluster_to_rp
                cluster_to_rp[icls_real] = irp
    return track_to_rp, cluster_to_rp


def get_reco_properties(prop_data, iev):
    reco_arr = prop_data["MergedRecoParticles"][iev]
    reco_arr = {k.replace("MergedRecoParticles.", ""): reco_arr[k] for k in reco_arr.fields}

    reco_p4 = vector.awk(
        awkward.zip(
            {"mass": reco_arr["mass"], "x": reco_arr["momentum.x"], "y": reco_arr["momentum.y"], "z": reco_arr["momentum.z"]}
        )
    )
    reco_arr["pt"] = reco_p4.pt
    reco_arr["eta"] = reco_p4.eta
    reco_arr["phi"] = reco_p4.phi
    reco_arr["energy"] = reco_p4.energy

    msk = reco_arr["type"] != 0
    reco_arr = awkward.Record({k: reco_arr[k][msk] for k in reco_arr.keys()})
    return reco_arr


def get_particle_feature_matrix(pfelem_to_particle, feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = feature_dict[feat]
        if len(feat_arr) == 0:
            feat_arr_reordered = feat_arr
        else:
            feat_arr_reordered = awkward.to_numpy(feat_arr[pfelem_to_particle])
            feat_arr_reordered[pfelem_to_particle == -1] = 0.0
        feats.append(feat_arr_reordered)
    feats = np.array(feats)
    return feats.T


def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T


def process_one_file(fn, ofn):

    # output exists, do not recreate
    if os.path.isfile(ofn):
        print("{} exists".format(ofn))
        return

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
    calohit_links = arrs.arrays(["CalohitMCTruthLink", "CalohitMCTruthLink#0", "CalohitMCTruthLink#1"])
    sitrack_links = arrs.arrays(["SiTracksMCTruthLink", "SiTracksMCTruthLink#0", "SiTracksMCTruthLink#1"])

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
    for iev in range(arrs.num_entries):

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
        gpdata_cleaned, gp_to_obj = assign_genparticles_to_obj_and_merge(gpdata)

        n_tracks = len(gpdata_cleaned.track_features["type"])
        n_clusters = len(gpdata_cleaned.cluster_features["type"])
        n_gps = len(gpdata_cleaned.gen_features["PDG"])

        assert len(gp_to_obj) == len(gpdata_cleaned.gen_features["PDG"])
        assert gp_to_obj.shape[1] == 2

        # for each reco particle, find the tracks and clusters associated with it
        # construct track/cluster -> recoparticle maps
        track_to_rp, cluster_to_rp = get_recoptcl_to_obj(n_rps, reco_arr, idx_rp_to_track[iev], idx_rp_to_cluster[iev])

        # get the track/cluster -> genparticle map
        track_to_gp = {itrk: igp for igp, itrk in enumerate(gp_to_obj[:, 0]) if itrk != -1}
        cluster_to_gp = {icl: igp for igp, icl in enumerate(gp_to_obj[:, 1]) if icl != -1}

        used_gps = np.zeros(n_gps, dtype=np.int64)
        track_to_gp_all = assign_to_recoobj(n_tracks, track_to_gp, used_gps)
        cluster_to_gp_all = assign_to_recoobj(n_clusters, cluster_to_gp, used_gps)
        # all genparticles must be assigned to some PFElement
        assert np.all(used_gps == 1)

        used_rps = np.zeros(n_rps, dtype=np.int64)
        track_to_rp_all = assign_to_recoobj(n_tracks, track_to_rp, used_rps)
        cluster_to_rp_all = assign_to_recoobj(n_clusters, cluster_to_rp, used_rps)
        # all reco particles must be assigned to some PFElement
        assert np.all(used_rps == 1)

        gps_track = get_particle_feature_matrix(track_to_gp_all, gpdata_cleaned.gen_features, particle_feature_order)
        gps_track[:, 0] = np.array(
            [map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(gps_track[:, 0], gps_track[:, 1])]
        )
        gps_cluster = get_particle_feature_matrix(cluster_to_gp_all, gpdata_cleaned.gen_features, particle_feature_order)
        gps_cluster[:, 0] = np.array(
            [map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(gps_cluster[:, 0], gps_cluster[:, 1])]
        )
        gps_cluster[:, 1] = 0

        rps_track = get_particle_feature_matrix(track_to_rp_all, reco_features, particle_feature_order)
        rps_track[:, 0] = np.array(
            [map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(rps_track[:, 0], rps_track[:, 1])]
        )
        rps_cluster = get_particle_feature_matrix(cluster_to_rp_all, reco_features, particle_feature_order)
        rps_cluster[:, 0] = np.array(
            [map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(rps_cluster[:, 0], rps_cluster[:, 1])]
        )
        rps_cluster[:, 1] = 0

        # all initial gen/reco particle energy must be reconstructable
        assert (
            abs(np.sum(gps_track[:, 6]) + np.sum(gps_cluster[:, 6]) - np.sum(gpdata_cleaned.gen_features["energy"])) < 1e-2
        )

        assert abs(np.sum(rps_track[:, 6]) + np.sum(rps_cluster[:, 6]) - np.sum(reco_features["energy"])) < 1e-2

        # we don"t want to try to reconstruct charged particles from primary clusters, make sure the charge is 0
        assert np.all(gps_cluster[:, 1] == 0)
        assert np.all(rps_cluster[:, 1] == 0)

        X_track = get_feature_matrix(gpdata_cleaned.track_features, track_feature_order)
        X_cluster = get_feature_matrix(gpdata_cleaned.cluster_features, cluster_feature_order)
        ygen_track = gps_track
        ygen_cluster = gps_cluster
        ycand_track = rps_track
        ycand_cluster = rps_cluster

        sanitize(X_track)
        sanitize(X_cluster)
        print("X_track={} X_cluster={}".format(len(X_track), len(X_cluster)))
        sanitize(ygen_track)
        sanitize(ygen_cluster)
        sanitize(ycand_track)
        sanitize(ycand_cluster)

        this_ev = awkward.Record(
            {
                "X_track": X_track,
                "X_cluster": X_cluster,
                "ygen_track": ygen_track,
                "ygen_cluster": ygen_cluster,
                "ycand_track": ycand_track,
                "ycand_cluster": ycand_cluster,
            }
        )
        ret.append(this_ev)

    ret = awkward.Record({k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields})
    awkward.to_parquet(ret, ofn)


def process_sample(sample):
    inp = "/local/joosep/clic_edm4hep_2023_02_27/"
    outp = "/local/joosep/mlpf/clic_edm4hep_2023_05_09/"

    pool = multiprocessing.Pool(16)

    inpath_samp = inp + sample
    outpath_samp = outp + sample
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
    else:
        process_one_file(sys.argv[1], sys.argv[2])
