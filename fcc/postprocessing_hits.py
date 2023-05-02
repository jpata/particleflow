import bz2
import numpy as np
import awkward
import matplotlib.pyplot as plt
import uproot
import vector
import glob
import networkx as nx
import tqdm
import numba
import os
import sys
import multiprocessing
from scipy.sparse import coo_matrix

track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

#the feature matrices will be saved in this order
particle_feature_order = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]

#arrange track and cluster features such that pt (et), eta, phi, p (energy) are in the same spot
#so we can easily use them in skip connections
track_feature_order = [
    "elemtype", "pt", "eta", "sin_phi", "cos_phi", "p",
    "chi2", "ndf",
    "radiusOfInnermostHit", "tanLambda", "D0", "omega",
    "referencePoint.x", "referencePoint.y", "referencePoint.z",
    "Z0", "time", "type"
]
hit_feature_order = [
    "elemtype", "et", "eta", "sin_phi", "cos_phi", "energy",
    "position.x", "position.y", "position.z", "time", "subdetector", "type"
]

def build_dummy_array(num, dtype=np.int64):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            awkward.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )

def track_pt(omega):
    a = 3 * 10**-4
    b = 4  # B-field in tesla, from clicRec_e4h_input

    return a * np.abs(b / omega)

def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    #photon, electron, muon
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
    def __init__(self,
        gen_features,
        hit_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster
        ):
        self.gen_features = gen_features 
        self.hit_features = hit_features 
        self.track_features = track_features 
        self.genparticle_to_hit = genparticle_to_hit 
        self.genparticle_to_track = genparticle_to_track 
        self.hit_to_cluster = hit_to_cluster

def get_cluster_subdet_energies(hit_list, hit_data, collectionIDs_reverse, iev):
    """
    This function calculates the energy contribution from each of four subdetectors in a particle physics experiment, based on a list of hits and their corresponding data.

    Args:
    hit_list: a list of tuples, where each tuple contains a collection ID and a hit index
    hit_data: a dictionary containing data for each hit in the experiment, organized by collection
    collectionIDs_reverse: a dictionary mapping collection IDs to collection names
    iev: the event number for the current event

    Returns:
    A tuple containing the energy contributions from each of the four subdetectors:
    (ecal_energy, hcal_energy, muon_energy, other_energy)
    """

    ecal_energy = 0.0
    hcal_energy = 0.0
    muon_energy = 0.0
    other_energy = 0.0

    for coll_id, hit_idx in hit_list:
        coll = collectionIDs_reverse[coll_id]
        hit_energy = hit_data[coll][iev][coll+".energy"][hit_idx]

        if coll.startswith("ECAL"):
            ecal_energy += hit_energy
        elif coll.startswith("HCAL"):
            hcal_energy += hit_energy
        elif coll == "MUON":
            muon_energy += hit_energy
        else:
            other_energy += hit_energy

    return ecal_energy, hcal_energy, muon_energy, other_energy

def hits_to_features(hit_data, iev, coll, feats):
    feat_arr = {f: hit_data[coll + "." + f][iev] for f in feats}

    #set the subdetector type
    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(len(feat_arr["type"]), dtype=np.int32)
    if coll.startswith("ECAL"):
        feat_arr[sdcoll][:] = 0
    elif coll.startswith("HCAL"):
        feat_arr[sdcoll][:] = 1
    else:
        feat_arr[sdcoll][:] = 2

    #hit elemtype is always 2
    feat_arr["elemtype"] = 2*np.ones(len(feat_arr["type"]), dtype=np.int32)

    #precompute some approximate et, eta, phi
    pos_mag = np.sqrt(feat_arr["position.x"]**2 + feat_arr["position.y"]**2 + feat_arr["position.z"]**2) 
    px = (feat_arr["position.x"] / pos_mag) * feat_arr["energy"]
    py = (feat_arr["position.y"] / pos_mag) * feat_arr["energy"]
    pz = (feat_arr["position.z"] / pos_mag) * feat_arr["energy"]
    feat_arr["et"] = np.sqrt(px**2+py**2)
    feat_arr["eta"] = 0.5*np.log((feat_arr["energy"] + pz)/(feat_arr["energy"] - pz))
    feat_arr["sin_phi"] = py/feat_arr["energy"]
    feat_arr["cos_phi"] = px/feat_arr["energy"]

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
        for ihit in range(len(hit_data[col][col+".energy"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1
    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = awkward.Record({
        k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))]) for k in hit_feature_matrix[0].fields})

    #add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links["CalohitMCTruthLink"]["CalohitMCTruthLink.weight"][iev]
    calohit_to_gen_calo_colid = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.index"][iev]
    calohit_to_gen_gen_idx = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.index"][iev]
    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, w in zip(calohit_to_gen_calo_colid, calohit_to_gen_calo_idx, calohit_to_gen_gen_colid, calohit_to_gen_gen_idx, calohit_to_gen_weight):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(w)

    return hit_feature_matrix, (genparticle_to_hit_matrix_coo0, genparticle_to_hit_matrix_coo1, genparticle_to_hit_matrix_w), hit_idx_local_to_global

def hit_cluster_adj(prop_data, hit_idx_local_to_global, iev):
    coll_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.collectionID"][iev]
    idx_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.index"][iev]
    hits_begin = prop_data["PandoraClusters"]["PandoraClusters.hits_begin"][iev]
    hits_end = prop_data["PandoraClusters"]["PandoraClusters.hits_end"][iev]

    #index in the array of all hits
    hit_to_cluster_matrix_coo0 = []
    #index in the cluster array
    hit_to_cluster_matrix_coo1 = []

    #weight
    hit_to_cluster_matrix_w = []

    #loop over all clusters
    for icluster in range(len(hits_begin)):

        #get the slice in the hit array corresponding to this cluster
        hbeg = hits_begin[icluster]
        hend = hits_end[icluster]
        idx_range = idx_arr[hbeg:hend]
        coll_range = coll_arr[hbeg:hend]

        #add edges from hit to cluster
        for icol, idx in zip(coll_range, idx_range):
            hit_to_cluster_matrix_coo0.append(hit_idx_local_to_global[(icol, idx)])
            hit_to_cluster_matrix_coo1.append(icluster)
            hit_to_cluster_matrix_w.append(1.0)
    return np.array(hit_to_cluster_matrix_coo0), np.array(hit_to_cluster_matrix_coo1), np.array(hit_to_cluster_matrix_w)

def gen_to_features(prop_data, iev):
    gen_arr = prop_data[mc_coll][iev]
    gen_arr = {k.replace(mc_coll+".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(awkward.zip({
        "mass": gen_arr["mass"],
        "x": gen_arr["momentum.x"],
        "y": gen_arr["momentum.y"],
        "z": gen_arr["momentum.z"]}))
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy

    return awkward.Record({
        "PDG": gen_arr["PDG"],
        "generatorStatus": gen_arr["generatorStatus"],
        "charge": gen_arr["charge"],
        "pt": gen_arr["pt"],
        "eta": gen_arr["eta"],
        "phi": gen_arr["phi"],
        "sin_phi": np.sin(gen_arr["phi"]),
        "cos_phi": np.cos(gen_arr["phi"]),
        "energy": gen_arr["energy"],
        })

def genparticle_track_adj(sitrack_links, iev):
    trk_to_gen_trkidx = sitrack_links["SiTracksMCTruthLink#0"]["SiTracksMCTruthLink#0.index"][iev]
    trk_to_gen_genidx = sitrack_links["SiTracksMCTruthLink#1"]["SiTracksMCTruthLink#1.index"][iev]
    trk_to_gen_w = sitrack_links["SiTracksMCTruthLink"]["SiTracksMCTruthLink.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)
    
    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w


def track_to_features(prop_data, iev):
    track_arr = prop_data[track_coll][iev]
    feats_from_track = ["type", "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit"]
    ret = {feat: track_arr[track_coll + "." + feat] for feat in feats_from_track}
    n_tr = len(ret["type"])

    #FIXME: add additional track features from track state

    #get the index of the first track state
    trackstate_idx = prop_data[track_coll][track_coll + ".trackStates_begin"][iev]
    #get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time", "referencePoint.x", "referencePoint.y", "referencePoint.z"]:
        ret[k] = prop_data["SiTracks_1"]["SiTracks_1." + k][iev][trackstate_idx]

    ret["pt"] = track_pt(ret["omega"])
    ret["px"] = np.cos(ret["phi"]) * ret["pt"]
    ret["py"] = np.sin(ret["phi"]) * ret["pt"]
    ret["pz"] = ret["tanLambda"] * ret["pt"]
    ret["p"] = np.sqrt(ret["px"]**2 + ret["py"]**2 + ret["pz"]**2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"]>0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt>0))
    eta[tt<=0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    #override track type with 1
    ret["elemtype"] = 1*np.ones(n_tr, dtype=np.int32)

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
    hit_features, genparticle_to_hit, hit_idx_local_to_global = get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs)
    hit_to_cluster = hit_cluster_adj(prop_data, hit_idx_local_to_global, iev)
    track_features = track_to_features(prop_data, iev)
    genparticle_to_track = genparticle_track_adj(sitrack_links, iev)

    n_gp = awkward.count(gen_features["PDG"])
    n_track = awkward.count(track_features["type"])
    n_hit = awkward.count(hit_features["type"])

    if len(genparticle_to_track[0])>0:
        gp_to_track = np.array(coo_matrix(
            (genparticle_to_track[2],
            (genparticle_to_track[0], genparticle_to_track[1])),
            shape=(n_gp, n_track)
        ).max(axis=1).todense())[:, 0]
    else:
        gp_to_track = np.zeros(n_gp)

    if len(genparticle_to_hit[0])>0:
        gp_to_calohit = np.array(coo_matrix(
            (genparticle_to_hit[2],
            (genparticle_to_hit[0], genparticle_to_hit[1])),
            shape=(n_gp, n_hit)
        ).max(axis=1).todense())[:, 0]
    else:
        gp_to_calohit = np.zeros(n_gp)

    #60% of the hits of a track must come from the genparticle
    gp_in_tracker = gp_to_track>=0.6

    #the particle should deposit energy to some calo hit
    gp_in_calo = gp_to_calohit>0.0

    gp_interacted_with_detector = gp_in_tracker | gp_in_calo

    #get status 1 particles that are not neutrinos
    #and have energy > 100 MeV
    mask_visible = (
        (gen_features["generatorStatus"]==1) & 
        (gen_features["PDG"]!=12) & 
        (gen_features["PDG"]!=14) & 
        (gen_features["PDG"]!=16) & 
        (gen_features["energy"]>0.1) &
        gp_interacted_with_detector
    )
    idx_all_masked = np.where(mask_visible)[0]
    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}

    gen_features = awkward.Record({
        feat: gen_features[feat][mask_visible] for feat in gen_features.fields
    })

    genparticle_to_hit = filter_adj(genparticle_to_hit, genpart_idx_all_to_filtered)
    genparticle_to_track = filter_adj(genparticle_to_track, genpart_idx_all_to_filtered)

    return EventData(
        gen_features,
        hit_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster
    )

def assign_genparticles_to_obj_and_merge(gpdata):

    n_gp = awkward.count(gpdata.gen_features["PDG"])
    n_track = awkward.count(gpdata.track_features["type"])
    n_hit = awkward.count(gpdata.hit_features["type"])

    gp_to_track = np.array(coo_matrix(
        (gpdata.genparticle_to_track[2],
        (gpdata.genparticle_to_track[0], gpdata.genparticle_to_track[1])),
        shape=(n_gp, n_track)
    ).todense())

    gp_to_calohit = np.array(coo_matrix(
        (gpdata.genparticle_to_hit[2],
        (gpdata.genparticle_to_hit[0], gpdata.genparticle_to_hit[1])),
        shape=(n_gp, n_hit)
    ).todense())

    #map each genparticle to a track or calohit
    gp_to_obj = -1*np.ones((n_gp, 2), dtype=np.int32)
    set_used_tracks = set([])
    set_used_calohits = set([])
    gps_sorted_energy = sorted(range(n_gp), key=lambda x: gpdata.gen_features["energy"][x], reverse=True)

    for igp in gps_sorted_energy:

        #first check if we can match the genparticle to a track
        matched_tracks = gp_to_track[igp]
        trks = np.where(matched_tracks)[0]
        trks = sorted(trks, key=lambda x: matched_tracks[x], reverse=True)
        for trk in trks:
            #if the track was not already used for something else
            if trk not in set_used_tracks:
                gp_to_obj[igp, 0] = trk
                set_used_tracks.add(trk)
                break

        #if there was no matched track, try a calohit
        if gp_to_obj[igp, 0] == -1:
            matched_calohits = np.where(gp_to_calohit[igp])[0]
            calohits = sorted(matched_calohits, key=lambda x: gp_to_calohit[igp, x], reverse=True)
            for calohit in calohits:
                if calohit not in set_used_calohits:
                    gp_to_obj[igp, 1] = calohit
                    set_used_calohits.add(calohit)
                    break

    unmatched = (gp_to_obj[:, 0]!=-1) & (gp_to_obj[:, 1]!=-1)
    return gp_to_obj


#for each PF element (track, cluster), get the index of the best-matched particle (gen or reco)
#if the PF element has no best-matched particle, returns -1
def assign_to_recoobj(n_obj, obj_to_ptcl, used_particles):
    obj_to_ptcl_all = -1 * np.ones(n_obj, dtype=np.int64)
    for iobj in range(n_obj):
        if iobj in obj_to_ptcl:
            iptcl = obj_to_ptcl[iobj]
            obj_to_ptcl_all[iobj] = iptcl
            assert(used_particles[iptcl] == 0)
            used_particles[iptcl] = 1
    return obj_to_ptcl_all

def get_recoptcl_to_obj(n_rps, reco_arr, gpdata, idx_rp_to_track, idx_rp_to_cluster):
    track_to_rp = {}
    calohit_to_rp = {}
    for irp in range(n_rps):
        assigned = False
        trks_begin = reco_arr["tracks_begin"][irp]
        trks_end = reco_arr["tracks_end"][irp]
        for itrk in range(trks_begin, trks_end):
            itrk_real = idx_rp_to_track[itrk]
            assert(itrk_real not in track_to_rp)
            track_to_rp[itrk_real] = irp
            assigned = True

        #only look for calohits if tracks were not found
        if not assigned:
            cls_begin = reco_arr["clusters_begin"][irp]
            cls_end = reco_arr["clusters_end"][irp]
            for icls in range(cls_begin, cls_end):
                icls_real = idx_rp_to_cluster[icls]

                #find hits of the cluster
                calohit_inds = gpdata.hit_to_cluster[0][gpdata.hit_to_cluster[1]==icls_real]

                #get the highest-energy hit
                calohits_e_ascending = np.argsort(gpdata.hit_features["energy"][calohit_inds])
                highest_e_hit = calohit_inds[calohits_e_ascending[-1]]
                assert(highest_e_hit not in calohit_to_rp)
                calohit_to_rp[highest_e_hit] = irp
                assigned = True
                break
    return track_to_rp, calohit_to_rp

def get_reco_properties(prop_data, iev):
    reco_arr = prop_data["MergedRecoParticles"][iev]
    reco_arr = {k.replace("MergedRecoParticles.", ""): reco_arr[k] for k in reco_arr.fields}

    reco_p4 = vector.awk(awkward.zip({
        "mass": reco_arr["mass"],
        "x": reco_arr["momentum.x"],
        "y": reco_arr["momentum.y"],
        "z": reco_arr["momentum.z"]}))
    reco_arr["pt"] = reco_p4.pt
    reco_arr["eta"] = reco_p4.eta
    reco_arr["phi"] = reco_p4.phi
    reco_arr["energy"] = reco_p4.energy

    msk = reco_arr["type"]!=0
    reco_arr = awkward.Record({k: reco_arr[k][msk] for k in reco_arr.keys()})
    return reco_arr

def get_particle_feature_matrix(pfelem_to_particle, feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = feature_dict[feat]
        if len(feat_arr)==0:
            feat_arr_reordered = feat_arr
        else:
            feat_arr_reordered = awkward.to_numpy(feat_arr[pfelem_to_particle])
            feat_arr_reordered[pfelem_to_particle==-1] = 0.0
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

    #output exists, do not recreate
    if os.path.isfile(ofn):
        return
    print(fn)

    fi = uproot.open(fn)
    
    arrs = fi["events"]
    
    collectionIDs = {k: v for k, v in
        zip(fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_names"][0],
        fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_collectionIDs"][0])}
    collectionIDs_reverse = {v: k for k, v in collectionIDs.items()}
    
    prop_data = arrs.arrays([mc_coll, track_coll, "SiTracks_1", "PandoraClusters", "PandoraClusters#1", "PandoraClusters#0", "MergedRecoParticles"])
    calohit_links = arrs.arrays(["CalohitMCTruthLink", "CalohitMCTruthLink#0", "CalohitMCTruthLink#1"])
    sitrack_links = arrs.arrays(["SiTracksMCTruthLink", "SiTracksMCTruthLink#0", "SiTracksMCTruthLink#1"])

    #maps the recoparticle track/cluster index (in tracks_begin,end and clusters_begin,end) to the index in the track/cluster collection
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
    ret_unused_pt = []
    for iev in range(arrs.num_entries):

        #get the reco particles
        reco_arr = get_reco_properties(prop_data, iev)
        reco_type = np.abs(reco_arr["type"])
        n_rps = len(reco_type)
        reco_features = awkward.Record({
            "PDG": np.abs(reco_type),
            "charge": reco_arr["charge"],
            "pt": reco_arr["pt"],
            "eta": reco_arr["eta"],
            "sin_phi": np.sin(reco_arr["phi"]),
            "cos_phi": np.cos(reco_arr["phi"]),
            "energy": reco_arr["energy"]
        })

        #get the genparticles and the links between genparticles and tracks/clusters
        gpdata = get_genparticles_and_adjacencies(prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs)

        #find the reconstructable genparticles and associate them to the best track/cluster
        gp_to_obj = assign_genparticles_to_obj_and_merge(gpdata)

        n_tracks = len(gpdata.track_features["type"])
        n_hits = len(gpdata.hit_features["type"])
        n_gps = len(gpdata.gen_features["PDG"])

        assert(len(gp_to_obj) == len(gpdata.gen_features["PDG"]))
        assert(gp_to_obj.shape[1] == 2)
        
        #for each reco particle, find the tracks and clusters associated with it
        #construct track/cluster -> recoparticle maps
        track_to_rp, hit_to_rp = get_recoptcl_to_obj(n_rps, reco_arr, gpdata, idx_rp_to_track[iev], idx_rp_to_cluster[iev])

        #get the track/cluster -> genparticle map
        track_to_gp = {itrk: igp for igp, itrk in enumerate(gp_to_obj[:, 0]) if itrk != -1}
        hit_to_gp = {ihit: igp for igp, ihit in enumerate(gp_to_obj[:, 1]) if ihit != -1}

        used_gps = np.zeros(n_gps, dtype=np.int64)
        track_to_gp_all = assign_to_recoobj(n_tracks, track_to_gp, used_gps)
        hit_to_gp_all = assign_to_recoobj(n_hits, hit_to_gp, used_gps)
        if not np.all(used_gps==1):
            print("unmatched gen", gpdata.gen_features["energy"][used_gps==0])
        #assert(np.all(used_gps == 1))

        used_rps = np.zeros(n_rps, dtype=np.int64)
        track_to_rp_all = assign_to_recoobj(n_tracks, track_to_rp, used_rps)
        hit_to_rp_all = assign_to_recoobj(n_hits, hit_to_rp, used_rps)
        if not np.all(used_rps==1):
            print("unmatched reco", reco_features["energy"][used_rps==0])
        #assert(np.all(used_rps == 1))

        gps_track = get_particle_feature_matrix(
            track_to_gp_all,
            gpdata.gen_features,
            particle_feature_order
        )
        gps_track[:, 0] = np.array([
            map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(gps_track[:, 0], gps_track[:, 1])]
        )
        gps_hit = get_particle_feature_matrix(
            hit_to_gp_all,
            gpdata.gen_features,
            particle_feature_order
        )
        gps_hit[:, 0] = np.array([
            map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(gps_hit[:, 0], gps_hit[:, 1])]
        )
        gps_hit[:, 1] = 0

        rps_track = get_particle_feature_matrix(
            track_to_rp_all,
            reco_features,
            particle_feature_order
        )
        rps_track[:, 0] = np.array([
            map_neutral_to_charged(map_pdgid_to_candid(p, c)) for p, c in zip(rps_track[:, 0], rps_track[:, 1])]
        )
        rps_hit = get_particle_feature_matrix(
            hit_to_rp_all,
            reco_features,
            particle_feature_order
        )
        rps_hit[:, 0] = np.array([
            map_charged_to_neutral(map_pdgid_to_candid(p, c)) for p, c in zip(rps_hit[:, 0], rps_hit[:, 1])]
        )
        rps_hit[:, 1] = 0

        #all initial gen/reco particle energy must be reconstructable
        #assert(abs(
        #    np.sum(gps_track[:, 6]) + np.sum(gps_hit[:, 6]) - np.sum(gpdata.gen_features["energy"])
        #    ) < 1e-2)

        #assert(abs(
        #    np.sum(rps_track[:, 6]) + np.sum(rps_hit[:, 6]) - np.sum(reco_features["energy"])
        #    ) < 1e-2)


        #we don't want to try to reconstruct charged particles from primary clusters, make sure the charge is 0
        assert(np.all(gps_hit[:, 1] == 0))
        assert(np.all(rps_hit[:, 1] == 0))

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
        if np.sum(used_gps==0)>0:
            ret_unused_pt.append(awkward.to_numpy(gpdata.gen_features["pt"][used_gps==0]))
        else:
            ret_unused_pt.append(np.array([], dtype=np.float32))
        this_ev = awkward.Record(this_ev)

        ret.append(this_ev)

    ret = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}

    ntot = sum([len(x) for x in ret_unused_pt])
    if ntot>0:
        ret["ygen_unused_pt"] = awkward.from_iter(ret_unused_pt)
    else:
        ret["ygen_unused_pt"] = build_dummy_array(len(ret_unused_pt), dtype=np.float32)
    ret = awkward.Record(ret)

    awkward.to_parquet(ret, ofn)

def process_sample(samp):
    inp = "/media/joosep/data/clic_edm4hep_2023_02_27/"
    outp = "/media/joosep/data/mlpf_hits/clic_edm4hep_2023_02_27/"

    pool = multiprocessing.Pool(15)

    inpath_samp = inp + samp
    outpath_samp = outp + samp
    infiles = list(glob.glob(inpath_samp + "/*.root"))[:10000]
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
