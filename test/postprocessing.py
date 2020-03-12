import numpy as np
import pandas
import ROOT

map_candid_to_pdgid = {
    0: [0],
    211: [211, 2212, 321, -3112, 3222, -3312, -3334],
    -211: [-211, -2212, -321, 3112, -3222, 3312, 3334],
    130: [130, 2112, -2112, 310, 3122, -3122, 3322, -3322],
    22: [22],
    11: [11],
    -11: [-11],
     13: [13],
     -13: [-13]
}
map_pdgid_to_candid = {}
for candid, pdgids in map_candid_to_pdgid.items():
    for p in pdgids:
        map_pdgid_to_candid[p] = candid

def prepare_reco_df(reco_objects, elements):
    ret = pandas.DataFrame()

    all_keys = list(elements.keys())
    
    data_vecs = {k: [] for k in all_keys}

    for ro in reco_objects:
        tp, i = ro
        ntype = -1
       
        for k in all_keys:
           data_vecs[k] += [elements[k][i]]

    for k in data_vecs.keys():
        ret[k] = data_vecs[k]
    return ret

def prepare_gen_df(gen_objects, trackingparticles, simclusters):
    ret = pandas.DataFrame()

    all_keys = list(trackingparticles.keys())
    
    data_vecs = {k: [] for k in all_keys}
    data_vecs["type"] = []
    data_vecs["idx_original"] = []
    data_vecs["idx_original2"] = []

    for go in gen_objects:
        tp, i, j = go
 
        if tp == "trackingparticle":
            coll = trackingparticles
            ntype = 1 
        elif tp == "simcluster":
            coll = simclusters 
            ntype = 0 
        else:
            raise Exception()
       
        for k in all_keys:
           data_vecs[k] += [coll[k][i]]
        data_vecs["type"] += [ntype]
        data_vecs["idx_original"] += [i]
        data_vecs["idx_original2"] += [j]

    for k in data_vecs.keys():
        ret[k] = data_vecs[k]
    return ret

def prepare_cand_df(cand_objects, candidates):
    ret = pandas.DataFrame()

    all_keys = list(trackingparticles.keys())
    
    data_vecs = {k: [] for k in all_keys}

    for go in cand_objects:
        tp, i = go
 
        for k in all_keys:
           data_vecs[k] += [candidates[k][i]]

    for k in data_vecs.keys():
        ret[k] = data_vecs[k]
    return ret

if __name__ == "__main__":
    tf = ROOT.TFile("pfntuple.root")
    tt = tf.Get("ana/pftree")
    
    for iev, ev in enumerate(tt):
        print("processing event {}".format(iev))
        
        trackingparticles_pt = ev.trackingparticle_pt
        trackingparticles_e = ev.trackingparticle_energy
        trackingparticles_eta = ev.trackingparticle_eta
        trackingparticles_phi = ev.trackingparticle_phi
        trackingparticles_pid = ev.trackingparticle_pid
        trackingparticles_dvx = ev.trackingparticle_dvx
        trackingparticle_to_element = ev.trackingparticle_to_element
        trackingparticle_to_element = [(x.first, x.second) for x in trackingparticle_to_element]
        trackingparticle_to_element_d = {a: b for (a, b) in trackingparticle_to_element}
        element_to_trackingparticle = [(b, a) for (a, b) in trackingparticle_to_element]
        element_to_trackingparticle_d = {}
        for (t, tp) in element_to_trackingparticle:
            if not (t in element_to_trackingparticle_d):
                element_to_trackingparticle_d[t] = []
            element_to_trackingparticle_d[t] += [tp]
     
        element_pt = ev.element_pt
        element_e = ev.element_energy
        element_eta = ev.element_eta
        element_phi = ev.element_phi
        element_layer = ev.element_layer
        element_type = ev.element_type
        nelements = len(element_e)
        
        pfcandidates_pt = ev.pfcandidate_pt
        pfcandidates_e = ev.pfcandidate_energy
        pfcandidates_eta = ev.pfcandidate_eta
        pfcandidates_phi = ev.pfcandidate_phi
        pfcandidates_pid = ev.pfcandidate_pdgid
        
        simclusters_e = ev.simcluster_energy
        simclusters_pid = ev.simcluster_pid
        simclusters_pt = ev.simcluster_pt
        simclusters_eta = ev.simcluster_eta
        simclusters_phi = ev.simcluster_phi
        simclusters_idx_trackingparticle = ev.simcluster_idx_trackingparticle
        simcluster_to_element = ev.simcluster_to_element
        simcluster_to_element_cmp = ev.simcluster_to_element_cmp
        simcluster_to_element = [(x.first, x.second, c) for x, c in zip(simcluster_to_element, simcluster_to_element_cmp)]
        element_to_simcluster = [(b, a, c) for (a, b, c) in simcluster_to_element]
        element_to_simcluster_d = {}
        for (cl, sc, comp) in element_to_simcluster:
            if not (cl in element_to_simcluster_d):
                element_to_simcluster_d[cl] = []
            element_to_simcluster_d[cl] += [(sc, comp)]
       
        element_to_candidate = ev.element_to_candidate

        element_to_candidate_d = {}
        for (cl, cnd) in element_to_candidate:
            if not (cl in element_to_candidate_d):
                element_to_candidate_d[cl] = []
            element_to_candidate_d[cl] += [cnd]
 
        reco_objects = []
        gen_objects = []
        cand_objects = []
        map_reco_to_gen = []
        map_reco_to_cand = []

        for ielem in range(nelements):
            #print("track {} pt={} eta={} phi={}".format(itrack, tracks_pt[itrack], tracks_eta[itrack], tracks_phi[itrack]))
            ro = ("elem", ielem)
            reco_objects += [ro]
            elem_e = element_e[ielem]
    
            idx_tps = element_to_trackingparticle_d.get(ielem, [])
            idx_scs = element_to_simcluster_d.get(ielem, [])
            #print(ielem, element_type[ielem], idx_tps, idx_scs)
            for idx_tp in idx_tps:
                go = ("trackingparticle", idx_tp, -1)
                if not (go in gen_objects): 
                    gen_objects += [go]
                map_reco_to_gen += [(ro, go, 1000.0)]
            for idx_sc, comp in idx_scs:
                sc_idx_tp = simclusters_idx_trackingparticle[idx_sc]
                if sc_idx_tp != -1:
                    go = ("trackingparticle", sc_idx_tp, idx_sc)
                else:
                    go = ("simcluster", idx_sc, -1)
                if not (go in gen_objects): 
                    gen_objects += [go]
                map_reco_to_gen += [(ro, go, comp)]
            
            idx_cnds = element_to_candidate_d.get(ielem, [])
            for idx_cnd in idx_cnds:
                #print("candidate {} pt={} eta={} phi={}".format(
                #    idx_cnd, pfcandidates_pt[idx_cnd], pfcandidates_eta[idx_cnd],
                #    pfcandidates_phi[idx_cnd], pfcandidates_pid[idx_cnd]))

                go = ("candidate", idx_cnd)
                if not (go in cand_objects): 
                    cand_objects += [go]
                #print("track {} candidate {} match".format(itrack, idx_cnd))
                map_reco_to_cand += [(ro, go, elem_e)]
 
#        for icluster in range(nclusters):
#            ro = ("cluster", icluster)
#            #print("cluster {} e={} eta={} phi={}".format(icluster, clusters_e[icluster], clusters_eta[icluster], clusters_phi[icluster]))
#            reco_objects += [ro]
#            reco_energy = clusters_e[icluster]
#
#            idx_scs = cluster_to_simcluster_d.get(icluster, [])
#            for idx_sc, comp in sorted(idx_scs): 
#                sc_idx_tp = simclusters_idx_trackingparticle[idx_sc]
#                if sc_idx_tp != -1:
#                    go = ("trackingparticle", sc_idx_tp, idx_sc)
#                    if ("track", sc_idx_tp) in reco_objects:
#                        continue
#                else:
#                    go = ("simcluster", idx_sc, -1)
#
#                if not (go in gen_objects): 
#                    gen_objects += [go]
#                #print("cluster={} simcluster={} sc_idx_tp={} igen={} comp={}".format(
#                #    icluster, idx_sc, sc_idx_tp, gen_objects.index(go), comp))
#                map_reco_to_gen += [(ro, go, comp)]
#            
#            idx_cnds = cluster_to_candidate_d.get(icluster, [])
#            for idx_cnd in idx_cnds:
#                #print("candidate {} pt={} eta={} phi={}".format(idx_cnd, pfcandidates_pt[idx_cnd], pfcandidates_eta[idx_cnd], pfcandidates_phi[idx_cnd], pfcandidates_pid[idx_cnd]))
#                if idx_cnd in idx_all_candidates:
#                    idx_all_candidates.remove(idx_cnd)
#                else:
#                    #print("candidate {} already removed".format(idx_cnd))
#                    continue
#                go = ("candidate", idx_cnd)
#                if not (go in cand_objects): 
#                    cand_objects += [go]
#                map_reco_to_cand += [(ro, go, reco_energy)]
#
#        #candidates that were not matched to reco objects  
#        for idx_cnd in idx_all_candidates:
#            print("unmatched candidate", idx_cnd, pfcandidates_pid[idx_cnd])
# 
        #reco_objects = sorted(reco_objects)
        #gen_objects = sorted(gen_objects)
 
        elements = {
            "eta": element_eta,
            "phi": element_phi, 
            "e": element_e,
            "pt": element_pt, 
            "layer": element_layer,
            "type": element_type,
        }
        trackingparticles = {
            "pid": trackingparticles_pid,
            "pt": trackingparticles_pt,
            "eta": trackingparticles_eta,
            "phi": trackingparticles_phi,
            "e": trackingparticles_e
        }
        simclusters = {
            "pid": simclusters_pid,
            "pt": simclusters_pt,
            "eta": simclusters_eta,
            "phi": simclusters_phi,
            "e": simclusters_e
        }
        candidates = {
            "pid": pfcandidates_pid,
            "pt": pfcandidates_pt,
            "eta": pfcandidates_eta,
            "phi": pfcandidates_phi,
            "e": pfcandidates_e 
        }
    
        reco_df = prepare_reco_df(reco_objects, elements)
        gen_df = prepare_gen_df(gen_objects, trackingparticles, simclusters)
        cand_df = prepare_cand_df(cand_objects, candidates)
 
        mat_reco_to_gen = np.zeros((len(reco_objects), len(gen_objects)), dtype=np.float64)
        for ro, go, comp in map_reco_to_gen:
            idx_ro = reco_objects.index(ro)
            idx_go = gen_objects.index(go)
            mat_reco_to_gen[idx_ro, idx_go] += comp
        print("reco-gen", len(reco_objects), len(gen_objects), len(map_reco_to_gen))
        
        mat_reco_to_cand = np.zeros((len(reco_objects), len(cand_objects)), dtype=np.float64)
        for ro, go, comp in map_reco_to_cand:
            idx_ro = reco_objects.index(ro)
            idx_go = cand_objects.index(go)
            mat_reco_to_cand[idx_ro, idx_go] += comp
        print("reco-cand", len(reco_objects), len(pfcandidates_pid), len(cand_objects), len(map_reco_to_cand))
 
#        #np.savez("ev.npz", reco_to_gen=mat_reco_to_gen)
#        #reco_df.to_csv("reco.csv")
#        #gen_df.to_csv("gen.csv")
#
        #loop over all genparticles in pt-descending order, find the best-matched reco-particle
        highest_pt_idx = np.argsort(gen_df["pt"].values)[::-1]
        pairs_reco_gen = {}
        for igen in highest_pt_idx:
            #skip genparticle below an energy and pT threshold
            if gen_df.loc[igen, "e"] < 1 or gen_df.loc[igen, "pt"]<0.2:
                continue

            best_reco_idx = np.argmax(mat_reco_to_gen[:, igen])
            #require at least 0.5 GeV energy from simcluster to be matched to reco object
            if best_reco_idx != 0 and mat_reco_to_gen[:, igen].sum() > 1.0:
                if not (best_reco_idx in pairs_reco_gen):
                    pairs_reco_gen[best_reco_idx] = []
                pairs_reco_gen[best_reco_idx] += [(igen, mat_reco_to_gen[best_reco_idx, igen])]
       
        for ireco in range(len(reco_objects)):
            if not (ireco in pairs_reco_gen):
                pairs_reco_gen[ireco] = []
 
        pairs_reco_gen_sorted = {}
        for k, v in pairs_reco_gen.items():
            v = sorted(v, key=lambda x: x[1], reverse=True)
            pairs_reco_gen_sorted[k] = v
        pairs_reco_gen = pairs_reco_gen_sorted
       
        #all the PFCandidates that could not be matched one-to-one to a reco object (one reco object had multiple pfcandidates?)
        unmatched_candidates = []
        #reco objects that were already matched to a candidate
        remaining_indices = np.ones(len(reco_objects), dtype=np.float32)
        pairs_reco_cand = {}
        
        #find reco to candidate matches 
        highest_pt_idx = np.argsort(cand_df["pt"].values)[::-1]
        for icand in highest_pt_idx:
            temp = remaining_indices*mat_reco_to_cand[:, icand]
            best_reco_idx = np.argmax(temp)

            if best_reco_idx != 0 and mat_reco_to_cand[best_reco_idx, icand] > 0.0:
                remaining_indices[best_reco_idx] = 0.0
                if not (best_reco_idx in pairs_reco_cand):
                    pairs_reco_cand[best_reco_idx] = []
                pairs_reco_cand[best_reco_idx] += [(icand, mat_reco_to_cand[best_reco_idx, icand])]
            else:
                unmatched_candidates += [icand]
  
        for ireco in range(len(reco_objects)):
            if not (ireco in pairs_reco_cand):
                pairs_reco_cand[ireco] = []
 
        pairs_reco_cand_sorted = {}
        for k, v in pairs_reco_cand.items():
            v = sorted(v, key=lambda x: x[1], reverse=True)
            pairs_reco_cand_sorted[k] = v
        pairs_reco_cand = pairs_reco_cand_sorted

        X = np.zeros((len(pairs_reco_gen), 6), dtype=np.float32)
        ygen = np.zeros((len(pairs_reco_gen), 6), dtype=np.float32)
        ycand = np.zeros((len(pairs_reco_gen), 5), dtype=np.float32)

        #loop over all reco-gen pairs
        for ireco, (reco, gens) in enumerate(pairs_reco_gen.items()):
            print("---")
            reco_arr = reco_df.loc[reco, ["pt", "eta", "phi", "e", "type", "layer"]]
            print("reco {} pt={:.5f} eta={:.5f} phi={:.5f} e={:.5f} l={} type={}".format(
                reco,
                reco_arr["pt"], reco_arr["eta"], reco_arr["phi"], reco_arr["e"], reco_arr["layer"], reco_arr["type"]
            ))

            #get all the genparticles associated to this reco particle
            igens = [g[0] for g in gens]
            gen_arr = gen_df.loc[igens, ["pt", "eta", "phi", "e", "pid"]]

            pid = 0

            #In case we have a single genparticle, use it's PID
            if len(gens) == 1:
                if gen_df.loc[igens[0], "pid"] in map_pdgid_to_candid:
                    pid = map_pdgid_to_candid[gen_df.loc[igens[0], "pid"]]
                else:
                    print("unknown or unhandled pid={}".format(gen_df.loc[igens[0], "pid"]))
                    pid = 0
            #In case of multiple genparticles overlapping, use a placeholder constant
            elif len(gens) > 1:
                count_em = 0
                count_had = 0

                for ig in igens:
                    pid = abs(gen_df.loc[ig, "pid"])

                    if pid in [11, 22]:
                        count_em += 1
                    else:
                        count_had += 1

                if count_em > count_had:
                    pid = 1
                else:
                    pid = 2

            all_pids = [gen_df.loc[i, "pid"] for i in igens]

            #add up the momentum vectors of the genparticles
            lvs = []
            for igen in igens:
                lv = ROOT.TLorentzVector()
                lv.SetPtEtaPhiE(
                    gen_df.loc[igen, "pt"],
                    gen_df.loc[igen, "eta"],
                    gen_df.loc[igen, "phi"],
                    gen_df.loc[igen, "e"]
                )
                lvs += [lv]
            lv = sum(lvs, ROOT.TLorentzVector())
            if len(igens) > 0:
                print("gen pt={:.2f} eta={:.2f} phi={:.2f} e={:.2f} pid={} ngen={} all_pids={}".format(
                    lv.Pt(), lv.Eta(), lv.Phi(), lv.E(), pid, len(gens), all_pids
                ))

            X[ireco, 0] = reco_arr["type"]
            X[ireco, 1] = reco_arr["pt"]
            X[ireco, 2] = reco_arr["eta"]
            X[ireco, 3] = reco_arr["phi"]
            X[ireco, 4] = reco_arr["e"]
            X[ireco, 5] = reco_arr["layer"]
            
            ygen[ireco, 0] = pid
            ygen[ireco, 1] = lv.Pt()
            ygen[ireco, 2] = lv.Eta()
            ygen[ireco, 3] = lv.Phi()
            ygen[ireco, 4] = lv.E()
            ygen[ireco, 5] = len(gens)
            
            cands = pairs_reco_cand[reco]
            if len(cands) > 1:
                print("ERROR! more than one candidate found for reco object {}".format(ireco))
            for icand, comp in cands: 
                print("cand pt={:.2f} eta={:.2f} phi={:.2f} e={:.2f} pid={} comp={}".format(
                    cand_df.loc[icand, "pt"], cand_df.loc[icand, "eta"], cand_df.loc[icand, "phi"], cand_df.loc[icand, "e"], cand_df.loc[icand, "pid"], comp
                ))
            
                ycand[ireco, 0] = cand_df.loc[icand, "pid"]
                ycand[ireco, 1] = cand_df.loc[icand, "pt"]
                ycand[ireco, 2] = cand_df.loc[icand, "eta"]
                ycand[ireco, 3] = cand_df.loc[icand, "phi"]
                ycand[ireco, 4] =  cand_df.loc[icand, "e"]

        #Mostly soft photons, a few neutral hadrons - we will need to solve this later
        print("unmatched pfcandidates", len(unmatched_candidates))
        for idx_cnd in unmatched_candidates:
            print("unmatched pfcandidate", cand_df.loc[idx_cnd, "pid"], cand_df.loc[idx_cnd, "pt"]) 

        np.savez("ev_{}.npz".format(iev), X=X, ygen=ygen, ycand=ycand, reco_gen=mat_reco_to_gen, reco_cand=mat_reco_to_cand)
