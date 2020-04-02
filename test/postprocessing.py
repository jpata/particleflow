import numpy as np
import pandas
import ROOT
import scipy
import scipy.sparse
import sys
import networkx as nx
import numba

debug = False

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

@numba.njit
def deltaphi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

@numba.njit
def associate_deltar(etaphi, dr2cut, ret):
    for i in range(len(etaphi)):
        for j in range(i+1, len(etaphi)):
            dphi = deltaphi(etaphi[i, 1], etaphi[j, 1])
            deta = etaphi[i, 0] - etaphi[j, 0]
            dr2 = dphi**2 + deta**2
            #dr = np.sqrt(dphi**2 + deta**2)
            if dr2 < dr2cut:
                ret[i,j] += np.sqrt(dr2)                   

def prepare_df(reco_objects, elements):
    ret = pandas.DataFrame()
    all_keys = list(elements.keys())

    if len(reco_objects) > 0:
        inds = np.array([ro[1] for ro in reco_objects])
        for k in all_keys:
            ret[k] = np.array(elements[k])[inds]
    else:
        for k in all_keys:
            ret[k] = np.array([], dtype=np.float64)
 
    return ret

if __name__ == "__main__":

    infile = sys.argv[1]
    outpath = infile.split(".")[0]
    tf = ROOT.TFile(infile)
    tt = tf.Get("ana/pftree")

    for iev, ev in enumerate(tt):
        print("processing event {}".format(iev))

        trackingparticles_pt = ev.trackingparticle_pt
        trackingparticles_px = ev.trackingparticle_px
        trackingparticles_py = ev.trackingparticle_py
        trackingparticles_pz = ev.trackingparticle_pz
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
        element_px = ev.element_px
        element_py = ev.element_py
        element_pz = ev.element_pz
        element_e = ev.element_energy
        element_eta = ev.element_eta
        element_phi = ev.element_phi
        element_layer = ev.element_layer
        element_depth = ev.element_depth
        element_charge = ev.element_charge
        element_eta_ecal = ev.element_eta_ecal
        element_phi_ecal = ev.element_phi_ecal
        element_eta_hcal = ev.element_eta_hcal
        element_phi_hcal = ev.element_phi_hcal
        element_type = ev.element_type
        element_trajpoint = ev.element_trajpoint
        nelements = len(element_e)

        pfcandidates_pt = ev.pfcandidate_pt
        pfcandidates_px = ev.pfcandidate_px
        pfcandidates_py = ev.pfcandidate_py
        pfcandidates_pz = ev.pfcandidate_pz
        pfcandidates_e = ev.pfcandidate_energy
        pfcandidates_eta = ev.pfcandidate_eta
        pfcandidates_phi = ev.pfcandidate_phi
        pfcandidates_pid = ev.pfcandidate_pdgid

        simclusters_e = ev.simcluster_energy
        simclusters_pid = ev.simcluster_pid
        simclusters_pt = ev.simcluster_pt
        #simclusters_px = ev.simcluster_px
        #simclusters_py = ev.simcluster_py
        #simclusters_pz = ev.simcluster_pz
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
            ro = ("elem", ielem)
            elem_e = element_e[ielem]
            #print("ielem={} elem_e={} type={}".format(ielem, elem_e, element_type[ielem]))
            reco_objects += [ro]
 
            idx_tps = element_to_trackingparticle_d.get(ielem, [])
            idx_scs = element_to_simcluster_d.get(ielem, [])

            for idx_tp in idx_tps:
                go = ("trackingparticle", idx_tp)
                if not (go in gen_objects): 
                    gen_objects += [go]
                map_reco_to_gen += [(ro, go, 1000.0)]

            for idx_sc, comp in idx_scs:
                sc_idx_tp = simclusters_idx_trackingparticle[idx_sc]
                if sc_idx_tp != -1:
                    go = ("trackingparticle", sc_idx_tp)
                    gen_pid = trackingparticles_pid[sc_idx_tp]
                    gen_e = trackingparticles_e[sc_idx_tp]
                else:
                    go = ("simcluster", idx_sc)
                    gen_pid = simclusters_pid[idx_sc]
                    gen_e = simclusters_e[idx_sc]

                if not (go in gen_objects): 
                    gen_objects += [go]
                #print("gen pid={} e={} comp={}".format(gen_pid, gen_e, comp))
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

        #All PFCandidates must be associated to something
        if len(cand_objects) != len(pfcandidates_pt):
            print("cand_objects != pfcandidates", len(cand_objects), len(pfcandidates_pt))

        #sort gen objects by index
        gen_objects = sorted(gen_objects)

        elements = {
            "eta": element_eta,
            "phi": element_phi, 
            "e": element_e,
            "pt": element_pt, 
            "layer": element_layer,
            "trajpoint": element_trajpoint,
            "depth": element_depth,
            "type": element_type,
            "charge": element_charge,
            "eta_ecal": element_eta_ecal,
            "phi_ecal": element_phi_ecal,
            "eta_hcal": element_eta_hcal,
            "phi_hcal": element_phi_hcal,
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

        #create dataframes
        assert(len(reco_objects) > 0) 
        reco_df = prepare_df(reco_objects, elements)
        gen_objects_sc = [go for go in gen_objects if go[0] == "simcluster"]
        gen_objects_tp = [go for go in gen_objects if go[0] == "trackingparticle"]
        gen_df_sc = prepare_df(gen_objects_sc, simclusters)
        gen_df_tp = prepare_df(gen_objects_tp, trackingparticles)
        gen_df = pandas.concat([gen_df_sc, gen_df_tp], ignore_index=True)
        #gen_df.index = np.arange(len(gen_df))
        cand_df = prepare_df(cand_objects, candidates)

        #create a matrix of reco to gen associations
        mat_reco_to_gen = np.zeros((len(reco_objects), len(gen_objects)), dtype=np.float64)
        reco_objects_d = {}
        for i in range(len(reco_objects)):
            reco_objects_d[reco_objects[i]] = i
        gen_objects_d = {}
        for i in range(len(gen_objects)):
            gen_objects_d[gen_objects[i]] = i
        cand_objects_d = {}
        for i in range(len(cand_objects)):
            cand_objects_d[cand_objects[i]] = i

        for ro, go, comp in map_reco_to_gen:
            idx_ro = reco_objects_d[ro]
            idx_go = gen_objects_d[go]
            mat_reco_to_gen[idx_ro, idx_go] += comp
       
        #create a matrix of reco to candidate associations
        mat_reco_to_cand = np.zeros((len(reco_objects), len(cand_objects)), dtype=np.float64)
        for ro, go, comp in map_reco_to_cand:
            idx_ro = reco_objects_d[ro]
            idx_go = cand_objects_d[go]
            mat_reco_to_cand[idx_ro, idx_go] += comp

        #reco_df.to_csv("reco_{}.csv".format(iev))
        #gen_df.to_csv("gen_{}.csv".format(iev))
        
        reco_graph_gen = nx.Graph()
        for ireco in range(len(reco_objects)):
            reco_graph_gen.add_node(ireco)

        #loop over all genparticles in pt-descending order, find the best-matched reco-particle
        highest_pt_idx = np.argsort(gen_df["pt"].values)[::-1]
        remaining_indices = np.ones(len(reco_objects), dtype=np.float32)
        pairs_reco_gen = {}
        for igen in highest_pt_idx:
            
            inds_elem = np.nonzero(mat_reco_to_gen[:, igen])[0]
            for i1 in inds_elem:
                for i2 in inds_elem:
                    if i1 != i2:
                        reco_graph_gen.add_edge(i1, i2)

            gen_e = gen_df.loc[igen, "e"]  
            gen_type = map_pdgid_to_candid.get(gen_df.loc[igen, "pid"], 0)

            #skip genparticle below an energy threshold
            #if gen_type == 22:
            #    if gen_e < 0.3:
            #        continue
            #elif gen_e < 0.1:
            #    continue

            temp = remaining_indices*mat_reco_to_gen[:, igen]
            best_reco_idx = np.argmax(temp)
            if best_reco_idx != 0 and mat_reco_to_gen[best_reco_idx, igen] > 0.0:
                remaining_indices[best_reco_idx] = 0.0
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

        #mask of reco objects that were already matched to a candidate
        remaining_indices = np.ones(len(reco_objects), dtype=np.float32)
       
        #reco index - candidate index 
        pairs_reco_cand = {}
      
        reco_graph_cand = nx.Graph()
        for ireco in range(len(reco_objects)):
            reco_graph_cand.add_node(ireco)
 
        #find reco to candidate matches 
        highest_pt_idx = np.argsort(cand_df["pt"].values)[::-1]
        for icand in highest_pt_idx:
            inds_elem = np.nonzero(mat_reco_to_cand[:, icand])[0]
            for i1 in inds_elem:
                for i2 in inds_elem:
                    if i1 != i2:
                        reco_graph_cand.add_edge(i1, i2)

            temp = remaining_indices*mat_reco_to_cand[:, icand]
            best_reco_idx = np.argmax(temp)

            if best_reco_idx != 0 and mat_reco_to_cand[best_reco_idx, icand] > 0.0:
                #no other pfcandidate can be matched to this reco object
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

        reco_arr = reco_df[[
            "type", "pt", "eta", "phi", "e",
            "layer", "depth", "charge", "trajpoint", 
            "eta_ecal", "phi_ecal", "eta_hcal", "phi_hcal"]].values

        X = np.zeros((len(pairs_reco_gen), reco_arr.shape[1]), dtype=np.float32)
        gen_arr = gen_df[["pid", "pt", "eta", "phi", "e"]].values
        cand_arr = cand_df[["pid", "pt", "eta", "phi", "e"]].values
        ygen = np.zeros((len(pairs_reco_gen), 6), dtype=np.float32)
        ycand = np.zeros((len(pairs_reco_gen), 5), dtype=np.float32)

        #loop over all reco-gen pairs
        for ireco, (reco, gens) in enumerate(pairs_reco_gen.items()):
            reco_type = reco_arr[ireco, 0]
            if debug:
                print("---")
                print("rec i={:<5} typ={:<5} pt={:.2f} e={:.2f} eta={:.2f} phi={:.2f}".format(ireco,
                    reco_df.loc[ireco, "type"],
                    reco_df.loc[ireco, "pt"],
                    reco_df.loc[ireco, "e"],
                    reco_df.loc[ireco, "eta"],
                    reco_df.loc[ireco, "phi"],
                ))

            #get all the genparticles associated to this reco particle
            igens = [g[0] for g in gens]
            igens = sorted(igens, key=lambda x, gen_arr=gen_arr: gen_arr[x, 4])

            pid = 0
            if len(igens) > 0:
                #get the PID of the highest-energy particle
                pid = map_pdgid_to_candid.get(gen_arr[igens[-1], 0], 0)

                #Assign HF PID in the forward region
                if abs(reco_arr[ireco, 2]) > 3.0:
                    #HFHAD -> always produce hadronic
                    if reco_type == 9:
                        pid = 1
                    #HFEM -> decide based on pid
                    elif reco_type == 8:
                        if abs(pid) in [11, 22]:
                            pid = 2 #produce EM candidate 
                        else:
                            pid = 1 #produce hadronic
                #remap PID in case of HCAL or ECAL cluster
                if reco_type == 5 and (pid == 22 or abs(pid) == 11):
                    pid = 130
                #if reco_type == 4 and (abs(pid) == 11):
                #    pid = 22

            #add up the momentum vectors of the genparticles
            lvs = []
            for igen in igens:
                lv = ROOT.TLorentzVector()
                lv.SetPtEtaPhiE(
                    gen_arr[igen, 1],
                    gen_arr[igen, 2],
                    gen_arr[igen, 3],
                    gen_arr[igen, 4]
                )
                lvs += [lv]
                if debug:
                    print("gen i={:<5} pid={:<5} pt={:.2f} e={:.2f} eta={:.2f} phi={:.2f} c={:.2f}".format(igen,
                        gen_df.loc[igen, "pid"],
                        gen_df.loc[igen, "pt"],
                        gen_df.loc[igen, "e"],
                        gen_df.loc[igen, "eta"],
                        gen_df.loc[igen, "phi"],
                        mat_reco_to_gen[ireco, igen],
                    ))

            #lv = sum(lvs, ROOT.TLorentzVector())

            lv = ROOT.TLorentzVector()
            if len(igens) > 0:
                lv.SetPtEtaPhiE(
                    gen_arr[igens[-1], 1],
                    gen_arr[igens[-1], 2],
                    gen_arr[igens[-1], 3],
                    gen_arr[igens[-1], 4]
                )

            if len(igens) > 0:
                if debug:
                    print("Gen i={:<5} pid={:<5} pt={:.2f} e={:.2f} eta={:.2f} phi={:.2f}".format(igens[0],
                        pid,
                        lv.Pt(),
                        lv.E(),
                        lv.Eta(),
                        lv.Phi(),
                    ))

            X[ireco, :] = reco_arr[reco, :]

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
                ycand[ireco, :] = cand_arr[icand, :]
                if debug:
                    print("cnd i={:<5} pid={:<5} pt={:.2f} e={:.2f} eta={:.2f} phi={:.2f}".format(ireco,
                        cand_df.loc[icand, "pid"],
                        cand_df.loc[icand, "pt"],
                        cand_df.loc[icand, "e"],
                        cand_df.loc[icand, "eta"],
                        cand_df.loc[icand, "phi"],
                    ))

        #Mostly soft photons, a few neutral hadrons - we will need to solve this later
        print("unmatched pfcandidates n={}/{}/{}".format(len(unmatched_candidates), len(cand_df), len(pfcandidates_pt)))
        for idx_cnd in unmatched_candidates:
            print("unmatched pfcandidate idx={:<5} pid={} pt={:.2f} e={:.2f}".format(
                idx_cnd, cand_df.loc[idx_cnd, "pid"],
                cand_df.loc[idx_cnd, "pt"], cand_df.loc[idx_cnd, "e"])
            ) 

        di = np.array(list(ev.element_distance_i))
        dj = np.array(list(ev.element_distance_j))
        d = np.array(list(ev.element_distance_d))
        etas = np.array(list(ev.element_eta), dtype=np.float32)
        phis = np.array(list(ev.element_phi), dtype=np.float32)
        etaphis = np.vstack([etas, phis]).T
        dm_dr = np.zeros((etaphis.shape[0], etaphis.shape[0]), dtype=np.float32)
        associate_deltar(etaphis, 0.2**2, dm_dr)
        n = len(X)
        dm = scipy.sparse.coo_matrix((d, (di, dj)), shape=(n,n)).todense()
        dm += dm_dr 
        dm += dm.T
        dm = scipy.sparse.coo_matrix(dm)

        with open("{}_dist_{}.npz".format(outpath, iev), "wb") as fi:
            scipy.sparse.save_npz(fi, dm)

        with open("{}_cand_{}.npz".format(outpath, iev), "wb") as fi:
            dm = scipy.sparse.coo_matrix(nx.to_numpy_matrix(reco_graph_cand))
            scipy.sparse.save_npz(fi, dm)

        with open("{}_gen_{}.npz".format(outpath, iev), "wb") as fi:
            dm = scipy.sparse.coo_matrix(nx.to_numpy_matrix(reco_graph_gen))
            scipy.sparse.save_npz(fi, dm)

        #np.savez("ev_{}.npz".format(iev), X=X, ygen=ygen, ycand=ycand, reco_gen=mat_reco_to_gen, reco_cand=mat_reco_to_cand)
        np.savez("{}_ev_{}.npz".format(outpath, iev), X=X, ygen=ygen, ycand=ycand)
