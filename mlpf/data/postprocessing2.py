import sys
import pickle
import networkx as nx
import numpy as np
import os
import awkward
import uproot
import vector
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm

from networkx.drawing.nx_pydot import graphviz_layout

map_candid_to_pdgid = {
    0: [0],
    211: [211, 2212, 321, 3112, 3222, 3312, 3334, 411, 521],
    130: [111, 130, 2112, 310, 3122, 3322, 511, 421],
    22: [22],
    11: [11],
    13: [13],
}

elem_branches = [
    "typ", "pt", "eta", "phi", "e",
    "layer", "depth", "charge", "trajpoint", 
    "eta_ecal", "phi_ecal", "eta_hcal", "phi_hcal", "muon_dt_hits", "muon_csc_hits", "muon_type",
    "px", "py", "pz", "deltap", "sigmadeltap",
    "gsf_electronseed_trkorecal", "gsf_electronseed_dnn1", "gsf_electronseed_dnn2", "gsf_electronseed_dnn3", "gsf_electronseed_dnn4", "gsf_electronseed_dnn5",
    "num_hits", "cluster_flags", "corr_energy", "corr_energy_err", "vx", "vy", "vz", "pterror", "etaerror", "phierror", "lambd", "lambdaerror", "theta", "thetaerror"
]

target_branches = ["typ", "charge", "pt", "eta", "sin_phi", "cos_phi", "e"]

map_pdgid_to_candid = {}

for candid, pdgids in map_candid_to_pdgid.items():
    for p in pdgids:
        map_pdgid_to_candid[p] = candid

def deltar_pairs(eta_vec, phi_vec, dr_cut):

    deta = np.abs(np.subtract.outer(eta_vec, eta_vec))
    dphi = np.mod(np.subtract.outer(phi_vec, phi_vec) + np.pi, 2 * np.pi) - np.pi

    dr2 = deta**2 + dphi**2
    dr2 *= np.tri(*dr2.shape)
    dr2[dr2==0] = 999

    ind_pairs = np.where(dr2<dr_cut)

    return ind_pairs

def get_charge(pid):
    abs_pid = abs(pid)
    if pid == 130 or pid == 22 or pid == 1 or pid == 2:
        return 0.0
    #13: mu-, 11: e-
    elif abs_pid == 13 or abs_pid == 11:
        return -math.copysign(1.0, pid)
    #211: pi+
    elif abs_pid == 211:
        return math.copysign(1.0, pid)

def save_ego_graph(g, node, radius=4, undirected=False):
    sg = nx.ego_graph(g, node, radius, undirected=undirected).reverse()

    #remove BREM PFElements from plotting 
    nodes_to_remove = [n for n in sg.nodes if (n[0]=="elem" and sg.nodes[n]["typ"] in [7,])]
    sg.remove_nodes_from(nodes_to_remove)

    fig = plt.figure(figsize=(2*len(sg.nodes)+2, 10))
    sg_pos = graphviz_layout(sg, prog='dot')
    
    edge_labels = {}
    for e in sg.edges:
        if e[1][iev] == "elem" and not (sg.nodes[e[1]]["typ"] in [1,10]):
            edge_labels[e] = "{:.2f} GeV".format(sg.edges[e].get("weight", 0))
        else:
            edge_labels[e] = ""
    
    node_labels = {}
    for node in sg.nodes:
        labels = {"sc": "CaloParticle", "elem": "PFElement", "tp": "TrackingParticle", "pfcand": "PFCandidate"}
        node_labels[node] = "[{label} {idx}] \ntype: {typ}\ne: {e:.4f} GeV\neta: {eta:.4f}".format(
            label=labels[node[0]], idx=node[1], **sg.nodes[node])
        tp = sg.nodes[node]["typ"]
            
    nx.draw_networkx(sg, pos=sg_pos, node_shape=".", node_color="grey", edge_color="grey", node_size=0, alpha=0.5, labels={})
    nx.draw_networkx_labels(sg, pos=sg_pos, labels=node_labels)
    nx.draw_networkx_edge_labels(sg, pos=sg_pos, edge_labels=edge_labels);
    plt.tight_layout()
    plt.axis("off")

    return fig

def draw_event(g):
    pos = {}
    for node in g.nodes:
        pos[node] = (g.nodes[node]["eta"], g.nodes[node]["phi"])

    fig = plt.figure(figsize=(10,10))
    
    nodes_to_draw = [n for n in g.nodes if n[0]=="elem"]
    nx.draw_networkx(g, pos=pos, with_labels=False, node_size=5, nodelist=nodes_to_draw, edgelist=[], node_color="red", node_shape="s", alpha=0.5)
    
    nodes_to_draw = [n for n in g.nodes if n[0]=="pfcand"]
    nx.draw_networkx(g, pos=pos, with_labels=False, node_size=10, nodelist=nodes_to_draw, edgelist=[], node_color="green", node_shape="x", alpha=0.5)
    
    nodes_to_draw = [n for n in g.nodes if (n[0]=="sc" or n[0]=="tp")]
    nx.draw_networkx(g, pos=pos, with_labels=False, node_size=1, nodelist=nodes_to_draw, edgelist=[], node_color="blue", node_shape=".", alpha=0.5)
   
    #draw edges between genparticles and elements
    edges_to_draw = [e for e in g.edges if e[0] in nodes_to_draw]
    nx.draw_networkx_edges(g, pos, edgelist=edges_to_draw, arrows=False, alpha=0.1)
    
    plt.xlim(-6,6)
    plt.ylim(-4,4)
    plt.tight_layout()
    plt.axis("on")
    return fig

def merge_photons_from_pi0(g):
    photons = [elem for elem in g.nodes if g.nodes[elem]["typ"]==22 and (elem[0]=="tp" or elem[0]=="sc")]
    phot_eta = [g.nodes[node]["eta"] for node in photons]
    phot_phi = [g.nodes[node]["phi"] for node in photons]
    merge_pairs = []

    pairs_0, pairs_1 = deltar_pairs(phot_eta, phot_phi, 0.001)
    merge_pairs = [(photons[p0], photons[p1]) for p0, p1 in zip(pairs_0, pairs_1)]

    for pair in merge_pairs:
        if pair[0] in g.nodes and pair[1] in g.nodes:
            lv = vector.obj(pt=0, eta=0, phi=0, E=0)
            for gp in pair:
                lv += vector.obj(
                    pt=g.nodes[gp]["pt"],
                    eta=g.nodes[gp]["eta"],
                    phi=g.nodes[gp]["phi"],
                    E=g.nodes[gp]["e"]
                )
                
            g.nodes[pair[0]]["pt"] = lv.pt
            g.nodes[pair[0]]["eta"] = lv.eta
            g.nodes[pair[0]]["phi"] = lv.phi
            g.nodes[pair[0]]["e"] = lv.energy
            
            #add edge weights from the deleted photon to the remaining photon
            for suc in g.successors(pair[1]):
                if (pair[0], suc) in g.edges:
                    g.edges[(pair[0], suc)]["weight"] += g.edges[(pair[1], suc)]["weight"]
            g.remove_nodes_from([pair[1]])

def cleanup_graph(g, edge_energy_threshold=0.0):
    g = g.copy()

    edges_to_remove = []
    nodes_to_remove = []

    #for each element, remove the incoming edge where the caloparticle deposited less than 5% of it's energy
    #edges_to_remove = []
    for node in g.nodes:
        if node[0] == "elem":
            #remove edges that don't contribute above a threshold 
            ew = [((gen, node), g.edges[gen, node]["weight"]) for gen in g.predecessors(node)]
            #if the edge weight is exactly 1, this was a trackingparticle
            ew = filter(lambda x: x[1] != 1.0, ew)
            ew = sorted(ew, key=lambda x: x[1], reverse=True)
            for edge, weight in ew:
                if weight/g.nodes[edge[0]]["e"] < edge_energy_threshold:
                    edges_to_remove += [edge] 
    
    g.remove_edges_from(edges_to_remove)
    
    #remove calopart/trackingpart not linked to any elements
    #as these are not reconstructable in principle
    nodes_to_remove = []
    for node in g.nodes:
        if node[0]=="sc" or node[0]=="tp":
            deg = g.degree[node]
            if deg==0:
                nodes_to_remove += [node]
    g.remove_nodes_from(nodes_to_remove)

    merge_photons_from_pi0(g)   

    #For each truth particle, compute the energy in tracks or calorimeter clusters
    for node in g.nodes:
        if node[0] == "sc" or node[0] == "tp":
            E_track = 0.0
            E_calo = 0.0
            E_other = 0.0
            E_hf = 0.0
            E_hfem = 0.0
            E_hfhad = 0.0

            #remap PID
            pid = map_pdgid_to_candid.get(abs(g.nodes[node]["typ"]), 0)
            g.nodes[node]["typ"] = pid

            for suc in g.successors(node):
                elem_type = g.nodes[suc]["typ"]
                if elem_type in [1,6]:
                    E_track += g.edges[node, suc]["weight"]
                elif elem_type in [4,5,10]:
                    E_calo += g.edges[node, suc]["weight"]
                elif elem_type in [8,9,11]:
                    if elem_type == 8:
                        E_hfem += g.edges[node, suc]["weight"] 
                    elif elem_type == 9:
                        E_hfhad += g.edges[node, suc]["weight"] 
                    E_hf += g.edges[node, suc]["weight"]
                else:
                    E_other += g.edges[node, suc]["weight"]

            g.nodes[node]["E_track"] = E_track
            g.nodes[node]["E_calo"] = E_calo
            g.nodes[node]["E_other"] = E_other
            g.nodes[node]["E_hf"] = E_hf
            g.nodes[node]["E_hfem"] = E_hfem
            g.nodes[node]["E_hfhad"] = E_hfhad
   
    for node in g.nodes:
        if node[0] == "sc" or node[0] == "tp":
            typ = g.nodes[node]["typ"]

            #print(typ, g.nodes[node]["E_track"], g.nodes[node]["E_calo"], g.nodes[node]["E_hf"], g.nodes[node]["E_other"])
            #for suc in g.successors(node):
            #    print("  {}={}".format(g.nodes[suc]["typ"], g.edges[node, suc]["weight"]))

            #charged particles that leave no track should not be reconstructed as charged 
            if typ in [211, 13, 11] and g.nodes[node]["E_track"]==0:
                g.nodes[node]["typ"] = 130
            
            #if a particle only leaves deposits in the HF, it should be reconstructed as an HF candidate
            if (g.nodes[node]["E_track"]==0) and (g.nodes[node]["E_calo"]==0) and (g.nodes[node]["E_other"]==0) and g.nodes[node]["E_hf"]>0:
                if g.nodes[node]["E_hfem"]>g.nodes[node]["E_hfhad"]:
                    g.nodes[node]["typ"] = 2
                else:
                    g.nodes[node]["typ"] = 1

    #CaloParticles contain a lot of electrons and muons with a soft pt spectrum
    #these should not be attempted to be reconstructed as ele/mu, but rather as charged or neutral hadrons
    for node in g.nodes:
        if node[0] == "sc" or node[0] == "tp":
            nd = g.nodes[node]
            if nd["pt"] < 1.0 and (abs(nd["typ"]) == 11 or abs(nd["typ"]) == 13):
                E_track = 0.0
                E_calo = 0.0
                for suc in g.successors(node):
                    elem_type = g.nodes[suc]["typ"]
                    if elem_type == 1:
                        E_track += g.edges[node, suc]["weight"]
                    elif elem_type == 4 or elem_type == 5:
                        E_calo += g.edges[node, suc]["weight"]
                if E_track > E_calo:
                    g.nodes[node]["typ"] = 211
                else:
                    g.nodes[node]["typ"] = 130

             
    return g

def prepare_normalized_table(g, genparticle_energy_threshold=0.2):
    #rg = g.reverse()

    all_genparticles = []
    all_elements = []
    all_pfcandidates = []
    for node in g.nodes:
        if node[0] == "elem":
            all_elements += [node]
            for parent in g.predecessors(node):
                all_genparticles += [parent]
        elif node[0] == "pfcand":
            all_pfcandidates += [node]
    all_genparticles = list(set(all_genparticles))
    all_elements = sorted(all_elements)

    #assign genparticles in reverse energy order uniquely to best element
    elem_to_gp = {} #map of element -> genparticles
    unmatched_gp = []
    for gp in sorted(all_genparticles, key=lambda x: g.nodes[x]["e"], reverse=True):
        elems = [e for e in g.successors(gp)]

        #sort elements by energy deposit from genparticle
        elems_sorted = sorted([(g.edges[gp, e]["weight"], e) for e in elems], key=lambda x: x[0], reverse=True)

        chosen_elem = None
        for weight, elem in elems_sorted:
            if not (elem in elem_to_gp):
                chosen_elem = elem
                elem_to_gp[elem] = []
                break

        if chosen_elem is None:
            unmatched_gp += [gp]
        else:
            elem_to_gp[elem] += [gp]

    #assign unmatched genparticles to best element, allowing for overlaps
    for gp in sorted(unmatched_gp, key=lambda x: g.nodes[x]["e"], reverse=True):
        elems = [e for e in g.successors(gp)]
        elems_sorted = sorted([(g.edges[gp, e]["weight"], e) for e in elems], key=lambda x: x[0], reverse=True)
        _, elem = elems_sorted[0]
        elem_to_gp[elem] += [gp]
 
    unmatched_cand = [] 
    elem_to_cand = {}

    #Find primary element for each PFCandidate
    for cand in sorted(all_pfcandidates, key=lambda x: g.nodes[x]["e"], reverse=True):
        tp = g.nodes[cand]["typ"]
        neighbors = list(g.predecessors(cand))

        chosen_elem = None

        #Pions, muons and electrons will be assigned to tracks
        if tp in [211, 13, 11]:
            for elem in neighbors:
                tp_neighbor = g.nodes[elem]["typ"]

                #track or gsf
                if tp_neighbor==1 or tp_neighbor==6:
                    if not (elem in elem_to_cand):
                        chosen_elem = elem
                        elem_to_cand[elem] = cand
                        break

        #other particles will be assigned to the highest-energy cluster (ECAL, HCAL, HFEM, HFHAD, SC)
        else:
            neighbors = [n for n in neighbors if g.nodes[n]["typ"] in [4,5,8,9,10]]
            sorted_neighbors = sorted(neighbors, key=lambda x: g.nodes[x]["e"], reverse=True)
            for elem in sorted_neighbors:
                if not (elem in elem_to_cand):
                    chosen_elem = elem
                    elem_to_cand[elem] = cand
                    break

        if chosen_elem is None:
            print("unmatched candidate {}, {}".format(cand, g.nodes[cand]))
            unmatched_cand += [cand]

    Xelem = np.recarray((len(all_elements),), dtype=[(name, np.float32) for name in elem_branches])
    Xelem.fill(0.0)
    ygen = np.recarray((len(all_elements),), dtype=[(name, np.float32) for name in target_branches])
    ygen.fill(0.0)
    ycand = np.recarray((len(all_elements),), dtype=[(name, np.float32) for name in target_branches])
    ycand.fill(0.0)
 
    for ielem, elem in enumerate(all_elements):
        elem_type = g.nodes[elem]["typ"]
        elem_eta = g.nodes[elem]["eta"]
        genparticles = sorted(elem_to_gp.get(elem, []), key=lambda x: g.edges[(x, elem)]["weight"], reverse=True)
        genparticles = [gp for gp in genparticles if g.nodes[gp]["e"] > genparticle_energy_threshold]
        candidate = elem_to_cand.get(elem, None)
       
        lv = vector.obj(x=0,y=0,z=0,t=0)

        for j in range(len(elem_branches)):
            Xelem[elem_branches[j]][ielem] = g.nodes[elem][elem_branches[j]]

        if not (candidate is None):
            for j in range(len(target_branches)):
                ycand[target_branches[j]][ielem] = g.nodes[candidate][target_branches[j]]

        if len(genparticles)>0:
            #print("elem type={} E={:.2f} eta={:.2f} phi={:.2f}".format(g.nodes[elem]["typ"], g.nodes[elem]["e"], g.nodes[elem]["eta"], g.nodes[elem]["phi"]))
            #for gp in genparticles:
            #    print("  gp type={} E={:.2f} eta={:.2f} phi={:.2f} w={:.2f}".format(g.nodes[gp]["typ"], g.nodes[gp]["e"], g.nodes[gp]["eta"], g.nodes[gp]["phi"], g.edges[(gp, elem)]["weight"]))
                
            pid = g.nodes[genparticles[0]]["typ"]

            for gp in genparticles:
                lv += vector.obj(
                    pt=g.nodes[gp]["pt"],
                    eta=g.nodes[gp]["eta"],
                    phi=g.nodes[gp]["phi"],
                    e=g.nodes[gp]["e"]
                )

            if elem_type in [8,9]:
                #HFHAD -> always produce hadronic candidate
                if elem_type == 9:
                    pid = 1
                #HFEM -> decide based on pid
                elif elem_type == 8:
                    if pid in [11, 22]:
                        pid = 2 #produce EM candidate 
                    else:
                        pid = 1 #produce hadronic

            #remap PID in case of HCAL cluster
            if elem_type == 5 and (pid == 22 or pid == 11):
                pid = 130
       
            gp = {
                "pt": lv.rho, "eta": lv.eta, "sin_phi": np.sin(lv.phi), "cos_phi": np.cos(lv.phi), "e": lv.t, "typ": pid, "px": lv.x, "py": lv.y, "pz": lv.z, "charge": get_charge(pid)
            }
            #print("{},{}".format(elem_type, pid))

            for j in range(len(target_branches)):
                ygen[target_branches[j]][ielem] = gp[target_branches[j]]

    return Xelem, ycand, ygen
#end of prepare_normalized_table

def make_graph(ev, iev):
    element_type = ev['element_type'][iev]
    element_pt = ev['element_pt'][iev]
    element_e = ev['element_energy'][iev]
    element_eta = ev['element_eta'][iev]
    element_phi = ev['element_phi'][iev]
    element_eta_ecal = ev['element_eta_ecal'][iev]
    element_phi_ecal = ev['element_phi_ecal'][iev]
    element_eta_hcal = ev['element_eta_hcal'][iev]
    element_phi_hcal = ev['element_phi_hcal'][iev]
    element_trajpoint = ev['element_trajpoint'][iev]
    element_layer = ev['element_layer'][iev]
    element_charge = ev['element_charge'][iev]
    element_depth = ev['element_depth'][iev]
    element_deltap = ev['element_deltap'][iev]
    element_sigmadeltap = ev['element_sigmadeltap'][iev]
    element_px = ev['element_px'][iev]
    element_py = ev['element_py'][iev]
    element_pz = ev['element_pz'][iev]
    element_muon_dt_hits = ev['element_muon_dt_hits'][iev]
    element_muon_csc_hits = ev['element_muon_csc_hits'][iev]
    element_muon_type = ev['element_muon_type'][iev]
    element_gsf_electronseed_trkorecal = ev['element_gsf_electronseed_trkorecal'][iev]
    element_gsf_electronseed_dnn1 = ev['element_gsf_electronseed_dnn1'][iev]
    element_gsf_electronseed_dnn2 = ev['element_gsf_electronseed_dnn2'][iev]
    element_gsf_electronseed_dnn3 = ev['element_gsf_electronseed_dnn3'][iev]
    element_gsf_electronseed_dnn4 = ev['element_gsf_electronseed_dnn4'][iev]
    element_gsf_electronseed_dnn5 = ev['element_gsf_electronseed_dnn5'][iev]
    element_num_hits = ev['element_num_hits'][iev]
    element_cluster_flags = ev['element_cluster_flags'][iev]
    element_corr_energy = ev['element_corr_energy'][iev]
    element_corr_energy_err = ev['element_corr_energy_err'][iev]
    element_pterror = ev['element_pterror'][iev]
    element_etaerror = ev['element_etaerror'][iev]
    element_phierror = ev['element_phierror'][iev]
    element_lambda = ev['element_lambda'][iev]
    element_theta = ev['element_theta'][iev]
    element_lambdaerror = ev['element_lambdaerror'][iev]
    element_thetaerror = ev['element_thetaerror'][iev]
    element_vx = ev['element_vx'][iev]
    element_vy = ev['element_vy'][iev]
    element_vz = ev['element_vz'][iev]

    trackingparticle_pid = ev['trackingparticle_pid'][iev]
    trackingparticle_pt = ev['trackingparticle_pt'][iev]
    trackingparticle_e = ev['trackingparticle_energy'][iev]
    trackingparticle_eta = ev['trackingparticle_eta'][iev]
    trackingparticle_phi = ev['trackingparticle_phi'][iev]
    trackingparticle_phi = ev['trackingparticle_phi'][iev]
    trackingparticle_px = ev['trackingparticle_px'][iev]
    trackingparticle_py = ev['trackingparticle_py'][iev]
    trackingparticle_pz = ev['trackingparticle_pz'][iev]

    caloparticle_pid = ev['caloparticle_pid'][iev]
    caloparticle_pt = ev['caloparticle_pt'][iev]
    caloparticle_e = ev['caloparticle_energy'][iev]
    caloparticle_eta = ev['caloparticle_eta'][iev]
    caloparticle_phi = ev['caloparticle_phi'][iev]
    caloparticle_idx_trackingparticle = ev['caloparticle_idx_trackingparticle'][iev]

    pfcandidate_pdgid = ev['pfcandidate_pdgid'][iev]
    pfcandidate_pt = ev['pfcandidate_pt'][iev]
    pfcandidate_e = ev['pfcandidate_energy'][iev]
    pfcandidate_eta = ev['pfcandidate_eta'][iev]
    pfcandidate_phi = ev['pfcandidate_phi'][iev]
    pfcandidate_px = ev['pfcandidate_px'][iev]
    pfcandidate_py = ev['pfcandidate_py'][iev]
    pfcandidate_pz = ev['pfcandidate_pz'][iev]

    g = nx.DiGraph()
    for iobj in range(len(element_type)):
        g.add_node(("elem", iobj),
            typ=element_type[iobj],
            pt=element_pt[iobj],
            e=element_e[iobj],
            eta=element_eta[iobj],
            phi=element_phi[iobj],
            eta_ecal=element_eta_ecal[iobj],
            phi_ecal=element_phi_ecal[iobj],
            eta_hcal=element_eta_hcal[iobj],
            phi_hcal=element_phi_hcal[iobj],
            trajpoint=element_trajpoint[iobj],
            layer=element_layer[iobj],
            charge=element_charge[iobj],
            depth=element_depth[iobj],
            deltap=element_deltap[iobj],
            sigmadeltap=element_sigmadeltap[iobj],
            px=element_px[iobj],
            py=element_py[iobj],
            pz=element_pz[iobj],
            muon_dt_hits=element_muon_dt_hits[iobj],
            muon_csc_hits=element_muon_csc_hits[iobj],
            muon_type=element_muon_type[iobj],
            gsf_electronseed_trkorecal=element_gsf_electronseed_trkorecal[iobj],
            gsf_electronseed_dnn1=element_gsf_electronseed_dnn1[iobj],
            gsf_electronseed_dnn2=element_gsf_electronseed_dnn2[iobj],
            gsf_electronseed_dnn3=element_gsf_electronseed_dnn3[iobj],
            gsf_electronseed_dnn4=element_gsf_electronseed_dnn4[iobj],
            gsf_electronseed_dnn5=element_gsf_electronseed_dnn5[iobj],
            num_hits=element_num_hits[iobj],
            cluster_flags=element_cluster_flags[iobj],
            corr_energy=element_corr_energy[iobj],
            corr_energy_err=element_corr_energy_err[iobj],
            pterror=element_pterror[iobj],
            etaerror=element_etaerror[iobj],
            phierror=element_phierror[iobj],
            lambd=element_lambda[iobj],
            theta=element_theta[iobj],
            lambdaerror=element_lambdaerror[iobj],
            thetaerror=element_thetaerror[iobj],
            vx=element_vx[iobj],
            vy=element_vy[iobj],
            vz=element_vz[iobj],
        )
    for iobj in range(len(trackingparticle_pid)):
        g.add_node(("tp", iobj),
            typ=trackingparticle_pid[iobj],
            pt=trackingparticle_pt[iobj],
            e=trackingparticle_e[iobj],
            eta=trackingparticle_eta[iobj],
            phi=trackingparticle_phi[iobj],
        )
    for iobj in range(len(caloparticle_pid)):
        g.add_node(("sc", iobj),
            typ=caloparticle_pid[iobj],
            pt=caloparticle_pt[iobj],
            e=caloparticle_e[iobj],
            eta=caloparticle_eta[iobj],
            phi=caloparticle_phi[iobj],
        )

    for iobj in range(len(pfcandidate_pdgid)):
        g.add_node(("pfcand", iobj),
            typ=abs(pfcandidate_pdgid[iobj]),
            pt=pfcandidate_pt[iobj],
            e=pfcandidate_e[iobj],
            eta=pfcandidate_eta[iobj],
            sin_phi=np.sin(pfcandidate_phi[iobj]),
            cos_phi=np.cos(pfcandidate_phi[iobj]),
            charge=get_charge(pfcandidate_pdgid[iobj]),
        )

    trackingparticle_to_element_first = ev['trackingparticle_to_element.first'][iev]
    trackingparticle_to_element_second = ev['trackingparticle_to_element.second'][iev]
    #for trackingparticles associated to elements, set a very high edge weight
    for tp, elem in zip(trackingparticle_to_element_first, trackingparticle_to_element_second):
        if not (g.nodes[("elem", elem)]["typ"] in [2,3,7]):
            g.add_edge(("tp", tp), ("elem", elem), weight=float("inf"))
 
    caloparticle_to_element_first = ev['caloparticle_to_element.first'][iev]
    caloparticle_to_element_second = ev['caloparticle_to_element.second'][iev]
    caloparticle_to_element_cmp = ev['caloparticle_to_element_cmp'][iev]
    for sc, elem, c in zip(caloparticle_to_element_first, caloparticle_to_element_second, caloparticle_to_element_cmp):
        if not (g.nodes[("elem", elem)]["typ"] in [2,3,7]):
            g.add_edge(("sc", sc), ("elem", elem), weight=c)

    #merge caloparticles and trackingparticles that refer to the same particle
    nodes_to_remove = []
    for idx_sc, idx_tp in enumerate(caloparticle_idx_trackingparticle):
        if idx_tp != -1:
            for elem in g.neighbors(("sc", idx_sc)):
                g.add_edge(("tp", idx_tp), elem, weight=g.edges[("sc", idx_sc), elem]["weight"]) 
            g.nodes[("tp", idx_tp)]["idx_sc"] = idx_sc   
            nodes_to_remove += [("sc", idx_sc)]
    g.remove_nodes_from(nodes_to_remove)

    element_to_candidate_first = ev['element_to_candidate.first'][iev]
    element_to_candidate_second = ev['element_to_candidate.second'][iev]
    for elem, pfcand in zip(element_to_candidate_first, element_to_candidate_second):
        g.add_edge(("elem", elem), ("pfcand", pfcand), weight=1.0)

    return g

def gen_e(g):
    etot_gen = 0.0
    etot_pf = 0.0
    for node in g.nodes:
        if node[0] == "tp" or node[0] == "sc":
            etot_gen += g.nodes[node]["e"]
        if node[0] == "pfcand":
            etot_pf += g.nodes[node]["e"]
    return etot_gen, etot_pf

def process(args):
    infile = args.input
    outpath = os.path.join(args.outpath, os.path.basename(infile).split(".")[0])
    tf = uproot.open(infile)
    tt = tf["ana/pftree"]
    if args.num_events == -1:
        args.num_events = tt.num_entries
    events_to_process = [i for i in range(args.num_events)] 

    all_data = []
    ifile = 0
    ev = tt.arrays(library="np")
    for iev in tqdm.tqdm(events_to_process):

        g = make_graph(ev, iev)
        g = cleanup_graph(g)

        #associate target particles to input elements
        Xelem, ycand, ygen = prepare_normalized_table(g)
        data = {}

        if args.save_normalized_table:
            data = {
                "Xelem": Xelem,
                "ycand": ycand,
                "ygen": ygen,
            }

        if args.save_full_graph:
            data["full_graph"] = g

        all_data += [data]

    with open(outpath + ".pkl", "wb") as fi:
        pickle.dump(all_data, fi)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file from PFAnalysis", required=True)
    parser.add_argument("--outpath", type=str, default="raw", help="output path")
    parser.add_argument("--save-full-graph", action="store_true", help="save the full event graph")
    parser.add_argument("--save-normalized-table", action="store_true", help="save the uniquely identified table")
    parser.add_argument("--num-events", type=int, help="number of events to process", default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    process(args)

