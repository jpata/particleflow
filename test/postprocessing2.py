import sys
import ROOT
import pickle
import networkx as nx
import numpy as np
import numba

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy
import scipy.sparse
from networkx.readwrite import json_graph
from networkx.drawing.nx_pydot import graphviz_layout

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


def prepare_elem_distance_matrix(ev):
    di = np.array(list(ev.element_distance_i))
    dj = np.array(list(ev.element_distance_j))
    d = np.array(list(ev.element_distance_d))
    etas = np.array(list(ev.element_eta), dtype=np.float32)
    phis = np.array(list(ev.element_phi), dtype=np.float32)
    etaphis = np.vstack([etas, phis]).T
    dm_dr = np.zeros((etaphis.shape[0], etaphis.shape[0]), dtype=np.float32)
    associate_deltar(etaphis, 0.2**2, dm_dr)
    n = len(etas)
    dm = scipy.sparse.coo_matrix((d, (di, dj)), shape=(n,n)).todense()
    dm += dm_dr 
    dm += dm.T
    dm = scipy.sparse.coo_matrix(dm)
    return dm

def save_ego_graph(g, node, radius=4, undirected=False):
    sg = nx.ego_graph(g, node, radius, undirected=undirected).reverse()

    #remove BREM PFElements from plotting 
    nodes_to_remove = [n for n in sg.nodes if (n[0]=="elem" and sg.nodes[n]["typ"] in [7,])]
    sg.remove_nodes_from(nodes_to_remove)

    fig = plt.figure(figsize=(2*len(sg.nodes)+2, 10))
    sg_pos = graphviz_layout(sg, prog='dot')
    
    edge_labels = {}
    for e in sg.edges:
        if e[1][0] == "elem" and not (sg.nodes[e[1]]["typ"] in [1,10]):
            edge_labels[e] = "{:.2f} GeV".format(sg.edges[e].get("weight", 0))
        else:
            edge_labels[e] = ""
    
    node_labels = {}
    for node in sg.nodes:
        labels = {"sc": "SimCluster", "elem": "PFElement", "tp": "TrackingParticle", "pfcand": "PFCandidate"}
        node_labels[node] = "[{label} {idx}] \ntype: {typ}\ne: {e:.4f} GeV\npt: {pt:.4f} GeV\neta: {eta:.4f}\nphi: {phi:.4f}\nc/p: {children}/{parents}".format(
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

def cleanup_graph(g, edge_energy_threshold=0.01, edge_fraction_threshold=0.05, genparticle_energy_threshold=0.2, genparticle_pt_threshold=0.01):
    g = g.copy()

    edges_to_remove = []
    nodes_to_remove = []

    #remove edges that contribute little
    for edge in g.edges:
        if edge[0][0] == "sc":
            w = g.edges[edge]["weight"]
            if w < edge_energy_threshold:
                edges_to_remove += [edge]
        if edge[0][0] == "sc" or edge[0][0] == "tp":
            if g.nodes[edge[1]]["typ"] == 10:
                g.edges[edge]["weight"] = 1.0
                
    #remove genparticles below energy threshold
    for node in g.nodes:
        if (node[0]=="sc" or node[0]=="tp") and g.nodes[node]["e"] < genparticle_energy_threshold:
            nodes_to_remove += [node]
    
    g.remove_edges_from(edges_to_remove)
    g.remove_nodes_from(nodes_to_remove)
    
    rg = g.reverse()
    
    #for each element, remove the incoming edges that contribute less than 5% of the total
    edges_to_remove = []
    nodes_to_remove = []
    for node in rg.nodes:
        if node[0] == "elem":
            ##check for generator pairs with very similar eta,phi, which can come from gamma->e+ e-
            #if rg.nodes[node]["typ"] == 4:
            #    by_eta_phi = {}
            #    for neigh in rg.neighbors(node):
            #        k = (round(rg.nodes[neigh]["eta"], 2), round(rg.nodes[neigh]["phi"], 2))
            #        if not k in by_eta_phi:
            #            by_eta_phi[k] = []
            #        by_eta_phi[k] += [neigh]
    
            #    for k in by_eta_phi:
            #        #if there were genparticles with the same eta,phi, assume it was a photon with nuclear interaction
            #        if len(by_eta_phi[k])>=2:
            #            #print(by_eta_phi[k][0])
            #            rg.nodes[by_eta_phi[k][0]]["typ"] = 22
            #            rg.nodes[by_eta_phi[k][0]]["e"] += sum(rg.nodes[n]["e"] for n in by_eta_phi[k][1:])
            #            rg.nodes[by_eta_phi[k][0]]["pt"] = 0 #fixme
            #            nodes_to_remove += by_eta_phi[k][1:]
                    
            ew = [((node, node2), rg.edges[node, node2]["weight"]) for node2 in rg.neighbors(node)]
            ew = filter(lambda x: x[1] != 1.0, ew)
            ew = sorted(ew, key=lambda x: x[1], reverse=True)
            if len(ew) > 1:
                max_in = ew[0][1]
                for e, w in ew[1:]:
                    if w / max_in < edge_fraction_threshold:
                        edges_to_remove += [e]
    
    rg.remove_edges_from(edges_to_remove)        
    rg.remove_nodes_from(nodes_to_remove)        
    g = rg.reverse()
    
    #remove genparticles not linked to any elements
    nodes_to_remove = []
    for node in g.nodes:
        if node[0]=="sc" or node[0]=="tp":
            deg = g.degree[node]
            if deg==0:
                nodes_to_remove += [node]
    g.remove_nodes_from(nodes_to_remove)
    
    for node in g.nodes:
        g.nodes[node]["children"] = len(list(g.neighbors(node)))
    
    rg = g.reverse()
    
    for node in rg.nodes:
        g.nodes[node]["parents"] = len(list(rg.neighbors(node)))
        rg.nodes[node]["parents"] = len(list(rg.neighbors(node)))

    return g

def prepare_normalized_table(g):
    rg = g.reverse()

    all_genparticles = []
    all_elements = []
    all_pfcandidates = []
    for node in rg.nodes:
        if node[0] == "elem":
            all_elements += [node]
            for parent in rg.neighbors(node):
                all_genparticles += [parent]
        elif node[0] == "pfcand":
            all_pfcandidates += [node]
    all_genparticles = list(set(all_genparticles))
    all_elements = sorted(all_elements)

    #assign genparticles in reverse pt order uniquely to best element
    elem_to_gp = {}
    unmatched_gp = []
    for gp in sorted(all_genparticles, key=lambda x: g.nodes[x]["pt"], reverse=True):
        elems = [e for e in g.neighbors(gp)]

        #don't assign any genparticle to these elements (PS, BREM, SC)
        elems = [e for e in elems if not (g.nodes[e]["typ"] in [2,3,7,10])]

        #sort elements by energy from genparticle
        elems_sorted = sorted([(g.edges[gp, e]["weight"], e) for e in elems], key=lambda x: x[0], reverse=True)
        
        if len(elems_sorted) == 0:
            continue

        chosen_elem = None
        for _, elem in elems_sorted:
            if not (elem in elem_to_gp):
                chosen_elem = elem
                elem_to_gp[elem] = []
                break
        if chosen_elem is None:
            unmatched_gp += [gp]
        else:
            elem_to_gp[elem] += [gp]

    #assign unmatched genparticles to best element, allowing for overlaps
    for gp in sorted(unmatched_gp, key=lambda x: g.nodes[x]["pt"], reverse=True):
        elems = [e for e in g.neighbors(gp)]
        elems = [e for e in elems if not (g.nodes[e]["typ"] in [2,3,7,10])]
        elems_sorted = sorted([(g.edges[gp, e]["weight"], e) for e in elems], key=lambda x: x[0], reverse=True)
        _, elem = elems_sorted[0]
        elem_to_gp[elem] += [gp]
 
    unmatched_cand = [] 
    elem_to_cand = {} 
    for cand in sorted(all_pfcandidates, key=lambda x: g.nodes[x]["pt"], reverse=True):
        tp = g.nodes[cand]["typ"]
        neighbors = list(rg.neighbors(cand))

        chosen_elem = None

        #Pions and muons will be assigned to tracks
        if abs(tp) == 211 or abs(tp) == 13:
            for elem in neighbors:
                tp_neighbor = g.nodes[elem]["typ"]
                if tp_neighbor == 1:
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

    elem_branches = [
        "typ", "pt", "eta", "phi", "e",
        "layer", "depth", "charge", "trajpoint", 
        "eta_ecal", "phi_ecal", "eta_hcal", "phi_hcal"
    ]
    target_branches = ["typ", "pt", "eta", "phi", "e", "px", "py", "pz"]

    Xelem = np.zeros((len(all_elements), len(elem_branches)), dtype=np.float32)
    ygen = np.zeros((len(all_elements), len(target_branches)), dtype=np.float32)
    ycand = np.zeros((len(all_elements), len(target_branches)), dtype=np.float32)
 
    #find which elements should be linked together in the output when regressing to PFCandidates or GenParticles
    graph_elem_cand = nx.Graph()  
    graph_elem_gen = nx.Graph()  
    for elem in all_elements:
        graph_elem_cand.add_node(elem) 
        graph_elem_gen.add_node(elem) 
 
    for cand in all_pfcandidates:
        for elem1 in rg.neighbors(cand):
            for elem2 in rg.neighbors(cand):
                if (elem1 != elem2):
                    graph_elem_cand.add_edge(elem1, elem2)

    for gp in all_genparticles:
        for elem1 in g.neighbors(gp):
            for elem2 in g.neighbors(gp):
                if (elem1 != elem2):
                    graph_elem_gen.add_edge(elem1, elem2)
 
    for ielem, elem in enumerate(all_elements):
        elem_type = g.nodes[elem]["typ"]
        elem_eta = g.nodes[elem]["eta"]
        genparticles = sorted(elem_to_gp.get(elem, []), key=lambda x: g.nodes[x]["pt"], reverse=True)
        candidate = elem_to_cand.get(elem, None)
       
        lvs = []
       
        pid = 0
        if len(genparticles) > 0:
            pid = map_pdgid_to_candid.get(g.nodes[genparticles[0]]["typ"], 0)

        for gp in genparticles:
            lv = ROOT.TLorentzVector()
            lv.SetPtEtaPhiE(
                g.nodes[gp]["pt"], 
                g.nodes[gp]["eta"], 
                g.nodes[gp]["phi"], 
                g.nodes[gp]["e"]
            )
            lvs += [lv]
        lv = sum(lvs, ROOT.TLorentzVector())

        if len(genparticles) > 0:
            if abs(elem_eta) > 3.0:
                #HFHAD -> always produce hadronic
                if elem_type == 9:
                    pid = 1
                #HFEM -> decide based on pid
                elif elem_type == 8:
                    if abs(pid) in [11, 22]:
                        pid = 2 #produce EM candidate 
                    else:
                        pid = 1 #produce hadronic

            #remap PID in case of HCAL cluster
            if elem_type == 5 and (pid == 22 or abs(pid) == 11):
                pid = 130

        gp = {
            "pt": lv.Pt(), "eta": lv.Eta(), "phi": lv.Phi(), "e": lv.E(), "typ": pid, "px": lv.Px(), "py": lv.Py(), "pz": lv.Pz()
        }

        for j in range(len(elem_branches)):
            Xelem[ielem, j] = g.nodes[elem][elem_branches[j]]

        for j in range(len(target_branches)):
            if not (candidate is None):
                ycand[ielem, j] = g.nodes[candidate][target_branches[j]]
            ygen[ielem, j] = gp[target_branches[j]]

    dm_elem_cand = scipy.sparse.coo_matrix(nx.to_numpy_matrix(graph_elem_cand, nodelist=all_elements))
    dm_elem_gen = scipy.sparse.coo_matrix(nx.to_numpy_matrix(graph_elem_gen, nodelist=all_elements))
    return Xelem, ycand, ygen, dm_elem_cand, dm_elem_gen
#end of prepare_normalized_table

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file from PFAnalysis", required=True)
    parser.add_argument("--event", type=int, default=None, help="event to process, omit to process all")
    parser.add_argument("--plot-candidates", type=int, default=0, help="number of PFCandidates to plot")
    parser.add_argument("--events-per-file", type=int, default=-1, help="number of events per output file")
    args = parser.parse_args()

    infile = args.input
    outpath = "raw/" + infile.split(".")[0]
    tf = ROOT.TFile(infile)
    tt = tf.Get("ana/pftree")

    events_to_process = [i for i in range(tt.GetEntries())] 
    if not (args.event is None):
        events_to_process = [args.event]

    all_data = []
    ifile = 0
    for iev, ev in enumerate(tt):
        if not (iev in events_to_process):
            continue
        print("processing event {}".format(iev))

        element_type = ev.element_type
        element_pt = ev.element_pt
        element_e = ev.element_energy
        element_eta = ev.element_eta
        element_phi = ev.element_phi
        element_eta_ecal = ev.element_eta_ecal
        element_phi_ecal = ev.element_phi_ecal
        element_eta_hcal = ev.element_eta_hcal
        element_phi_hcal = ev.element_phi_hcal
        element_trajpoint = ev.element_trajpoint
        element_layer = ev.element_layer
        element_charge = ev.element_charge
        element_depth = ev.element_depth
        element_deltap = ev.element_deltap
        element_sigmadeltap = ev.element_sigmadeltap
        element_px = ev.element_px
        element_py = ev.element_py
        element_pz = ev.element_pz

        trackingparticle_pid = ev.trackingparticle_pid
        trackingparticle_pt = ev.trackingparticle_pt
        trackingparticle_e = ev.trackingparticle_energy
        trackingparticle_eta = ev.trackingparticle_eta
        trackingparticle_phi = ev.trackingparticle_phi
        trackingparticle_phi = ev.trackingparticle_phi
        trackingparticle_px = ev.trackingparticle_px
        trackingparticle_py = ev.trackingparticle_py
        trackingparticle_pz = ev.trackingparticle_pz

        simcluster_pid = ev.simcluster_pid
        simcluster_pt = ev.simcluster_pt
        simcluster_e = ev.simcluster_energy
        simcluster_eta = ev.simcluster_eta
        simcluster_phi = ev.simcluster_phi
        simcluster_px = ev.simcluster_px
        simcluster_py = ev.simcluster_py
        simcluster_pz = ev.simcluster_pz

        simcluster_idx_trackingparticle = ev.simcluster_idx_trackingparticle
        pfcandidate_pdgid = ev.pfcandidate_pdgid
        pfcandidate_pt = ev.pfcandidate_pt
        pfcandidate_e = ev.pfcandidate_energy
        pfcandidate_eta = ev.pfcandidate_eta
        pfcandidate_phi = ev.pfcandidate_phi
        pfcandidate_px = ev.pfcandidate_px
        pfcandidate_py = ev.pfcandidate_py
        pfcandidate_pz = ev.pfcandidate_pz

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
            )
        for iobj in range(len(trackingparticle_pid)):
            g.add_node(("tp", iobj),
                typ=trackingparticle_pid[iobj],
                pt=trackingparticle_pt[iobj],
                e=trackingparticle_e[iobj],
                eta=trackingparticle_eta[iobj],
                phi=trackingparticle_phi[iobj],
                px=trackingparticle_px[iobj],
                py=trackingparticle_py[iobj],
                pz=trackingparticle_pz[iobj],
            )
        for iobj in range(len(simcluster_pid)):
            g.add_node(("sc", iobj),
                typ=simcluster_pid[iobj],
                pt=simcluster_pt[iobj],
                e=simcluster_e[iobj],
                eta=simcluster_eta[iobj],
                phi=simcluster_phi[iobj],
                px=simcluster_px[iobj],
                py=simcluster_py[iobj],
                pz=simcluster_pz[iobj],
            )

        trackingparticle_to_element = ev.trackingparticle_to_element
        #for trackingparticles associated to elements, set a very high edge weight
        for obj in trackingparticle_to_element:
            g.add_edge(("tp", obj.first), ("elem", obj.second), weight=99999.0)
 
        simcluster_to_element = ev.simcluster_to_element
        simcluster_to_element_cmp = ev.simcluster_to_element_cmp
        for obj, c in zip(simcluster_to_element, simcluster_to_element_cmp):
            g.add_edge(("sc", obj.first), ("elem", obj.second), weight=c)

        print("contracting nodes: trackingparticle to simcluster")
        nodes_to_remove = []
        for idx_sc, idx_tp in enumerate(simcluster_idx_trackingparticle):
            if idx_tp != -1:
                for elem in g.neighbors(("sc", idx_sc)):
                    g.add_edge(("tp", idx_tp), elem, weight=g.edges[("sc", idx_sc), elem]["weight"]) 
                g.nodes[("tp", idx_tp)]["idx_sc"] = idx_sc   
                nodes_to_remove += [("sc", idx_sc)]
        g.remove_nodes_from(nodes_to_remove)

        for iobj in range(len(pfcandidate_pdgid)):
            g.add_node(("pfcand", iobj),
                typ=pfcandidate_pdgid[iobj],
                pt=pfcandidate_pt[iobj],
                e=pfcandidate_e[iobj],
                eta=pfcandidate_eta[iobj],
                phi=pfcandidate_phi[iobj],
                px=pfcandidate_px[iobj],
                py=pfcandidate_py[iobj],
                pz=pfcandidate_pz[iobj],
            )

        element_to_candidate = ev.element_to_candidate
        for obj in element_to_candidate:
            g.add_edge(("elem", obj.first), ("pfcand", obj.second), weight=1.0)
        print("Graph created: {} nodes, {} edges".format(len(g.nodes), len(g.edges)))
 
        g =  cleanup_graph(g)
        rg = g.reverse()

        ncand = 0 
        for node in sorted(filter(lambda x: x[0]=="pfcand", g.nodes), key=lambda x: g.nodes[x]["pt"], reverse=True):
            if ncand < args.plot_candidates:
                print(node, g.nodes[node]["pt"])
                fig = save_ego_graph(rg, node, 3, False)
                plt.savefig(outpath + "_ev_{}_cand_{}_idx_{}.pdf".format(iev, ncand, node[1]), bbox_inches="tight")
                plt.clf()
                del fig
            ncand += 1

        #fig = draw_event(g)
        #plt.savefig(outpath + "_ev_{}.pdf".format(iev))
        #plt.clf()

        Xelem, ycand, ygen, dm_elem_cand, dm_elem_gen = prepare_normalized_table(g)
        dm = prepare_elem_distance_matrix(ev)
        data = {
            "Xelem": Xelem,
            "ycand": ycand,
            "ygen": ygen,
            "dm": dm,
            "dm_elem_cand": dm_elem_cand,
            "dm_elem_gen": dm_elem_gen
        }
        all_data += [data]

        if args.events_per_file > 0:
            if len(all_data) == args.events_per_file:
                with open(outpath + "_{}.pkl".format(ifile), "wb") as fi:
                    pickle.dump(all_data, fi)
                ifile += 1
                all_data = []

    if args.events_per_file == -1:
        with open(outpath + ".pkl", "wb") as fi:
            pickle.dump(all_data, fi)
