import uproot
import networkx as nx
import math
from collections import Counter
import numpy as np
import sys

def load_file(fn):
    fi = uproot.open(fn)
    tree = fi.get("pftree")
    data = tree.arrays(tree.keys())
    data = {str(k, 'ascii'): v for k, v in data.items()}
    
    #get the list of element (iblock, ielem) to candidate associations
    linktree = fi.get("linktree_elemtocand")
    data_elemtocand = linktree.arrays(linktree.keys())
    data_elemtocand = {str(k, 'ascii'): v for k, v in data_elemtocand.items()}

    return data, data_elemtocand

def create_graph_elements_candidates(data, data_elemtocand, iev):

    pfgraph = nx.Graph()
    node_pos = {}
    node_colors = {}

    for i in range(len(data["clusters_iblock"][iev])):
        ibl = data["clusters_iblock"][iev][i]
        iel = data["clusters_ielem"][iev][i]
        this = ("E", ibl, iel)
        node_pos[this] = data["clusters_eta"][iev][i], data["clusters_phi"][iev][i]
        node_colors[this] = "green"
        pfgraph.add_node(this, type=data["clusters_type"][iev][i])
        
    for i in range(len(data["tracks_iblock"][iev])):
        ibl = data["tracks_iblock"][iev][i]
        iel = data["tracks_ielem"][iev][i]
        this = ("E", ibl, iel)
        #node_pos[this] = data["tracks_eta"][iev][i], data["tracks_phi"][iev][i]
        #node_pos[this] = data["tracks_inner_eta"][iev][i], data["tracks_inner_phi"][iev][i]
        node_pos[this] = data["tracks_outer_eta"][iev][i], data["tracks_outer_phi"][iev][i]
        if node_pos[this][0] == 0 and node_pos[this][1] == 0:
            node_pos[this] = data["tracks_inner_eta"][iev][i], data["tracks_inner_phi"][iev][i]
        if node_pos[this][0] == 0 and node_pos[this][1] == 0:
            node_pos[this] = data["tracks_eta"][iev][i], data["tracks_phi"][iev][i]
        node_colors[this] = "r"
        pfgraph.add_node(this, type=1)
        
    for i in range(len(data["pfcands_iblock"][iev])):
        this = ("C", i)
        node_pos[this] = data["pfcands_eta"][iev][i], data["pfcands_phi"][iev][i]
        node_colors[this] = "black"
        pfgraph.add_node(this, type=-1)
        
    for i in range(len(data_elemtocand["linkdata_elemtocand_ielem"][iev])):
        ibl = data_elemtocand["linkdata_elemtocand_iblock"][iev][i]
        iel = data_elemtocand["linkdata_elemtocand_ielem"][iev][i]
        ic = data_elemtocand["linkdata_elemtocand_icand"][iev][i]
        u = ("E", ibl, iel)
        v = ("C", ic)
        if u in pfgraph.nodes and v in pfgraph.nodes:
            p0 = node_pos[u]
            p1 = node_pos[v]
            dist = math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
            pfgraph.add_edge(u, v, weight=dist)
    
    return pfgraph

def analyze_graph_subgraph_elements(pfgraph):
    sub_graphs = list(nx.connected_component_subgraphs(pfgraph))

    subgraph_element_types = []

    elem_to_newblock = {}
    cand_to_newblock = {}
    for inewblock, sg in enumerate(sub_graphs):
        element_types = []
        for node in sg.nodes:
            node_type = node[0]
            if node_type == "E":
                ibl, iel = node[1], node[2]
                k = (ibl, iel)
                assert(not (k in elem_to_newblock))
                elem_to_newblock[k] = inewblock
                element_type = sg.nodes[node]["type"]
                element_types += [element_type]
            elif node_type == "C":
                icand = node[1]
                k = icand
                assert(not (k in cand_to_newblock))
                cand_to_newblock[k] = inewblock
                
        element_types = tuple(sorted(element_types))
        subgraph_element_types += [element_types] 
    return subgraph_element_types, elem_to_newblock, cand_to_newblock

def assign_cand(iblocks, ielems, elem_to_newblock, _i):
    icands = elem_to_newblock.get((iblocks[_i], ielems[_i]), -1)
    return icands

def prepare_data(data, data_elemtocand, elem_to_newblock, cand_to_newblock, iev):
   
    #clusters
    X1 = np.vstack([
        data["clusters_type"][iev],
        data["clusters_energy"][iev],
        data["clusters_eta"][iev],
        data["clusters_phi"][iev]]
    ).T
    ys1 = np.array([assign_cand(
        data["clusters_iblock"][iev],
        data["clusters_ielem"][iev],
        elem_to_newblock, i)
        for i in range(len(data["clusters_phi"][iev]))
    ])
    
    #tracks
    X2 = np.vstack([
        1*np.ones_like(data["tracks_qoverp"][iev]),
        data["tracks_qoverp"][iev],
        data["tracks_eta"][iev],
        data["tracks_phi"][iev],
        data["tracks_inner_eta"][iev],
        data["tracks_inner_phi"][iev],
        data["tracks_outer_eta"][iev],
        data["tracks_outer_phi"][iev]]
    ).T
    ys2 = np.array([assign_cand(
        data["tracks_iblock"][iev],
        data["tracks_ielem"][iev],
        elem_to_newblock, i)
    for i in range(len(data["tracks_phi"][iev]))])

    X1p = np.pad(X1, ((0,0),(0, X2.shape[1] - X1.shape[1])), mode="constant")
    X = np.vstack([X1p, X2])
    y = np.concatenate([ys1, ys2])
        
    cand_data = np.vstack([
        data["pfcands_pdgid"][iev],
        data["pfcands_pt"][iev],
        data["pfcands_eta"][iev],
        data["pfcands_phi"][iev],
    ]).T
    cand_block_id = np.array([cand_to_newblock.get(ic, -1) for ic in range(len(data["pfcands_phi"][iev]))], dtype=np.int64)
    
    genpart_data = np.vstack([
        data["genparticles_pdgid"][iev],
        data["genparticles_pt"][iev],
        data["genparticles_eta"][iev],
        data["genparticles_phi"][iev],
    ]).T
    
    return X, y, cand_data, cand_block_id

def get_unique_X_y(X, Xbl, y, ybl):
    uniqs = np.unique(Xbl)
    
    Xs = []
    ys = []
    for bl in uniqs:
        subX = X[Xbl==bl]
        suby = y[ybl==bl]
        
        #choose only miniblocks with 3 elements to simplify the problem
        if len(subX) >= 3:
            continue
            
        subX = np.pad(subX, ((0, 3 - subX.shape[0]), (0,0)), mode="constant")
        suby = np.pad(suby, ((0, 3 - suby.shape[0]), (0,0)), mode="constant")
        
        Xs += [subX]
        ys += [suby]
        
    return Xs, ys

if __name__ == "__main__":
    fn = sys.argv[1]
    data, data_elemtocand = load_file(fn)

    all_sgs = []
    for iev in range(len(data)):
        pfgraph = create_graph_elements_candidates(data, data_elemtocand, iev)
        sgs, elem_to_newblock, cand_to_newblock = analyze_graph_subgraph_elements(pfgraph)
        elements, block_id, pfcands, cand_block_id = prepare_data(data, data_elemtocand, elem_to_newblock, cand_to_newblock, iev)

        cache_filename = fn.replace(".root", "_ev{0}.npz".format(iev))
        with open(cache_filename, "wb") as fi:
            np.savez(fi, elements=elements, element_block_id=block_id, candidates=pfcands, candidate_block_id=cand_block_id)
     
        Xs, ys = get_unique_X_y(elements, block_id, pfcands, cand_block_id)
        cache_filename = fn.replace(".root", "_cl{0}.npz".format(iev))
        with open(cache_filename, "wb") as fi:
            np.savez(fi, Xs=Xs, ys=ys)

        all_sgs += sgs
   
    block_sizes = Counter([len(sg) for sg in all_sgs])
    print("block sizes", block_sizes)

    for blocksize in range(1,5): 
        blocks_nelem = Counter([tuple(sg) for sg in all_sgs if len(sg)==blocksize])
        print("{0}-element blocks".format(blocksize), blocks_nelem)
