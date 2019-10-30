import uproot
import networkx as nx
import math
from collections import Counter
import numpy as np
import sys
import scipy
import numba
print(numba.__version__)

def load_file(fn):
    fi = uproot.open(fn)
    tree = fi.get("pftree")
    data = tree.arrays(tree.keys())
    data = {str(k, 'ascii'): v for k, v in data.items()}
    
    #get the list of element (iblock, ielem) to candidate associations
    linktree = fi.get("linktree_elemtocand")
    data_elemtocand = linktree.arrays(linktree.keys())
    data_elemtocand = {str(k, 'ascii'): v for k, v in data_elemtocand.items()}
    
    linktree2 = fi.get("linktree")
    data_elemtoelem = linktree2.arrays(linktree2.keys())
    data_elemtoelem = {str(k, 'ascii'): v for k, v in data_elemtoelem.items()}

    return data, data_elemtocand, data_elemtoelem

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

def fill_dist_matrix(dist_matrix, elem_blk, elem_ielem, nelem, elem_to_elem):
    for i in range(nelem):
        bl = elem_blk[i]
        iel1 = elem_ielem[i]
        for j in range(i+1, nelem):
            if elem_blk[j] == bl:
                iel2 = elem_ielem[j]
                k = (bl, iel1, iel2)
                if k in elem_to_elem:
                    dist_matrix[i,j] = elem_to_elem[k]
    
def prepare_data(data, data_elemtocand, data_elemtoelem, elem_to_newblock, cand_to_newblock, iev):
   
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

    #Fill the distance matrix between all elements
    nelem = len(X1) + len(X2)
    #dist_matrix = scipy.sparse.dok_matrix((nelem, nelem), dtype=np.float32)
    dist_matrix = np.zeros((nelem, nelem))
    bls = data_elemtoelem["linkdata_iblock"][iev]
    el1 = data_elemtoelem["linkdata_ielem"][iev]
    el2 = data_elemtoelem["linkdata_jelem"][iev]
    dist = data_elemtoelem["linkdata_distance"][iev]
    elem_to_elem = {(bl, e1, e2): d for bl, e1, e2, d in zip(bls, el1, el2, dist)}
    elem_blk = np.hstack([data["clusters_iblock"][iev], data["tracks_iblock"][iev]])
    elem_ielem = np.hstack([data["clusters_ielem"][iev], data["tracks_ielem"][iev]])

    fill_dist_matrix(dist_matrix, elem_blk, elem_ielem, nelem, elem_to_elem)     
    dist_matrix_sparse = scipy.sparse.dok_matrix(dist_matrix)
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
    
    return X, y, cand_data, cand_block_id, dist_matrix_sparse

def get_unique_X_y(X, Xbl, y, ybl, maxn=3):
    uniqs = np.unique(Xbl)
    
    Xs = []
    ys = []
    for bl in uniqs:
        subX = X[Xbl==bl]
        suby = y[ybl==bl]
        
        #choose only miniblocks with 3 elements to simplify the problem
        if len(subX) > maxn:
            continue
            
        subX = np.pad(subX, ((0, maxn - subX.shape[0]), (0,0)), mode="constant")
        suby = np.pad(suby, ((0, maxn - suby.shape[0]), (0,0)), mode="constant")
        
        Xs += [subX]
        ys += [suby]
        
    return Xs, ys

if __name__ == "__main__":
    fn = sys.argv[1]
    data, data_elemtocand, data_elemtoelem = load_file(fn)

    nev = len(data["pfcands_pt"])
    print("Loaded {0} events".format(nev))

    all_sgs = []
    for iev in range(nev):
        print("{0}/{1}".format(iev, nev))

        #Create a graph of the elements and candidates
        pfgraph = create_graph_elements_candidates(data, data_elemtocand, iev)
        
        #Find disjoint subgraphs
        sgs, elem_to_newblock, cand_to_newblock = analyze_graph_subgraph_elements(pfgraph)
        
        #Create arrays from subgraphs
        elements, block_id, pfcands, cand_block_id, dist_matrix = prepare_data(data, data_elemtocand, data_elemtoelem, elem_to_newblock, cand_to_newblock, iev)

        #save the all the elements, candidates and the miniblock id
        cache_filename = fn.replace(".root", "_ev{0}.npz".format(iev))
        with open(cache_filename, "wb") as fi:
            np.savez(fi, elements=elements, element_block_id=block_id, candidates=pfcands, candidate_block_id=cand_block_id)
        cache_filename = fn.replace(".root", "_dist{0}.npz".format(iev))
        with open(cache_filename, "wb") as fi:
            scipy.sparse.save_npz(fi, dist_matrix.tocoo())
    
        #save the miniblocks separately (Xs - all miniblocks in event, ys - all candidates made from each block) 
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
