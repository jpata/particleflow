import numpy as np
import glob
import matplotlib.pyplot as plt
import numba
from collections import Counter
import math
import sklearn
import sklearn.metrics
import sklearn.ensemble
import scipy.sparse
import keras
import pickle
from train_clustering import encode_triu, decode_triu

@numba.njit
def fill_input_matrix(els, dm):
    Nel = len(els)
    l = np.zeros((int(Nel*(Nel-1)/2), 3), dtype=np.float32)
    for i0 in range(len(els)):
        for i1 in range(i0+1, len(els)):
            elid0 = els[i0, 0]
            elid1 = els[i1, 0]
            if dm[i0,i1]>0:
                idx = encode_triu(i0, i1, Nel)
                l[idx, 0] = elid0
                l[idx, 1] = elid1
                l[idx, 2] = dm[i0,i1]
    return l

@numba.njit
def pred_vector_to_matrix(pred_vector, N):
    mat = np.zeros((N,N), dtype=np.float32)
    for k in range(len(pred_vector)):
        i, j = decode_triu(k, N)
        mat[i,j] = pred_vector[k]
    return mat

"""
The pointer-tree structure used by `find_root` and `union` uses 1D (i,e, flat) indices for
the nodes of the graph structure. A negative value indicates the element in the matrix is a
root of a connected component; the magnitude of that value indicates the total number of
elements in that connected component. Otherwise a non-negative value is the flat-index that
points to another element in its connected component.
For example, the pointer-tree:
    [-3, 0, 0, -2, 3]
corresponds to a graph with two connected components. element-0 is the root of the component
containing {0, 1, 2}. element-3 is the root of the component containing {3, 4}.
"""
@numba.jit(nopython=True)
def find_root(x, pntr_tree):
    """ Returns the root node-ID of the connected component containing `x`
        Performs path compression. I.e. redirects all pointer values
        along the recursive path to point directly to the root node, compressing
        future root finding paths.
        Parameters
        ----------
        x : int
            A valid node-ID.
        pntr_tree : Sequence[int, ...]
            A pointer-tree, indicating the connected component membership of nodes in
            the graph.
        Returns
        -------
        int
            The root node-ID of the connected component `x` """
    if pntr_tree[x] < 0:  # x is the root of a connected component
        return x
    pntr_tree[x] = find_root(pntr_tree[x], pntr_tree)  # find the root that x points to, and update tree
    return pntr_tree[x]


@numba.jit(nopython=True)
def union(x, y, pntr_tree):
    """ Joins the connected components containing `x` and `y`, respectively.
        Performs union by rank: the root of the smaller component is pointed
        to the root of the larger one.
        Parameters
        ----------
        x : int
            A valid node-ID
        y : int
            A valid node-ID.
        pntr_tree : Sequence[int, ...]
            A pointer-tree, indicating the connected component membership of nodes in
            the graph."""
    r_x = find_root(x, pntr_tree)
    r_y = find_root(y, pntr_tree)

    if r_x != r_y:
        if pntr_tree[r_x] <= pntr_tree[r_y]:  # subgraph containing x is bigger (in magnitude!!)
            pntr_tree[r_x] += pntr_tree[r_y]  # add the smaller subgraph to the larger
            pntr_tree[r_y] = r_x  # point root of cluster y to x
        else:
            pntr_tree[r_y] += pntr_tree[r_x]
            pntr_tree[r_x] = r_y
    return None

@numba.njit
def create_graph(pntr_tree, connection_matrix):
    Nel = connection_matrix.shape[0]
    for i0 in range(Nel):
        for i1 in range(i0+1, Nel):
            c = connection_matrix[i0, i1]
            if c == 1:
                union(i0, i1, pntr_tree)

def get_unique_X(X, Xbl, blsize=3):
    uniqs = np.unique(Xbl)

    Xs = []
    Xs_ids = []
    for bl in uniqs:
        subX = X[Xbl==bl][:blsize]

        subX = np.pad(subX, ((0, blsize - subX.shape[0]), (0,0)), mode="constant")
        Xs += [subX]
        Xs_ids += [bl]
    return Xs, Xs_ids

@numba.njit
def set_pred_to_zero(p, ncand):
    for i in range(len(ncand)):
        p0 = np.copy(p[i, :])
        p[i, :] = 0
        p[i, :ncand[i]*3] = p0[:ncand[i]*3]

@numba.njit
def fill_cand_vector(cand_types, ncand, cand_momenta, cluster_id):
    ret = np.zeros((10000,4))
    ret_cand_block_id = np.zeros((10000,1), dtype=np.int32)
    n = 0
    for ibl in range(len(cand_types)):
        for icand in range(ncand[ibl]):
            ret[n, 0] = cand_types[ibl, icand]
            ret[n, 1:4] = cand_momenta[ibl, icand*3:(icand+1)*3]
            ret_cand_block_id[n, 0] = cluster_id[ibl, 0]
            n += 1
            assert(n < ret.shape[0])
    return ret[:n], ret_cand_block_id[:n]

def run_prediction(model1, model2, preproc, num_onehot_y, elements, distancematrix):
    X = elements
    dm = distancematrix
    Nel = len(X)
    assert(dm.shape == (Nel, Nel))

    inp_matrix = fill_input_matrix(X, dm)
    
    pred_vector = model1.predict(inp_matrix, batch_size=10000)
    
    pred_matrix = pred_vector_to_matrix(pred_vector[:, 0], Nel)
    
    pred_matrix[dm==0] = 0
    pred_matrix[pred_matrix > 0.9] = 1.0
    
    pntr_tree = -1*np.ones(Nel, dtype=np.int)
    create_graph(pntr_tree, pred_matrix)
    
    clid = np.zeros(Nel, dtype=np.int32)
    for iel in range(Nel):
        root = find_root(iel, pntr_tree)
        clid[iel] = root

    scaler_X = preproc["scaler_X"]
    scaler_y = preproc["scaler_y"]
    
    enc_X = preproc["enc_X"]
    enc_y = preproc["enc_y"]

    X, X_ids = get_unique_X(data["elements"], clid)
    X = np.stack(X, axis=0)
    X_ids = np.vstack(X_ids)
    
    X_types = X[:, :, 0]
    X_kin = X[:, :, 1:]
    X_kin = X_kin.reshape((X_kin.shape[0], X_kin.shape[1]*X_kin.shape[2]))
    
    trf = enc_X.transform(X_types)
    X = np.hstack([trf, scaler_X.transform(X_kin)])
    
    pred2 = model2.predict(X)

    cand_types = enc_y.inverse_transform(pred2[:, :num_onehot_y])
    ncand = (cand_types!=0).sum(axis=1)

    cand_momenta = scaler_y.inverse_transform(pred2[:, num_onehot_y:])
    set_pred_to_zero(cand_momenta, ncand)

    pred_cands, pred_cand_block_id = fill_cand_vector(cand_types, ncand, cand_momenta, X_ids)
    return clid, pred_cands, pred_cand_block_id

if __name__ == "__main__":
    model1 = keras.models.load_model("data/clustering.h5")
    model2 = keras.models.load_model("data/regression.h5")
    with open("data/preprocessing.pkl", "rb") as fi:
        preproc = pickle.load(fi)
    num_onehot_y = 27 #determined by enc_y in train_regression

    for iev in range(500):
        fn = "data/TTbar/191009_155100/step3_AOD_2_ev{0}.npz".format(iev)
        fi = open(fn, "rb")
        data = np.load(fi)
        X = data["elements"]
        Nel = len(X)
        
        fi = open(fn.replace("ev", "dist"), "rb")
        dm = scipy.sparse.load_npz(fi).todense()

        element_block_id, pred_cand, pred_cand_block_id = run_prediction(model1, model2, preproc, num_onehot_y, X, dm)

        print("elem", X.shape)
        print("elem_block_id", element_block_id.shape)
        print("cand", pred_cand.shape)
        print("cand_block_id", pred_cand_block_id.shape)
        with open(fn.replace("ev", "pred"), "wb") as fi:
            np.savez(fi, elements=X, element_block_id=element_block_id, candidates=pred_cand, candidate_block_id=pred_cand_block_id)
