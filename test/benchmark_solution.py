import numpy as np
import sklearn
import sklearn.metrics
import keras
import scipy
import numba
import networkx
import pickle

import sklearn.cluster
from train_clustering import fill_elem_pairs, fill_target_matrix
from train_regression import get_unique

@numba.njit
def vector_to_triu_matrix(vec, nelem):
    n = 0
    ret = np.zeros((nelem, nelem))
    for i in range(nelem):
        for j in range(i+1, nelem):
            ret[i, j] = vec[n]
            n += 1
    return ret

@numba.njit
def set_pred_to_zero(p, ncand):
    for i in range(len(ncand)):
        p0 = np.copy(p[i, :])
        p[i, :] = 0
        p[i, :ncand[i]*3] = p0[:ncand[i]*3]

@numba.njit
def fill_cand_vector(cand_types, ncand, cand_momenta, cluster_id):
    ret = np.zeros((10000,4))
    ret_cand_block_id = np.zeros(10000, dtype=np.int32)
    n = 0
    for ibl in range(len(cand_types)):
        for icand in range(ncand[ibl]):
            ret[n, 0] = cand_types[ibl, icand]
            ret[n, 1:4] = cand_momenta[ibl, icand*3:(icand+1)*3]
            ret_cand_block_id[n] = cluster_id[ibl]
            n += 1
            assert(n < ret.shape[0])
    return ret[:n], ret_cand_block_id[:n]

#Ultra simple PF model requiring no training, just to test the baseline
class DummyPFAlgo:
    #assign an integer cluster label for each element based on a simple DBScan
    def predict_clusters(self, elements, distance_matrix):
        dbscan = sklearn.cluster.DBSCAN(eps=0.28, min_samples=1, metric="precomputed")

        distance_matrix_copy = distance_matrix.copy()
        distance_matrix_copy[distance_matrix_copy==0] = 999

        pred_clusters = dbscan.fit_predict(distance_matrix_copy)
        return pred_clusters

    def predict_candidates(self, elements, element_block_id):
        ncands = len(np.unique(len(element_block_id)))
        ret_cands = np.zeros((ncands, 4))
        ret_cands_blid = np.zeros(ncands, dtype=np.int32)
        return ret, ret_cands_blid

    def assess_candidates(self, cands_true, cands_pred, cands_true_blockid, cands_pred_blockid):

        #number of candidates produced per block
        ncands_true = []
        ncands_pred = []

        #Check for each block how many candidates were produced (true & predicted)
        for ibl in np.unique(cands_true_blockid):
            msk1 = cands_true_blockid==ibl
            msk2 = cands_pred_blockid==ibl

            nc_true = np.sum(msk1)
            nc_pred = np.sum(msk2)
            ncands_true += [nc_true]
            ncands_pred += [nc_pred]

        return {
            "num_cands_true": len(cands_true),
            "num_cands_pred": len(cands_pred),
            "pt_avg_true": cands_true[:, 1].mean(),
            "pt_avg_pred": cands_pred[:, 1].mean(),
            "ncand_r2": sklearn.metrics.r2_score(ncands_true, ncands_pred)
        }

    def assess_clustering(self, distance_matrix, true_block_id, pred_block_id):
        num_blocks_pred = len(np.unique(pred_block_id))
        num_blocks_true = len(np.unique(true_block_id))
        m1 = sklearn.metrics.adjusted_rand_score(pred_block_id, true_block_id)
        m2 = sklearn.metrics.adjusted_mutual_info_score(pred_block_id, true_block_id)

        #compute precision and recall between all elements that can be connected (distance != 0)
        edge_prec, edge_rec = self.assess_connectable_edges(distance_matrix, true_block_id, pred_block_id)

        return {
            "num_clusters_true": num_blocks_true,
            "num_clusters_pred": num_blocks_pred,
            "adjusted_rand_score": m1,
            "adjusted_mutual_info_score": m2,
            "edge_precision": edge_prec,
            "edge_recall": edge_rec,
        }

    def unique_blockids_to_adj_matrix(self, element_blockids):
        nelem = len(element_blockids)
        mat = np.zeros((nelem, nelem))
        fill_target_matrix(mat, element_blockids)
        return mat

    def assess_connectable_edges(self, distance_matrix, true_blockids, pred_blockids):
        nonzero = distance_matrix>0
        m1 = self.unique_blockids_to_adj_matrix(true_blockids)
        m2 = self.unique_blockids_to_adj_matrix(pred_blockids)
        v1 = m1[nonzero]
        v2 = m2[nonzero]

        prec = sklearn.metrics.precision_score(v1, v2)
        rec = sklearn.metrics.recall_score(v1, v2)
        return prec, rec

#Simple feedforward-DNN based PF model
class BaselineDNN(DummyPFAlgo):
    def __init__(self):
        pass
        self.model_clustering = keras.models.load_model("clustering.h5")
        self.model_regression = keras.models.load_model("regression.h5")
        with open("preprocessing.pkl", "rb") as fi:
            self.preprocessing_reg = pickle.load(fi)
        self.num_onehot_y = 27

    def predict_candidates(self, elements, element_block_id):
        Xs, Xs_blid = get_unique(elements, element_block_id, np.unique(element_block_id))
        Xs2 = np.stack(Xs, axis=0)
        Xs_types = Xs2[:, :, 0]
        Xs_kin = Xs2[:, :, 1:]
        Xs_kin = Xs_kin.reshape((Xs_kin.shape[0], Xs_kin.shape[1]*Xs_kin.shape[2]))

        transformed_type = self.preprocessing_reg["enc_X"].transform(Xs_types)
        transformed_kin = self.preprocessing_reg["scaler_X"].transform(Xs_kin)
        X = np.hstack([transformed_type, transformed_kin])

        pred = self.model_regression.predict(X, batch_size=10000)

        cand_types = self.preprocessing_reg["enc_y"].inverse_transform(pred[:, :self.num_onehot_y])
        ncand = (cand_types!=0).sum(axis=1)

        cand_momenta = self.preprocessing_reg["scaler_y"].inverse_transform(pred[:, self.num_onehot_y:])
        set_pred_to_zero(cand_momenta, ncand)

        pred_cands, pred_cand_blid = fill_cand_vector(cand_types, ncand, cand_momenta, Xs_blid)
        print(pred_cands)
        return pred_cands, pred_cand_blid

    def predict_clusters(self, elements, distance_matrix):
        
        nelem = len(elements)

        #integer cluster label for each element
        ret = np.zeros(nelem, dtype=np.int32)

        #number of upper triangular elements without diagonal
        num_pairs = int(nelem*(nelem-1)/2)
        i1, i2 = np.triu_indices(nelem, k=1)

        target_matrix = np.zeros_like(distance_matrix)
        elem_pairs_X = np.zeros((num_pairs, 5), dtype=np.float32)
        
        elem_pairs_X[:, 0] = elements[i1, 0]
        elem_pairs_X[:, 1] = elements[i1, 1]
        elem_pairs_X[:, 2] = elements[i2, 0]
        elem_pairs_X[:, 3] = elements[i2, 1]
        elem_pairs_X[:, 4] = distance_matrix[i1, i2]

        #Predict linkage proba for each element pair
        pred = self.model_clustering.predict_proba(elem_pairs_X, batch_size=100000)

        #Create adjacency matrix from element pairs which had predicted value greater than a threshold
        pred_matrix = vector_to_triu_matrix(pred[:, 0], nelem)
        pred_matrix[dm==0] = 0
        print("pred mean=", pred_matrix[pred_matrix>0].mean(), "std=", pred_matrix[pred_matrix>0].std())
        pred_matrix[pred_matrix>=0.9] = 1
        pred_matrix[pred_matrix<0.9] = 0

        #Find connected subgraphs based on adjacency matrix
        g = networkx.from_numpy_matrix(pred_matrix)
        for isg, nodes in enumerate(networkx.connected_components(g)):
            for node in nodes:
                ret[node] = isg

        return ret

def load_elements_candidates(fn):
    fi = open(fn, "rb")
    data = np.load(fi)
    els = data["elements"]
    els_blid = data["element_block_id"]
    cands = data["candidates"]
    cands_blid = data["candidate_block_id"]

    fi = open(fn.replace("ev", "dist"), "rb")
    dm = scipy.sparse.load_npz(fi).todense()

    return els, els_blid, cands, cands_blid, dm

if __name__ == "__main__":
    m = BaselineDNN()

    fn = "data/TTbar/191009_155100/step3_AOD_{0}_ev{1}.npz".format(1, 0)
    els, els_blid, cands, cands_blid, dm = load_elements_candidates(fn)
    #els_blid_pred = m.predict_clusters(els, dm)
    #score_clustering = m.assess_clustering(dm, els_blid, els_blid_pred)
    cands_pred, cands_pred_blid = m.predict_candidates(els, els_blid)
    score_cands = m.assess_candidates(cands, cands_pred, cands_blid, cands_pred_blid)
    #print(score_clustering)
    print(score_cands)
    #print(cands)