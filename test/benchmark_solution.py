import numpy as np
import sklearn
import sklearn.metrics
import keras
import scipy
import numba
import networkx

from train_clustering import fill_elem_pairs

@numba.njit
def vector_to_triu_matrix(vec, nelem):
    n = 0
    ret = np.zeros((nelem, nelem))
    for i in range(nelem):
        for j in range(i+1, nelem):
            ret[i, j] = vec[n]
            n += 1
    return ret

class DummyPFAlgo:
    def predict_clusters(self, elements, distance_matrix):
        #integer cluster label for each element
        pred_clusters = np.zeros(len(elements), dtype=np.int32)
        return pred_clusters

    def assess_clustering(self, true_clusters, pred_clusters):
        num_clusters_pred = len(np.unique(pred_clusters))
        num_clusters_true = len(np.unique(true_clusters))
        m1 = sklearn.metrics.adjusted_rand_score(pred_clusters, true_clusters)
        m2 = sklearn.metrics.adjusted_mutual_info_score(pred_clusters, true_clusters)

        return {
            "num_clusters_true": num_clusters_true,
            "num_clusters_pred": num_clusters_pred,
            "adjusted_rand_score": m1,
            "adjusted_mutual_info_score": m2
        }

class BaselineDNN(DummyPFAlgo):
    def __init__(self):
        pass
        self.model_clustering = keras.models.load_model("clustering.h5")

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
        pred_matrix[pred_matrix>0.5] = 1

        #Find connected subgraphs based on adjacency matrix
        g = networkx.from_numpy_matrix(pred_matrix)
        for isg, subgraph in enumerate(networkx.connected_component_subgraphs(g)):
            for node in subgraph.nodes:
                ret[node] = isg

        return ret

def load_elements(fn):
    fi = open(fn, "rb")
    data = np.load(fi)
    els = data["elements"]
    els_blid = data["element_block_id"]

    fi = open(fn.replace("ev", "dist"), "rb")
    dm = scipy.sparse.load_npz(fi).todense()

    return els, els_blid, dm

if __name__ == "__main__":
    m = BaselineDNN()

    fn = "data/TTbar/191009_155100/step3_AOD_{0}_ev{1}.npz".format(1, 0)
    els, els_blid, dm = load_elements(fn)
    els_blid_pred = m.predict_clusters(els, dm)
    score = m.assess_clustering(els_blid, els_blid_pred)
    print(score)