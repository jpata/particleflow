import sys, os

import numpy as np
import sklearn
import sklearn.metrics
import scipy
import numba
import networkx
import pickle
from collections import Counter

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
def fill_cand_vector(cand_types, ncand, cand_momenta):
    ret = np.zeros((10000,4))
    n = 0
    for ibl in range(len(cand_types)):
        for icand in range(ncand[ibl]):
            ret[n, 0] = cand_types[ibl, icand]
            ret[n, 1:4] = cand_momenta[ibl, icand*3:(icand+1)*3]
            n += 1
            assert(n < ret.shape[0])
    return ret[:n]

"""
For each particle in cand1, finds the first matched particle in cand2.
cand1 (np.array): array with shape (Ncand1, Nfeat_cand) for the first set of candidates
cand2 (np.array): array with shape (Ncand2, Nfeat_cand) for the second set of candidates

Returns: np.array with shape(Ncand1) with the indices of the match in cand2.
"""
@numba.njit
def deltar_match(cand1, cand2, drcut=0.01):
    masked_cand2 = np.zeros(len(cand2), dtype=np.int32)
    inds_cand1_to_cand2 = np.zeros(len(cand1), dtype=np.int32)
    inds_cand1_to_cand2[:] = -1
    
    for i in range(len(cand1)):
        eta1 = cand1[i, 2]
        phi1 = cand1[i, 3]
        for j in range(len(cand2)):
            if masked_cand2[j]==0:
                eta2 = cand2[j, 2]
                phi2 = cand2[j, 3]
                
                dr2 = (eta1-eta2)**2 + (np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi)**2
                dr = np.sqrt(dr2)
                
                if dr < drcut:
                    masked_cand2[j] = 1
                    inds_cand1_to_cand2[np.int32(i)] = np.int32(j)
                    break
    return inds_cand1_to_cand2


"""
Get the data from cand2 that corresponds to cand1
"""
@numba.njit
def get_matched_pairs(cands1, cands2, matched_inds):
    ret = np.zeros_like(cands1)
    
    for i in range(len(cands1)):
        if matched_inds[i] >= 0:
            ret[i, :] = cands2[matched_inds[i], :]
    return ret

#Ultra simple PF model requiring no training, just to test the baseline
class DummyPFAlgo:
    """
    Assign an integer block label for each element. This dummy solution just uses the same
    big blocks as standard PF

    elements (np.array): (Nelem, Nfeat) array of all the elements
    distance_matrix (np.array): (Nelem, Nelem) array of the distances between all elements.
        Distance 0 is set to a large number.

    Returns: np.array with shape (Nelem), block label for each element
    """
    def predict_blocks(self, elements, distance_matrix):
        dm2 = distance_matrix.copy()
        dm2[dm2>0] = 1
        g = networkx.from_numpy_matrix(dm2)

        block_id_aspf = np.zeros((len(elements), ), dtype=np.int32)
        for ibl, conn in enumerate(networkx.connected_components(g)):
            block_id_aspf[np.array(list(conn), dtype=np.int32)] = ibl

        return block_id_aspf

    """
    Given arrays of true and predicted particles, computes metrics
    on how well the predicted candidates reproduce the true particles.
    """
    def assess_candidates(self, cands_true, cands_pred):

        with open("cands.npz", "wb") as fi:
            np.savez(fi, true=cands_true, pred=cands_pred)

        matched_inds = deltar_match(cands_true, cands_pred, drcut=0.05)
        msk = matched_inds!=-1
        num_cands_matched = np.sum(msk)

        cands_true_matched = cands_true[msk]
        cands_pred_matched = get_matched_pairs(cands_true, cands_pred, matched_inds)[msk]

        ret = {
            "num_cands_true": len(cands_true),
            "num_cands_pred": len(cands_pred),
            "num_cands_matched": num_cands_matched,
            "pt_avg_true": cands_true[:, 1].mean(),
            "pt_avg_pred": cands_pred[:, 1].mean(),
        }
        ret.update(self.assess_matched_candidates(cands_true_matched, cands_pred_matched))
        return ret

    """
    Given the true and predicted block ids for elements, assess the quality of the element to block clustering.
    
    true_block_id (np.array): integer label for each element from the true block, shape (Nelem) 
    pred_block_id (np.array): integer label for each element from the prediction algo, shape (Nelem)
    distance_matrix (np.array): the initial distance matrix, used to assess which edges even can be connected
        at all (dist > 0)
    
    Returns: dict of various clustering metrics
    """
    def assess_blocks(self, true_block_id, pred_block_id, distance_matrix):
        #Compute the number of true and predicted blocks
        num_blocks_pred = len(np.unique(pred_block_id))
        num_blocks_true = len(np.unique(true_block_id))

        block_sizes_true = self.compute_block_sizes(true_block_id)
        block_sizes_pred = self.compute_block_sizes(pred_block_id)

        #Compute clustering metrics based on scikit-learn
        m1 = sklearn.metrics.adjusted_rand_score(pred_block_id, true_block_id)
        #m2 = sklearn.metrics.adjusted_mutual_info_score(pred_block_id, true_block_id, average_method='arithmetic')

        #compute precision and recall between all elements that can be connected (distance > 0)
        edge_prec, edge_rec = self.assess_connectable_edges(distance_matrix, true_block_id, pred_block_id)

        return {
            "num_blocks_true": num_blocks_true,
            "num_blocks_pred": num_blocks_pred,
            "mean_block_size_true": np.mean(block_sizes_true),
            "mean_block_size_pred": np.mean(block_sizes_pred),
            "max_block_size_true": np.max(block_sizes_true),
            "max_block_size_pred": np.max(block_sizes_pred),
            "num_blocks_pred": num_blocks_pred,
            "adjusted_rand_score": m1,
            #"adjusted_mutual_info_score": m2,
            "edge_precision": edge_prec,
            "edge_recall": edge_rec,
        }

    def compute_block_sizes(self, block_ids):
        counts = Counter(block_ids)
        return list(counts.values())

    """
    Given the block ids of N elements, fill an NxN adjacency matrix for the elements.
    """
    def unique_blockids_to_adj_matrix(self, element_blockids):
        nelem = len(element_blockids)
        mat = np.zeros((nelem, nelem))
        fill_target_matrix(mat, element_blockids)
        return mat

    """
    For edges that have dist>0, compute the precision and recall given the true and predicted edges.
    """
    def assess_connectable_edges(self, distance_matrix, true_blockids, pred_blockids):
        nonzero = distance_matrix>0
        m1 = self.unique_blockids_to_adj_matrix(true_blockids)
        m2 = self.unique_blockids_to_adj_matrix(pred_blockids)
        v1 = m1[nonzero]
        v2 = m2[nonzero]

        prec = sklearn.metrics.precision_score(v1, v2)
        rec = sklearn.metrics.recall_score(v1, v2)
        return prec, rec
   
    """
    Given an array of the element data and a unique block ID for each element,
    predict the particle candidates for each block given the elements in the block.
    elements (np.array): (Nelem, Nfeat) array
    elements_blockids (np.array): (Nelem) array with block IDs for each element
    Returns: cands (np.array): (Ncand, Nfeatcand) array of all the predicted candidates
    """ 
    def predict_candidates(self, elements, elements_blockids, maxel=3):
        cands = []
        for ibl in np.unique(elements_blockids):
            #print("predict_candidates", ibl)
            msk = elements_blockids==ibl
            els_in_block = elements[msk][:maxel]
            els_in_block = np.pad(els_in_block, ((0, maxel - els_in_block.shape[0]), (0,0)), mode="constant")
            cands_pred = self.predict_one_block(els_in_block)
            cands += [cands_pred]
        cands = np.vstack(cands)
        return cands

    """
    Given arrays of predicted candidates that were matched to true candidates, assess the quality
    of the reconstruction.
    cands_true_matched (np.array): array with shape (Ncand, Nfeat_cand) of the truth-level candidates
    cands_pred_matched (np.array): as above, but predicted candidates that were matched to the truth candidates, in the same order
    
    Returns: dict of 2d histograms (pt_true, pt_pred)
    """
    def assess_matched_candidates(self, cands_true_matched, cands_pred_matched):
        all_pdgids = [-211, -13, 0, 1, 2, 13, 22, 130, 211]
        pdgid_confusion_matrix = sklearn.metrics.confusion_matrix(cands_true_matched[:, 0], cands_pred_matched[:, 0], labels=all_pdgids)
        ptbins = np.linspace(0, 10, 20)
        pt_matrix, _, _ = np.histogram2d(cands_true_matched[:, 1], cands_pred_matched[:, 1], bins=(ptbins, ptbins))
        
        etabins = np.linspace(-6, 6, 20)
        eta_matrix, _, _ = np.histogram2d(cands_true_matched[:, 2], cands_pred_matched[:, 2], bins=(etabins, etabins))
        
        phibins = np.linspace(-4, 4, 20)
        phi_matrix, _, _ = np.histogram2d(cands_true_matched[:, 3], cands_pred_matched[:, 3], bins=(phibins, phibins))

        return {
            "pdgid_confusion_matrix": pdgid_confusion_matrix,
            "pt_matrix": pt_matrix,
            "eta_matrix": eta_matrix,
            "phi_matrix": phi_matrix,
        }

    """
    Predicts the candidates given the true block IDs. In this case, we can compare
    the true and predicted candidates for each block, which allows to make direct candidate-by-candidate
    comparisons.
    """ 
    def predict_with_true_blocks(self, elements, elements_blockids, candidates, candidates_blockids, maxel=3):
        all_ncands_true = []  
        all_ncands_pred = []

        cands_true_matched = []
        cands_pred_matched = []
        for ibl in np.unique(elements_blockids):
            #print("predict_with_true_blocks", ibl)
            msk = elements_blockids==ibl
            els_in_block = elements[msk][:maxel]
            els_in_block = np.pad(els_in_block, ((0, maxel - els_in_block.shape[0]), (0,0)), mode="constant")
            cands_pred = self.predict_one_block(els_in_block)
            cands_true = cands[cands_blid==ibl]
            ncands_true = cands_true.shape[0]
            ncands_pred = cands_pred.shape[0]
            all_ncands_true += [ncands_true]
            all_ncands_pred += [ncands_pred]

            #For blocks with the right number of predicted candidates, compare the pdgIds
            if ncands_true == ncands_pred:
                cands_true_matched += [cands_true]
                cands_pred_matched += [cands_pred]

        ncand_confusion_matrix = sklearn.metrics.confusion_matrix(all_ncands_true, all_ncands_pred, labels=range(4))
        ret = {
            "num_cands_true": np.sum(all_ncands_true),
            "num_cands_pred": np.sum(all_ncands_pred),
            "ncand_r2": sklearn.metrics.r2_score(all_ncands_true, all_ncands_pred),
            "ncand_confusion_matrix": ncand_confusion_matrix,
        }
       
        #Check candidates that were well-matched 
        cands_true_matched = np.vstack(cands_true_matched)
        cands_pred_matched = np.vstack(cands_pred_matched)
        ret.update(self.assess_matched_candidates(cands_true_matched, cands_pred_matched))
        
        return ret

    """
    Given the element data for one block, predict the candidates from this block.
    This dummy method predicts 0 candidates - overridden in child classes.
    """ 
    def predict_one_block(self, elem_data):
        ret = np.zeros((0, 4))
        return ret

@numba.njit
def set_adj_matrix(adj_matrix, clid):
    for iel in range(len(clid)):
        for jel in range(iel+1, len(clid)):
            if clid[iel] != -1 and clid[jel] != -1:
                if clid[iel] == clid[jel]:
                    adj_matrix[iel, jel] = 1

"""
Given the PF elements, create a list of points, where each point is just one hit in a layer.
This is needed to do clustering layer-by-layer.
"""
def create_points(elements):
    Npoints = 100000
    Nlinks = 100000

    #id, type, layer
    points_data = np.zeros((Npoints, 3), dtype=np.int64)

    #eta, phi, energy
    points_pos = np.zeros((Npoints, 3), dtype=np.float32)

    #which points are linked (e.g. track points belonging to the same track)
    point_to_point_link = np.zeros((Nlinks, 2), dtype=np.int64)

    #map the points back to the original elements
    point_to_elem = np.zeros((Nlinks, ), dtype=np.int64)

    ip = 0
    ilink = 0
    for iel in range(len(elements)):
        tp = elements[iel, 0]

        if tp == 1 or tp == 6:
            ip_in_tracker = -1
            ip_in_ecal = -1
            ip_in_hcal = -1

            in_tracker = (elements[iel, 2]!=0) and (elements[iel, 3]!=0)
            in_ecal = (elements[iel, 4]!=0) and (elements[iel, 5]!=0)
            in_hcal = (elements[iel, 6]!=0) and (elements[iel, 7]!=0)
            if in_tracker:
                ip_in_tracker = ip
                points_data[ip, 0] = ip
                points_data[ip, 1] = tp
                points_data[ip, 2] = 0
                points_pos[ip, 0] = elements[iel, 2]
                points_pos[ip, 1] = elements[iel, 3]
                points_pos[ip, 2] = 1.0/abs(elements[iel, 1]) if elements[iel, 1] != 0 else 0.0
                point_to_elem[ip] = iel
                ip += 1
            if in_ecal:
                ip_in_ecal = ip
                points_data[ip, 0] = ip
                points_data[ip, 1] = tp
                points_data[ip, 2] = 1
                points_pos[ip, 0] = elements[iel, 4]
                points_pos[ip, 1] = elements[iel, 5]
                points_pos[ip, 2] = 1.0/abs(elements[iel, 1]) if elements[iel, 1] != 0 else 0.0
                if in_tracker:
                    point_to_point_link[ilink, 0] = ip_in_tracker
                    point_to_point_link[ilink, 1] = ip
                    ilink += 1
                point_to_elem[ip] = iel
                ip += 1
            if in_hcal:
                ip_in_hcal = ip
                points_data[ip, 0] = ip
                points_data[ip, 1] = tp
                points_data[ip, 2] = 2
                points_pos[ip, 0] = elements[iel, 6]
                points_pos[ip, 1] = elements[iel, 7]
                points_pos[ip, 2] = 1.0/abs(elements[iel, 1]) if elements[iel, 1] != 0 else 0.0
                if in_tracker:
                    point_to_point_link[ilink, 0] = ip_in_tracker
                    point_to_point_link[ilink, 1] = ip
                    ilink += 1
                if in_ecal:
                    point_to_point_link[ilink, 0] = ip_in_ecal
                    point_to_point_link[ilink, 1] = ip
                    ilink += 1
                point_to_elem[ip] = iel
                ip += 1
        else:
            layer = 1
            if tp == 5:
                layer = 2
            elif tp >= 8:
                layer = 3
            points_data[ip, 0] = ip
            points_data[ip, 1] = tp
            points_data[ip, 2] = layer
            points_pos[ip, 0] = elements[iel, 2]
            points_pos[ip, 1] = elements[iel, 3]
            points_pos[ip, 2] = elements[iel, 1]
            point_to_elem[ip] = iel
            ip += 1

    points_data = points_data[:ip]
    points_pos = points_pos[:ip]
    point_to_elem = point_to_elem[:ip]
    point_to_point_link = point_to_point_link[:ilink]
    return points_data, points_pos, point_to_point_link, point_to_elem

@numba.njit
def dist(points, i, j):
    p0 = points[i]
    p1 = points[j]
    dphi = p0[1] - p1[1]
    dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
    deta = p0[0] - p1[0]
    return np.sqrt(dphi**2 + deta**2)

@numba.njit
def fill_local_density(points, points_data, delta_crit=0.3):
    points_data[:, 0] = 0
    
    Np = len(points)
    for i in range(Np):
        for j in range(Np):
            d = dist(points, i, j)
            if d < delta_crit:
                #weight multiplier
                fact = 1.0 if i==j else 0.5
                #density += weight * weight multiplier
                points_data[i, 0] += points_data[j, 1]*fact

@numba.njit
def find_nearest_higher(points, points_data):
    Np = len(points)
    
    for i in range(Np):
        delta = 999.0
        nearestHigher = -1
        
        for j in range(Np):
            d = dist(points, i, j)
            if d < delta and points_data[j, 0] > points_data[i, 0]:
                nearestHigher = j
                delta = d

        points_data[i, 2] = delta
        points_data[i, 3] = nearestHigher

@numba.njit
def fill_adj_matrix(adj_matrix, point_to_point_link, clid0, clid1, clid2, clid3):
    #all track points are connected across the layers
    for ip in range(len(point_to_point_link)):
        adj_matrix[point_to_point_link[ip][0], point_to_point_link[ip][1]] = 1
    
    #create connections between elements in the same cluster
    set_adj_matrix(adj_matrix, clid0)
    set_adj_matrix(adj_matrix, clid1)
    set_adj_matrix(adj_matrix, clid2)
    set_adj_matrix(adj_matrix, clid3)

    #set lower triangular part of adjacency matrix
    for i in range(adj_matrix.shape[0]):
        for j in range(i, adj_matrix.shape[0]):
            adj_matrix[j][i] = adj_matrix[i][j]

@numba.njit
def clid_point_to_element(elements, point_to_elem, new_clid):
    clid_elem = -1*np.ones((len(elements), ), dtype=np.int64)
    for ip in range(len(point_to_elem)):
        ie = point_to_elem[ip]
        if clid_elem[ie] != -1 and new_clid[ip] != clid_elem[ie]:
            print(new_clid[ip], clid_elem[ie])
        clid_elem[ie] = new_clid[ip]
    return clid_elem

class CLUE(DummyPFAlgo):

    def __init__(self, rho_ecal=0.5, rho_hcal=0.5, rho_hf=0.5, delta_ecal=0.05, delta_hcal=0.05, delta_hf=0.05):
        self.rho_ecal = rho_ecal
        self.rho_hcal = rho_hcal
        self.rho_hf = rho_hf
        self.delta_ecal = delta_ecal
        self.delta_hcal = delta_hcal
        self.delta_hf = delta_hf
        pass
    
    def predict_blocks(self, elements, distance_matrix):
        points_data, points_pos, point_to_point_link, point_to_elem = create_points(elements)

        #Run the CLUE clustering in each layer
        clid1 = self.get_clusters_clue_layer(points_data, points_pos, points_data[:, 2]==1, 0.1, self.rho_ecal, self.delta_ecal)
        clid2 = self.get_clusters_clue_layer(points_data, points_pos, points_data[:, 2]==2, 0.1, self.rho_hcal, self.delta_hcal)
        clid3 = self.get_clusters_clue_layer(points_data, points_pos, points_data[:, 2]==3, 0.1, self.rho_hf, self.delta_hf)

        #Assign each track point on the tracker surface to it's own cluster
        clid0 = -1*np.ones_like(clid1)
        clid0[points_data[:, 2]==0] = points_data[points_data[:, 2]==0, 0]

        #Adjacency matrix for all points across the layers
        adj_matrix = np.zeros((len(points_data), len(points_data)), dtype=np.int32)
        fill_adj_matrix(adj_matrix, point_to_point_link, clid0, clid1, clid2, clid3)

        #Assign cluster id to points based on the connected subgraphs from the adjacency matrix
        new_clid = -1*np.ones_like(clid0)
        icl = 0
        for sg in networkx.connected_components(networkx.from_numpy_matrix(adj_matrix)):
            for s in sg:
                new_clid[s] = icl
            icl += 1

        #Assign cluster id to PF elements
        clid_elem = clid_point_to_element(elements, point_to_elem, new_clid)

        return clid_elem


    @staticmethod
    def assign_cluster_id(points_data, rho_crit=10, delta_crit=0.2):
        cluster_id = -1*np.ones((len(points_data),), dtype=np.int32)
        Np = len(points_data)
        nClusters = 0

        buffer_seeds = []
        followers = {i: [] for i in range(Np)}

        for i in range(Np):
            isSeed = (points_data[i, 0] > rho_crit) and (points_data[i, 2] > delta_crit)
            isOutlier = (points_data[i, 0] <= rho_crit) and (points_data[i, 2] > 2*delta_crit)
            #isOutlier = False

            if isSeed:
                cluster_id[i] = nClusters
                nClusters += 1
                buffer_seeds += [i]
            else:
                if not isOutlier:
                    #add as a follower to the nearest highest point
                    nearestHighest = points_data[i, 3]
                    if nearestHighest != -1:
                        followers[nearestHighest] += [i]
        
        #Now set the cluster ID for all children of all seeds
        while len(buffer_seeds) > 0:
            i = buffer_seeds.pop()
            for fl in followers[i]:
                cluster_id[fl] = cluster_id[i]
                buffer_seeds += [fl]
    
        return cluster_id

    @staticmethod
    def get_clusters_clue_layer(points_data, points_pos, mask, track_weight, rho_crit, delta_crit):
        clid_all = -1 * np.ones((len(points_pos), ), dtype=np.int32)
    
        Np = np.sum(mask)
    
        #rho, weight, delta, nearestHigher
        points_data_clue = np.zeros((Np, 4))
        points_data_clue[:, 1] = 1.0
        points_data_clue[points_data[mask, 1]==1, 1] = track_weight
        points_data_clue[points_data[mask, 1]==6, 1] = track_weight
    #     points_data_clue[points_types==5, 1] = 1
    
        fill_local_density(points_pos[mask], points_data_clue, delta_crit=delta_crit)
        find_nearest_higher(points_pos[mask], points_data_clue)
        clid = CLUE.assign_cluster_id(points_data_clue, rho_crit=rho_crit, delta_crit=delta_crit)
        clid_all[np.where(mask)] = clid
    
        return clid_all

#Simple feedforward-DNN based PF model
class BaselineDNN(DummyPFAlgo):
    def __init__(self):
        self.model_blocks = keras.models.load_model("data/clustering.h5")
        self.model_regression = keras.models.load_model("data/regression.h5")
        with open("data/preprocessing.pkl", "rb") as fi:
            self.preprocessing_reg = pickle.load(fi)
        self.num_onehot_y = 27

    #Predict the elements from one block
    def predict_one_block(self, elem_data):
        Xs2 = elem_data.reshape((1,elem_data.shape[0], elem_data.shape[1]))
        Xs_types = Xs2[:, :, 0]
        Xs_kin = Xs2[:, :, 1:]
        Xs_kin = Xs_kin.reshape((Xs_kin.shape[0], Xs_kin.shape[1]*Xs_kin.shape[2]))
        transformed_type = self.preprocessing_reg["enc_X"].transform(Xs_types)
        transformed_kin = self.preprocessing_reg["scaler_X"].transform(Xs_kin)
        X = np.hstack([transformed_type, transformed_kin])

        pred = self.model_regression.predict(X, batch_size=X.shape[0])

        cand_types = self.preprocessing_reg["enc_y"].inverse_transform(pred[:, :self.num_onehot_y])
        ncand = (cand_types!=0).sum(axis=1)

        cand_momenta = self.preprocessing_reg["scaler_y"].inverse_transform(pred[:, self.num_onehot_y:])
        set_pred_to_zero(cand_momenta, ncand)

        pred_cands = fill_cand_vector(cand_types, ncand, cand_momenta)
        return pred_cands

    #Predict the element to block clustering
    def predict_blocks(self, elements, distance_matrix):
        
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

        #Predict linkage proba for each element pair with a nonzero distance
        good_inds = np.nonzero(elem_pairs_X[:, -1])
        pred = self.model_blocks.predict_proba(elem_pairs_X[good_inds], batch_size=10000)
        pred2 = np.zeros(elem_pairs_X.shape[0], dtype=np.float64)
        pred2[good_inds] = pred[:, 0]

        #Create adjacency matrix from element pairs which had predicted value greater than a threshold
        pred_matrix = vector_to_triu_matrix(pred2, nelem)
        pred_matrix[dm==0] = 0
        pred_matrix[pred_matrix>=0.9] = 1
        pred_matrix[pred_matrix<0.9] = 0

        #Find connected subgraphs based on adjacency matrix, set the label in the output vector
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

#Graph NN based PF model
class BaselineGNN(DummyPFAlgo):
    def __init__(self):
        input_dim = 8
        hidden_dim = 32
        n_iters = 1
        from models import EdgeNet
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_blocks = EdgeNet(input_dim=input_dim,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
        self.model_blocks.load_state_dict(torch.load('EdgeNet_13873_10e465f628_jduarte.best.pth',map_location=device))
        self.model_regression = keras.models.load_model("data/regression.h5")
        with open("data/preprocessing.pkl", "rb") as fi:
            self.preprocessing_reg = pickle.load(fi)
        self.num_onehot_y = 27

    #Predict the element to block clustering
    def predict_blocks(self, elements, distance_matrix):
        
        nelem = len(elements)

        #number of upper triangular elements without diagonal
        num_pairs = int(nelem*(nelem-1)/2)
        i1, i2 = np.triu_indices(nelem, k=1)

        #integer cluster label for each element
        ret = np.zeros(nelem, dtype=np.int32)

        sparse_distance_matrix = scipy.sparse.coo_matrix(distance_matrix)
        num_edges = sparse_distance_matrix.nnz
        row_index = sparse_distance_matrix.row
        col_index = sparse_distance_matrix.col

        edge_index = np.zeros((2, 2*num_edges))
        edge_index[0,:num_edges] = row_index
        edge_index[1,:num_edges] = col_index
        edge_index[0,num_edges:] = col_index
        edge_index[1,num_edges:] = row_index
        
        bidir_row = edge_index[0].astype(int)
        bidir_col = edge_index[1].astype(int)

        import torch
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        x = torch.tensor(elements, dtype=torch.float)
        #y = [X_element_block_id[i]==X_element_block_id[j] for (i,j) in edge_index.t().contiguous()]
        #y = torch.tensor(y, dtype=torch.float)

        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index)
        pred = self.model_blocks(data)
        pred_numpy = pred.detach().numpy()

        sparse_pred_matrix = scipy.sparse.coo_matrix((pred_numpy, (bidir_row, bidir_col)))

        pred_matrix = sparse_pred_matrix.todense()
        #Create adjacency matrix from element pairs which had predicted value greater than a threshold
        pred_matrix[pred_matrix>=0.9] = 1
        pred_matrix[pred_matrix<0.9] = 0

        #Find connected subgraphs based on adjacency matrix, set the label in the output vector
        g = networkx.from_numpy_matrix(pred_matrix)
        for isg, nodes in enumerate(networkx.connected_components(g)):
            for node in nodes:
                ret[node] = isg

        return ret

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)

    import keras
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    
    from keras.backend.tensorflow_backend import set_session
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config)) 

    m = BaselineDNN()
    m0 = DummyPFAlgo()
    m1 = CLUE(0.2, 0.7, 0.6, 0.003, 0.125, 0.16)
    m2 = BaselineGNN()


    fns = sys.argv[1:]
    for fn in fns:
        els, els_blid, cands, cands_blid, dm = load_elements_candidates(fn)
        
        #Run the dummy block algo
        els_blid_pred_dummy = m0.predict_blocks(els, dm)
        score_blocks_dummy = m0.assess_blocks(els_blid, els_blid_pred_dummy, dm)

        #Run CLUE block algo
        els_blid_pred_glue = m1.predict_blocks(els, dm)
        score_blocks_glue = m1.assess_blocks(els_blid, els_blid_pred_glue, dm)
   
        #Run the DNN block algo
        els_blid_pred = m.predict_blocks(els, dm)
        score_blocks = m.assess_blocks(els_blid, els_blid_pred, dm)

        #Run the GNN block algo
        els_blid_pred_gnn = m2.predict_blocks(els, dm)
        score_blocks_gnn = m2.assess_blocks(els_blid, els_blid_pred_gnn, dm)
        
        #Run candidate regression with the true blocks 
        #score_true_blocks = m.predict_with_true_blocks(els, els_blid, cands, cands_blid)

        #Run candidate regression with the predicted blocks 
        #cands_pred = m.predict_candidates(els, els_blid_pred)
        #score_cands = m.assess_candidates(cands, cands_pred)
        
        ret = {
            "blocks": score_blocks,
            "blocks_dummy": score_blocks_dummy,
            "blocks_glue": score_blocks_glue,
            "blocks_gnn": score_blocks_gnn,
            #"cand_true_blocks": score_true_blocks,
            #"cand_pred_blocks": score_cands,
        }
        
        print("score_blocks", score_blocks)
        print("score_blocks_dummy", score_blocks_dummy)
        print("score_blocks_glue", score_blocks_glue)
        print("score_blocks_gnn", score_blocks_gnn)

        output_file = fn.replace(".npz", "_res.pkl")
        with open(output_file, "wb") as fi:
            print("saving output to {0}".format(output_file))
            pickle.dump(ret, fi)
