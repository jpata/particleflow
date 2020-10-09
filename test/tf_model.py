import os
import sys
import random
os.environ["KERAS_BACKEND"] = "tensorflow"

import glob
try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        print("importing setGPU")
        import setGPU
except:
    print("Could not import setGPU, please make sure you configure CUDA_VISIBLE_DEVICES manually")
    pass

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas
import time
from tqdm import tqdm
import itertools
import io
import sklearn
import sklearn.cluster
import tensorflow as tf
from plot_utils import plot_confusion_matrix
from numpy.lib.recfunctions import append_fields

import scipy
import scipy.special

from mpnn import MessagePassing, ReadoutGraph, Aggregation

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 11, 13, 22, 130, 211]

#LSH bins
nbins = 8
bins = tf.cast(tf.linspace(-1.0, 1.0, nbins), tf.float32)

#truncate elements beyond this cutoff in a single bin (to avoid too large dense-dense multiplications)
max_per_bin = tf.constant(256)
top_n_bins = nbins*nbins
batch_size = 10
num_max_elems = 5000
use_eager = False
attention_layer_cutoff = 0.2

#@tf.function
def sparse_dense_matmult_batch(sp_a, b):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)

#@tf.function
def valid_sparse_mat(n_points, sinds, bin_idx, bi, subpoints):
    #dm = tf.reduce_sum(tf.math.squared_difference(sp0, sp1), axis=2)
    dm = tf.matmul(subpoints, subpoints, transpose_a=True)
    dm = tf.nn.softmax(dm)

    mask = tf.cast(dm>attention_layer_cutoff, tf.float32)
    dm = dm * mask
    spt_small = tf.sparse.from_dense(dm)
    spt_this = tf.sparse.SparseTensor(tf.cast(tf.gather(sinds, spt_small.indices), tf.int64), tf.exp(-1.0*spt_small.values), (n_points, n_points))

    #print(len(spt_this.indices))
    #tf.print(spt_this.indices)
    # dists, top_inds = tf.nn.top_k(-dm, k=4)

    # #top_inds[:, 1:2] are the indices of the nearest neighbor, excluding the element itself
    # #create a [src, dst] pair array by concatenating the original index and the found neighbor 
    # nn_inds1 = tf.concat([tf.expand_dims(sinds, 1), tf.gather(sinds, top_inds[:, 1:2])], axis=1)
    # nn_inds1 = tf.cast(nn_inds1, dtype=tf.int64)

    # #top_inds[:, 2:3] are the indices of the second nearest neighbor, excluding the element itself
    # nn_inds2 = tf.concat([tf.expand_dims(sinds, 1), tf.gather(sinds, top_inds[:, 2:3])], axis=1)
    # nn_inds2 = tf.cast(nn_inds2, dtype=tf.int64)

    # nn_inds3 = tf.concat([tf.expand_dims(sinds, 1), tf.gather(sinds, top_inds[:, 3:4])], axis=1)
    # nn_inds3 = tf.cast(nn_inds3, dtype=tf.int64)

    # spt_this1 = tf.sparse.reorder(tf.sparse.SparseTensor(
    #     nn_inds1, tf.cast(-dists[:, 1], tf.float32), (n_points, n_points))
    # )
    # spt_this2 = tf.sparse.reorder(tf.sparse.SparseTensor(
    #     nn_inds2, tf.cast(-dists[:, 2], tf.float32), (n_points, n_points))
    # )
    # spt_this3 = tf.sparse.reorder(tf.sparse.SparseTensor(
    #     nn_inds2, tf.cast(-dists[:, 3], tf.float32), (n_points, n_points))
    # )
    # spt_this = tf.sparse.add(spt_this1, spt_this2)
    # spt_this = tf.sparse.add(spt_this, spt_this3)

    return spt_this

#@tf.function
def loop_cond(spt, inds, bin_idx, lsh_bin_index, good_bin_inds, points):
    return tf.math.less(lsh_bin_index, tf.math.minimum(tf.shape(good_bin_inds)[0], top_n_bins))

#@tf.function
def loop_body(spt, inds, bin_idx, lsh_bin_index, good_bin_inds, points):
    n_points = inds.shape[0]
    #in case the bin index is out of range, take the last bin
    mask = bin_idx == good_bin_inds[tf.minimum(lsh_bin_index, tf.shape(good_bin_inds)[0]-1)]
    sinds = inds[mask][:max_per_bin]
    subpoints = tf.gather(points, sinds, axis=0)

    spt_this = valid_sparse_mat(n_points, sinds, bin_idx, lsh_bin_index, subpoints)
    tf.sparse.add(spt, spt_this)
    return [spt, inds, bin_idx, lsh_bin_index, good_bin_inds, points]

#@tf.function(input_signature=[tf.TensorSpec(shape=(6000,254)), tf.TensorSpec(shape=(6000, 2)), ])
def construct_sparse_dm(points, points_lsh):
    n_points = tf.constant(points.shape[0])
    inds = tf.range(n_points)

    #put each input item into a bin defined by the softmax ouptut across the LSH embedding
    bin_idx = tf.argmax(tf.nn.softmax(points_lsh, axis=-1), axis=-1)
    uniqs = tf.unique_with_counts(bin_idx)

    sort_inds = tf.argsort(uniqs.count, direction="DESCENDING")[:top_n_bins]
    good_bin_inds = tf.gather(uniqs.y, sort_inds)
    good_bin_counts = tf.gather(uniqs.count, sort_inds)

    sparse_distance_matrix = tf.sparse.SparseTensor(indices=tf.zeros((0,2), dtype=np.int64), values=tf.zeros(0, tf.float32), dense_shape=(n_points, n_points))

    #loop over each LSH bin, prepare sparse distance matrix in bin, update final sparse distance matrix
    lsh_bin_index = tf.constant(0)

    #manually unrolled while_loop, otherwise can't profile or run in graph mode
    for i in range(top_n_bins):
        loop_body(sparse_distance_matrix, inds, bin_idx, i, good_bin_inds, points)
    #tf.while_loop(loop_cond, loop_body, [sparse_distance_matrix, inds, bin_idx, lsh_bin_index, good_bin_inds, points], parallel_iterations=1)

    return sparse_distance_matrix

def summarize_dataset(dataset):
    yclasses = []
    nev = 0.0
    ntot = 0.0
    sizes = []

    for X, y, w in dataset:
        yclasses += [y[:, 0]]
        nev += 1
        ntot += len(y)
        sizes += [len(y)]
    
    yclasses = np.concatenate(yclasses)
    values, counts= np.unique(yclasses, return_counts=True)
    print("nev={}".format(nev))
    print("sizes={}".format(np.percentile(sizes, [25, 50, 95, 99])))
    for v, c in zip(values, counts):
        print("label={} count={} frac={:.6f}".format(class_labels[int(v)], c, c/ntot))

#https://arxiv.org/pdf/1901.05555.pdf
beta = 0.9999 #beta -> 1 means weight by inverse frequency, beta -> 0 means no reweighting
def compute_weights_classbalanced(X, y, w):
    wn = (1.0 - beta)/(1.0 - tf.pow(beta, w))
    wn /= tf.reduce_sum(wn)
    return X, y, wn

#uniform weights
def compute_weights_uniform(X, y, w):
    wn = tf.ones_like(w)
    wn /= tf.reduce_sum(wn)
    return X, y, wn

#weight proportional to 1/sqrt(N)
def compute_weights_inverse(X, y, w):
    wn = 1.0/tf.sqrt(w)
    wn /= tf.reduce_sum(wn)
    return X, y, wn

weight_schemes = {
    "uniform": compute_weights_uniform,
    "inverse": compute_weights_inverse,
    "classbalanced": compute_weights_classbalanced,
}

def load_one_file(fn, num_clusters=10):
    Xs = []
    ys = []
    ys_cand = []
    dms = []

    data = pickle.load(open(fn, "rb"), encoding='iso-8859-1')
    for event in data:
        Xelem = event["Xelem"]
        ygen = event["ygen"]
        ycand = event["ycand"]

        #remove PS from inputs, they don't seem to be very useful
        msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3)

        Xelem = Xelem[~msk_ps]
        ygen = ygen[~msk_ps]
        ycand = ycand[~msk_ps]

        Xelem = append_fields(Xelem, "typ_idx", np.array([elem_labels.index(int(i)) for i in Xelem["typ"]], dtype=np.float32))
        ygen = append_fields(ygen, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32))
        ycand = append_fields(ycand, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32))
    
        Xelem_flat = np.stack([Xelem[k].view(np.float32).data for k in [
            'typ_idx',
            'pt', 'eta', 'phi', 'e',
            'layer', 'depth', 'charge', 'trajpoint',
            'eta_ecal', 'phi_ecal', 'eta_hcal', 'phi_hcal',
            'muon_dt_hits', 'muon_csc_hits']], axis=-1
        )
        ygen_flat = np.stack([ygen[k].view(np.float32).data for k in [
            'typ_idx',
            'eta', 'phi', 'e', 'charge',
            ]], axis=-1
        )
        ycand_flat = np.stack([ycand[k].view(np.float32).data for k in [
            'typ_idx',
            'eta', 'phi', 'e', 'charge',
            ]], axis=-1
        )

        #take care of outliers
        Xelem_flat[np.isnan(Xelem_flat)] = 0
        Xelem_flat[np.abs(Xelem_flat) > 1e4] = 0
        ygen_flat[np.isnan(ygen_flat)] = 0
        ygen_flat[np.abs(ygen_flat) > 1e4] = 0
        ycand_flat[np.isnan(ycand_flat)] = 0
        ycand_flat[np.abs(ycand_flat) > 1e4] = 0

        Xs += [Xelem_flat[:num_max_elems]]
        ys += [ygen_flat[:num_max_elems]]
        ys_cand += [ycand_flat[:num_max_elems]]
    
    print("created {} blocks, max size {}".format(len(Xs), max([len(X) for X in Xs])))
    return Xs, ys, ys_cand


class InputEncoding(tf.keras.layers.Layer):
    def __init__(self, num_input_classes):
        super(InputEncoding, self).__init__()
        self.num_input_classes = num_input_classes
        
    #@tf.function
    def call(self, X):
        #X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=tf.float32)

        #X[:, :, 1:] - all the other non-categorical features
        Xprop = X[:, :, 1:]
        return tf.concat([Xid, Xprop], axis=-1)

#https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConv(tf.keras.layers.Layer):
    def __init__(self, k, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.hidden_dim = args[0]
        self.k = k

        super(GHConv, self).__init__(*args, **kwargs)

        self.W_t = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_t", initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t", initializer="zeros")
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_h", initializer="random_normal")
        self.theta = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="theta", initializer="random_normal")
 
    #@tf.function
    def call(self, x, adj):
        #compute the normalization of the adjacency matrix
        in_degrees = tf.sparse.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        #norm_k = tf.pow(norm, self.k)
        #adj_k = tf.pow(adj, self.k)

        f_hom = tf.linalg.matmul(x, self.theta)
        f_hom = sparse_dense_matmult_batch(adj, f_hom*norm)*norm

        f_het = tf.linalg.matmul(x, self.W_h)
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)
        #tf.print(tf.reduce_mean(f_hom), tf.reduce_mean(f_het), tf.reduce_mean(gate))

        out = gate*f_hom + (1-gate)*f_het
        return out

class SGConv(tf.keras.layers.Dense):
    def __init__(self, k, *args, **kwargs):
        super(SGConv, self).__init__(*args, **kwargs)
        self.k = k
    
    #@tf.function
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]

        #compute the normalization of the adjacency matrix
        in_degrees = tf.sparse.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)

        support = tf.linalg.matmul(inputs, W)
     
        #k-th power of the normalized adjacency matrix is nearly equivalent to k consecutive GCN layers
        #adj_k = tf.pow(adj, self.k)
        out = sparse_dense_matmult_batch(adj, support*norm)*norm

        return self.activation(out + b)

#@tf.function
def predict_distancematrix(inputs):
    distcoords = inputs

    dms = []
    for ibatch in range(batch_size):
        dms.append(tf.sparse.expand_dims(construct_sparse_dm(distcoords[ibatch, :, top_n_bins:], distcoords[ibatch, :, :top_n_bins]), 0))

    dms = tf.sparse.concat(0, dms)
    return dms

#Simple message passing based on a matrix multiplication
class PFNet(tf.keras.Model):
    
    def __init__(self, activation=tf.nn.selu, hidden_dim=256, distance_dim=256, num_conv=4, convlayer="ghconv", dropout=0.1):
        super(PFNet, self).__init__()
        self.activation = activation

        self.enc = InputEncoding(len(elem_labels))

        self.layer_input_dist = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input_dist")

        self.layer_input1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input1")
        self.layer_input1_do = tf.keras.layers.Dropout(dropout)
        self.layer_input2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2")
        self.layer_input2_do = tf.keras.layers.Dropout(dropout)
        self.layer_input3 = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="input3")
        self.layer_input3_do = tf.keras.layers.Dropout(dropout)
        
        self.layer_input1_momentum = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input1_momentum")
        self.layer_input1_momentum_do = tf.keras.layers.Dropout(dropout)
        self.layer_input2_momentum = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2_momentum")
        self.layer_input2_momentum_do = tf.keras.layers.Dropout(dropout)
        self.layer_input3_momentum = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="input3_momentum")
        self.layer_input3_momentum_do = tf.keras.layers.Dropout(dropout)
        
        #self.layer_dist = Distance(distance_dim, name="distance")

        if convlayer == "sgconv":
            self.layer_conv1 = SGConv(num_conv, 2*hidden_dim, activation=activation, name="conv1")
            self.layer_conv2 = SGConv(num_conv, 2*hidden_dim+len(class_labels), activation=activation, name="conv2")
        elif convlayer == "ghconv":
            self.layer_conv1 = GHConv(num_conv, 2*hidden_dim, activation=activation, name="conv1")
            self.layer_conv2 = GHConv(num_conv, 2*hidden_dim+len(class_labels), activation=activation, name="conv2")

        self.layer_id1 = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="id1")
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        self.layer_charge = tf.keras.layers.Dense(1, activation="linear", name="out_charge")
        
        self.layer_momentum1 = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="momentum1")
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="linear", name="out_momentum")

    def create_model(self):
        inputs = tf.keras.Input(shape=(num_max_elems,15,))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs), name="MLPFNet")

    #@tf.function#(input_signature=[tf.TensorSpec(shape=(4,6000,15)), ])
    def call(self, inputs, training=True):
        X = tf.cast(inputs, tf.float32)
        msk_input = tf.expand_dims(tf.cast(X[:, :, 0] != 0, tf.float32), -1)
        enc = self.enc(inputs)

        x = self.layer_input_dist(enc)
        dm = predict_distancematrix(x)

        x = self.layer_input1(enc)
        x = self.layer_input1_do(x, training)
        x = self.layer_input2(x)
        x = self.layer_input2_do(x, training)
        x = self.layer_input3(x)
        x = self.layer_input3_do(x, training)
        x = self.layer_conv1(x, dm)
        x = self.layer_id1(x)
        x = self.layer_id2(x)
        x = self.layer_id3(x)
        out_id_logits = self.layer_id(x)
        out_charge = self.layer_charge(x)
        
        x = self.layer_input1_momentum(enc)
        x = self.layer_input1_momentum_do(x, training)
        x = self.layer_input2_momentum(x)
        x = self.layer_input2_momentum_do(x, training)
        x = self.layer_input3_momentum(x)
        x = self.layer_input3_momentum_do(x, training)
        x = tf.concat([x, out_id_logits], axis=-1)
        x = self.layer_conv2(x, dm)
        x = self.layer_momentum1(x)
        x = self.layer_momentum2(x)
        x = self.layer_momentum3(x)
        pred_corr = self.layer_momentum(x)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        msk_good = tf.cast(out_id != 0, tf.float32)
        out_momentum_eta = X[:, :, 2] + pred_corr[:, :, 0]
        out_momentum_phi = X[:, :, 3] + pred_corr[:, :, 1] 
        out_momentum_E = X[:, :, 4] + pred_corr[:, :, 2]

        out_momentum = tf.stack([
            out_momentum_eta,
            out_momentum_phi,
            out_momentum_E,
        ], axis=-1)

        ret = tf.concat([out_id_logits, out_momentum, out_charge], axis=-1)*msk_input
        return ret

#Based on MPNN, implemented by KX
class PFNet2(tf.keras.Model):
    def __init__(self, hidden_sizes=[128, 128], num_outputs=128, state_dim=16, update_steps=3, activation=tf.nn.selu, hidden_dim=256):
        #super(PFNet2, self).__init__()
        self.activation = activation

        self.enc = InputEncoding(len(elem_labels))

        self.update_steps = int(update_steps)
        self.node_embedding = tf.keras.layers.Dense(units=state_dim, activation=activation)
        self.message_passing = MessagePassing(state_dim=state_dim)
        self.readout_func = ReadoutGraph(hidden_sizes, num_outputs, Aggregation('sum', 2))

        self.layer_id1 = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="id1")
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        self.layer_charge = tf.keras.layers.Dense(1, activation="linear", name="out_charge")
        
        self.layer_momentum1 = tf.keras.layers.Dense(2*hidden_dim, activation=activation, name="momentum1")
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="linear", name="out_momentum")
 

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, 15], dtype=tf.float32)])
    def call(self, inputs, training=True):
        x = self.enc(inputs)
        nodes = x
        bs = nodes.shape[0] 
        bs = 10
        node_elem = nodes.shape[1] # number of particles (elements)
        node_cols = nodes.shape[2] #25 is the node.shape[1] after encoding
        edges = tf.constant(0, dtype = "float32", shape= [bs,  np.power(node_elem, 2), 1])
        edge_masks = tf.cast(tf.random.uniform([bs,np.power(node_elem,2), 1], 
                                        minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.float32)
        states = self.node_embedding(nodes)

        for time_step in range(self.update_steps):
            states = self.message_passing(states, edges, edge_masks, training=training)
        node_masks = tf.constant(1., dtype = "float32", shape= [bs, node_elem,1]) 
        readout = self.readout_func(states, node_masks, training=training)
        
        x = self.layer_id1(readout)
        x = self.layer_id2(x)
        x = self.layer_id3(x)
        out_id_logits = self.layer_id(x)
        out_charge = self.layer_charge(x)

        x = self.layer_momentum1(readout)
        x = self.layer_momentum2(x)
        x = self.layer_momentum3(x)
        pred_corr = self.layer_momentum(x)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        msk_good = tf.cast(out_id != 0, tf.float32)

        out_momentum_eta = inputs[:, :, 2] + pred_corr[:, :, 0]
        out_momentum_phi = inputs[:, :, 3] + pred_corr[:, :, 1] 
        out_momentum_E = inputs[:, :, 4] + pred_corr[:, :, 2]

        out_momentum = tf.stack([
            out_momentum_eta,
            out_momentum_phi,
            out_momentum_E,
        ], axis=-1)

        ret = tf.concat([out_id_logits, out_momentum, out_charge], axis=-1)
        return ret

#@tf.function
def separate_prediction(y_pred):
    N = len(class_labels)
    pred_id_onehot = y_pred[:, :, :N]
    pred_momentum = y_pred[:, :, N:N+3]
    pred_charge = y_pred[:, :, N+3:N+4]
    return pred_id_onehot, pred_charge, pred_momentum

#@tf.function
def separate_truth(y_true):
    true_id = tf.cast(y_true[:, :, :1], tf.int32)
    true_momentum = y_true[:, :, 1:4]
    true_charge = y_true[:, :, 4:5]
    return true_id, true_charge, true_momentum

def mse_unreduced(true, pred):
    return tf.math.pow(true-pred,2)

def msle_unreduced(true, pred):
    return tf.math.pow(tf.math.log(tf.math.abs(true) + 1.0) - tf.math.log(tf.math.abs(pred) + 1.0), 2)

def my_loss_cls(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    #predict the particle class labels
    l1 = 1e4*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
    #l1 = 1e4*tf.keras.losses.categorical_crossentropy(true_id_onehot[:, :, 0], pred_id_onehot, from_logits=True)
    return 1e3*l1

def my_loss_reg(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))

    l2_0 = mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])*10
    l2_1 = mse_unreduced(tf.math.floormod(true_momentum[:, :, 1] - pred_momentum[:, :, 1] + np.pi, 2*np.pi) - np.pi, 0.0)*10
    l2_2 = mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])/100.0

    l2 = (l2_0 + l2_1 + l2_2)
    
    return l2

#@tf.function
def my_loss_full(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    #tf.print(pred_id_onehot)
    l1 = 1e3*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
  
    #msk_good = (true_id[:, 0] == pred_id)
    #nsamp = tf.cast(tf.size(y_pred), tf.float32)

    l2_0 = mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mse_unreduced(tf.math.floormod(true_momentum[:, :, 1] - pred_momentum[:, :, 1] + np.pi, 2*np.pi) - np.pi, 0.0)
    l2_2 = mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])/100.0

    l2 = (l2_0 + l2_1 + l2_2)
    #l2 = tf.multiply(tf.cast(msk_good, tf.float32), l2)

    l3 = mse_unreduced(true_charge, pred_charge)[:, :, 0]

    #tf.debugging.check_numerics(l1, "l1")
    #tf.debugging.check_numerics(l2_0, "l2_0")
    #tf.debugging.check_numerics(l2_1, "l2_1")
    #tf.debugging.check_numerics(l2_2, "l2_2")

    #tf.print("l1", tf.reduce_mean(l1))
    #tf.print("l2_0", tf.reduce_mean(l2_0))
    #tf.print("l2_1", tf.reduce_mean(l2_1))
    #tf.print("l2_2", tf.reduce_mean(l2_2))
    #tf.print("l2", tf.reduce_mean(l2))
    #tf.print("l3", tf.reduce_mean(l3))

    #tf.print("\n")
    l = (l1 + l2 + l3)

    return 1e3*l

#TODO: put these in a class
def cls_130(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk_true = true_id[:, :, 0] == class_labels.index(130)
    msk_pos = pred_id == class_labels.index(130)
    num_true_pos = tf.reduce_sum(tf.cast(msk_true&msk_pos, tf.float32))
    num_true = tf.reduce_sum(tf.cast(msk_true, tf.float32))
    return num_true_pos/num_true

def cls_211(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk_true = true_id[:, :, 0] == class_labels.index(211)
    msk_pos = pred_id == class_labels.index(211)
    num_true_pos = tf.reduce_sum(tf.cast(msk_true&msk_pos, tf.float32))
    num_true = tf.reduce_sum(tf.cast(msk_true, tf.float32))

    return num_true_pos/num_true

def cls_22(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk_true = true_id[:, :, 0] == class_labels.index(22)
    msk_pos = pred_id == class_labels.index(22)
    num_true_pos = tf.reduce_sum(tf.cast(msk_true&msk_pos, tf.float32))
    num_true = tf.reduce_sum(tf.cast(msk_true, tf.float32))

    return num_true_pos/num_true

def cls_11(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk_true = true_id[:, :, 0] == class_labels.index(11)
    msk_pos = pred_id == class_labels.index(11)
    num_true_pos = tf.reduce_sum(tf.cast(msk_true&msk_pos, tf.float32))
    num_true = tf.reduce_sum(tf.cast(msk_true, tf.float32))

    return num_true_pos/num_true

def cls_13(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk_true = true_id[:, :, 0] == class_labels.index(13)
    msk_pos = pred_id == class_labels.index(13)
    num_true_pos = tf.reduce_sum(tf.cast(msk_true&msk_pos, tf.float32))
    num_true = tf.reduce_sum(tf.cast(msk_true, tf.float32))

    return num_true_pos/num_true

#@tf.function
def num_pred(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    ntrue = tf.reduce_sum(tf.cast(true_id[:, :, 0]!=0, tf.int32))
    npred = tf.reduce_sum(tf.cast(pred_id!=0, tf.int32))
    return tf.cast(ntrue - npred, tf.float32)

#@tf.function
def eta_resolution(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk = true_id[:, :, 0]!=0
    return tf.reduce_mean(mse_unreduced(true_momentum[msk][:, 0], pred_momentum[msk][:, 0]))

#@tf.function
def phi_resolution(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk = true_id[:, :, 0]!=0
    return tf.reduce_mean(mse_unreduced(tf.math.floormod(true_momentum[msk][:, 1] - pred_momentum[msk][:, 1] + np.pi, 2*np.pi) - np.pi, 0.0))

#@tf.function(experimental_relax_shapes=True)
def energy_resolution(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk = true_id[:, :, 0]!=0
    return tf.reduce_mean(mse_unreduced(true_momentum[msk][:, 2], pred_momentum[msk][:, 2]))

def get_unique_run():
    previous_runs = os.listdir('experiments')
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    return run_number

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PFNet", help="type of model to train", choices=["PFNet", "PFNet2"])
    parser.add_argument("--ntrain", type=int, default=100, help="number of training events")
    parser.add_argument("--ntest", type=int, default=100, help="number of testing events")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--nhidden", type=int, default=256, help="hidden dimension")
    parser.add_argument("--num-conv", type=int, default=1, help="number of convolution layers (powers)")
    parser.add_argument("--distance-dim", type=int, default=256, help="distance dimension")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    parser.add_argument("--weights", type=str, choices=["uniform", "inverse", "classbalanced"], help="Sample weighting scheme to use", default="inverse")
    parser.add_argument("--name", type=str, default=None, help="where to store the output")
    parser.add_argument("--convlayer", type=str, default="sgconv", choices=["sgconv", "ghconv"], help="Type of graph convolutional layer")
    parser.add_argument("--load", type=str, default=None, help="model to load")
    parser.add_argument("--datapath", type=str, help="Input data path", required=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.0, help="learning rate decay")
    parser.add_argument("--train-cls", action="store_true", help="Train only the classification part")
    parser.add_argument("--train-reg", action="store_true", help="Train only the regression part")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()
    return args

def assign_label(pred_id_onehot_linear):
    ret2 = np.argmax(pred_id_onehot_linear, axis=-1)
    return ret2

def prepare_df(epoch, model, data, outdir, target, save_raw=False):
    tf.print("\nprepare_df")

    dfs = []
    for iev, d in enumerate(data):
        if iev%50==0:
            tf.print(".", end="")
        X, y, w = d
        pred = model(X).numpy()
        pred_id_onehot, pred_charge, pred_momentum = separate_prediction(pred)
        pred_id = assign_label(pred_id_onehot).flatten()
 
        if save_raw:
            np.savez_compressed("ev_{}.npz".format(iev), X=X.numpy(), y=y.numpy(), w=w.numpy(), y_pred=pred)

        pred_charge = pred_charge[:, :, 0].flatten()
        pred_momentum = pred_momentum.reshape((pred_momentum.shape[0]*pred_momentum.shape[1], pred_momentum.shape[2]))

        true_id, true_charge, true_momentum = separate_truth(y)
        true_id = true_id.numpy()[:, :, 0].flatten()
        true_charge = true_charge.numpy()[:, :, 0].flatten()
        true_momentum = true_momentum.numpy().reshape((true_momentum.shape[0]*true_momentum.shape[1], true_momentum.shape[2]))
       
        df = pandas.DataFrame()
        df["pred_pid"] = np.array([int(class_labels[p]) for p in pred_id])
        df["pred_eta"] = np.array(pred_momentum[:, 0], dtype=np.float64)
        df["pred_phi"] = np.array(pred_momentum[:, 1], dtype=np.float64)
        df["pred_e"] = np.array(pred_momentum[:, 2], dtype=np.float64)

        df["{}_pid".format(target)] = np.array([int(class_labels[p]) for p in true_id])
        df["{}_eta".format(target)] = np.array(true_momentum[:, 0], dtype=np.float64)
        df["{}_phi".format(target)] = np.array(true_momentum[:, 1], dtype=np.float64)
        df["{}_e".format(target)] = np.array(true_momentum[:, 2], dtype=np.float64)

        df["iev"] = iev
        dfs += [df]
    df = pandas.concat(dfs, ignore_index=True)
    fn = outdir + "/df_{}.pkl.bz2".format(epoch + 1)
    df.to_pickle(fn)
    tf.print("\nprepare_df done", fn)

class DataFrameCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, outdir, freq=5):
        self.dataset = dataset
        self.outdir = outdir
        self.freq = freq

    def on_epoch_end(self, epoch, logs):
        if epoch > 0 and (epoch + 1)%self.freq == 0:
            prepare_df(epoch, self.model, self.dataset, self.outdir)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def load_dataset_gun():
    globs = [
        "test/SingleGammaFlatPt10To100_pythia8_cfi/tfr/cand/chunk_0.tfrecords",
        #"test/SingleElectronFlatPt1To100_pythia8_cfi/tfr/cand/chunk_0.tfrecords",
        #"test/SingleMuFlatPt0p7To10_cfi/tfr/cand/chunk_0.tfrecords",
        #"test/SinglePi0E10_pythia8_cfi/tfr/cand/chunk_0.tfrecords",
        #"test/SinglePiFlatPt0p7To10_cfi/tfr/cand/chunk_0.tfrecords",
        #"test/SingleTauFlatPt2To150_cfi/tfr/cand/chunk_0.tfrecords",
    ]

    tfr_files = []
    for g in globs:
        tfr_files += [g]
    tfr_files = sorted(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(5000)
    return dataset

def load_dataset_ttbar(datapath):
    path = datapath + "/tfr/{}/*.tfrecords".format(args.target)
    tfr_files = glob.glob(path)
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(path))
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    #tf.debugging.enable_check_numerics()
    tf.config.run_functions_eagerly(use_eager)

    args = parse_args()

    try:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        print("num_gpus=", num_gpus)
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        print("fallback to CPU")
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    filelist = sorted(glob.glob(args.datapath + "/raw/*.pkl"))[:args.ntrain+args.ntest]

    from tf_data import _parse_tfr_element

    #dataset = load_dataset_gun()
    dataset = load_dataset_ttbar(args.datapath)

    ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
    ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).padded_batch(batch_size, padded_shapes=ps)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).padded_batch(batch_size, padded_shapes=ps)

    ds_train_r = ds_train.repeat(args.nepochs)
    ds_test_r = ds_test.repeat(args.nepochs)

    if args.lr_decay > 0:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=10*int(args.ntrain/batch_size),
            decay_rate=args.lr_decay
        )
    else:
        lr_schedule = args.lr

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        if args.model == "PFNet":
            model = PFNet(hidden_dim=args.nhidden, distance_dim=args.distance_dim, num_conv=args.num_conv, convlayer=args.convlayer, dropout=args.dropout)
        elif args.model == "PFNet2":
            model = PFNet2(hidden_sizes = [args.nhidden, args.nhidden], num_outputs=128, state_dim=16, update_steps=3, hidden_dim=args.nhidden)
        if use_eager:
            model(np.random.randn(batch_size, num_max_elems, 15).astype(np.float32))
        else:
            model = model.create_model()
        model.summary()

    if not os.path.isdir("experiments"):
        os.makedirs("experiments")

    if args.name is None:
        args.name =  'run_{:02}'.format(get_unique_run())

    outdir = 'experiments/' + args.name

    if os.path.isdir(outdir):
        print("Output directory exists: {}".format(outdir), file=sys.stderr)
        sys.exit(1)

    print(outdir)
    callbacks = []
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=outdir, histogram_freq=10, write_graph=True, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,90),
        profile_batch=0,
    )
    tb.set_model(model)
    callbacks += [tb]

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir + "/weights.{epoch:02d}-{val_loss:.6f}.hdf5",
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model)
    callbacks += [cp_callback]

    loss_fn = my_loss_full
    if args.train_cls:
        loss_fn = my_loss_cls
        model.layer_conv2.trainable = False
        model.layer_momentum1.trainable = False
        model.layer_momentum2.trainable = False
        model.layer_momentum3.trainable = False
        model.layer_momentum4.trainable = False
        model.layer_momentum.trainable = False
    elif args.train_reg:
        loss_fn = my_loss_reg
        for layer in model.layers:
            layer.trainable = False
        model.layer_input1_momentum.trainable = True
        model.layer_input2_momentum.trainable = True
        model.layer_input3_momentum.trainable = True
        model.layer_conv2.trainable = True
        model.layer_momentum1.trainable = True
        model.layer_momentum2.trainable = True
        model.layer_momentum3.trainable = True
        model.layer_momentum.trainable = True

    with strategy.scope():
        model.compile(optimizer=opt, loss=loss_fn,
            metrics=[cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution],
            sample_weight_mode="temporal")

        # for X, y, w in ds_train:
        #     ypred = model(X)
        #     l = loss_fn(y, ypred)
        #     cls_130(y, ypred)

        if args.load:
            #ensure model input size is known
            for X, y, w in ds_train:
                model(X)
                break
   
            model.load_weights(args.load)
        if args.nepochs > 0:
            ret = model.fit(ds_train_r,
                validation_data=ds_test_r, epochs=args.nepochs,
                steps_per_epoch=args.ntrain/batch_size, validation_steps=args.ntest/batch_size,
                verbose=True,
                callbacks=callbacks
            )
