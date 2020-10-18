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
import itertools
import io
import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from numpy.lib.recfunctions import append_fields

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 11, 13, 22, 130, 211]

num_max_elems = 5000

mult_classification_loss = 1e3
mult_charge_loss = 1.0
mult_energy_loss = 10.0
mult_phi_loss = 10.0
mult_eta_loss = 10.0
mult_total_loss = 1e3

def my_softmax_split(cmul, nbins):
    bin_idx = tf.argmax(cmul, axis=-1)
    bins_split = tf.split(tf.cast(tf.argsort(bin_idx), tf.int64), nbins)
    return bins_split

def softargmax(x, beta, xrange):
    return tf.reduce_sum(tf.nn.softmax(x*beta) * xrange, axis=-1)

"""
sp_a: (nbatch, nelem, nelem) sparse distance matrices
b: (nbatch, nelem, ncol) dense per-elemenet feature matrices
"""
def sparse_dense_matmult_batch(sp_a, b):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    ret = tf.map_fn(map_function, elems, fn_output_signature=tf.float32, back_prop=True)
    return ret 

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

def load_one_file(fn):
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

    """
        X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data
    """        
    @tf.function
    def call(self, X):

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=tf.float32)

        #X[:, :, 1:] - all the other non-categorical features
        Xprop = X[:, :, 1:]
        return tf.concat([Xid, Xprop], axis=-1)

#https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConv(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.hidden_dim = args[0]

        super(GHConv, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.W_t = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_t", initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t", initializer="random_normal")
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_h", initializer="random_normal")
        self.theta = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="theta", initializer="random_normal")
 
    #@tf.function
    def call(self, inputs):
        x, adj = inputs

        #compute the normalization of the adjacency matrix
        in_degrees = tf.sparse.reduce_sum(adj, axis=-1)
        in_degrees = tf.reshape(in_degrees, (tf.shape(x)[0], tf.shape(x)[1]))

        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)

        f_hom = tf.linalg.matmul(x, self.theta)
        f_hom = sparse_dense_matmult_batch(adj, f_hom*norm)*norm

        f_het = tf.linalg.matmul(x, self.W_h)
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)

        out = gate*f_hom + (1-gate)*f_het
        return self.activation(out)

class SGConv(tf.keras.layers.Dense):
    def __init__(self, k, *args, **kwargs):
        super(SGConv, self).__init__(*args, **kwargs)
        self.activation = kwargs.pop("activation")
    
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


class SparseAttentionDistance(tf.keras.layers.Layer):
    def __init__(self, distance_dim, nbins=10, batch_size=10, num_neighbors=5):
        super(SparseAttentionDistance, self).__init__()
        self.nbins = nbins
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors

        self.random_rotations = tf.constant(tf.random.normal((distance_dim, nbins//2)))

    @tf.function
    def call(self, inputs, training=True):
        point_embedding = inputs

        #cannot concat sparse tensors directly as that incorrectly destroys the gradient, see
        #https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/python/ops/sparse_grad.py#L33
        #therefore, for training, we implement sparse concatenation by hand 
        if training:
            indices_all = []
            values_all = []
            for ibatch in range(self.batch_size):
                dm = self.construct_sparse_dm(point_embedding[ibatch])
                indices_all.append(
                    tf.concat([tf.expand_dims(ibatch*tf.ones(tf.shape(dm.indices)[0], dtype=tf.int64), -1), dm.indices], axis=-1)
                )
                values_all.append(dm.values)

            #now create a new sparsetensor that is a concatenation of the previous ones
            shp = tf.shape(inputs)
            dms = tf.SparseTensor(
                tf.concat(indices_all, axis=0),
                tf.concat(values_all, axis=0),
                (shp[0], shp[1], shp[1])
            )
        else:
            for ibatch in range(self.batch_size):
                dms.append(tf.sparse.expand_dims(self.construct_sparse_dm(
                    point_embedding[ibatch]), 0))
            dms = tf.sparse.concat(0, dms)

        return tf.sparse.reorder(dms)

    @tf.function
    def valid_sparse_mat(self, n_points, subindices, subpoints):

        #find the cosine distance between the given points using dense matrix multiplication
        normed = tf.nn.l2_normalize(subpoints, axis=1)
        dm = tf.matmul(normed, normed, transpose_b=True)
        dm = tf.nn.softmax(dm, axis=-1)

        #run KNN in the distance matrix, accumulate each index pair into a sparse distance matrix
        top_k = tf.nn.top_k(dm, k=self.num_neighbors)
        sp_sum = tf.sparse.SparseTensor(indices=tf.zeros((0,2), dtype=np.int64), values=tf.zeros(0, tf.float32), dense_shape=(n_points, n_points))
        for i in range(self.num_neighbors):
            inds_to_gather = tf.transpose(tf.stack([tf.range(tf.shape(dm)[0]), top_k.indices[:, i]]))
            indices_in_full = tf.gather(subindices, inds_to_gather)
            sp_sum = tf.sparse.add(sp_sum, tf.sparse.SparseTensor(indices_in_full, top_k.values[:, i], (n_points, n_points)))

        spt_this = tf.sparse.reorder(sp_sum)

        return spt_this

    @tf.function
    def loop_body(self, subindices, points):
        n_points = points.shape[0]

        #get the embedding data for the chosen points
        subpoints = tf.gather(points, subindices, axis=0)

        #generate a sparse distance matrix with size [n_points, n_points] between these points 
        spt_this = self.valid_sparse_mat(n_points, subindices, subpoints)

        #add the distance matrix between the chosen points to the total distance matrix
        return spt_this

    @tf.function
    def construct_sparse_dm(self, points):
        n_points = tf.shape(points)[0]

        #put each input item into a bin defined by the softmax output across the LSH embedding
        mul = tf.linalg.matmul(points, self.random_rotations)
        cmul = tf.concat([mul, -mul], axis=-1)
        bins_split = my_softmax_split(cmul, self.nbins)

        #loop over each LSH bin, prepare sparse distance matrix in bin, update final sparse distance matrix
        sparse_distance_matrix = tf.sparse.SparseTensor(indices=tf.zeros((0,2), dtype=np.int64), values=tf.zeros(0, tf.float32), dense_shape=(n_points, n_points))
        for bin_inds in bins_split:
            sparse_distance_matrix = tf.sparse.add(sparse_distance_matrix, self.loop_body(bin_inds, points))

        return tf.sparse.reorder(sparse_distance_matrix)

class EncoderDecoderGNN(tf.keras.layers.Layer):
    def __init__(self, encoders, decoders, dropout, activation, conv, **kwargs):
        super(EncoderDecoderGNN, self).__init__(**kwargs)
        name = kwargs.get("name")

        assert(encoders[-1] == decoders[0])
        self.encoders = encoders
        self.decoders = decoders

        self.encoding_layers = []
        for ilayer, nunits in enumerate(encoders):
            self.encoding_layers.append(
                tf.keras.layers.Dense(nunits, activation=activation, name="encoding_{}_{}".format(name, ilayer)))
            if dropout > 0.0:
                self.encoding_layers.append(tf.keras.layers.Dropout(dropout))

        self.conv = conv

        self.decoding_layers = []
        for ilayer, nunits in enumerate(decoders):
            self.decoding_layers.append(
                tf.keras.layers.Dense(nunits, activation=activation, name="decoding_{}_{}".format(name, ilayer)))
            if dropout > 0.0:
                self.decoding_layers.append(tf.keras.layers.Dropout(dropout))

    @tf.function
    def call(self, inputs, distance_matrix, training=True):
        x = inputs

        for layer in self.encoding_layers:
            x = layer(x)

        for convlayer in self.conv:
            x = convlayer([x, distance_matrix])

        for layer in self.decoding_layers:
            x = layer(x)

        return x

#Simple message passing based on a matrix multiplication
class PFNet(tf.keras.Model):
    def __init__(self,
        activation=tf.nn.selu,
        hidden_dim_id=256,
        hidden_dim_reg=256,
        distance_dim=256,
        convlayer="ghconv",
        dropout=0.1,
        nbins=10,
        batch_size=10,
        num_convs_id=1,
        num_convs_reg=1,
        num_hidden_id=1,
        num_hidden_reg=1,
        num_neighbors=5):

        super(PFNet, self).__init__()
        self.activation = activation

        encoding_id = []
        decoding_id = []
        encoding_reg = []
        decoding_reg = []
        #the encoder outputs and decoder inputs have to have the hidden dim (convlayer size)
        for ihidden in range(num_hidden_id):
            encoding_id.append(hidden_dim_id)
            decoding_id.append(hidden_dim_id)

        for ihidden in range(num_hidden_reg):
            encoding_reg.append(hidden_dim_reg)
            decoding_reg.append(hidden_dim_reg)

        self.enc = InputEncoding(len(elem_labels))
        self.layer_embedding = tf.keras.layers.Dense(distance_dim, activation=activation, name="embedding_attention")
        self.dist = SparseAttentionDistance(26+distance_dim, nbins, batch_size, num_neighbors)

        convs_id = []
        convs_reg = []
        if convlayer == "sgconv":
            for iconv in range(num_convs_id):
                convs_id.append(SGConv(hidden_dim_id, activation=activation, name="conv_id{}".format(iconv)))
            for iconv in range(num_convs_reg):
                convs_reg.append(SGConv(hidden_dim_reg, activation=activation, name="conv_reg{}".format(iconv)))
        elif convlayer == "ghconv":
            for iconv in range(num_convs_id):
                convs_id.append(GHConv(hidden_dim_id, activation=activation, name="conv_id{}".format(iconv)))
            for iconv in range(num_convs_reg):
                convs_reg.append(GHConv(hidden_dim_reg, activation=activation, name="conv_reg{}".format(iconv)))

        self.gnn_id = EncoderDecoderGNN(encoding_id, decoding_id, dropout, activation, convs_id, name="gnn_id")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        self.layer_charge = tf.keras.layers.Dense(1, activation="linear", name="out_charge")
        
        self.gnn_reg = EncoderDecoderGNN(encoding_reg, decoding_reg, dropout, activation, convs_reg, name="gnn_reg")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="linear", name="out_momentum")

    def create_model(self):
        inputs = tf.keras.Input(shape=(num_max_elems,15,))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs), name="MLPFNet")

    def call(self, inputs, training=True):
        X = tf.cast(inputs, tf.float32)
        msk_input = tf.expand_dims(tf.cast(X[:, :, 0] != 0, tf.float32), -1)

        enc = self.enc(inputs)

        #embed inputs for graph structure prediction
        embedding_attention = self.layer_embedding(enc)

        #create graph structure by predicting a sparse distance matrix
        dm = self.dist(tf.concat([enc, embedding_attention], axis=-1), training)

        #run graph net for multiclass id prediction
        x_id = self.gnn_id(tf.concat([enc, embedding_attention], axis=-1), dm, training)
        out_id_logits = self.layer_id(x_id)
        out_charge = self.layer_charge(x_id)

        #run graph net for regression output prediction, taking as an additonal input the ID predictions
        x_reg = self.gnn_reg(tf.concat([enc, embedding_attention, out_id_logits, out_charge], axis=-1), dm, training)
        pred_corr = self.layer_momentum(x_reg)

        #soft-mask elements for which the id prediction was 0  
        probabilistic_mask_good = 1.0 - tf.keras.activations.softmax(out_id_logits)[:, :, 0]

        out_momentum_eta = X[:, :, 2] + pred_corr[:, :, 0]
        out_momentum_phi = X[:, :, 3] + pred_corr[:, :, 1] 
        out_momentum_E = X[:, :, 4] + pred_corr[:, :, 2]

        out_momentum = tf.stack([
            out_momentum_eta * probabilistic_mask_good,
            out_momentum_phi * probabilistic_mask_good,
            out_momentum_E * probabilistic_mask_good,
        ], axis=-1)

        ret = tf.concat([out_id_logits, out_momentum, out_charge], axis=-1)*msk_input
        return ret

#@tf.function
def separate_prediction(y_pred):
    N = len(class_labels)
    pred_id_logits = y_pred[:, :, :N]
    pred_momentum = y_pred[:, :, N:N+3]
    pred_charge = y_pred[:, :, N+3:N+4]
    return pred_id_logits, pred_charge, pred_momentum

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
    pred_id_logits, pred_charge, _ = separate_prediction(y_pred)
    true_id, true_charge, _ = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    #predict the particle class labels
    l1 = mult_classification_loss*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_logits)
    l3 = mult_charge_loss*mse_unreduced(true_charge, pred_charge)[:, :, 0]

    loss = l1 + l3
    return mult_total_loss*loss

def my_loss_reg(y_true, y_pred):
    _, _, pred_momentum = separate_prediction(y_pred)
    _, true_charge, true_momentum = separate_truth(y_true)

    l2_0 = mult_eta_loss*mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mult_phi_loss*mse_unreduced(tf.math.floormod(true_momentum[:, :, 1] - pred_momentum[:, :, 1] + np.pi, 2*np.pi) - np.pi, 0.0)
    l2_2 = mult_energy_loss*mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])

    loss = (l2_0 + l2_1 + l2_2)
    
    return 1e3*loss

def my_loss_full(y_true, y_pred):
    pred_id_logits, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_logits, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)
    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    
    l1 = mult_classification_loss*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_logits)
  
    l2_0 = mult_eta_loss*mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mult_phi_loss*mse_unreduced(tf.math.floormod(true_momentum[:, :, 1] - pred_momentum[:, :, 1] + np.pi, 2*np.pi) - np.pi, 0.0)
    l2_2 = mult_energy_loss*mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])

    l2 = (l2_0 + l2_1 + l2_2)

    l3 = mult_charge_loss*mse_unreduced(true_charge, pred_charge)[:, :, 0]
    loss = l1 + l2 + l3

    return mult_total_loss*loss

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

def accuracy(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    is_true = true_id[:, :, 0]!=0
    is_same = true_id[:, :, 0] == pred_id

    acc = tf.reduce_sum(tf.cast(is_true&is_same, tf.int32)) / tf.reduce_sum(tf.cast(is_true, tf.int32))
    return tf.cast(acc, tf.float32)

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
    parser.add_argument("--model", type=str, default="PFNet", help="type of model to train", choices=["PFNet"])
    parser.add_argument("--ntrain", type=int, default=100, help="number of training events")
    parser.add_argument("--ntest", type=int, default=100, help="number of testing events")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--hidden-dim-id", type=int, default=256, help="hidden dimension")
    parser.add_argument("--hidden-dim-reg", type=int, default=256, help="hidden dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="number of events in training batch")
    parser.add_argument("--num-convs-id", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--num-convs-reg", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--num-hidden-id", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num-hidden-reg", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num-neighbors", type=int, default=5, help="number of knn neighbors")
    parser.add_argument("--distance-dim", type=int, default=256, help="distance dimension")
    parser.add_argument("--nbins", type=int, default=10, help="number of locality-sensitive hashing (LSH) bins")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
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
    parser.add_argument("--eager", action="store_true", help="Run in eager mode for debugging")
    args = parser.parse_args()
    return args

def assign_label(pred_id_onehot_linear):
    ret2 = np.argmax(pred_id_onehot_linear, axis=-1)
    return ret2

def prepare_df(model, data, outdir, target, save_raw=False):
    print("prepare_df")

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
    fn = outdir + "/df.pkl.bz2"
    df.to_pickle(fn)
    print("prepare_df done", fn)

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

def load_dataset_ttbar(datapath, target):
    from tf_data import _parse_tfr_element
    path = datapath + "/tfr/{}/*.tfrecords".format(target)
    tfr_files = glob.glob(path)
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(path))
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    args = parse_args()

    #tf.debugging.enable_check_numerics()
    tf.config.run_functions_eagerly(args.eager)

    #batch size for loading data must be configured according to the number of distributed GPUs 
    global_batch_size = args.batch_size
    try:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        print("num_gpus=", num_gpus)
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            global_batch_size = num_gpus * args.batch_size
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        print("fallback to CPU")
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    filelist = sorted(glob.glob(args.datapath + "/raw/*.pkl"))[:args.ntrain+args.ntest]

    dataset = load_dataset_ttbar(args.datapath, args.target)

    #create padded input data
    ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
    ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps).cache().prefetch(tf.data.experimental.AUTOTUNE)

    #repeat needed for keras api
    ds_train_r = ds_train.repeat(args.nepochs)
    ds_test_r = ds_test.repeat(args.nepochs)

    if args.lr_decay > 0:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=10*int(args.ntrain/global_batch_size),
            decay_rate=args.lr_decay
        )
    else:
        lr_schedule = args.lr

    loss_fn = my_loss_full

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model = PFNet(
            hidden_dim_id=args.hidden_dim_id,
            hidden_dim_reg=args.hidden_dim_reg,
            num_convs_id=args.num_convs_id,
            num_convs_reg=args.num_convs_reg,
            num_hidden_id=args.num_hidden_id,
            num_hidden_reg=args.num_hidden_reg,
            distance_dim=args.distance_dim,
            convlayer=args.convlayer,
            dropout=args.dropout,
            batch_size=args.batch_size,
            nbins=args.nbins,
            num_neighbors=args.num_neighbors
        )

        if args.train_cls:
            loss_fn = my_loss_cls
            model.gnn_reg.trainable = False
            model.layer_momentum.trainable = False
        elif args.train_reg:
            loss_fn = my_loss_reg
            for layer in model.layers:
                layer.trainable = False
            model.gnn_reg.trainable = True
            model.layer_momentum.trainable = True

        #model(np.random.randn(args.batch_size, num_max_elems, 15).astype(np.float32))
        if not args.eager:
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
        log_dir=outdir, histogram_freq=0, write_graph=False, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,40),
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

    with strategy.scope():
        model.compile(optimizer=opt, loss=loss_fn,
            metrics=[accuracy, cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution],
            sample_weight_mode="temporal")

        if args.load:
            #ensure model input size is known
            for X, y, w in ds_train:
                model(X)
                break
   
            model.load_weights(args.load)

        if args.nepochs > 0:
            ret = model.fit(ds_train_r,
                validation_data=ds_test_r, epochs=args.nepochs,
                steps_per_epoch=args.ntrain/global_batch_size, validation_steps=args.ntest/global_batch_size,
                verbose=True,
                callbacks=callbacks
            )
