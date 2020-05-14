import os
import sys
import random
os.environ["KERAS_BACKEND"] = "tensorflow"

import glob

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
except:
    print("Coult not import setGPU, please make sure you configure CUDA_VISIBLE_DEVICES manually")
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

elem_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
class_labels = [0, 1, 2, 11, 13, 22, 130, 211]

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
    "classbalanced": compute_weights_inverse,
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

        Xelem = append_fields(Xelem, "typ_idx", np.array([elem_labels.index(int(i)) for i in Xelem["typ"]], dtype=np.float32))
        ygen = append_fields(ygen, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32))
        ycand = append_fields(ycand, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32))
    
        #now preprocess the PFElements event up to num_clusters to simplify batched training
        #this means that the network sees only a subset of the elements in the event in training during each weight update
        clusters = sklearn.cluster.KMeans(n_clusters=num_clusters).fit_predict(np.stack([Xelem["eta"], Xelem["phi"]], axis=-1))

        print("clustered {} inputs for preprocessing".format(len(Xelem)))
        #save each cluster separately
        for cl in np.unique(clusters):
            msk = clusters==cl
            Xelem_flat = np.stack([Xelem[msk][k].view(np.float32).data for k in [
                'typ_idx',
                'pt', 'eta', 'phi', 'e',
                'layer', 'depth', 'charge', 'trajpoint',
                'eta_ecal', 'phi_ecal', 'eta_hcal', 'phi_hcal',
                'muon_dt_hits', 'muon_csc_hits']], axis=-1
            )
            ygen_flat = np.stack([ygen[msk][k].view(np.float32).data for k in [
                'typ_idx',
                'eta', 'phi', 'e', 'charge',
                ]], axis=-1
            )
            ycand_flat = np.stack([ycand[msk][k].view(np.float32).data for k in [
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

            Xs += [Xelem_flat]
            ys += [ygen_flat]
            ys_cand += [ycand_flat]
    
    print("created {} blocks, max size {}".format(len(Xs), max([len(X) for X in Xs])))
    return Xs, ys, ys_cand

def dist(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [tf.shape(na)[0], -1, 1])
    nb = tf.reshape(nb, [tf.shape(na)[0], 1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, transpose_a=False, transpose_b=True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D

class InputEncoding(tf.keras.layers.Layer):
    def __init__(self, num_input_classes):
        super(InputEncoding, self).__init__()
        self.num_input_classes = num_input_classes
        
    def call(self, X):
        #X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes)

        #X[:, :, 1:] - all the other non-categorical features
        Xprop = X[:, :, 1:]
        return tf.concat([Xid, Xprop], axis=-1)

#Given a list of [Nbatch, Nelem, Nfeat] input nodes, computes the dense [Nbatch, Nelem, Nelem] adjacency matrices
class Distance(tf.keras.layers.Layer):

    def __init__(self, dist_shape, *args, **kwargs):
        super(Distance, self).__init__(*args, **kwargs)

    def call(self, inputs1, inputs2):
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        #inputs1, inputs2: [Nbatch, Nelem, distance_dim] embedded coordinates used for element-to-element distance calculation
        D =  dist(inputs1, inputs2)
      
        #adjacency between two elements should be high if the distance is small.
        #this is equivalent to radial basis functions. 
        #self-loops adj_{i,i}=1 are included, as D_{i,i}=0 by construction
        adj = tf.math.exp(-1.0*D)

        #optionally set the adjacency matrix to 0 for low values in order to make the matrix sparse.
        #need to test if this improves the result.
        #adj = tf.keras.activations.relu(adj, threshold=0.01)

        return adj
    
class GraphConv(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(GraphConv, self).__init__(*args, **kwargs)
        self.k = 4
    
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]

        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)

        support = (tf.linalg.matmul(inputs, W))
     
        #k-th power of the normalized adjacency matrix is nearly equivalent to k consecutive GCN layers
        adj_k = tf.pow(adj, self.k)
        out = tf.linalg.matmul(adj_k, support)

        return self.activation(out + b)

class PFNet(tf.keras.Model):
    
    def __init__(self, activation=tf.nn.selu, hidden_dim=256, distance_dim=32, num_conv=1):
        super(PFNet, self).__init__()
        self.activation = activation

        self.enc = InputEncoding(len(elem_labels))

        #self.layer_conv0 = GraphConv(hidden_dim, activation=activation, name="initial_conv")

        self.layer_distcoords1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="distcoords11")
        self.layer_distcoords21 = tf.keras.layers.Dense(distance_dim, activation="linear", name="distcoords21")
        self.layer_distcoords22 = tf.keras.layers.Dense(distance_dim, activation="linear", name="distcoords22")

        self.layer_input1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input1")
        self.layer_input2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2")
        self.layer_input3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input3")
        
        self.layer_dist = Distance(distance_dim, name="distance")

        self.num_conv = num_conv
        self.convlayers = []
        for iconv in range(self.num_conv):
            self.convlayers += [GraphConv(hidden_dim, activation=activation, name="conv{}".format(iconv))]

        self.layer_id1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id1")
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        self.layer_charge = tf.keras.layers.Dense(2, activation="linear", name="out_charge")
        
        self.layer_momentum1 = tf.keras.layers.Dense(hidden_dim+len(class_labels), activation=activation, name="momentum1")
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="linear", name="out_momentum")
        self.inp_X = tf.keras.Input(batch_size=None, shape=(15, ), name="X")
        #self.inp_dm = tf.keras.Input(batch_size=None, shape=(None, ), sparse=True, name="dm")
        #self._set_inputs(self.inp_X)
 
    def predict_distancematrix(self, inputs):

        enc = self.activation(self.enc(inputs))

        x = self.layer_distcoords1(enc)
        distcoords1 = self.layer_distcoords21(x)
        distcoords2 = self.layer_distcoords22(x)

        dm = self.layer_dist(distcoords1, distcoords2)
        return enc, dm

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, 15], dtype=tf.float32)])
    def call(self, inputs):
        X = inputs
        #tf.print(X.shape)
        enc, dm = self.predict_distancematrix(X)

        x = self.layer_input1(enc)
        x = self.layer_input2(x)
        x = self.layer_input3(x)
        
        for conv in self.convlayers:
            x = conv(x, dm)
 
        a = self.layer_id1(tf.concat(x, axis=-1))
        a = self.layer_id2(a)
        a = self.layer_id3(a)
        out_id_logits = self.layer_id(a)
        out_charge = self.layer_charge(a)
        
        x = tf.concat([x, self.activation(out_id_logits)], axis=-1)
        b = self.layer_momentum1(x)
        b = self.layer_momentum2(b)
        b = self.layer_momentum3(b)
        pred_corr = self.layer_momentum(b)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        #msk_good = tf.cast(out_id != 0, tf.float32)

        out_momentum_eta = X[:, :, 2] + pred_corr[:, :, 0]
        new_phi = X[:, :, 3] + pred_corr[:, :, 1]
        out_momentum_phi = new_phi
        out_momentum_E = X[:, :, 4] + pred_corr[:, :, 2]

        out_momentum = tf.stack([
            #tf.multiply(out_momentum_eta, msk_good),
            #tf.multiply(out_momentum_phi, msk_good),
            #tf.multiply(out_momentum_E, msk_good)
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

def my_loss_cls(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))[:, 0, :]
    #predict the particle class labels
    l1 = 1e3*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
    return 1e3*l1

#@tf.function
def my_loss_full(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    #tf.print(pred_id_onehot)
    l1 = 1e4 * tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
  
    #msk_good = (true_id[:, 0] == pred_id)
    #nsamp = tf.cast(tf.size(y_pred), tf.float32)

    l2_0 = mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mse_unreduced(tf.math.floormod(true_momentum[:, :, 1] - pred_momentum[:, :, 1] + np.pi, 2*np.pi) - np.pi, 0.0)
    l2_2 = mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])/10.0

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
    l = l1 + l2 + l3
    return 1e3*l

#@tf.function
def cls_accuracy(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)
    msk = true_id[:, :, 0]!=0
    return tf.keras.metrics.categorical_accuracy(true_id[msk][:, 0], pred_id_onehot[msk])

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

def loss(model, inputs, targets, weights, epoch, training, lossfn):
    pred = model(inputs)
    l = weights*lossfn(targets, pred)
    return tf.reduce_mean(l)

def grad(model, inputs, targets, weights, epoch, lossfn):
    epoch_tf = tf.cast(tf.constant(epoch), tf.float32)
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, weights, epoch_tf, True, lossfn)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

def custom_training_loop(loss_fn, opt, model, ds_training, ds_testing, num_training, num_testing, num_epochs, callbacks=[], num_accumulate=2):
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in model.trainable_variables]

    metrics = {
        "num_pred": num_pred, 
        "cls_accuracy": cls_accuracy,
        "energy_resolution": energy_resolution,
        "phi_resolution": phi_resolution,
        "eta_resolution": eta_resolution,
    }

    for iepoch in range(num_epochs):
        
        ibatch = 0
        t0 = time.time()
        logs = {
            "loss": [],
            "val_loss": []
        }
        for metric in metrics.keys():        
            logs[metric] = []
            logs["val_" + metric] = []

        nsamp = 0 
        looper = tqdm(ds_training, total=num_training, ascii=" =")
        for Xelem, ygen, ws in looper:
            ibatch += 1
            loss_value, grads = grad(model, Xelem, ygen, ws, iepoch, loss_fn)
            for igrad, gv in enumerate(grads):
                accum_vars[igrad].assign_add(gv)
            
            logs["loss"] += [loss_value.numpy()]
            if ibatch == num_accumulate:
                opt.apply_gradients([(accum_vars[igrad] / num_accumulate, model.trainable_variables[igrad]) for igrad in range(len(accum_vars))])
                ibatch = 0
                for igrad in range(len(accum_vars)):
                    accum_vars[igrad].assign(tf.zeros_like(accum_vars[igrad]))
    
            ypred, _ = model(Xelem)
            for metric, func in metrics.items():        
                logs[metric] += [func(ygen, ypred).numpy()]
            looper.set_postfix(
                nelem="{:.0f}".format(np.mean(logs["num_pred"])),
                acc="{:.2f}".format(np.mean(logs["cls_accuracy"])),
                e="{:.2f}".format(np.mean(logs["energy_resolution"]))
            )

            nsamp += 1 
        t1 = time.time()

        looper = tqdm(ds_testing, total=num_testing, ascii=" =")
        for Xelem, ygen, ws in looper:
            ypred, _ = model(Xelem)
            for metric, func in metrics.items():
                logs["val_" + metric] += [func(ygen, ypred).numpy()]
       
            loss_value = loss(model, Xelem, ygen, ws, iepoch, False, loss_fn)
            logs["val_loss"] += [loss_value.numpy()]
            looper.set_postfix(
                nelem="{:.0f}".format(np.mean(logs["val_num_pred"])),
                acc="{:.2f}".format(np.mean(logs["val_cls_accuracy"])),
                e="{:.2f}".format(np.mean(logs["val_energy_resolution"]))
            )

        for k in logs.keys():
            logs[k] = np.mean(logs[k])

        #for k, v in logs.items():
        #    tf.summary.scalar('epoch_{}'.format(k), v, step=iepoch)
        
        dt = 0
        if nsamp > 0:
            dt = 1000.0*(t1-t0)/nsamp
        s = ""
        for metric in sorted(metrics.keys()):
            s += "{}={:.2f}/{:.2f} ".format(metric, logs[metric], logs["val_" + metric]) 
        print("epoch={epoch}/{maxepoch} t={t:.2f}s dt={dt:.0f}ms loss={loss_train:.2f}/{loss_test:.2f} {metrics}".format(
            epoch=iepoch, maxepoch=num_epochs, t=(t1-t0),
            dt=dt, loss_train=logs["loss"], loss_test=logs["val_loss"], metrics=s)
        )
   
        for callback in callbacks:
            callback.on_epoch_end(iepoch, logs)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=80, help="number of training events")
    parser.add_argument("--ntest", type=int, default=20, help="number of testing events")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--nhidden", type=int, default=256, help="hidden dimension")
    parser.add_argument("--num-conv", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--distance-dim", type=int, default=256, help="distance dimension")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    parser.add_argument("--weights", type=str, choices=["uniform", "inverse", "classbalanced"], help="Sample weighting scheme to use", default="inverse")
    parser.add_argument("--name", type=str, default=None, help="where to store the output")
    parser.add_argument("--load", type=str, default=None, help="model to load")
    #parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--custom-training-loop", action="store_true", help="Run a custom training loop")
    parser.add_argument("--train-cls", action="store_true", help="Train only the classification part")
    #parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    #parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet")
    args = parser.parse_args()
    return args

def prepare_df(epoch, model, data, outdir):
    tf.print("\nprepare_df")

    dfs = []
    for iev, d in enumerate(data):
        if iev%50==0:
            tf.print(".", end="")
        X, y, w = d
        pred = model(X).numpy()
        pred_id_onehot, pred_charge, pred_momentum = separate_prediction(pred)
        pred_id = np.argmax(pred_id_onehot, axis=-1).flatten()
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

        df["target_pid"] = np.array([int(class_labels[p]) for p in true_id])
        df["target_eta"] = np.array(true_momentum[:, 0], dtype=np.float64)
        df["target_phi"] = np.array(true_momentum[:, 1], dtype=np.float64)
        df["target_e"] = np.array(true_momentum[:, 2], dtype=np.float64)

        df["iev"] = iev
        dfs += [df]
    assert(len(dfs) > 0)
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

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, ntest, file_writer_cm):
        self.dataset = dataset
        self.ntest = ntest
        self.file_writer_cm = file_writer_cm

    def on_epoch_end(self, epoch, logs):
        if True or (epoch%10==0 and epoch>0):
            true_ids = []
            pred_ids = []
            for iev, data in enumerate(self.dataset):
                if iev>=self.ntest:
                    break
                X, y, w = data
                pred = self.model(X).numpy()
                pred_id_onehot, pred_charge, pred_momentum = separate_prediction(pred)
                pred_id = np.argmax(pred_id_onehot, axis=-1)
                true_id, true_charge, true_momentum = separate_truth(y)
                true_id = true_id.numpy()
                pred_ids += [pred_id]
                true_ids += [true_id]
   
            true_ids = np.concatenate(true_ids) 
            pred_ids = np.concatenate(pred_ids)
 
            # Calculate the confusion matrix.
            cm = confusion_matrix(true_ids, pred_ids, labels=range(len(class_labels)))

            figure, _ = plot_confusion_matrix(cm, [int(x) for x in class_labels], cmap="Blues")
            cm_image = plot_to_image(figure)
            
            figure2 = plot_confusion_matrix(np.round(100.0*cm/np.sum(cm), 1), [int(x) for x in class_labels], cmap="Blues", normalize=False)
            cm_image2 = plot_to_image(figure2)
    
            # Log the confusion matrix as an image summary.
            with self.file_writer_cm.as_default():
              tf.summary.image("Confusion Matrix", cm_image, step=epoch)
              tf.summary.image("Confusion Matrix (unnormalized)", cm_image2, step=epoch)
              for pdgid in [211, 130, 13, 11, 22, 1, 2, 0]:
                  idx = class_labels.index(pdgid)
                  eff = cm[idx, idx] / np.sum(cm[idx, :])
                  fake = (np.sum(cm[:, idx]) - cm[idx, idx]) / np.sum(cm[:, idx])
                  tf.summary.scalar("eff_{}".format(pdgid), eff, step=epoch)
                  tf.summary.scalar("fake_{}".format(pdgid), fake, step=epoch)

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
    tfr_files = glob.glob(datapath + "/tfr2/{}/*.tfrecords".format(args.target))
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = tf.data.Dataset.from_tensor_slices(tfr_files).interleave(lambda x: tf.data.TFRecordDataset(x)).map(_parse_tfr_element)
    return dataset

if __name__ == "__main__":
    #tf.debugging.enable_check_numerics()
    tf.config.experimental_run_functions_eagerly(False)

    args = parse_args()

    #datapath = "/storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi"
    datapath = "data/TTbar_14TeV_TuneCUETP8M1_cfi"

    try:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    filelist = sorted(glob.glob(datapath + "/raw/*.pkl"))[:args.ntrain+args.ntest]

    from tf_data import _parse_tfr_element

    #dataset = load_dataset_gun()
    dataset = load_dataset_ttbar(datapath)
 
    ps = (tf.TensorShape([None, 15]), tf.TensorShape([None, 5]), tf.TensorShape([None, ]))
    batch_size = 50
    ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).shuffle(10000).padded_batch(batch_size, padded_shapes=ps)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).shuffle(10000).padded_batch(batch_size, padded_shapes=ps)

    #print("train")
    #summarize_dataset(ds_train)
    #print("test")
    #summarize_dataset(ds_test)
 
    if not args.custom_training_loop:
        ds_train_r = ds_train.repeat(args.nepochs)
        ds_test_r = ds_test.repeat(args.nepochs)

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model = PFNet(hidden_dim=args.nhidden, distance_dim=args.distance_dim, num_conv=args.num_conv)

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
    file_writer_cm = tf.summary.create_file_writer(outdir + '/cm')
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
    
    #confusion_cb = ConfusionMatrixCallback(ds_test.repeat(), args.ntest, file_writer_cm)
    #confusion_cb.set_model(model)
    #callbacks += [confusion_cb]

    loss_fn = my_loss_full
    if args.train_cls:
        loss_fn = my_loss_cls
        model.layer_momentum1.trainable = False
        model.layer_momentum2.trainable = False
        model.layer_momentum3.trainable = False
        model.layer_momentum.trainable = False

    if args.custom_training_loop:
        assert(num_gpus == 1)
        custom_training_loop(loss_fn, opt, model, ds_train, ds_test, args.ntrain, args.ntest, args.nepochs, callbacks=callbacks)
    else:
        with strategy.scope():
            model.compile(optimizer=opt, loss=loss_fn,
                metrics=[cls_accuracy, energy_resolution, eta_resolution, phi_resolution],
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
                    steps_per_epoch=args.ntrain/batch_size, validation_steps=args.ntest/batch_size,
                    verbose=True,
                    callbacks=callbacks
                )

    #prepare_df(args.nepochs, model, ds_test, outdir)

    #ensure model is compiled
    #model.predict((X, dm))
    #tf.keras.models.save_model(model, outdir + "/model.tf", save_format="tf")
