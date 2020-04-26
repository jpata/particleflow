import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

import keras
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.python.keras import backend as K

elem_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

from tensorflow.keras.optimizers import Optimizer
import tensorflow_addons as tfa

def load_one_file(fn):
    Xs = []
    ys = []
    ys_cand = []

    data = pickle.load(open(fn, "rb"), encoding='iso-8859-1')
    for event in data:
        Xelem = event["Xelem"]
        ygen = event["ygen"]
        ycand = event["ycand"]

        #skip PS, GSF and BREM for now
        #elem_id = Xelem[:, 0]
        #msk = (elem_id != 2) & (elem_id != 3) & (elem_id != 7)
        #Xelem = Xelem[msk]
        #ygen = ygen[msk]
        #ycand = ycand[msk]

        Xelem[:, 0] = [int(elem_labels.index(i)) for i in Xelem[:, 0]]
        ygen[:, 0] = [int(class_labels.index(i)) for i in ygen[:, 0]]
        ycand[:, 0] = [int(class_labels.index(i)) for i in ycand[:, 0]]
        
        #inds_sort_energy = np.argsort(Xelem[:, 4])[::-1]
        #Xelem = Xelem[inds_sort_energy, :]
        #ygen = ygen[inds_sort_energy, :]
        #ycand = ycand[inds_sort_energy, :]
    
        #max_elem = 10000
        #Xelem = Xelem[:max_elem, :]
        #ygen = ygen[:max_elem, :]
        #ycand = ygen[:max_elem, :]

        ygen = np.hstack([ygen[:, :1], ygen[:, 2:5]])
        ycand = np.hstack([ycand[:, :1], ycand[:, 2:5]])
        Xelem[np.isnan(Xelem)] = 0
        ygen[np.isnan(ygen)] = 0
        Xelem[np.abs(Xelem) > 1e4] = 0
        ygen[np.abs(ygen) > 1e4] = 0

        #Xelem = np.pad(Xelem, ((0, max_elem - Xelem.shape[0]), (0,0)))
        #ygen = np.pad(ygen, ((0, max_elem - ygen.shape[0]), (0,0)))
        #ycand = np.pad(ycand, ((0, max_elem - ycand.shape[0]), (0,0)))
        #print(Xelem.shape, ygen.shape, ycand.shape)

        assert(Xelem.shape[0] == ygen.shape[0])
        assert(Xelem.shape[0] == ycand.shape[0])
        Xs += [Xelem]
        ys += [ygen]
        ys_cand += [ycand]

    return Xs[0], ys[0], ys_cand[0]

def load_data(filelist, target):
    Xs = []
    ys = []
    ws = []

    for ifi, fi in enumerate(filelist):
        print(fi)
        X, y, ycand = load_one_file(fi)
        Xs += [X]
        if target == "cand":
            ys += [ycand]
        elif target == "gen":
            ys += [y]
    
    uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
    for i in range(len(ys)):
        w = np.ones(len(ys[i]), dtype=np.float32)
        for uv, uc in zip(uniq_vals, uniq_counts):
            w[ys[i][:, 0]==uv] = 1.0/uc
        ids = ys[i][:, 0]

        #equalize total weights between (0 and !=0 PIDs) to correctly predict the number of particles
        w[ids==0] *= w[ids!=0].sum()/w[ids==0].sum()
        #w *= len(ys[i])

        ws += [w]

    return Xs, ys, ws

def dist(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, False, True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D

class InputEncoding(tf.keras.layers.Layer):
    def __init__(self, num_input_classes):
        super(InputEncoding, self).__init__()
        self.num_input_classes = num_input_classes
        
    def call(self, X):
        Xid = tf.one_hot(tf.cast(X[:, 0], tf.int32), self.num_input_classes)
        Xprop = X[:, 1:]
        return tf.concat([Xid, Xprop], axis=-1)

class Distance(tf.keras.layers.Layer):

    def __init__(self, distance_dim, *args, **kwargs):
        super(Distance, self).__init__(*args, **kwargs)
        #self.a = tf.Variable(0.1, trainable=True, name="pf_net/distance/a")
        #self.b = tf.Variable(-0.5, trainable=True, name="pf_net/distance/b")
        self.distance_dim = tf.constant(distance_dim, tf.int32)

    def call(self, inputs):
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        D =  dist(inputs[:, :self.distance_dim], inputs[:, :self.distance_dim])
        
        #closer nodes have higher weight, could also consider exp(-D) or such here
        #D = tf.math.abs(tf.math.divide_no_nan(1.0, tf.clip_by_value(D, 1e-6, 1e6)))
        D = tf.math.exp(-1.0*D)
        
        #turn edges on or off based on activation with an arbitrary shift parameter
        #D = tf.clip_by_value(tf.nn.relu(D + self.b), 0.0, 10.0)
        #D = tf.nn.sigmoid(self.a*D + self.b)
        #tf.print("D=", tf.reduce_sum(D))

        #rowsum = tf.reduce_sum(D, axis=0) 
        #colsum = tf.reduce_sum(D, axis=1)
        #deg = tf.expand_dims(tf.sqrt(rowsum+colsum), axis=-1)
        #D = D - deg

        #D = tf.nn.sigmoid(self.a*D + self.b)
        #tf.print(tf.reduce_sum(D)/tf.cast(tf.shape(D)[0]*tf.shape(D)[1], tf.float32))

        #keep only upper triangular matrix (unidirectional edges)
        #D = tf.linalg.band_part(D, 0, -1)

        #spD = tf.sparse.from_dense(D)

        return D
    
class GraphConv(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(GraphConv, self).__init__(*args, **kwargs)
    
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]

        in_degrees = tf.reduce_sum(adj, axis=-1) 
        norm = tf.expand_dims(tf.pow(in_degrees, -0.5), 1)

        support = (tf.linalg.matmul(inputs, W) + b)

        out = tf.linalg.matmul(adj, support*norm)*norm

        return self.activation(out)

class PFNet(tf.keras.Model):
    
    def __init__(self, activation=tf.nn.selu, hidden_dim=256, distance_dim=32):
        super(PFNet, self).__init__()
        self.activation = activation
        self.inp = tf.keras.Input(shape=(None,))
        self.enc = InputEncoding(len(elem_labels))

        self.layer_input1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input1")
        self.layer_input2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2")
        self.layer_input3 = tf.keras.layers.Dense(hidden_dim, activation="linear", name="input3")
        
        self.layer_dist = Distance(distance_dim, name="distance")

        self.layer_conv1 = GraphConv(hidden_dim, activation=activation, name="conv1")
        
        self.layer_id1 = tf.keras.layers.Dense(24 + hidden_dim, activation=activation, name="id1")
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        
        self.layer_momentum1 = tf.keras.layers.Dense(24 + hidden_dim+len(class_labels), activation=activation, name="momentum1")
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="linear", name="out_momentum")
        
    def call(self, inputs):
        enc = self.activation(self.enc(inputs))

        x = self.layer_input1(enc)
        x = self.layer_input2(x)
        x = self.layer_input3(x)
        
        dm = self.layer_dist(x)

        x = self.layer_conv1(self.activation(x), dm)
        
        a = self.layer_id1(tf.concat([enc, x], axis=-1))
        a = self.layer_id2(a)
        a = self.layer_id3(a)
        out_id_logits = self.layer_id(a)
        
        x = tf.concat([enc, x, tf.nn.selu(out_id_logits)], axis=-1)
        b = self.layer_momentum1(x)
        b = self.layer_momentum2(b)
        b = self.layer_momentum3(b)
        pred_corr = self.layer_momentum(b)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        msk_good = tf.cast(out_id != 0, tf.float32)

        #out_momentum_eta = inputs[:, 2] + inputs[:, 2]*pred_corr[:, 0]
        #new_phi = inputs[:, 3] + inputs[:, 3]*pred_corr[:, 1]
        #out_momentum_phi = tf.math.atan2(tf.math.sin(new_phi), tf.math.cos(new_phi))
        #out_momentum_E = inputs[:, 4] + inputs[:, 4]*pred_corr[:, 2]

        #tf.debugging.check_numerics(pred_corr, "pred_corr")
        out_momentum_eta = inputs[:, 2] + pred_corr[:, 0]
        new_phi = inputs[:, 3] + pred_corr[:, 1]
        out_momentum_phi = new_phi
        out_momentum_E = inputs[:, 4] + pred_corr[:, 2]
        
        #out_momentum = tf.stack([
        #    out_momentum_eta,
        #    out_momentum_phi,
        #    out_momentum_E,
        #], axis=-1)

        out_momentum = tf.stack([
            tf.multiply(out_momentum_eta, msk_good),
            tf.multiply(out_momentum_phi, msk_good),
            tf.multiply(out_momentum_E, msk_good)
        ], axis=-1)

        ret = tf.concat([out_id_logits, out_momentum], axis=-1)
        return ret

#@tf.function
def separate_prediction(y_pred):
    N = tf.constant(len(class_labels))
    pred_id_onehot = y_pred[:, :N]
    pred_momentum = y_pred[:, N:]
    return pred_id_onehot, pred_momentum

#@tf.function
def separate_truth(y_true):
    true_id = tf.cast(y_true[:, :1], tf.int32)
    true_momentum = y_true[:, 1:]
    return true_id, true_momentum

def mse_unreduced(true, pred):
    return tf.math.pow(true-pred,2)

def my_loss_cls(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    l1 = 1000.0 * tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
    l3 = 0.01*tf.math.pow(tf.reduce_sum(tf.cast(true_id[:, 0]!=0, tf.float32)) - tf.reduce_sum(tf.cast(pred_id != 0, tf.float32)), 2)
    return 1000.0*(l1 + l3)

#@tf.function
def my_loss_full(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels))
    #tf.print(pred_id_onehot)
    l1 = 1000.0 * tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
  
    msk_good = (true_id[:, 0] == pred_id)
    #nsamp = tf.cast(tf.size(y_pred), tf.float32)

    #tf.print(tf.reduce_mean(pred_momentum[:, 0]))
    l2_0 = mse_unreduced(true_momentum[:, 0], pred_momentum[:, 0])
    l2_1 = mse_unreduced(tf.math.floormod(true_momentum[:, 1] - pred_momentum[:, 1] + np.pi, 2*np.pi) - np.pi, 0.0)
    l2_2 = mse_unreduced(true_momentum[:, 2], pred_momentum[:, 2])

    l2 = (l2_0 + l2_1 + l2_2)
    #l2 = tf.multiply(tf.cast(msk_good, tf.float32), l2)

    l3 = 0.1*tf.math.pow(tf.reduce_sum(tf.cast(true_id[:, 0]!=0, tf.float32)) - tf.reduce_sum(tf.cast(pred_id != 0, tf.float32)), 2)

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
    return 1000.0*l

#@tf.function
def cls_accuracy(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    nmatch = tf.reduce_sum(tf.cast(tf.math.equal(true_id[:, 0], pred_id), tf.float32))
    nall = tf.cast(tf.size(pred_id), tf.float32)
    v = nmatch/nall
    return v

#@tf.function
def num_pred(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    ntrue = tf.reduce_sum(tf.cast(true_id[:, 0]!=0, tf.int32))
    npred = tf.reduce_sum(tf.cast(pred_id!=0, tf.int32))
    return tf.cast(ntrue - npred, tf.float32)

#@tf.function
def eta_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    return tf.reduce_mean(mse_unreduced(true_momentum[:, 0], pred_momentum[:, 0]))

#@tf.function
def phi_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    return tf.reduce_mean(mse_unreduced(tf.math.floormod(true_momentum[:, 1] - pred_momentum[:, 1] + np.pi, 2*np.pi) - np.pi, 0.0))

#@tf.function(experimental_relax_shapes=True)
def energy_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    return tf.reduce_mean(mse_unreduced(true_momentum[:, 2], pred_momentum[:, 2]))

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
    parser.add_argument("--nplot", type=int, default=5, help="make plots every iterations")
    parser.add_argument("--nhidden", type=int, default=512, help="hidden dimension")
    parser.add_argument("--distance-dim", type=int, default=32, help="distance dimension")
    #parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    #parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
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
        pred_id_onehot, pred_momentum = separate_prediction(pred)
        pred_id = np.argmax(pred_id_onehot, axis=-1)
        true_id, true_momentum = separate_truth(y)
        true_id = true_id.numpy()
       
        df = pandas.DataFrame()
        df["pred_pid"] = np.array([int(class_labels[p]) for p in pred_id])
        df["pred_eta"] = np.array(pred_momentum[:, 0], dtype=np.float64)
        df["pred_phi"] = np.array(pred_momentum[:, 1], dtype=np.float64)
        df["pred_e"] = np.array(pred_momentum[:, 2], dtype=np.float64)

        df["target_pid"] = np.array([int(class_labels[p]) for p in true_id[:, 0]])
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
 
if __name__ == "__main__":
    #tf.debugging.enable_check_numerics()
    #tf.config.experimental_run_functions_eagerly(True)

    args = parse_args()

    #datapath = "/storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi"
    datapath = "/storage/user/jpata/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy("gpu:0")

    filelist = sorted(glob.glob(datapath + "/raw/*.pkl"))[:args.ntrain+args.ntest]
    X, y, ycand = load_one_file(filelist[0])

    from tf_data import _parse_tfr_element, NUM_EVENTS_PER_TFR
    tfr_files = sorted(glob.glob(datapath + "/tfr/{}/chunk_*.tfrecords".format(args.target)))

    dataset = tf.data.TFRecordDataset(tfr_files)
    ds_train = dataset.take(args.ntrain).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
    if not args.custom_training_loop:
        ds_train_r = ds_train.repeat(args.nepochs)
        ds_test_r = ds_test.repeat(args.nepochs)

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model = PFNet(hidden_dim=args.nhidden, distance_dim=args.distance_dim)
        model(X)

    if args.name is None:
        args.name =  'run_{:02}'.format(get_unique_run())
    outdir = 'experiments/' + args.name
    if os.path.isdir(outdir):
        print("Output directory exists: {}".format(outdir), file=sys.stderr)
        sys.exit(1)

    print(outdir)
    callbacks = []
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=outdir, histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch'
    )
    tb.set_model(model)
    callbacks += [tb]

    #if args.nplot > 0:
    #    prepare_df_cb = DataFrameCallback(ds_test, outdir, freq=args.nplot)
    #    prepare_df_cb.set_model(model)
    #    callbacks += [prepare_df_cb]

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model)
    callbacks += [cp_callback]

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
            model.compile(optimizer=opt, loss=loss_fn, metrics=[num_pred, cls_accuracy, energy_resolution, eta_resolution, phi_resolution])
            if args.load:
                model.load_weights(args.load)
            if args.nepochs > 0:
                ret = model.fit(ds_train_r,
                    validation_data=ds_test_r, epochs=args.nepochs, steps_per_epoch=args.ntrain, validation_steps=args.ntest,
                    verbose=True, callbacks=callbacks
                )

    prepare_df(args.nepochs, model, ds_test, outdir)

    #ensure model is compiled
    model.predict(X)
    tf.keras.models.save_model(model, outdir + "/model.tf", save_format="tf")
