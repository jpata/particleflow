import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
#import setGPU

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas

import keras
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.python.keras import backend as K

elem_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

def compute_weights(ys):
    ws = []
    uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
    for i in range(len(Xs)):
        w = np.ones(len(ys[i]), dtype=np.float32)
        #for uv, uc in zip(uniq_vals, uniq_counts):
        #    w[ys[i][:, 0]==uv] = len(ys[i])/uc
        ws += [w]
    return ws

def load_data(nfiles):
    Xs = []
    ys = []
    ys_cand = []
    filelist = sorted(glob.glob("/storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi/raw/*.pkl"))
    print("Found {} input .pkl files".format(len(filelist)))
    print("loading data from {} files".format(nfiles)) 
    for fi in filelist[:nfiles]:
        print("loading {}".format(fi)) 
        data = pickle.load(open(fi, "rb"), encoding='iso-8859-1')
        for event in data:
            Xelem = event["Xelem"]
            ygen = event["ygen"]
            ycand = event["ycand"]
            Xelem[:, 0] = [int(elem_labels.index(i)) for i in Xelem[:, 0]]
            ygen[:, 0] = [int(class_labels.index(i)) for i in ygen[:, 0]]
            ycand[:, 0] = [int(class_labels.index(i)) for i in ycand[:, 0]]

            ygen = np.hstack([ygen[:, :1], ygen[:, 2:5]])
            ycand = np.hstack([ycand[:, :1], ycand[:, 2:5]])
            Xelem[np.isnan(Xelem)] = 0
            ygen[np.isnan(ygen)] = 0
            Xelem[np.abs(Xelem) > 1e4] = 0
            ygen[np.abs(ygen) > 1e4] = 0

            Xs += [Xelem.copy()]
            ys += [ygen.copy()]
            ys_cand += [ycand.copy()]

    return Xs, ys, ys_cand

def split_test_train(Xs, ys, ws, nepochs, ntrain):
    Xs_training = Xs[:ntrain]
    ys_training = ys[:ntrain]
    ws_training = ws[:ntrain]
    
    Xs_testing = Xs[ntrain:]
    ys_testing = ys[ntrain:]
    ws_testing = ws[ntrain:]

    ds_training = tf.data.Dataset.from_generator(
        lambda: [(yield x) for x in zip(Xs_training, ys_training, ws_training)], (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape((None, Xs_training[0].shape[1])),
            tf.TensorShape((None, ys_training[0].shape[1])),
            tf.TensorShape((None,)),
        ),
    ).repeat(nepochs)
    
    ds_testing = tf.data.Dataset.from_generator(
        lambda: [(yield x) for x in zip(Xs_testing, ys_testing, ws_testing)], (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape((None, Xs_training[0].shape[1])),
            tf.TensorShape((None, ys_training[0].shape[1])),
            tf.TensorShape((None,)),
        ),
    ).repeat(nepochs)
 
    return len(Xs_training), len(Xs_testing), ds_training, ds_testing


def dist(A,B):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
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

    def __init__(self, *args, **kwargs):
        super(Distance, self).__init__(*args, **kwargs)

    def call(self, inputs):
        
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        D =  dist(inputs[:, :2], inputs[:, :2])
        
        #closer nodes have higher weight, could also consider exp(-D) or such here
        D = tf.math.divide_no_nan(1.0, D)
        
        #turn edges on or off based on activation with an arbitrary shift parameter
        D = tf.keras.activations.sigmoid(D - 5.0)
        
        #keep only upper triangular matrix (unidirectional edges)
        D = tf.linalg.band_part(D, 0, -1)
        return D
    
class GraphConv(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(GraphConv, self).__init__(*args, **kwargs)
    
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]
        support = tf.matmul(inputs, W) + b
        out = tf.matmul(adj, support)
        return self.activation(out)

class PFNet(tf.keras.Model):
    
    def __init__(self, activation=tf.keras.activations.selu, hidden_dim=256):
        super(PFNet, self).__init__()
        self.inp = tf.keras.Input(shape=(None,))
        self.enc = InputEncoding(len(elem_labels))
        self.layer_input1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input1")
        self.layer_input2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2")
        self.layer_input3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input3")
        
        self.layer_dist = Distance(name="distance")
        self.layer_conv = GraphConv(hidden_dim, activation=activation, name="conv")
        
        self.layer_id1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id1")
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        
        self.layer_momentum1 = tf.keras.layers.Dense(hidden_dim+len(class_labels), activation=activation, name="momentum1")
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        self.layer_momentum = tf.keras.layers.Dense(3, activation="tanh", name="out_momentum")
        
    def call(self, inputs):
        x = self.enc(inputs)
        x = self.layer_input1(x)
        x = self.layer_input2(x)
        x = self.layer_input3(x)
        
        dm = self.layer_dist(x)
        x = self.layer_conv(x, dm)
        
        a = self.layer_id1(x)
        a = self.layer_id2(a)
        a = self.layer_id3(a)
        out_id_logits = self.layer_id(a)
      
        x = tf.concat([x, out_id_logits], axis=-1)
        b = self.layer_momentum1(x)
        b = self.layer_momentum2(b)
        b = self.layer_momentum3(b)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        msk_good = tf.cast(out_id != 0, tf.float32)
        pred_corr = self.layer_momentum(b)

        out_momentum_eta = (inputs[:, 2] + inputs[:, 2]*pred_corr[:, 0])*msk_good
        out_momentum_phi = (inputs[:, 3] + inputs[:, 3]*pred_corr[:, 1])*msk_good
        out_momentum_E = tf.keras.activations.relu(inputs[:, 4] + inputs[:, 4]*pred_corr[:, 2])*msk_good

        out_momentum = tf.stack([out_momentum_eta, out_momentum_phi, out_momentum_E], axis=-1)
        ret = tf.concat([out_id_logits, out_momentum], axis=-1)
        return ret

def separate_prediction(y_pred):
    pred_id_onehot = y_pred[:, :len(class_labels)]
    pred_momentum = y_pred[:, len(class_labels):]
    return pred_id_onehot, pred_momentum

def separate_truth(y_true):
    true_id = tf.cast(y_true[:, :1], tf.int32)
    true_momentum = y_true[:, 1:]
    return true_id, true_momentum

def my_loss(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)

    true_id, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels)) 
    l1 = 1000.0*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
  
    msk_good = (true_id[:, 0] == pred_id)

    l2 = tf.keras.losses.mse(true_momentum[:, 0], pred_momentum[:, 0])
    l2 += tf.keras.losses.mse(tf.math.floormod(true_momentum[:, 1] - pred_momentum[:, 1] + np.pi, 2*np.pi), 0.0)
    #l2 += tf.keras.losses.mse(tf.math.log(tf.abs(true_momentum[:, 2] + 0.00001), tf.math.log(tf.abs(pred_momentum[:, 2] + 0.00001)))
    l2 += tf.keras.losses.mse(true_momentum[:, 2], pred_momentum[:, 2])
    l2 = tf.multiply(l2, tf.cast(msk_good, tf.float32))
 
    return l1 + l2

def cls_accuracy(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    nmatch = tf.reduce_sum(tf.cast(tf.math.equal(true_id[:, 0], pred_id), tf.float32))
    nall = tf.cast(tf.size(pred_id), tf.float32)
    v = nmatch/nall
    return v

def energy_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    msk = ((true_id[:, 0]==1) & (pred_id==1)) | ((true_id[:, 0]==10) & (pred_id==10))
    #tf.print("true", true_momentum[msk][:10])
    #tf.print("pred", pred_momentum[msk][:10])
    #tf.print("pred", y_pred[msk][:10, len(class_labels):])
    #tf.print("true", y_true[msk][:10, 1:])
    return tf.math.reduce_std(tf.math.divide_no_nan(pred_momentum[msk][:, 0], true_momentum[msk][:, 0])) 

if __name__ == "__main__":

    #tf.config.experimental_run_functions_eagerly(True)

    nepochs = 50
    Xs, ys, ys_cand = load_data(100)
    ws = compute_weights(ys)

    model = PFNet(hidden_dim=256)
    opt = tf.keras.optimizers.Adam(lr=0.00001)
    model.compile(optimizer=opt, loss=my_loss, metrics = [cls_accuracy, energy_resolution])

    ntrain, ntest, ds_training, ds_testing = split_test_train(Xs, ys, ws, nepochs, 80)
    
    #validation_metrics = ValidationMetrics()
    #validation_metrics.dataset = ds_testing 
    ret = model.fit(ds_training,
        validation_data=ds_testing, epochs=nepochs,
        steps_per_epoch=ntrain, validation_steps=ntest, batch_size=None, workers=None, verbose=True
    )
    tf.keras.models.save_model(model, "model.tf", save_format="tf")
#    model = tf.keras.models.load_model("model.tf", compile=False)
#    ntrain = 0

    dfs = []
    for i in range(ntrain, len(Xs)):
        pred = model(Xs[i])
        pred_id_onehot, pred_momentum = separate_prediction(pred)
        pred_id = np.argmax(pred_id_onehot, axis=-1)
        true_id, true_momentum = separate_truth(ys[i])
        true_id = true_id.numpy()
        
        df = pandas.DataFrame()
        df["pred_pid"] = [int(class_labels[p]) for p in pred_id]
        df["pred_eta"] = pred_momentum[:, 0]
        df["pred_phi"] = pred_momentum[:, 1]
        df["pred_e"] = pred_momentum[:, 2]

        if i==ntrain:
            import pdb;pdb.set_trace()

        df["gen_pid"] = [int(class_labels[p]) for p in true_id[:, 0]]
        df["gen_eta"] = true_momentum[:, 0]
        df["gen_phi"] = true_momentum[:, 1]
        df["gen_e"] = true_momentum[:, 2]
        
        df["cand_pid"] = [int(class_labels[int(p)]) for p in ys_cand[i][:, 0]]
        df["cand_eta"] = ys_cand[i][:, 1]
        df["cand_phi"] = ys_cand[i][:, 2]
        df["cand_e"] = ys_cand[i][:, 2]

        df["iev"] = i
        dfs += [df]

    df = pandas.concat(dfs, ignore_index=True)
    #Print some stats for each target particle type 
    for pid in [211, -211, 130, 22, -11, 11, 13, -13, 1, 2]:
        msk_gen = df["gen_pid"] == pid
        msk_pred = df["pred_pid"] == pid

        npred = int(np.sum(msk_pred))
        ngen = int(np.sum(msk_gen))
        tpr = np.sum(msk_gen & msk_pred) / npred
        fpr = np.sum(~msk_gen & msk_pred) / npred
        eff = np.sum(msk_gen & msk_pred) / ngen

        mu = 0.0
        sigma = 0.0
        if np.sum(msk_pred) > 0:
            energies = df[msk_gen & msk_pred][["gen_e", "pred_e"]].values
            r = energies[:, 1]/energies[:, 0]
            mu, sigma = np.mean(r), np.std(r)
        print("pid={pid} Ngen={ngen} Npred={npred} eff={eff:.4f} tpr={tpr:.4f} fpr={fpr:.4f} E_m={E_m:.4f} E_s={E_s:.4f}".format(
            pid=pid, ngen=ngen, npred=npred, eff=eff, tpr=tpr, fpr=fpr, E_m=mu, E_s=sigma
        ))
    df.to_pickle("df.pkl.bz2")
    tf.keras.models.save_model(model, "model.tf", save_format="tf")
