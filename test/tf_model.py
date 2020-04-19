import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import setGPU

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas
import time

import keras
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.python.keras import backend as K

elem_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

from tensorflow.keras.optimizers import Optimizer
import tensorflow_addons as tfa

def compute_weights(ys):
    ws = []
    uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
    for i in range(len(Xs)):
        w = np.ones(len(ys[i]), dtype=np.float32)
        for uv, uc in zip(uniq_vals, uniq_counts):
            w[ys[i][:, 0]==uv] = len(ys[i])/uc
        ws += [w]
    return ws

def load_data(nfiles):
    Xs = []
    ys = []
    ys_cand = []
    filelist = sorted(glob.glob("/storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi/raw/*.pkl"))
    print("Found {} input .pkl files".format(len(filelist)))
    print("loading data from {} files".format(nfiles)) 
    for ifi, fi in enumerate(filelist[:nfiles]):
        print("loading {} {}/{}".format(fi, ifi, len(filelist[:nfiles]))) 
        data = pickle.load(open(fi, "rb"), encoding='iso-8859-1')
        for event in data:
            Xelem = event["Xelem"]
            ygen = event["ygen"]
            ycand = event["ycand"]

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

            Xs += [Xelem]
            ys += [ygen]
            ys_cand += [ycand]

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
        self.a = tf.Variable(0.1, trainable=True)
        self.b = tf.Variable(-5.0, trainable=True)

    def call(self, inputs):
        
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        D =  dist(inputs[:, :3], inputs[:, :3])
        
        #closer nodes have higher weight, could also consider exp(-D) or such here
        D = tf.math.divide_no_nan(1.0, D)
        
        #turn edges on or off based on activation with an arbitrary shift parameter
        D = self.a*tf.keras.activations.sigmoid(D + self.b)
        
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
        #self.layer_input1_d = tf.keras.layers.Dropout(0.2)
        self.layer_input2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input2")
        #self.layer_input2_d = tf.keras.layers.Dropout(0.2)
        self.layer_input3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="input3")
        #self.layer_input3_d = tf.keras.layers.Dropout(0.2)
        
        self.layer_dist = Distance(name="distance")
        self.layer_conv = GraphConv(hidden_dim, activation=activation, name="conv")
        
        self.layer_id1 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id1")
        #self.layer_id1_d = tf.keras.layers.Dropout(0.2)
        self.layer_id2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id2")
        #self.layer_id2_d = tf.keras.layers.Dropout(0.2)
        self.layer_id3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="id3")
        #self.layer_id3_d = tf.keras.layers.Dropout(0.2)
        self.layer_id = tf.keras.layers.Dense(len(class_labels), activation="linear", name="out_id")
        
        self.layer_momentum1 = tf.keras.layers.Dense(hidden_dim+len(class_labels), activation=activation, name="momentum1")
        #self.layer_momentum1_d = tf.keras.layers.Dropout(0.2)
        self.layer_momentum2 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum2")
        #self.layer_momentum2_d = tf.keras.layers.Dropout(0.2)
        self.layer_momentum3 = tf.keras.layers.Dense(hidden_dim, activation=activation, name="momentum3")
        #self.layer_momentum3_d = tf.keras.layers.Dropout(0.2)
        self.layer_momentum = tf.keras.layers.Dense(3, activation="tanh", name="out_momentum")
        
    def call(self, inputs, training=False):
        x = self.enc(inputs)
        x = self.layer_input1(x)
        #x = self.layer_input1_d(x, training)
        x = self.layer_input2(x)
        #x = self.layer_input2_d(x, training)
        x = self.layer_input3(x)
        #x = self.layer_input3_d(x, training)
        
        dm = self.layer_dist(x)
        x = self.layer_conv(x, dm)
        
        a = self.layer_id1(x)
        #a = self.layer_id1_d(a, training)
        a = self.layer_id2(a)
        #a = self.layer_id2_d(a, training)
        a = self.layer_id3(a)
        #a = self.layer_id3_d(a, training)
        out_id_logits = self.layer_id(a)
      
        x = tf.concat([x, tf.keras.activations.selu(out_id_logits)], axis=-1)
        b = self.layer_momentum1(x)
        #b = self.layer_momentum1_d(b, training)
        b = self.layer_momentum2(b)
        #b = self.layer_momentum2_d(b, training)
        b = self.layer_momentum3(b)
        #b = self.layer_momentum3_d(b, training)
        pred_corr = self.layer_momentum(b)

        #add predicted momentum correction to original momentum components (2,3,4) = (eta, phi, E) 
        out_id = tf.argmax(out_id_logits, axis=-1)
        #msk_good = tf.cast(out_id != 0, tf.float32)

        out_momentum_eta = inputs[:, 2] + inputs[:, 2]*pred_corr[:, 0]
        new_phi = inputs[:, 3] + inputs[:, 3]*pred_corr[:, 1]
        out_momentum_phi = tf.math.atan2(tf.math.sin(new_phi), tf.math.cos(new_phi))
        out_momentum_E = inputs[:, 4] + inputs[:, 4]*pred_corr[:, 2]
        #out_momentum_eta = (inputs[:, 2] + pred_corr[:, 0])
        #new_phi = (inputs[:, 3] + pred_corr[:, 1])
        #out_momentum_phi = tf.math.atan2(tf.math.sin(new_phi), tf.math.cos(new_phi))
        #out_momentum_phi = new_phi
        out_momentum_E = (inputs[:, 4] + pred_corr[:, 2])

        out_momentum = tf.stack([out_momentum_eta, out_momentum_phi, out_momentum_E], axis=-1)
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

#@tf.function
def my_loss(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=len(class_labels)) 
    l1 = 1000.0*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_onehot)
  
    msk_good = (true_id[:, 0] == pred_id)
    #nsamp = tf.cast(tf.size(y_pred), tf.float32)

    l2_0 = 10*tf.math.pow(true_momentum[:, 0] - pred_momentum[:, 0], 2)
    l2_1 = 100*tf.math.pow(tf.math.floormod(true_momentum[:, 1] - pred_momentum[:, 1] + np.pi, 2*np.pi) - np.pi, 2)
    l2_2 = tf.math.pow(true_momentum[:, 2] - pred_momentum[:, 2], 2)
    #tf.print("l1", tf.reduce_mean(l1))
    #tf.print("l2_0", tf.reduce_mean(l2_0))
    #tf.print("l2_1", tf.reduce_mean(l2_1))
    #tf.print("l2_2", tf.reduce_mean(l2_2))
    #loss_mult = tf.math.minimum(epoch/tf.constant(20.0), tf.constant(1.0))

    l2 = (l2_0 + l2_1 + l2_2)
    l2 = tf.multiply(tf.cast(msk_good, tf.float32), l2)
    l = l1 + l2
 
    return l

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
def eta_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    msk = ((true_id[:, 0]==1) & (pred_id==1)) | ((true_id[:, 0]==10) & (pred_id==10))
    return tf.math.reduce_std(tf.math.divide(pred_momentum[:, 0], true_momentum[:, 0])[msk]) 

#@tf.function
def phi_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    msk = ((true_id[:, 0]==1) & (pred_id==1)) | ((true_id[:, 0]==10) & (pred_id==10))
    return tf.math.reduce_std(tf.math.divide(pred_momentum[:, 1], true_momentum[:, 1])[msk]) 

#@tf.function(experimental_relax_shapes=True)
def energy_resolution(y_true, y_pred):
    pred_id_onehot, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_momentum = separate_truth(y_true)

    msk = ((true_id[:, 0]==1) & (pred_id==1)) | ((true_id[:, 0]==10) & (pred_id==10))
    return tf.math.reduce_std(tf.math.divide(pred_momentum[:, 2], true_momentum[:, 2])[msk]) 

def get_unique_run():
    previous_runs = os.listdir('experiments')
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    return run_number

def loss(model, inputs, targets, weights, epoch, training):
    pred = model(inputs, training=training)
    l = weights*my_loss(targets, pred, epoch)
    return tf.reduce_mean(l)

def grad(model, inputs, targets, weights, epoch):
    epoch_tf = tf.cast(tf.constant(epoch), tf.float32)
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, weights, epoch_tf, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

def custom_training_loop(model, ds_training, ds_testing, num_epochs, callbacks=[]):

    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in model.trainable_variables]

    metrics = {
        "cls_accuracy": cls_accuracy,
        "energy_resolution": energy_resolution,
        "phi_resolution": phi_resolution,
        "eta_resolution": eta_resolution,
    }

    max_dim_X = 10000
 
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
        for Xelem, ygen, ws in ds_training:
            ibatch += 1
            loss_value, grads = grad(model, Xelem, ygen, ws, iepoch)
            for igrad, gv in enumerate(grads):
                accum_vars[igrad].assign_add(gv)
            
            logs["loss"] += [loss_value.numpy()]
            if ibatch == 5:
                opt.apply_gradients([(accum_vars[igrad] / 5.0, model.trainable_variables[igrad]) for igrad in range(len(accum_vars))])
                ibatch = 0
                for igrad in range(len(accum_vars)):
                    accum_vars[igrad].assign(tf.zeros_like(accum_vars[igrad]))
    
            ypred = model(Xelem)
            for metric, func in metrics.items():        
                logs[metric] += [func(ygen, ypred).numpy()]
            nsamp += 1 

        for Xelem, ygen, ws in ds_testing:
            ypred = model(Xelem)
            for metric, func in metrics.items():        
                logs["val_" + metric] += [func(ygen, ypred).numpy()]
       
            loss_value = loss(model, Xelem, ygen, ws, iepoch, False)
            logs["val_loss"] += [loss_value.numpy()]
            nsamp += 1 

        for k in logs.keys():
            logs[k] = np.mean(logs[k])

        #for k, v in logs.items():
        #    tf.summary.scalar('epoch_{}'.format(k), v, step=iepoch)

        t1 = time.time()
        s = ""
        for metric in sorted(metrics.keys()):
            s += "{}={:.2f}/{:.2f} ".format(metric, logs[metric], logs["val_" + metric]) 
        print("epoch={epoch}/{maxepoch} t={t:.2f}s dt={dt:.0f}ms loss={loss_train:.2f}/{loss_test:.2f} {metrics}".format(
            epoch=iepoch, maxepoch=num_epochs, t=(t1-t0),
            dt=1000.0*(t1-t0)/nsamp, loss_train=logs["loss"], loss_test=logs["val_loss"], metrics=s)
        )
   
        for callback in callbacks:
            callback.on_epoch_end(iepoch, logs)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=80, help="number of training events")
    parser.add_argument("--ntest", type=int, default=20, help="number of testing events")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")
    #parser.add_argument("--nplot", type=int, default=10, help="make plots every iterations")
    parser.add_argument("--nhidden", type=int, default=512, help="hidden dimension")
    #parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    #parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    #parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    #parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    #parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #tf.config.experimental_run_functions_eagerly(True)

    args = parse_args()

    Xs, ys, ys_cand = load_data(args.ntrain + args.ntest)

    model = PFNet(hidden_dim=args.nhidden)
    model(Xs[0])

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    outdir = 'experiments/run_{:02}'.format(get_unique_run())
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=outdir, histogram_freq=10, write_graph=True, write_images=False,
        update_freq='epoch'
    )
    tb.set_model(model)

    model.compile(optimizer=opt, loss=my_loss, metrics=[cls_accuracy, energy_resolution, eta_resolution, phi_resolution])

    if args.target == "gen":
        ws = compute_weights(ys)
        ntrain, ntest, ds_training, ds_testing = split_test_train(Xs, ys, ws, args.nepochs, args.ntrain)
    elif args.target == "cand":
        ws = compute_weights(ys_cand)
        ntrain, ntest, ds_training, ds_testing = split_test_train(Xs, ys_cand, ws, args.nepochs, args.ntrain)

    #custom_training_loop(model, ds_training, ds_testing, nepochs, callbacks=[tb])
    
    ret = model.fit(ds_training,
        validation_data=ds_testing, epochs=args.nepochs,
        steps_per_epoch=args.ntrain, validation_steps=args.ntest, batch_size=None, workers=None, verbose=True, callbacks=[tb]
    )

#    tf.keras.models.save_model(model, "model.tf", save_format="tf")
#    model = tf.keras.models.load_model("model.tf", compile=False)
#    ntrain = 0

    dfs = []
    for i in range(args.ntrain, len(Xs)):
        pred = model(Xs[i])
        pred_id_onehot, pred_momentum = separate_prediction(pred)
        pred_id = np.argmax(pred_id_onehot, axis=-1)
        true_id, true_momentum = separate_truth(ys[i])
        true_id = true_id.numpy()
        
        df = pandas.DataFrame()
        df["pred_pid"] = np.array([int(class_labels[p]) for p in pred_id])
        df["pred_eta"] = np.array(pred_momentum[:, 0].numpy(), dtype=np.float64)
        df["pred_phi"] = np.array(pred_momentum[:, 1].numpy(), dtype=np.float64)
        df["pred_e"] = np.array(pred_momentum[:, 2].numpy(), dtype=np.float64)

        df["gen_pid"] = np.array([int(class_labels[p]) for p in true_id[:, 0]])
        df["gen_eta"] = np.array(true_momentum[:, 0], dtype=np.float64)
        df["gen_phi"] = np.array(true_momentum[:, 1], dtype=np.float64)
        df["gen_e"] = np.array(true_momentum[:, 2], dtype=np.float64)
        
        df["cand_pid"] = np.array([int(class_labels[int(p)]) for p in ys_cand[i][:, 0]])
        df["cand_eta"] = np.array(ys_cand[i][:, 1], dtype=np.float64)
        df["cand_phi"] = np.array(ys_cand[i][:, 2], dtype=np.float64)
        df["cand_e"] = np.array(ys_cand[i][:, 3], dtype=np.float64)

        df["iev"] = i
        dfs += [df]

    df = pandas.concat(dfs, ignore_index=True)
    df.to_pickle(outdir + "/df.pkl.bz2")
    #Print some stats for each target particle type 
    for pid in [211, -211, 130, 22, -11, 11, 13, -13, 1, 2]:
        if args.target == "gen":
            msk_gen = df["gen_pid"] == pid
        elif args.target == "cand":
            msk_gen = df["cand_pid"] == pid
        msk_pred = df["pred_pid"] == pid

        npred = int(np.sum(msk_pred))
        ngen = int(np.sum(msk_gen))
        tpr = np.sum(msk_gen & msk_pred) / npred
        fpr = np.sum(~msk_gen & msk_pred) / npred
        eff = np.sum(msk_gen & msk_pred) / ngen

        mu = 0.0
        sigma = 0.0
        if np.sum(msk_pred) > 0:

            if args.target == "gen":
                energies = df[msk_gen & msk_pred][["gen_e", "pred_e"]].values
            elif args.target == "cand":
                energies = df[msk_gen & msk_pred][["cand_e", "pred_e"]].values

            r = energies[:, 1]/energies[:, 0]
            mu, sigma = np.mean(r), np.std(r)
        print("pid={pid} Ngen={ngen} Npred={npred} eff={eff:.4f} tpr={tpr:.4f} fpr={fpr:.4f} E_m={E_m:.4f} E_s={E_s:.4f}".format(
            pid=pid, ngen=ngen, npred=npred, eff=eff, tpr=tpr, fpr=fpr, E_m=mu, E_s=sigma
        ))

    #ensure model is compiled
    model.predict(Xs[0])
    tf.keras.models.save_model(model, outdir + "/model.tf", save_format="tf")
