from .model import DummyNet, PFNetDense

import tensorflow as tf
import tensorflow_probability
import tensorflow_addons as tfa
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
import glob
import io
import os
import yaml
import uuid
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from argparse import Namespace
import time
import json
import random
import math
import platform
from tqdm import tqdm
from pathlib import Path
from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler
from tfmodel.callbacks import CustomTensorBoard
from tfmodel.utils import get_lr_schedule, make_weight_function, targets_multi_output


from tensorflow.keras.metrics import Recall, CategoricalAccuracy

def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap="Blues")
    plt.xlabel("Predicted PID")
    plt.ylabel("Target PID")
    plt.colorbar()
    plt.tight_layout()
    return fig

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    return image

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset_def, outpath, X, y, dataset_transform, num_output_classes, plot_freq=1):
        super(CustomCallback, self).__init__()
        self.X = X
        self.y = y
        self.plot_freq = plot_freq

        self.dataset_def = dataset_def

        #transform the prediction target from an array into a dictionary for easier access
        self.ytrue = dataset_transform(self.X, self.y, None)[1]
        self.ytrue_id = np.argmax(self.ytrue["cls"], axis=-1)

        self.outpath = outpath
        self.num_output_classes = num_output_classes

        #ch.had, n.had, HFEM, HFHAD, gamma, ele, mu
        self.color_map = {
            1: "black",
            2: "green",
            3: "red",
            4: "orange",
            5: "blue",
            6: "cyan",
            7: "purple",
            8: "gray",
            9: "gray",
            10: "gray",
            11: "gray"
        }

        self.reg_bins = {
            "pt": np.linspace(0, 50, 100),
            "eta": np.linspace(-5, 5, 100),
            "sin_phi": np.linspace(-1,1,100),
            "cos_phi": np.linspace(-1,1,100),
            "energy": np.linspace(0,1000,100),
        }

    def plot_cm(self, outpath, ypred_id, msk):

        ytrue_id_flat = self.ytrue_id[msk].astype(np.int64).flatten()
        ypred_id_flat = ypred_id[msk].flatten()

        cm = sklearn.metrics.confusion_matrix(
            ytrue_id_flat,
            ypred_id_flat, labels=list(range(self.num_output_classes)), normalize="true"
        )
        figure = plot_confusion_matrix(cm)

        acc = sklearn.metrics.accuracy_score(
            ytrue_id_flat,
            ypred_id_flat
        )
        balanced_acc = sklearn.metrics.balanced_accuracy_score(
            ytrue_id_flat,
            ypred_id_flat
        )
        plt.title("acc={:.3f} bacc={:.3f}".format(acc, balanced_acc))
        plt.savefig(str(outpath / "cm_normed.pdf"), bbox_inches="tight")
        plt.close("all")

    def plot_event_visualization(self, outpath, ypred, ypred_id, msk, ievent=0):

        X_eta, X_phi, X_energy = self.dataset_def.get_X_eta_phi_energy(self.X)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*5, 5))

        #Plot the input PFElements
        plt.axes(ax1)
        msk = self.X[ievent, :, 0] != 0
        eta = X_eta[ievent][msk]
        phi = X_phi[ievent][msk]
        energy = X_energy[ievent][msk]
        typ = self.X[ievent][msk][:, 0]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in typ], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        #Plot the predicted particles
        plt.axes(ax3)
        msk = ypred_id[ievent] != 0
        eta = ypred["eta"][ievent][msk]
        sphi = ypred["sin_phi"][ievent][msk]
        cphi = ypred["cos_phi"][ievent][msk]
        phi = np.arctan2(sphi, cphi)
        energy = ypred["energy"][ievent][msk]
        pdgid = ypred_id[ievent][msk]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in pdgid], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        #Plot the target particles
        plt.axes(ax2)
        
        msk = self.ytrue_id[ievent] != 0
        eta = self.ytrue["eta"][ievent][msk]
        sphi = self.ytrue["sin_phi"][ievent][msk]
        cphi = self.ytrue["cos_phi"][ievent][msk]
        phi = np.arctan2(sphi, cphi)
        energy = self.ytrue["energy"][ievent][msk]
        pdgid = self.ytrue_id[ievent][msk]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in pdgid], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        plt.savefig(str(outpath / "event_iev{}.png".format(ievent)), bbox_inches="tight")
        plt.close("all")

    def plot_reg_distribution(self, outpath, ypred, ypred_id, msk, icls, reg_variable):

        if icls==0:
            vals_pred = ypred[reg_variable][msk][ypred_id[msk]!=icls].flatten()
            vals_true = self.ytrue[reg_variable][msk][self.ytrue_id[msk]!=icls].flatten()
        else:
            vals_pred = ypred[reg_variable][msk][ypred_id[msk]==icls].flatten()
            vals_true = self.ytrue[reg_variable][msk][self.ytrue_id[msk]==icls].flatten()

        bins = self.reg_bins[reg_variable]
        plt.hist(vals_true, bins=bins, histtype="step", lw=2, label="true")
        plt.hist(vals_pred, bins=bins, histtype="step", lw=2, label="predicted")

        if reg_variable in ["pt", "energy"]:
            plt.yscale("log")
            plt.ylim(bottom=1e-2)

        plt.xlabel(reg_variable)
        plt.ylabel("Number of particles")
        plt.legend(loc="best")
        plt.title("Regression output, cls {}".format(icls))
        plt.savefig(str(outpath / "{}_cls{}.png".format(reg_variable, icls)), bbox_inches="tight")
        plt.close("all")

    def plot_corr(self, epoch, outpath, ypred, ypred_id, msk, icls, reg_variable, log=False):

        if icls==0:
            sel = (self.ytrue_id[msk]!=0) & (ypred_id[msk]!=0)
        else:
            sel = (ypred_id[msk]==icls) & (self.ytrue_id[msk]==icls)

        vals_pred = ypred[reg_variable][msk][sel].flatten()
        vals_true = self.ytrue[reg_variable][msk][sel].flatten()

        #FIXME: propagate from configuration
        if reg_variable == "energy" or reg_variable == "pt":
            delta = 1.0
        else:
            delta = 0.1
            
        loss = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
        loss_vals = loss(np.expand_dims(vals_true, -1), np.expand_dims(vals_pred, axis=-1)).numpy()

        #suffix for log-transformed variable
        s = ""
        if log:
            vals_pred = np.log(vals_pred)
            vals_true = np.log(vals_true)
            s = "_log"

        plt.scatter(vals_pred, vals_true, marker=".", alpha=0.8, s=loss_vals)
        if len(vals_true) > 0:
            minval = np.min(vals_true)
            maxval = np.max(vals_true)
            if not (math.isnan(minval) or math.isnan(maxval) or math.isinf(minval) or math.isinf(maxval)):
                plt.plot([minval, maxval], [minval, maxval], color="black", ls="--", lw=0.5)
                plt.xlim(minval, maxval)
                plt.ylim(minval, maxval)

        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title("{}, L={:.4f}".format(reg_variable, np.sum(loss_vals)))
        plt.savefig(str(outpath / "{}_cls{}_corr{}.png".format(reg_variable, icls, s)), bbox_inches="tight")
        plt.close("all")

        #Also plot the residuals, as we have the true and predicted values already available here
        plt.figure()
        residual = vals_true - vals_pred
        residual[np.isnan(residual)] = 0
        residual[np.isinf(residual)] = 0
        plt.hist(residual, bins=100)
        plt.xlabel("true - pred")
        plt.title("{} residual, m={:.4f} s={:.4f}".format(reg_variable, np.mean(residual), np.std(residual)))
        plt.savefig(str(outpath / "{}_residual{}.png".format(reg_variable, s)), bbox_inches="tight")
        plt.close("all")

        # FIXME: for some reason, these don't end up on the tensorboard
        # tf.summary.scalar('residual_{}{}_mean'.format(reg_variable, s), data=np.mean(residual), step=epoch)
        # tf.summary.scalar('residual_{}{}_std'.format(reg_variable, s), data=np.std(residual), step=epoch)

    def on_epoch_end(self, epoch, logs=None):

        if epoch%self.plot_freq!=0:
            return

        #save the training logs (losses) for this epoch
        with open("{}/history_{}.json".format(self.outpath, epoch), "w") as fi:
            json.dump(logs, fi)

        cp_dir = Path(self.outpath) / "epoch_{}".format(epoch)
        cp_dir.mkdir(parents=True, exist_ok=True)

        #run the model inference on the validation dataset
        ypred = self.model.predict(self.X, batch_size=1)

        #choose the class with the highest probability as the prediction
        #this is a shortcut, in actual inference, we may want to apply additional per-class thresholds        
        ypred_id = np.argmax(ypred["cls"], axis=-1)
       
        #exclude padded elements from the plotting
        msk = self.X[:, :, 0] != 0

        self.plot_cm(cp_dir, ypred_id, msk)
        for ievent in range(min(5, self.X.shape[0])):
            self.plot_event_visualization(cp_dir, ypred, ypred_id, msk, ievent=ievent)

        for icls in range(self.num_output_classes):
            cp_dir_cls = cp_dir / "cls_{}".format(icls)
            cp_dir_cls.mkdir(parents=True, exist_ok=True)
            for variable in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
                self.plot_reg_distribution(cp_dir_cls, ypred, ypred_id, msk, icls, variable)
                self.plot_corr(epoch, cp_dir_cls, ypred, ypred_id, msk, icls, variable)
            self.plot_corr(epoch, cp_dir_cls, ypred, ypred_id, msk, icls, "energy", log=True)
            self.plot_corr(epoch, cp_dir_cls, ypred, ypred_id, msk, icls, "pt", log=True)

        np.savez(str(cp_dir/"pred.npz"), X=self.X, ytrue=self.y, **ypred)

def prepare_callbacks(model, outdir, X_val, y_val, dataset_transform, num_output_classes, dataset_def, plot_freq=1):
    callbacks = []
    tb = CustomTensorBoard(
        log_dir=outdir + "/tensorboard_logs", histogram_freq=1, write_graph=False, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,90),
        profile_batch=0,
    )
    tb.set_model(model)
    callbacks += [tb]

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    cp_dir = Path(outdir) / "weights"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cp_dir / "weights-{epoch:02d}-{val_loss:.6f}.hdf5"),
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model)
    callbacks += [cp_callback]

    history_path = Path(outdir) / "history"
    history_path.mkdir(parents=True, exist_ok=True)
    history_path = str(history_path)
    cb = CustomCallback(dataset_def, history_path, X_val, y_val, dataset_transform, num_output_classes, plot_freq=plot_freq)
    cb.set_model(model)

    callbacks += [cb]

    return callbacks

def get_rundir(base='experiments'):
    if not os.path.exists(base):
        os.makedirs(base)

    previous_runs = os.listdir(base)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    logdir = 'run_%02d' % run_number
    return '{}/{}'.format(base, logdir)


def scale_outputs(X,y,w):
    ynew = y-out_m
    ynew = ynew/out_s
    return X, ynew, w


def make_model(config, dtype):
    model = config['parameters']['model']

    if model == 'dense':
        return make_dense(config, dtype)
    elif model == 'gnn_dense':
        return make_gnn_dense(config, dtype)

    raise KeyError("Unknown model type {}".format(model))

def make_gnn_dense(config, dtype):

    parameters = [
        "layernorm",
        "hidden_dim",
        "bin_size",
        "num_node_messages",
        "num_graph_layers",
        "distance_dim",
        "dropout",
        "input_encoding",
        "graph_kernel",
        "skip_connection",
        "regression_use_classification",
        "node_message",
        "debug"
    ]

    kwargs = {par: config['parameters'][par] for par in parameters}

    model = PFNetDense(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        schema=config["dataset"]["schema"],
        **kwargs
    )

    return model

def make_dense(config, dtype):
    model = DummyNet(
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
    )
    return model

def eval_model(X, ygen, ycand, model, config, outdir, global_batch_size):
    import scipy
    for ibatch in tqdm(range(X.shape[0]//global_batch_size), desc="Evaluating model"):
        nb1 = ibatch*global_batch_size
        nb2 = (ibatch+1)*global_batch_size

        y_pred = model.predict(X[nb1:nb2], batch_size=global_batch_size)
        if type(y_pred) is dict:  # for e.g. when the model is multi_output
            y_pred_raw_ids = y_pred['cls']
        else:
            y_pred_raw_ids = y_pred[:, :, :config["dataset"]["num_output_classes"]]
        
        #softmax score must be over a threshold 0.6 to call it a particle (prefer low fake rate to high efficiency)
        # y_pred_id_sm = scipy.special.softmax(y_pred_raw_ids, axis=-1)
        # y_pred_id_sm[y_pred_id_sm < 0.] = 0.0

        msk = np.ones(y_pred_raw_ids.shape, dtype=np.bool)

        #Use thresholds for charged and neutral hadrons based on matching the DelphesPF fake rate
        # msk[y_pred_id_sm[:, :, 1] < 0.8, 1] = 0
        # msk[y_pred_id_sm[:, :, 2] < 0.025, 2] = 0
        y_pred_raw_ids = y_pred_raw_ids*msk

        y_pred_id = np.argmax(y_pred_raw_ids, axis=-1)

        if type(y_pred) is dict:
            y_pred_rest = np.concatenate([y_pred["charge"], y_pred["pt"], y_pred["eta"], y_pred["sin_phi"], y_pred["cos_phi"], y_pred["energy"]], axis=-1)
            y_pred_id = np.concatenate([np.expand_dims(y_pred_id, axis=-1), y_pred_rest], axis=-1)
        else:
            y_pred_id = np.concatenate([np.expand_dims(y_pred_id, axis=-1), y_pred[:, :, config["dataset"]["num_output_classes"]:]], axis=-1)

        np_outfile = "{}/pred_batch{}.npz".format(outdir, ibatch)
        np.savez(
            np_outfile,
            X=X[nb1:nb2],
            ygen=ygen[nb1:nb2],
            ycand=ycand[nb1:nb2],
            ypred=y_pred_id, ypred_raw=y_pred_raw_ids
        )

def freeze_model(model, config, outdir):

    model.compile(loss="mse", optimizer="adam")
    model.save(outdir + "/model_full", save_format="tf")

    full_model = tf.function(lambda x: model(x, training=False))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec((None, None, config["dataset"]["num_input_features"]), tf.float32))
    from tensorflow.python.framework import convert_to_constants
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(full_model)
    graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_func.graph.as_graph_def())
    
    tf.io.write_graph(graph_or_graph_def=graph,
      logdir="{}/model_frozen".format(outdir),
      name="frozen_graph.pb",
      as_text=False)
    tf.io.write_graph(graph_or_graph_def=graph,
      logdir="{}/model_frozen".format(outdir),
      name="frozen_graph.pbtxt",
      as_text=True)

class FlattenedCategoricalAccuracy(tf.keras.metrics.CategoricalAccuracy):
    def __init__(self, use_weights=False, **kwargs):
        super(FlattenedCategoricalAccuracy, self).__init__(**kwargs)
        self.use_weights = use_weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        #flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0]*tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        sample_weights = None

        if self.use_weights:
            sample_weights = _y_true*tf.reduce_sum(_y_true, axis=0)
            sample_weights = 1.0/sample_weights[sample_weights!=0]

        super(FlattenedCategoricalAccuracy, self).update_state(_y_true, _y_pred, sample_weights)

class SingleClassRecall(Recall):
    def __init__(self, icls, **kwargs):
        super(SingleClassRecall, self).__init__(**kwargs)
        self.icls = icls

    def update_state(self, y_true, y_pred, sample_weight=None):
        #flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.argmax(tf.reshape(y_pred, (tf.shape(y_pred)[0]*tf.shape(y_pred)[1], tf.shape(y_pred)[2])), axis=-1)
        super(SingleClassRecall, self).update_state(
            _y_true[:, self.icls],
            tf.cast(_y_pred==self.icls, tf.float32)
        )

class FlattenedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, use_weights=False, **kwargs):
        super(FlattenedMeanIoU, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        #flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0]*tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        super(FlattenedMeanIoU, self).update_state(_y_true, _y_pred, None)

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, numpy_logs):
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            tf.summary.scalar('learning rate', data=lr, step=epoch)
        except AttributeError as e:
            pass

def configure_model_weights(model, trainable_layers):
    print("setting trainable layers: {}".format(trainable_layers))
    if (trainable_layers is None):
        trainable_layers = "all"
    if trainable_layers == "all":
        model.trainable = True
    elif trainable_layers == "classification":
        model.set_trainable_classification()
    elif trainable_layers == "regression":
        model.set_trainable_regression()
    else:
        if isinstance(trainable_layers, str):
            trainable_layers = [trainable_layers]
        model.set_trainable_named(trainable_layers)

    model.compile()
    trainable_count = sum([np.prod(tf.keras.backend.get_value(w).shape) for w in model.trainable_weights])
    non_trainable_count = sum([np.prod(tf.keras.backend.get_value(w).shape) for w in model.non_trainable_weights])
    print("trainable={} non_trainable={}".format(trainable_count, non_trainable_count))

def make_focal_loss(config):
    def loss(x,y):
        return tfa.losses.sigmoid_focal_crossentropy(x,y,
            alpha=float(config["setup"].get("focal_loss_alpha", 0.25)),
            gamma=float(config["setup"].get("focal_loss_gamma", 2.0)),
            from_logits=bool(config["setup"].get("focal_loss_from_logits", False))
        )
    return loss