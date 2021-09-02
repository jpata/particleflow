from .model import DummyNet, PFNetDense

import tensorflow as tf
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
import mplhep
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
    def __init__(self, dataset_def, outpath, X, y, dataset_transform, num_output_classes, plot_freq=1, comet_experiment=None):
        super(CustomCallback, self).__init__()
        self.X = X
        self.y = y
        self.plot_freq = plot_freq
        self.comet_experiment = comet_experiment

        self.dataset_def = dataset_def

        #transform the prediction target from an array into a dictionary for easier access
        self.ytrue = dataset_transform(self.X, self.y, None)[1]
        self.ytrue = {k: np.array(v) for k, v in self.ytrue.items()}
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
            "pt": np.linspace(0, 100, 100),
            "eta": np.linspace(-6, 6, 100),
            "sin_phi": np.linspace(-1,1,100),
            "cos_phi": np.linspace(-1,1,100),
            "energy": None,
        }

    def plot_cm(self, epoch, outpath, ypred_id, msk):

        ytrue_id_flat = self.ytrue_id[msk].astype(np.int64).flatten()
        ypred_id_flat = ypred_id[msk].flatten()

        cm = sklearn.metrics.confusion_matrix(
            ytrue_id_flat,
            ypred_id_flat, labels=list(range(self.num_output_classes)), normalize="true"
        )
        if self.comet_experiment:
            self.comet_experiment.log_confusion_matrix(
                file_name="confusion-matrix-epoch{}.json".format(epoch), matrix=cm, epoch=epoch
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

        image_path = str(outpath / "cm_normed.png")
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def plot_event_visualization(self, epoch, outpath, ypred, ypred_id, msk, ievent=0):

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

        image_path = str(outpath / "event_iev{}.png".format(ievent))
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def plot_reg_distribution(self, epoch, outpath, ypred, ypred_id, icls, reg_variable):

        if icls==0:
            vals_pred = ypred[reg_variable][ypred_id!=icls].flatten()
            vals_true = self.ytrue[reg_variable][self.ytrue_id!=icls].flatten()
        else:
            vals_pred = ypred[reg_variable][ypred_id==icls].flatten()
            vals_true = self.ytrue[reg_variable][self.ytrue_id==icls].flatten()

        bins = self.reg_bins[reg_variable]
        if bins is None:
            bins = 100

        plt.figure()
        plt.hist(vals_true, bins=bins, histtype="step", lw=2, label="true")
        plt.hist(vals_pred, bins=bins, histtype="step", lw=2, label="predicted")

        if reg_variable in ["pt", "energy"]:
            plt.yscale("log")
            plt.ylim(bottom=1e-2)

        plt.xlabel(reg_variable)
        plt.ylabel("Number of particles")
        plt.legend(loc="best")
        plt.title("Regression output, cls {}".format(icls))
        image_path = str(outpath / "{}_cls{}.png".format(reg_variable, icls))
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def plot_corr(self, epoch, outpath, ypred, ypred_id, icls, reg_variable):

        if icls==0:
            sel = (ypred_id!=0) & (self.ytrue_id!=0)
        else:
            sel = (ypred_id==icls) & (self.ytrue_id==icls)

        vals_pred = ypred[reg_variable][sel].flatten()
        vals_true = self.ytrue[reg_variable][sel].flatten()

        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss_vals = loss(np.expand_dims(vals_true, -1), np.expand_dims(vals_pred, axis=-1)).numpy()

        #save scatterplot of raw values
        plt.figure()
        bins = self.reg_bins[reg_variable]
        if bins is None:
            bins = 100
        plt.scatter(vals_true, vals_pred, marker=".", alpha=0.4)

        if len(vals_true) > 0:
            minval = np.min(vals_true)
            maxval = np.max(vals_true)
            if not (math.isnan(minval) or math.isnan(maxval) or math.isinf(minval) or math.isinf(maxval)):
                plt.plot([minval, maxval], [minval, maxval], color="black", ls="--", lw=0.5)
        plt.xlabel("true")
        plt.ylabel("predicted")
        plt.title("{}, particle weighted, L={:.4f}".format(reg_variable, np.sum(loss_vals)))
        image_path = str(outpath / "{}_cls{}_corr.png".format(reg_variable, icls))
        plt.savefig(image_path, bbox_inches="tight")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)
        plt.close("all")

        #save loss-weighted correlation histogram
        plt.figure()
        plt.hist2d(vals_true, vals_pred, bins=(bins, bins), weights=loss_vals, cmap="Blues")
        plt.colorbar()
        if len(vals_true) > 0:
            minval = np.min(vals_true)
            maxval = np.max(vals_true)
            if not (math.isnan(minval) or math.isnan(maxval) or math.isinf(minval) or math.isinf(maxval)):
                plt.plot([minval, maxval], [minval, maxval], color="black", ls="--", lw=0.5)
        plt.xlabel("true")
        plt.ylabel("predicted")
        plt.title("{}, loss weighted, L={:.4f}".format(reg_variable, np.sum(loss_vals)))
        image_path = str(outpath / "{}_cls{}_corr_weighted.png".format(reg_variable, icls))
        plt.savefig(image_path, bbox_inches="tight")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

        #Also plot the residuals, as we have the true and predicted values already available here
        plt.figure()
        residual = vals_true - vals_pred
        residual[np.isnan(residual)] = 0
        residual[np.isinf(residual)] = 0
        plt.hist(residual, bins=100)
        plt.yscale("log")
        plt.xlabel("true - pred")
        plt.title("{} residual, m={:.4f} s={:.4f}".format(reg_variable, np.mean(residual), np.std(residual)))

        image_path = str(outpath / "{}_cls{}_residual.png".format(reg_variable, icls))
        plt.savefig(image_path, bbox_inches="tight")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)
        plt.close("all")

        if self.comet_experiment:
            self.comet_experiment.log_metric('residual_{}_cls{}_mean'.format(reg_variable, icls), np.mean(residual), step=epoch)
            self.comet_experiment.log_metric('residual_{}_cls{}_std'.format(reg_variable, icls), np.std(residual), step=epoch)
            self.comet_experiment.log_metric('val_loss_{}_cls{}'.format(reg_variable, icls), np.sum(loss_vals), step=epoch)

    def plot_elem_to_pred(self, epoch, cp_dir, msk, ypred_id):
        X_id = self.X[msk][:, 0]
        max_elem = int(np.max(X_id))
        cand_id = self.ytrue_id[msk]
        pred_id = ypred_id[msk]
        cm1 = sklearn.metrics.confusion_matrix(X_id, cand_id, labels=range(max_elem))
        cm2 = sklearn.metrics.confusion_matrix(X_id, pred_id, labels=range(max_elem))

        plt.figure(figsize=(10,4))

        ax = plt.subplot(1,2,1)
        plt.title("Targets")
        plt.imshow(cm1, cmap="Blues", norm=matplotlib.colors.LogNorm())
        plt.xticks(range(12));
        plt.yticks(range(12));
        plt.xlabel("Particle id")
        plt.ylabel("PFElement id")
        plt.colorbar()

        ax = plt.subplot(1,2,2)
        plt.title("Predictions")
        plt.imshow(cm2, cmap="Blues", norm=matplotlib.colors.LogNorm())
        plt.xticks(range(12));
        plt.yticks(range(12));
        plt.xlabel("Particle id")
        plt.ylabel("PFElement id")
        plt.colorbar()

        image_path = str(cp_dir / "elem_to_pred.png")
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")

        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def plot_eff_and_fake_rate(
        self,
        epoch,
        icls,
        msk,
        ypred_id,
        cp_dir,
        ivar=4,
        bins=np.linspace(0, 200, 100),
        xlabel="PFElement E",
        log_var=False,
        do_log_y=True
        ):
        
        values = self.X[msk][:, ivar]
        cand_id = self.ytrue_id[msk]
        pred_id = ypred_id[msk]

        if log_var:
            values = np.log(values)
            
        hist_cand = np.histogram(values[(cand_id==icls)], bins=bins);
        hist_cand_true = np.histogram(values[(cand_id==icls) & (pred_id==icls)], bins=bins);

        hist_pred = np.histogram(values[(pred_id==icls)], bins=bins);
        hist_pred_fake = np.histogram(values[(cand_id!=icls) & (pred_id==icls)], bins=bins);

        eff = hist_cand_true[0]/hist_cand[0]
        fake = hist_pred_fake[0]/hist_pred[0]

        plt.figure(figsize=(8,8))
        ax = plt.subplot(2,1,1)
        mplhep.histplot(hist_cand, label="PF")
        mplhep.histplot(hist_pred, label="MLPF")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("Number of particles")
        if do_log_y:
            ax.set_yscale("log")

        ax = plt.subplot(2,1,2, sharex=ax)
        mplhep.histplot(eff, bins=hist_cand[1], label="efficiency", color="black")
        mplhep.histplot(fake, bins=hist_cand[1], label="fake rate", color="red")
        plt.legend(frameon=False)
        plt.ylim(0, 1.4)
        plt.xlabel(xlabel)
        plt.ylabel("Fraction of particles / bin")

        image_path = str(cp_dir / "eff_fake_cls{}.png".format(icls))
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")

        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def on_epoch_end(self, epoch, logs=None):

        #first epoch is 1, not 0
        epoch = epoch + 1

        #save the training logs (losses) for this epoch
        with open("{}/history_{}.json".format(self.outpath, epoch), "w") as fi:
            json.dump(logs, fi)

        if self.plot_freq>1:
            if epoch%self.plot_freq!=0 or epoch==1:
                return

        cp_dir = Path(self.outpath) / "epoch_{}".format(epoch)
        cp_dir.mkdir(parents=True, exist_ok=True)

        #run the model inference on the validation dataset
        ypred = self.model.predict(self.X, batch_size=1)
        #ypred = self.model(self.X, training=False)
        #ypred = {k: v.numpy() for k, v in ypred.items()}

        #choose the class with the highest probability as the prediction
        #this is a shortcut, in actual inference, we may want to apply additional per-class thresholds        
        ypred_id = np.argmax(ypred["cls"], axis=-1)
       
        #exclude padded elements from the plotting
        msk = self.X[:, :, 0] != 0

        self.plot_elem_to_pred(epoch, cp_dir, msk, ypred_id)

        self.plot_cm(epoch, cp_dir, ypred_id, msk)
        for ievent in range(min(5, self.X.shape[0])):
            self.plot_event_visualization(epoch, cp_dir, ypred, ypred_id, msk, ievent=ievent)

        for icls in range(self.num_output_classes):
            cp_dir_cls = cp_dir / "cls_{}".format(icls)
            cp_dir_cls.mkdir(parents=True, exist_ok=True)

            if icls!=0:
                self.plot_eff_and_fake_rate(epoch, icls, msk, ypred_id, cp_dir_cls)

            for variable in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
                self.plot_reg_distribution(epoch, cp_dir_cls, ypred, ypred_id, icls, variable)
                self.plot_corr(epoch, cp_dir_cls, ypred, ypred_id, icls, variable)

        np.savez(str(cp_dir/"pred.npz"), X=self.X, ytrue=self.y, **ypred)

def prepare_callbacks(
    model, outdir,
    X_val, y_val,
    dataset_transform,
    num_output_classes,
    dataset_def,
    plot_freq=1, comet_experiment=None):
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
    cb = CustomCallback(
        dataset_def, history_path,
        X_val, y_val,
        dataset_transform,
        num_output_classes,
        plot_freq=plot_freq,
        comet_experiment=comet_experiment
    )
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
        "do_node_encoding",
        "hidden_dim",
        "dropout",
        "activation",
        "num_graph_layers_common",
        "num_graph_layers_energy",
        "input_encoding",
        "skip_connection",
        "output_decoding",
        "combined_graph_layer",
        "debug"
    ]

    kwargs = {}
    for par in parameters:
        if par in config['parameters'].keys():
            kwargs[par] = config['parameters'][par]

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
    for ibatch in tqdm(range(max(1, X.shape[0]//global_batch_size)), desc="Evaluating model"):
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
    elif trainable_layers == "regression":
        for cg in model.cg:
            cg.trainable = False
        for cg in model.cg_energy:
            cg.trainable = True

        model.output_dec.set_trainable_regression()
    elif trainable_layers == "classification":
        for cg in model.cg:
            cg.trainable = True
        for cg in model.cg_energy:
            cg.trainable = False

        model.output_dec.set_trainable_classification()
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