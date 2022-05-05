from .model import PFNetTransformer, PFNetDense

import tensorflow as tf
import tensorflow_addons as tfa
import pickle
import numpy as np
import os
import io
import os
import yaml
import uuid
import matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
import time
import json
import random
import math
import platform
import mplhep
from tqdm import tqdm
from pathlib import Path

import tf2onnx
import sklearn
import sklearn.metrics

from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler
from tfmodel.callbacks import CustomTensorBoard
from tfmodel.utils import get_lr_schedule, get_optimizer, make_weight_function, targets_multi_output
from tfmodel.datasets.BaseDatasetFactory import unpack_target
import tensorflow_datasets as tfds

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

class ModelOptimizerCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super(ModelOptimizerCheckpoint, self).on_epoch_end(epoch, logs=logs)
        weightfile_path = self.opt_path.format(epoch=epoch+1, **logs)
        try:
            with open(weightfile_path, "wb") as fi:
                pickle.dump({
                    #"lr": self.model.optimizer.lr,
                    "weights": self.model.optimizer.get_weights()
                    }, fi
                )
        except Exception as e:
            print("Could not save optimizer state: {}".format(e))
            os.remove(weightfile_path)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, outpath, dataset, dataset_info, plot_freq=1, comet_experiment=None):
        super(CustomCallback, self).__init__()
        self.plot_freq = plot_freq
        self.comet_experiment = comet_experiment

        self.X = []
        self.ytrue = {}
        for inputs, targets, weights in tfds.as_numpy(dataset):
            self.X.append(inputs)
            for target_name in targets.keys():
                if not (target_name in self.ytrue):
                    self.ytrue[target_name] = []
                self.ytrue[target_name].append(targets[target_name])

        self.X = np.concatenate(self.X)
        for target_name in self.ytrue.keys():
            self.ytrue[target_name] = np.concatenate(self.ytrue[target_name])
        self.ytrue_id = np.argmax(self.ytrue["cls"], axis=-1)
        self.dataset_info = dataset_info

        self.num_output_classes = self.ytrue["cls"].shape[-1]

        self.outpath = outpath

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
            "pt": np.linspace(-100, 200, 100),
            "eta": np.linspace(-6, 6, 100),
            "sin_phi": np.linspace(-1,1,100),
            "cos_phi": np.linspace(-1,1,100),
            "energy": np.linspace(-100, 1000, 100),
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

    def plot_sumperevent_corr(self, epoch, outpath, ypred, var):
        pred_per_event = np.sum(ypred[var], axis=-2)[:, 0]
        true_per_event = np.sum(self.ytrue[var], axis=-2)[:, 0]

        plt.figure()
        plt.hist2d(true_per_event, pred_per_event, bins=100, cmap="Blues")
        minval = min(np.min(pred_per_event), np.min(true_per_event))
        maxval = max(np.max(pred_per_event), np.max(true_per_event))
        plt.plot([minval, maxval], [minval, maxval], color="black")
        image_path = str(outpath / "event_{}.png".format(var))
        plt.savefig(image_path, bbox_inches="tight")
        plt.close("all")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)

    def plot_event_visualization(self, epoch, outpath, ypred, ypred_id, msk, ievent=0):

        x_feat = self.dataset_info.metadata.get("x_features")
        X_energy = self.X[:, :, x_feat.index("e")]
        X_eta = self.X[:, :, x_feat.index("eta")]

        if "phi" in x_feat:
            X_phi = self.X[:, :, x_feat.index("phi")]
        else:
            X_phi = np.arctan2(
                self.X[:, :, x_feat.index("sin_phi")],
                self.Xs[:, :, x_feat.index("cos_phi")]
            )

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

        #save scatterplot of raw values
        plt.figure(figsize=(6,5))
        bins = self.reg_bins[reg_variable]

        if bins is None:
            bins = 100

        if reg_variable == "pt" or reg_variable == "energy":
            bins = np.logspace(-2,3,100)
            vals_true = np.log10(vals_true)
            vals_pred = np.log10(vals_pred)
            vals_true[np.isnan(vals_true)] = 0.0
            vals_pred[np.isnan(vals_pred)] = 0.0

        plt.hist2d(vals_true, vals_pred, bins=(bins, bins), cmin=1, cmap="Blues", norm=matplotlib.colors.LogNorm())
        if reg_variable == "pt" or reg_variable == "energy":
            plt.xscale("log")
            plt.yscale("log")
        plt.colorbar()
 
        if len(vals_true) > 0:
            minval = np.min(vals_true)
            maxval = np.max(vals_true)
            if not (math.isnan(minval) or math.isnan(maxval) or math.isinf(minval) or math.isinf(maxval)):
                plt.plot([minval, maxval], [minval, maxval], color="black", ls="--", lw=0.5)
        plt.xlabel("true")
        plt.ylabel("predicted")
        plt.title(reg_variable)
        image_path = str(outpath / "{}_cls{}_corr.png".format(reg_variable, icls))
        plt.savefig(image_path, bbox_inches="tight")
        if self.comet_experiment:
            self.comet_experiment.log_image(image_path, step=epoch)
        plt.close("all")

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

        image_path = str(cp_dir / "eff_fake_cls{}_ivar{}.png".format(icls, ivar))
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

        if self.plot_freq==0:
            return
        if self.plot_freq>1:
            if epoch%self.plot_freq!=0 or epoch==1:
                return

        cp_dir = Path(self.outpath) / "epoch_{}".format(epoch)
        cp_dir.mkdir(parents=True, exist_ok=True)

        #run the model inference on the validation dataset
        ypred = self.model.predict(self.X, batch_size=1)

        #choose the class with the highest probability as the prediction
        #this is a shortcut, in actual inference, we may want to apply additional per-class thresholds        
        ypred_id = np.argmax(ypred["cls"], axis=-1)
       
        #exclude padded elements from the plotting
        msk = self.X[:, :, 0] != 0

        self.plot_elem_to_pred(epoch, cp_dir, msk, ypred_id)

        self.plot_sumperevent_corr(epoch, cp_dir, ypred, "energy")
        self.plot_sumperevent_corr(epoch, cp_dir, ypred, "pt")

        self.plot_cm(epoch, cp_dir, ypred_id, msk)
        for ievent in range(min(5, self.X.shape[0])):
            self.plot_event_visualization(epoch, cp_dir, ypred, ypred_id, msk, ievent=ievent)

        for icls in range(self.num_output_classes):
            cp_dir_cls = cp_dir / "cls_{}".format(icls)
            cp_dir_cls.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(4,4))
            npred = np.sum(ypred_id == icls, axis=1)
            ntrue = np.sum(self.ytrue_id == icls, axis=1)
            maxval = max(np.max(npred), np.max(ntrue))
            plt.scatter(ntrue, npred, marker=".")
            plt.plot([0,maxval], [0, maxval], color="black", ls="--")

            image_path = str(cp_dir_cls/"num_cls{}.png".format(icls))
            plt.savefig(image_path, bbox_inches="tight")
            plt.close("all")
            if self.comet_experiment:
                self.comet_experiment.log_image(image_path, step=epoch)
                num_ptcl_err = np.sqrt(np.sum((npred-ntrue)**2))
                self.comet_experiment.log_metric('num_ptcl_cls{}'.format(icls), num_ptcl_err, step=epoch)

            if icls!=0:
                self.plot_eff_and_fake_rate(epoch, icls, msk, ypred_id, cp_dir_cls)
                self.plot_eff_and_fake_rate(epoch, icls, msk, ypred_id, cp_dir_cls, ivar=2, bins=np.linspace(-5,5,100))

            for variable in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
                self.plot_reg_distribution(epoch, cp_dir_cls, ypred, ypred_id, icls, variable)
                try:
                    self.plot_corr(epoch, cp_dir_cls, ypred, ypred_id, icls, variable)
                except ValueError as e:
                    print("Could not draw corr plot: {}".format(e))

def prepare_callbacks(
        callbacks_cfg, outdir,
        dataset,
        dataset_info,
        comet_experiment=None
    ):

    callbacks = []
    tb = CustomTensorBoard(
        log_dir=outdir + "/logs", histogram_freq=callbacks_cfg["tensorboard"]["hist_freq"], write_graph=False, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,90),
        profile_batch=0,
        dump_history=callbacks_cfg["tensorboard"]["dump_history"],
    )
    # Change the class name of CustomTensorBoard TensorBoard to make keras_tuner recognise it
    tb.__class__.__name__ = "TensorBoard"
    callbacks += [tb]

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    cp_dir = Path(outdir) / "weights"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_callback = ModelOptimizerCheckpoint(
        filepath=str(cp_dir / "weights-{epoch:02d}-{val_loss:.6f}.hdf5"),
        save_weights_only=True,
        verbose=0,
        monitor=callbacks_cfg["checkpoint"]["monitor"],
        save_best_only=False,
    )
    cp_callback.opt_path = str(cp_dir / "opt-{epoch:02d}-{val_loss:.6f}.pkl")
    callbacks += [cp_callback]

    history_path = Path(outdir) / "history"
    history_path.mkdir(parents=True, exist_ok=True)
    history_path = str(history_path)
    cb = CustomCallback(
        history_path,
        dataset,
        dataset_info,
        plot_freq=callbacks_cfg["plot_freq"],
        comet_experiment=comet_experiment
    )

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

    if model == 'transformer':
        return make_transformer(config, dtype)
    elif model == 'gnn_dense':
        return make_gnn_dense(config, dtype)

    raise KeyError("Unknown model type {}".format(model))

def make_gnn_dense(config, dtype):

    parameters = [
        "do_node_encoding",
        "node_update_mode",
        "node_encoding_hidden_dim",
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

def make_transformer(config, dtype):
    parameters = [
        "input_encoding",
        "output_decoding"
    ]
    kwargs = {}
    for par in parameters:
        if par in config['parameters'].keys():
            kwargs[par] = config['parameters'][par]

    model = PFNetTransformer(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        schema=config["dataset"]["schema"],
        **kwargs
    )
    return model

#Given a model, evaluates it on each batch of the validation dataset
#For each batch, save the inputs, the generator-level target, the candidate-level target, and the prediction
def eval_model(model, dataset, config, outdir):

    ibatch = 0
    for elem in tqdm(dataset, desc="Evaluating model"):
        y_pred = model.predict(elem["X"])

        np_outfile = "{}/pred_batch{}.npz".format(outdir, ibatch)

        ygen = unpack_target(elem["ygen"], config["dataset"]["num_output_classes"])
        ycand = unpack_target(elem["ycand"], config["dataset"]["num_output_classes"])

        outs = {}
        for key in y_pred.keys():
            outs["gen_{}".format(key)] = ygen[key]
            outs["cand_{}".format(key)] = ycand[key]
            outs["pred_{}".format(key)] = y_pred[key]
        np.savez(
            np_outfile,
            X=elem["X"],
            **outs
        )
        ibatch += 1

def freeze_model(model, config, outdir):
    bin_size = config["parameters"]["combined_graph_layer"]["bin_size"]
    num_features = config["dataset"]["num_input_features"]
    num_out_classes = config["dataset"]["num_output_classes"]

    def model_output(ret):
        return tf.concat([ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1)
    full_model = tf.function(lambda x: model_output(model(x, training=False)))

    #we need to use opset 12 for the version of ONNXRuntime in CMSSW
    #the warnings "RuntimeError: Opset (12) must be >= 13 for operator 'batch_dot'." do not seem to be critical
    model_proto, _ = tf2onnx.convert.from_function(
        full_model,
        opset=12,
        input_signature=(tf.TensorSpec((None, None, num_features), tf.float32, name="x:0"), ),
        output_path=str(Path(outdir) / "model.onnx")
    )

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
