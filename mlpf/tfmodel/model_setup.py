try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    print("hvd not enabled, ignoring")

import glob
import json
import os
import pickle
from pathlib import Path

import awkward
import fastjet
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
import tf2onnx
import vector
from tensorflow.keras.metrics import Recall
from tfmodel.callbacks import CustomTensorBoard
from tfmodel.datasets.BaseDatasetFactory import unpack_target
from tqdm import tqdm

from .model import PFNetDense, PFNetTransformer


class ModelOptimizerCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super(ModelOptimizerCheckpoint, self).on_epoch_end(epoch, logs=logs)
        weightfile_path = self.opt_path.format(epoch=epoch + 1, **logs)
        try:
            # PCGrad is derived from the legacy optimizer
            if self.model.optimizer.__class__.__module__ == "keras.optimizers.optimizer_v1":
                # lr = self.model.optimizer.optimizer.optimizer.lr
                weights = self.model.optimizer.optimizer.optimizer.get_weights()
            else:
                # lr = self.model.optimizer.lr
                weights = self.model.optimizer.get_weights()

            with open(weightfile_path, "wb") as fi:
                pickle.dump(
                    {
                        # "lr": lr,
                        "weights": weights
                    },
                    fi,
                )
        except Exception as e:
            print("Could not save optimizer state: {}".format(e))
            if os.path.isfile(weightfile_path):
                os.remove(weightfile_path)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, outpath, dataset, config, plot_freq=1, horovod_enabled=False, comet_experiment=None):
        super(CustomCallback, self).__init__()
        self.plot_freq = plot_freq
        self.dataset = dataset
        self.outpath = outpath
        self.config = config
        self.horovod_enabled = horovod_enabled
        self.comet_experiment = comet_experiment

        self.writer = tf.summary.create_file_writer(outpath)

    def on_epoch_end(self, epoch, logs=None):
        if not self.horovod_enabled or hvd.rank() == 0:
            epoch_end(self, epoch, logs, comet_experiment=self.comet_experiment)


def epoch_end(self, epoch, logs, comet_experiment=None):
    # first epoch is 1, not 0
    epoch = epoch + 1

    # save the training logs (losses) for this epoch
    with open("{}/history_{}.json".format(self.outpath, epoch), "w") as fi:
        json.dump(logs, fi)

    if self.plot_freq <= 0:
        return

    if self.plot_freq >= 1:
        if epoch % self.plot_freq != 0:
            return

        cp_dir = Path(self.outpath) / "epoch_{}".format(epoch)
        cp_dir.mkdir(parents=True, exist_ok=True)

        # run the model inference on the validation dataset
        eval_model(self.model, self.dataset, self.config, cp_dir)

        yvals = {}
        for fi in glob.glob(str(cp_dir / "*.npz")):
            dd = np.load(fi)
            keys_in_file = list(dd.keys())
            for k in keys_in_file:
                if k == "X":
                    continue
                if not (k in yvals):
                    yvals[k] = []
                yvals[k].append(dd[k])
        yvals = {k: np.concatenate(v) for k, v in yvals.items()}

        gen_px = yvals["gen_pt"] * yvals["gen_cos_phi"]
        gen_py = yvals["gen_pt"] * yvals["gen_sin_phi"]
        pred_px = yvals["pred_pt"] * yvals["pred_cos_phi"]
        pred_py = yvals["pred_pt"] * yvals["pred_sin_phi"]
        cand_px = yvals["cand_pt"] * yvals["cand_cos_phi"]
        cand_py = yvals["cand_pt"] * yvals["cand_sin_phi"]

        gen_met = np.sqrt(np.sum(gen_px**2 + gen_py**2, axis=1))
        pred_met = np.sqrt(np.sum(pred_px**2 + pred_py**2, axis=1))
        cand_met = np.sqrt(np.sum(cand_px**2 + cand_py**2, axis=1))

        with self.writer.as_default():
            jet_ratio_pred = (yvals["jets_pt_gen_to_pred"][:, 1] - yvals["jets_pt_gen_to_pred"][:, 0]) / yvals[
                "jets_pt_gen_to_pred"
            ][:, 0]
            jet_ratio_cand = (yvals["jets_pt_gen_to_cand"][:, 1] - yvals["jets_pt_gen_to_cand"][:, 0]) / yvals[
                "jets_pt_gen_to_cand"
            ][:, 0]
            met_ratio_pred = (pred_met[:, 0] - gen_met[:, 0]) / gen_met[:, 0]
            met_ratio_cand = (cand_met[:, 0] - cand_met[:, 0]) / gen_met[:, 0]

            plt.figure()
            b = np.linspace(-2, 5, 100)
            plt.hist(jet_ratio_cand, bins=b, histtype="step", lw=2, label="PF")
            plt.hist(jet_ratio_pred, bins=b, histtype="step", lw=2, label="MLPF")
            plt.xlabel("jet pT (reco-gen)/gen")
            plt.ylabel("number of matched jets")
            plt.legend(loc="best")
            image_path = str(cp_dir / "jet_res.png")
            plt.savefig(image_path, bbox_inches="tight", dpi=100)
            plt.clf()
            if comet_experiment:
                comet_experiment.log_image(image_path, step=epoch - 1)

            plt.figure()
            b = np.linspace(-2, 5, 100)
            plt.hist(met_ratio_cand, bins=b, histtype="step", lw=2, label="PF")
            plt.hist(met_ratio_pred, bins=b, histtype="step", lw=2, label="MLPF")
            plt.xlabel("MET (reco-gen)/gen")
            plt.ylabel("number of events")
            plt.legend(loc="best")
            image_path = str(cp_dir / "met_res.png")
            plt.savefig(image_path, bbox_inches="tight", dpi=100)
            plt.clf()
            if comet_experiment:
                comet_experiment.log_image(image_path, step=epoch - 1)

            jet_pred_wd = scipy.stats.wasserstein_distance(
                yvals["jets_pt_gen_to_pred"][:, 0], yvals["jets_pt_gen_to_pred"][:, 1]
            )
            jet_pred_p25 = np.percentile(jet_ratio_pred, 25)
            jet_pred_p50 = np.percentile(jet_ratio_pred, 50)
            jet_pred_p75 = np.percentile(jet_ratio_pred, 75)
            jet_pred_iqr = jet_pred_p75 - jet_pred_p25

            met_pred_wd = scipy.stats.wasserstein_distance(gen_met[:, 0], pred_met[:, 0])
            met_pred_p25 = np.percentile(met_ratio_pred, 25)
            met_pred_p50 = np.percentile(met_ratio_pred, 50)
            met_pred_p75 = np.percentile(met_ratio_pred, 75)
            met_pred_iqr = met_pred_p75 - met_pred_p25

            for name, val in [
                ("jet_wd", jet_pred_wd),
                ("jet_iqr", jet_pred_iqr),
                ("jet_med", jet_pred_p50),
                ("met_wd", met_pred_wd),
                ("met_iqr", met_pred_iqr),
                ("met_med", met_pred_p50),
            ]:
                tf.summary.scalar(name, val, step=epoch - 1, description=None)

                if comet_experiment:
                    comet_experiment.log_metric(name, val, step=epoch - 1)


def prepare_callbacks(
    config,
    outdir,
    dataset,
    comet_experiment=None,
    horovod_enabled=False,
):

    callbacks = []
    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    if not horovod_enabled or hvd.rank() == 0:
        callbacks += get_checkpoint_history_callback(outdir, config, dataset, comet_experiment, horovod_enabled)

    return callbacks


def get_checkpoint_history_callback(outdir, config, dataset, comet_experiment, horovod_enabled):
    callbacks = []
    cp_dir = Path(outdir) / "weights"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_callback = ModelOptimizerCheckpoint(
        filepath=str(cp_dir / "weights-{epoch:02d}-{val_loss:.6f}.hdf5"),
        save_weights_only=True,
        verbose=0,
        monitor=config["callbacks"]["checkpoint"]["monitor"],
        save_best_only=False,
    )
    cp_callback.opt_path = str(cp_dir / "opt-{epoch:02d}-{val_loss:.6f}.pkl")
    callbacks += [cp_callback]

    history_path = Path(outdir) / "history"
    history_path.mkdir(parents=True, exist_ok=True)
    history_path = str(history_path)
    cb = CustomCallback(
        history_path,
        dataset.take(config["setup"]["num_events_validation"]),
        config,
        plot_freq=config["callbacks"]["plot_freq"],
        horovod_enabled=horovod_enabled,
        comet_experiment=comet_experiment,
    )

    callbacks += [cb]
    tb = CustomTensorBoard(
        log_dir=outdir + "/logs",
        histogram_freq=config["callbacks"]["tensorboard"]["hist_freq"],
        write_graph=False,
        write_images=False,
        update_freq="epoch",
        # profile_batch=(10,90),
        profile_batch=0,
        dump_history=config["callbacks"]["tensorboard"]["dump_history"],
    )
    # Change the class name of CustomTensorBoard TensorBoard to make keras_tuner recognise it
    tb.__class__.__name__ = "TensorBoard"
    callbacks += [tb]

    return callbacks


def get_rundir(base="experiments"):
    if not os.path.exists(base):
        os.makedirs(base)

    previous_runs = os.listdir(base)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split("run_")[1]) for s in previous_runs]) + 1

    logdir = "run_%02d" % run_number
    return "{}/{}".format(base, logdir)


def make_model(config, dtype):
    model = config["parameters"]["model"]

    if model == "transformer":
        return make_transformer(config, dtype)
    elif model == "gnn_dense":
        return make_gnn_dense(config, dtype)

    raise KeyError("Unknown model type {}".format(model))


def make_gnn_dense(config, dtype):

    parameters = [
        "do_node_encoding",
        "node_update_mode",
        "node_encoding_hidden_dim",
        "dropout",
        "activation",
        "num_graph_layers_id",
        "num_graph_layers_reg",
        "input_encoding",
        "skip_connection",
        "output_decoding",
        "combined_graph_layer",
        "debug",
    ]

    kwargs = {}
    for par in parameters:
        if par in config["parameters"].keys():
            kwargs[par] = config["parameters"][par]

    model = PFNetDense(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        schema=config["dataset"]["schema"],
        event_set_output=config["loss"]["event_loss"] != "none",
        **kwargs
    )

    return model


def make_transformer(config, dtype):
    parameters = ["input_encoding", "output_decoding"]
    kwargs = {}
    for par in parameters:
        if par in config["parameters"].keys():
            kwargs[par] = config["parameters"][par]

    model = PFNetTransformer(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        schema=config["dataset"]["schema"],
        event_set_output=config["loss"]["event_loss"] != "none",
        **kwargs
    )
    return model


def deltar(a, b):
    return a.deltaR(b)


# Given a model, evaluates it on each batch of the validation dataset
# For each batch, save the inputs, the generator-level target, the candidate-level target, and the prediction
def eval_model(model, dataset, config, outdir):

    ibatch = 0

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    for elem in tqdm(dataset, desc="Evaluating model"):
        y_pred = model.predict(elem["X"], verbose=False)

        np_outfile = "{}/pred_batch{}.npz".format(outdir, ibatch)

        ygen = unpack_target(elem["ygen"], config["dataset"]["num_output_classes"], config)
        ycand = unpack_target(elem["ycand"], config["dataset"]["num_output_classes"], config)

        outs = {}

        for key in y_pred.keys():
            outs["gen_{}".format(key)] = ygen[key].numpy()
            outs["cand_{}".format(key)] = ycand[key].numpy()
            outs["pred_{}".format(key)] = y_pred[key]

        jets_coll = {}
        jets_const = {}
        for typ in ["gen", "cand", "pred"]:
            cls_id = np.argmax(outs["{}_cls".format(typ)], axis=-1)
            valid = cls_id != 0
            pt = awkward.from_iter([y[m][:, 0] for y, m in zip(outs["{}_pt".format(typ)], valid)])
            eta = awkward.from_iter([y[m][:, 0] for y, m in zip(outs["{}_eta".format(typ)], valid)])

            phi = np.arctan2(outs["{}_sin_phi".format(typ)], outs["{}_cos_phi".format(typ)])
            phi = awkward.from_iter([y[m][:, 0] for y, m in zip(phi, valid)])
            e = awkward.from_iter([y[m][:, 0] for y, m in zip(outs["{}_energy".format(typ)], valid)])

            vec = vector.arr({"pt": pt, "eta": eta, "phi": phi, "e": e})

            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)

            jets = cluster.inclusive_jets()
            jet_constituents = cluster.constituent_index()
            jets_coll[typ] = jets[jets.pt > 5.0]
            jets_const[typ] = jet_constituents[jets.pt > 5.0]

        for key in ["pt", "eta", "phi", "energy"]:
            outs["jets_gen_{}".format(key)] = awkward.to_numpy(awkward.flatten(getattr(jets_coll["gen"], key)))
            outs["jets_cand_{}".format(key)] = awkward.to_numpy(awkward.flatten(getattr(jets_coll["cand"], key)))
            outs["jets_pred_{}".format(key)] = awkward.to_numpy(awkward.flatten(getattr(jets_coll["pred"], key)))

        # DeltaR match between genjets and PF/MLPF jets
        cart = awkward.cartesian([jets_coll["gen"], jets_coll["pred"]], nested=True)
        jets_a, jets_b = awkward.unzip(cart)
        drs = deltar(jets_a, jets_b)
        match_gen_to_pred = [awkward.where(d < 0.1) for d in drs]
        m0 = awkward.from_iter([m[0] for m in match_gen_to_pred])
        m1 = awkward.from_iter([m[1] for m in match_gen_to_pred])
        j1s = jets_coll["gen"][m0]
        j2s = jets_coll["pred"][m1]

        outs["jets_pt_gen_to_pred"] = np.stack(
            [awkward.to_numpy(awkward.flatten(j1s.pt)), awkward.to_numpy(awkward.flatten(j2s.pt))], axis=-1
        )

        cart = awkward.cartesian([jets_coll["gen"], jets_coll["cand"]], nested=True)
        jets_a, jets_b = awkward.unzip(cart)
        drs = deltar(jets_a, jets_b)
        match_gen_to_pred = [awkward.where(d < 0.1) for d in drs]
        m0 = awkward.from_iter([m[0] for m in match_gen_to_pred])
        m1 = awkward.from_iter([m[1] for m in match_gen_to_pred])
        j1s = jets_coll["gen"][m0]
        j2s = jets_coll["cand"][m1]

        outs["jets_pt_gen_to_cand"] = np.stack(
            [awkward.to_numpy(awkward.flatten(j1s.pt)), awkward.to_numpy(awkward.flatten(j2s.pt))], axis=-1
        )

        np.savez(np_outfile, X=elem["X"], **outs)

        ibatch += 1


def freeze_model(model, config, outdir):
    num_features = config["dataset"]["num_input_features"]

    def model_output(ret):
        return tf.concat(
            [ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1
        )

    full_model = tf.function(lambda x: model_output(model(x, training=False)))

    # we need to use opset 12 for the version of ONNXRuntime in CMSSW
    # the warnings "RuntimeError: Opset (12) must be >= 13 for operator 'batch_dot'." do not seem to be critical
    model_proto, _ = tf2onnx.convert.from_function(
        full_model,
        opset=12,
        input_signature=(tf.TensorSpec((None, None, num_features), tf.float32, name="x:0"),),
        output_path=str(Path(outdir) / "model.onnx"),
    )


class FlattenedCategoricalAccuracy(tf.keras.metrics.CategoricalAccuracy):
    def __init__(self, use_weights=False, **kwargs):
        super(FlattenedCategoricalAccuracy, self).__init__(**kwargs)
        self.use_weights = use_weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        # flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        sample_weights = None

        if self.use_weights:
            sample_weights = _y_true * tf.reduce_sum(_y_true, axis=0)
            sample_weights = 1.0 / sample_weights[sample_weights != 0]

        super(FlattenedCategoricalAccuracy, self).update_state(_y_true, _y_pred, sample_weights)


class SingleClassRecall(Recall):
    def __init__(self, icls, **kwargs):
        super(SingleClassRecall, self).__init__(**kwargs)
        self.icls = icls

    def update_state(self, y_true, y_pred, sample_weight=None):
        # flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.argmax(tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2])), axis=-1)
        super(SingleClassRecall, self).update_state(_y_true[:, self.icls], tf.cast(_y_pred == self.icls, tf.float32))


class FlattenedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, use_weights=False, **kwargs):
        super(FlattenedMeanIoU, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # flatten the batch dimension
        _y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
        _y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        super(FlattenedMeanIoU, self).update_state(_y_true, _y_pred, None)


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, numpy_logs):
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            tf.summary.scalar("learning rate", data=lr, step=epoch)
        except AttributeError as e:
            print(e)
            pass


def configure_model_weights(model, trainable_layers):
    print("setting trainable layers: {}".format(trainable_layers))

    if trainable_layers is None:
        trainable_layers = "all"

    if trainable_layers == "all":
        model.trainable = True
    elif trainable_layers == "regression":
        for cg in model.cg_id:
            cg.trainable = False
        for cg in model.cg_reg:
            cg.trainable = True
        model.output_dec.set_trainable_regression()
    elif trainable_layers == "classification":
        for cg in model.cg_id:
            cg.trainable = True
        for cg in model.cg_reg:
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
    def loss(x, y):
        return tfa.losses.sigmoid_focal_crossentropy(
            x,
            y,
            alpha=float(config["setup"].get("focal_loss_alpha", 0.25)),
            gamma=float(config["setup"].get("focal_loss_gamma", 2.0)),
            from_logits=bool(config["setup"].get("focal_loss_from_logits", False)),
        )

    return loss
