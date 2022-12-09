import logging

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    logging.warning("horovod not found, ignoring")

import glob
import json
import os
import pickle
from pathlib import Path

import awkward
import fastjet
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
import vector
from tfmodel.callbacks import BenchmarkLoggerCallback, CustomTensorBoard
from tfmodel.datasets.BaseDatasetFactory import unpack_target
from tqdm import tqdm

from .model import PFNetDense, PFNetTransformer


class ModelOptimizerCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super(ModelOptimizerCheckpoint, self).on_epoch_end(epoch, logs=logs)
        weightfile_path = self.opt_path.format(epoch=epoch + 1, **logs)
        weights = self.model.optimizer.get_weights()

        with open(weightfile_path, "wb") as fi:
            pickle.dump(
                {
                    # "lr": lr,
                    "weights": weights
                },
                fi,
            )


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, outpath, dataset, config, plot_freq=1, horovod_enabled=False, comet_experiment=None):
        super(CustomCallback, self).__init__()
        self.plot_freq = plot_freq
        self.dataset = dataset
        self.outpath = outpath
        self.config = config
        self.horovod_enabled = horovod_enabled
        self.comet_experiment = comet_experiment

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

        # load the evaluation data
        yvals = []
        for fi in glob.glob(str(cp_dir / "*.parquet")):
            dd = awkward.from_parquet(fi)
            # os.remove(fi)
            yvals.append(dd)

        yvals_awk = awkward.concatenate(yvals, axis=0)

        particles = {k: awkward.flatten(yvals_awk["particles"][k], axis=1) for k in yvals_awk["particles"].fields}

        msk_gen = np.argmax(particles["gen"]["cls"], axis=-1) != 0
        gen_px = particles["gen"]["pt"][msk_gen] * particles["gen"]["cos_phi"][msk_gen]
        gen_py = particles["gen"]["pt"][msk_gen] * particles["gen"]["sin_phi"][msk_gen]

        msk_pred = np.argmax(particles["pred"]["cls"], axis=-1) != 0
        pred_px = particles["pred"]["pt"][msk_pred] * particles["pred"]["cos_phi"][msk_pred]
        pred_py = particles["pred"]["pt"][msk_pred] * particles["pred"]["sin_phi"][msk_pred]

        msk_cand = np.argmax(particles["cand"]["cls"], axis=-1) != 0
        cand_px = particles["cand"]["pt"][msk_cand] * particles["cand"]["cos_phi"][msk_cand]
        cand_py = particles["cand"]["pt"][msk_cand] * particles["cand"]["sin_phi"][msk_cand]

        gen_met = np.sqrt(np.sum(gen_px, axis=1) ** 2 + np.sum(gen_py, axis=1) ** 2)
        pred_met = np.sqrt(np.sum(pred_px, axis=1) ** 2 + np.sum(pred_py, axis=1) ** 2)
        cand_met = np.sqrt(np.sum(cand_px, axis=1) ** 2 + np.sum(cand_py, axis=1) ** 2)

        met_ratio_pred = awkward.to_numpy((pred_met - gen_met) / gen_met)
        met_ratio_cand = awkward.to_numpy((cand_met - gen_met) / gen_met)

        # flatten across file and event dimension
        gen_to_pred_genpt = awkward.flatten(
            awkward.flatten(vector.arr(yvals_awk["matched_jets"]["gen_to_pred"]["gen_jet"]).pt, axis=1)
        )
        gen_to_pred_predpt = awkward.flatten(
            awkward.flatten(vector.arr(yvals_awk["matched_jets"]["gen_to_pred"]["pred_jet"]).pt, axis=1)
        )

        gen_to_cand_genpt = awkward.flatten(
            awkward.flatten(vector.arr(yvals_awk["matched_jets"]["gen_to_cand"]["gen_jet"]).pt, axis=1)
        )
        gen_to_cand_candpt = awkward.flatten(
            awkward.flatten(vector.arr(yvals_awk["matched_jets"]["gen_to_cand"]["cand_jet"]).pt, axis=1)
        )

        jet_ratio_pred = (gen_to_pred_predpt - gen_to_pred_genpt) / gen_to_pred_genpt
        jet_ratio_cand = (gen_to_cand_candpt - gen_to_cand_genpt) / gen_to_cand_genpt

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
        b = np.linspace(-1, 1, 100)
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

        jet_pred_wd = scipy.stats.wasserstein_distance(gen_to_pred_genpt, gen_to_pred_predpt)
        jet_pred_p25 = np.percentile(jet_ratio_pred, 25)
        jet_pred_p50 = np.percentile(jet_ratio_pred, 50)
        jet_pred_p75 = np.percentile(jet_ratio_pred, 75)
        jet_pred_iqr = jet_pred_p75 - jet_pred_p25

        met_pred_wd = scipy.stats.wasserstein_distance(gen_met, pred_met)
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
            logs["val_" + name] = val

            if comet_experiment:
                comet_experiment.log_metric(name, val, step=epoch - 1)


def prepare_callbacks(
    config,
    outdir,
    dataset,
    comet_experiment=None,
    horovod_enabled=False,
    benchmark_dir=None,
    num_train_steps=None,
    num_cpus=None,
    num_gpus=None,
    train_samples=None,
):

    callbacks = []
    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    if not horovod_enabled or hvd.rank() == 0:
        callbacks += get_checkpoint_history_callback(outdir, config, dataset, comet_experiment, horovod_enabled)

    if benchmark_dir:
        if benchmark_dir == "exp_dir":  # save benchmarking results in experiment output folder
            benchmark_dir = outdir
        if config["dataset"]["schema"] == "delphes":
            bmk_bs = config["train_test_datasets"]["delphes"]["batch_per_gpu"]
        elif config["dataset"]["schema"] == "cms":
            assert (
                len(config["train_test_datasets"]) == 1
            ), "Expected exactly 1 key, physical OR delphes, \
                found {}".format(
                config["train_test_datasets"].keys()
            )
            bmk_bs = config["train_test_datasets"]["physical"]["batch_per_gpu"]
        else:
            raise ValueError(
                "Benchmark callback only supports delphes or \
                cms dataset schema. {}".format(
                    config["dataset"]["schema"]
                )
            )
        Path(benchmark_dir).mkdir(exist_ok=True, parents=True)
        callbacks.append(
            BenchmarkLoggerCallback(
                outdir=benchmark_dir,
                steps_per_epoch=num_train_steps,
                batch_size_per_gpu=bmk_bs,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                train_set_size=train_samples,
            )
        )

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
        dataset.tensorflow_dataset.take(config["validation_num_events"]),
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
        update_freq="batch",
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
        met_output=config["loss"]["met_loss"] != "none",
        cls_output_as_logits=config["setup"].get("cls_output_as_logits", False),
        small_graph_opt=config["setup"].get("small_graph_opt", False),
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
        met_output=config["loss"]["met_loss"] != "none",
        cls_output_as_logits=config["setup"]["cls_output_as_logits"],
        **kwargs
    )
    return model


def deltar(a, b):
    return a.deltaR(b)


@numba.njit
def match_jets(jets1, jets2, deltaR_cut):
    iev = len(jets1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = jets1[ev]
        j2 = jets2[ev]

        jet_inds_1 = []
        jet_inds_2 = []
        # print(ev, j1.pt, j2.pt)
        for ij1 in range(len(j1)):
            drs = np.zeros(len(j2), dtype=np.float64)
            for ij2 in range(len(j2)):
                dr = j1[ij1].deltaR(j2[ij2])
                drs[ij2] = dr
            min_idx_dr = np.argmin(drs)
            if drs[min_idx_dr] < deltaR_cut:
                jet_inds_1.append(ij1)
                jet_inds_2.append(min_idx_dr)
        jet_inds_1_ev.append(jet_inds_1)
        jet_inds_2_ev.append(jet_inds_2)
    return jet_inds_1_ev, jet_inds_2_ev


def squeeze_if_one(arr):
    if arr.shape[-1] == 1:
        return np.squeeze(arr, axis=-1)
    else:
        return arr


def build_dummy_array(num):
    form = """
{
  "class": "ListOffsetArray64",
  "offsets": "i64",
  "content": "int64",
  "form_key": "node0"
}
"""
    builder = awkward.layout.LayoutBuilder32(form)

    for i in range(num):
        builder.begin_list()
        builder.end_list()

    return builder.snapshot()


def match_two_jet_collections(jets_coll, name1, name2, jet_match_dr):
    num_events = len(jets_coll[name1])
    ret = match_jets(jets_coll[name1], jets_coll[name2], jet_match_dr)
    j1_idx = awkward.from_iter(ret[0])
    j2_idx = awkward.from_iter(ret[1])
    if awkward.count(j1_idx) > 0:
        c1_to_c2 = awkward.layout.RecordArray([j1_idx.layout, j2_idx.layout], [name1, name2])
    else:
        dummy = build_dummy_array(num_events)
        c1_to_c2 = awkward.layout.RecordArray([dummy, dummy], [name1, name2])
    c1_to_c2 = awkward.Array(c1_to_c2)
    return c1_to_c2


# Given a model, evaluates it on each batch of the validation dataset
# For each batch, save the inputs, the generator-level target, the candidate-level target, and the prediction
def eval_model(model, dataset, config, outdir, jet_ptcut=5.0, jet_match_dr=0.1, verbose=False):

    ibatch = 0

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    for elem in tqdm(dataset, desc="Evaluating model"):

        if verbose:
            print("evaluating model")
        ypred = model.predict(elem["X"], verbose=verbose)

        keys_particle = [k for k in ypred.keys() if k != "met"]

        if verbose:
            print("unpacking outputs")

        ygen = unpack_target(elem["ygen"], config["dataset"]["num_output_classes"], config)
        ycand = unpack_target(elem["ycand"], config["dataset"]["num_output_classes"], config)

        X = awkward.Array(elem["X"].numpy())
        ygen = awkward.Array({k: squeeze_if_one(ygen[k].numpy()) for k in keys_particle})
        ycand = awkward.Array({k: squeeze_if_one(ycand[k].numpy()) for k in keys_particle})
        ypred = awkward.Array({k: squeeze_if_one(ypred[k]) for k in keys_particle})

        awkvals = {
            "gen": ygen,
            "cand": ycand,
            "pred": ypred,
        }

        jets_coll = {}
        if verbose:
            print("clustering jets")

        for typ in ["gen", "cand", "pred"]:
            phi = np.arctan2(awkvals[typ]["sin_phi"], awkvals[typ]["cos_phi"])

            cls_id = awkward.argmax(awkvals[typ]["cls"], axis=-1, mask_identity=False)
            valid = cls_id != 0

            # mask the particles in each event in the batch that were not predicted
            pt = awkward.from_iter([np.array(v[m], np.float32) for v, m in zip(awkvals[typ]["pt"], valid)])
            eta = awkward.from_iter([np.array(v[m], np.float32) for v, m in zip(awkvals[typ]["eta"], valid)])
            energy = awkward.from_iter([np.array(v[m], np.float32) for v, m in zip(awkvals[typ]["energy"], valid)])
            phi = awkward.from_iter([np.array(v[m], np.float32) for v, m in zip(phi, valid)])

            vec = vector.arr(awkward.zip({"pt": pt, "eta": eta, "phi": phi, "e": energy}))
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)

            jets_coll[typ] = cluster.inclusive_jets(min_pt=jet_ptcut)

            if verbose:
                print("jets {}".format(typ), awkward.to_numpy(awkward.count(jets_coll[typ].px, axis=1)))

        # DeltaR match between genjets and MLPF jets
        gen_to_pred = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)
        gen_to_cand = match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr)

        matched_jets = awkward.Array({"gen_to_pred": gen_to_pred, "gen_to_cand": gen_to_cand})

        # Save output file
        outfile = "{}/pred_batch{}.parquet".format(outdir, ibatch)
        if verbose:
            print("saving to {}".format(outfile))

        awkward.to_parquet(
            awkward.Array({"inputs": X, "particles": awkvals, "jets": jets_coll, "matched_jets": matched_jets}),
            outfile,
        )

        ibatch += 1


def freeze_model(model, config, outdir):
    import tf2onnx

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
            from_logits=config["setup"]["cls_output_as_logits"],
        )

    return loss
