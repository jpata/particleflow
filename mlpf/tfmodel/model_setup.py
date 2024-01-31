import logging

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    logging.warning("horovod not found, ignoring")

import json
import os
import pickle
from pathlib import Path

import time
import awkward
import fastjet
import numpy as np
import tensorflow as tf
import vector
from plotting.plot_utils import (
    compute_distances,
    compute_met_and_ratio,
    load_eval_data,
    plot_jet_ratio,
    plot_met,
    plot_met_ratio,
    plot_jets,
)
from tfmodel.callbacks import BenchmarkLoggerCallback, CustomTensorBoard
from tfmodel.datasets.BaseDatasetFactory import unpack_target
from tqdm import tqdm

from .model import PFNetDense, PFNetTransformer

from jet_utils import match_two_jet_collections, build_dummy_array, squeeze_if_one


class ModelOptimizerCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def save_opt_weights(self, logs):
        weightfile_path = self.opt_path.format(epoch=self._current_epoch + 1, **logs)
        weights = {}

        try:
            self.model.optimizer.save_own_variables(weights)
        except Exception as e:
            print("could not save optimizer weights with save_own_variables: {}".format(e))

        # TF 2.12 compatibility
        if len(weights) == 0:
            for i, variable in enumerate(self.model.optimizer.variables):
                weights[str(i)] = variable.numpy()

        with open(weightfile_path, "wb") as fi:
            print("saving {} optimizer weights to {}".format(len(weights), weightfile_path))
            pickle.dump(
                {
                    # "lr": lr,
                    "weights": weights
                },
                fi,
            )

    def on_epoch_end(self, epoch, logs=None):
        super(ModelOptimizerCheckpoint, self).on_epoch_end(epoch, logs=logs)
        self.save_opt_weights(logs)

    def on_train_batch_end(self, batch, logs=None):
        super(ModelOptimizerCheckpoint, self).on_train_batch_end(batch, logs=logs)
        if isinstance(self.save_freq, int) and batch > 0 and batch % self.save_freq == 0:
            self.save_opt_weights(logs)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        outpath,
        dataset,
        config,
        plot_freq=1,
        horovod_enabled=False,
        comet_experiment=None,
        is_hpo_run=False,
    ):
        super(CustomCallback, self).__init__()
        self.plot_freq = plot_freq
        self.dataset = dataset
        self.outpath = outpath
        self.config = config
        self.horovod_enabled = horovod_enabled
        self.comet_experiment = comet_experiment
        self.is_hpo_run = is_hpo_run

    def on_epoch_end(self, epoch, logs=None):
        if not self.horovod_enabled or hvd.rank() == 0:
            epoch_end(self, epoch, logs, comet_experiment=self.comet_experiment)


def epoch_end(self, epoch, logs, comet_experiment=None):
    # first epoch is 1, not 0
    epoch = epoch + 1

    # save the training logs (losses) for this epoch
    with open("{}/history_{}.json".format(self.outpath, epoch), "w") as fi:
        json.dump(logs, fi)

    if self.is_hpo_run:
        # comet does not log metrics automatically when running HPO using Ray Tune,
        # hence doing it manually here
        comet_experiment.log_metrics(logs, epoch=epoch)

    if self.plot_freq <= 0:
        return

    if self.plot_freq >= 1:
        if epoch % self.plot_freq != 0:
            return

        cp_dir = Path(self.outpath) / "epoch_{}".format(epoch)
        cp_dir.mkdir(parents=True, exist_ok=True)

        # run the model inference on the validation dataset
        eval_model(self.model, self.dataset, self.config, cp_dir)

        yvals, X, filenames = load_eval_data(str(cp_dir / "*.parquet"))
        for fi in filenames:
            os.remove(fi)
        met_data = compute_met_and_ratio(yvals)

        plot_jets(yvals, epoch, cp_dir, comet_experiment)
        plot_jet_ratio(yvals, epoch, cp_dir, comet_experiment)
        plot_met(met_data, epoch, cp_dir, comet_experiment)
        plot_met_ratio(met_data, epoch, cp_dir, comet_experiment)

        jet_distances = compute_distances(
            yvals["jet_gen_to_pred_genpt"],
            yvals["jet_gen_to_pred_predpt"],
            yvals["jet_ratio_pred"],
        )
        met_distances = compute_distances(
            met_data["gen_met"],
            met_data["pred_met"],
            met_data["ratio_pred"],
        )

        N_jets = len(awkward.flatten(yvals["jets_gen_pt"]))
        N_jets_matched_pred = len(yvals["jet_gen_to_pred_genpt"])
        for name, val in [
            ("jet_matched_frac", N_jets_matched_pred / N_jets if N_jets > 0 else float("nan")),
            ("jet_wd", jet_distances["wd"]),
            ("jet_iqr", jet_distances["iqr"]),
            ("jet_med", jet_distances["p50"]),
            ("met_wd", met_distances["wd"]),
            ("met_iqr", met_distances["iqr"]),
            ("met_med", met_distances["p50"]),
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
    is_hpo_run=False,
):
    callbacks = []
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks += get_checkpoint_history_callback(outdir, config, dataset, comet_experiment, horovod_enabled, is_hpo_run)

    if not horovod_enabled or hvd.rank() == 0:
        if benchmark_dir:
            if benchmark_dir == "exp_dir":  # save benchmarking results in experiment output folder
                benchmark_dir = outdir
            if config["dataset"]["schema"] == "delphes":
                bmk_bs = config["train_test_datasets"]["delphes"]["batch_per_gpu"]
            elif (config["dataset"]["schema"] == "cms") or (config["dataset"]["schema"] == "clic"):
                assert (
                    len(config["train_test_datasets"]) == 1
                ), "Expected exactly 1 key, physical OR delphes, \
                    found {}".format(
                    config["train_test_datasets"].keys()
                )
                bmk_bs = config["train_test_datasets"]["physical"]["batch_per_gpu"]
            else:
                raise ValueError(
                    "Benchmark callback only supports delphes \
                    cms or clic dataset schema. {}".format(
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
                    horovod_enabled=horovod_enabled,
                )
            )

    return callbacks


def get_checkpoint_history_callback(outdir, config, dataset, comet_experiment, horovod_enabled, is_hpo_run=False):
    callbacks = []

    if not horovod_enabled or hvd.rank() == 0:
        cp_dir = Path(outdir) / "weights"
        cp_dir.mkdir(parents=True, exist_ok=True)
        cp_callback = ModelOptimizerCheckpoint(
            filepath=str(cp_dir / "weights-{epoch:02d}-{val_loss:.6f}.hdf5"),
            save_weights_only=True,
            verbose=1,
            monitor=config["callbacks"]["checkpoint"]["monitor"],
            save_best_only=False,
        )
        cp_callback.opt_path = str(cp_dir / "opt-{epoch:02d}-{val_loss:.6f}.pkl")
        if config.get("do_checkpoint_callback", True):
            callbacks += [cp_callback]

        cp_callback = ModelOptimizerCheckpoint(
            filepath=str(cp_dir / "weights-{epoch:02d}-step.hdf5"), save_weights_only=True, verbose=1, save_freq=100
        )
        cp_callback.opt_path = str(cp_dir / "opt-{epoch:02d}-step.pkl")
        callbacks += [cp_callback]

    if not horovod_enabled:
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
            is_hpo_run=is_hpo_run,
        )

        if config.get("do_validation_callback", True):
            callbacks += [cb]

        tb = CustomTensorBoard(
            log_dir=outdir + "/logs",
            histogram_freq=config["callbacks"]["tensorboard"]["hist_freq"],
            write_graph=False,
            write_images=False,
            update_freq="batch",
            profile_batch=config["callbacks"]["tensorboard"]["profile_batch"]
            if "profile_batch" in config["callbacks"]["tensorboard"].keys()
            else 0,
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
        use_normalizer=config["setup"].get("use_normalizer", True),
        **kwargs,
    )

    return model


def make_transformer(config, dtype):
    parameters = [
        "input_encoding",
        "output_decoding",
        "num_layers_encoder",
        "num_layers_decoder_reg",
        "num_layers_decoder_cls",
        "hidden_dim",
        "num_heads",
        "num_random_features",
    ]
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
        **kwargs,
    )
    return model


# Given a model, evaluates it on each batch of the validation dataset
# For each batch, save the inputs, the generator-level target, the candidate-level target, and the prediction
def eval_model(
    model,
    dataset,
    config,
    outdir,
    jet_ptcut=5.0,
    jet_match_dr=0.1,
    verbose=False,
):
    ibatch = 0

    if config["evaluation_jet_algo"] == "ee_genkt_algorithm":
        jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
    elif config["evaluation_jet_algo"] == "antikt_algorithm":
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    else:
        raise KeyError("Unknown evaluation_jet_algo: {}".format(config["evaluation_jet_algo"]))

    for elem in tqdm(dataset, desc="Evaluating model"):
        if verbose:
            print("evaluating model")
        ypred = model.predict(elem["X"], verbose=verbose)
        ypred["charge"] = np.argmax(ypred["charge"], axis=-1) - 1

        if verbose:
            print("unpacking outputs")

        ygen = [unpack_target(x, config["dataset"]["num_output_classes"], config) for x in elem["ygen"]]
        ycand = [unpack_target(x, config["dataset"]["num_output_classes"], config) for x in elem["ycand"]]
        ygen = {k: tf.stack([x[k] for x in ygen]) for k in ygen[0].keys()}
        ycand = {k: tf.stack([x[k] for x in ycand]) for k in ycand[0].keys()}

        # 0, 1, 2 -> -1, 0, 1
        ygen["charge"] = tf.expand_dims(tf.math.argmax(ygen["charge"], axis=-1), axis=-1) - 1
        ycand["charge"] = tf.expand_dims(tf.math.argmax(ycand["charge"], axis=-1), axis=-1) - 1

        ygen["cls_id"] = tf.math.argmax(ygen["cls"], axis=-1)
        ycand["cls_id"] = tf.math.argmax(ycand["cls"], axis=-1)
        ypred["cls_id"] = tf.math.argmax(ypred["cls"], axis=-1).numpy()

        keys_particle = [k for k in ypred.keys() if k != "met"]

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

            if verbose:
                print(typ, pt)

            # If there were no particles, build dummy arrays with the correct datatype
            if len(awkward.flatten(pt)) == 0:
                pt = build_dummy_array(len(pt), np.float64)
                eta = build_dummy_array(len(pt), np.float64)
                phi = build_dummy_array(len(pt), np.float64)
                energy = build_dummy_array(len(pt), np.float64)

            vec = vector.awk(awkward.zip({"pt": pt, "eta": eta, "phi": phi, "e": energy}))
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)

            jets_coll[typ] = cluster.inclusive_jets(min_pt=jet_ptcut)

            if verbose:
                print(
                    "jets {}".format(typ),
                    awkward.to_numpy(awkward.count(jets_coll[typ].px, axis=1)),
                )

        # DeltaR match between genjets and MLPF jets
        gen_to_pred = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)
        gen_to_cand = match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr)

        matched_jets = awkward.Array({"gen_to_pred": gen_to_pred, "gen_to_cand": gen_to_cand})

        # Save output file
        outfile = "{}/pred_batch{}.parquet".format(outdir, ibatch)
        if verbose:
            print("saving to {}".format(outfile))

        awkward.to_parquet(
            awkward.Array(
                {
                    "inputs": X,
                    "particles": awkvals,
                    "jets": jets_coll,
                    "matched_jets": matched_jets,
                }
            ),
            outfile,
        )

        ibatch += 1


def freeze_model(model, config, outdir):
    def model_output(ret):
        return tf.concat(
            [
                ret["cls"],
                ret["charge"],
                ret["pt"],
                ret["eta"],
                ret["sin_phi"],
                ret["cos_phi"],
                ret["energy"],
            ],
            axis=-1,
        )

    full_model = tf.function(lambda x: model_output(model(x, training=False)))

    niter = 10
    nfeat = config["dataset"]["num_input_features"]

    if "combined_graph_layer" in config["parameters"]:
        bin_size = config["parameters"]["combined_graph_layer"]["bin_size"]
        elem_range = list(range(bin_size, 5 * bin_size, bin_size))
    else:
        elem_range = range(100, 1000, 200)

    for ibatch in [1, 2, 4]:
        for nptcl in elem_range:
            X = np.random.rand(ibatch, nptcl, nfeat)
            full_model(X)

            t0 = time.time()
            for i in range(niter):
                full_model(X)
            t1 = time.time()

            print(ibatch, nptcl, (t1 - t0) / niter)

    # we need to use opset 12 for the version of ONNXRuntime in CMSSW
    # the warnings "RuntimeError: Opset (12) must be >= 13 for operator 'batch_dot'." do not seem to be critical
    import tf2onnx

    model_proto, _ = tf2onnx.convert.from_function(
        full_model,
        opset=12,
        input_signature=(tf.TensorSpec((None, None, nfeat), tf.float32, name="x:0"),),
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
        from .tfa import sigmoid_focal_crossentropy

        return sigmoid_focal_crossentropy(
            x,
            y,
            alpha=float(config["setup"].get("focal_loss_alpha", 0.25)),
            gamma=float(config["setup"].get("focal_loss_gamma", 2.0)),
            from_logits=config["setup"]["cls_output_as_logits"],
        )

    return loss
