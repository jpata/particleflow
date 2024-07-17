import logging

logging.basicConfig(level=logging.INFO)

from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    logging.warning("horovod not found, ignoring")

import os
import pickle
import platform
import random
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
import ctypes

import boost_histogram as bh
import click
import numpy as np
import tensorflow as tf
import tqdm
from customizations import customization_functions
from tfmodel import hypertuning
from tfmodel.datasets.BaseDatasetFactory import (
    mlpf_dataset_from_config,
    unpack_target,
)
from tfmodel.lr_finder import LRFinder
from tfmodel.model_setup import (
    eval_model,
    freeze_model,
    prepare_callbacks,
)
from utils import create_experiment_dir, create_comet_experiment
from tfmodel.utils import (
    delete_all_but_best_checkpoint,
    get_best_checkpoint,
    get_datasets,
    get_latest_checkpoint,
    get_singlenode_strategy,
    get_train_test_val_datasets,
    get_tuner,
    initialize_horovod,
    load_config,
    model_scope,
    parse_config,
)
from tfmodel.utils_analysis import (
    analyze_ray_experiment,
    count_skipped_configurations,
    plot_ray_analysis,
    summarize_top_k,
    topk_summary_plot_v2,
)
from tfmodel.callbacks import NpEncoder


@click.group()
@click.help_option("-h", "--help")
def main():
    pass


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option(
    "-w",
    "--weights",
    default=None,
    help="trained weights to load",
    type=click.Path(),
)
@click.option(
    "--ntrain",
    default=None,
    help="override the number of training steps",
    type=int,
)
@click.option(
    "--ntest",
    default=None,
    help="override the number of testing steps",
    type=int,
)
@click.option(
    "--nepochs",
    default=None,
    help="override the number of training epochs",
    type=int,
)
@click.option(
    "-r",
    "--recreate",
    help="force creation of new experiment dir",
    is_flag=True,
)
@click.option(
    "-p",
    "--prefix",
    default="",
    help="prefix to put at beginning of training dir name",
    type=str,
)
@click.option(
    "--plot-freq",
    default=None,
    help="plot detailed validation every N epochs",
    type=int,
)
@click.option("--customize", help="customization function", type=str, default=None)
@click.option("--comet-offline", help="log comet-ml experiment locally", is_flag=True)
@click.option(
    "-j",
    "--jobid",
    help="log the Slurm job ID in experiments dir",
    type=str,
    default=None,
)
@click.option(
    "-m",
    "--horovod-enabled",
    help="Enable multi-node training using Horovod",
    is_flag=True,
)
@click.option(
    "-g",
    "--habana-enabled",
    help="Enable training on Habana Gaudi",
    is_flag=True,
)
@click.option(
    "-b",
    "--benchmark_dir",
    help="dir to save benchmark results. If 'exp_dir' results will be saved in the \
    experiment folder",
    type=str,
    default=None,
)
@click.option(
    "--batch-multiplier",
    help="batch size per device",
    type=float,
    default=None,
)
@click.option("--num-cpus", help="number of CPU threads to use", type=int, default=None)
@click.option("--seeds", help="set the random seeds", is_flag=True, default=True)
@click.option(
    "--comet-exp-name",
    help="comet experiment name",
    type=str,
    default="particleflow-tf",
)
def train(
    config,
    weights,
    ntrain,
    ntest,
    nepochs,
    recreate,
    prefix,
    plot_freq,
    customize,
    comet_offline,
    jobid,
    horovod_enabled,
    habana_enabled,
    benchmark_dir,
    batch_multiplier,
    num_cpus,
    seeds,
    comet_exp_name,
):
    # tf.debugging.enable_check_numerics()

    # Configure GPU threads according to TensorFlow's best practices for optimal model performance
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "2"

    # According to TensorFlow's best practices for optimal model performance, set GPU memory growth to True
    physical_devices = tf.config.list_physical_devices("GPU")
    for pd in physical_devices:
        tf.config.experimental.set_memory_growth(pd, True)

    if seeds:
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    """Train a model defined by config"""
    config_file_path = config
    config, config_file_stem = parse_config(config, nepochs=nepochs, weights=weights)
    logging.info(f"loaded config file: {config_file_path}")

    if plot_freq:
        config["callbacks"]["plot_freq"] = plot_freq

    if not os.path.isdir(config["cache"]):
        os.makedirs(config["cache"])

    if batch_multiplier:
        if config["batching"]["bucket_by_sequence_length"]:
            logging.info(
                "Dynamic batching is enabled, changing batch size multiplier from {} to {}".format(
                    config["batching"]["batch_multiplier"],
                    config["batching"]["batch_multiplier"] * batch_multiplier,
                )
            )
            config["batching"]["batch_multiplier"] *= batch_multiplier
        else:
            for key in config["train_test_datasets"]:
                logging.info(
                    "Static batching is enabled, changing batch size for dataset {} from {} to {}".format(
                        key,
                        config["train_test_datasets"][key]["batch_per_gpu"],
                        config["train_test_datasets"][key]["batch_per_gpu"] * batch_multiplier,
                    )
                )
                config["train_test_datasets"][key]["batch_per_gpu"] *= batch_multiplier

    if customize:
        config = customization_functions[customize](config)

    if habana_enabled:
        import habana_frameworks.tensorflow as htf

        htf.load_habana_module()

    # Decide tf.distribute.strategy depending on number of available GPUs
    horovod_enabled = horovod_enabled or config["setup"]["horovod_enabled"]

    if horovod_enabled:
        num_gpus, num_batches_multiplier = initialize_horovod(habana_enabled)
    else:
        strategy, num_gpus, num_batches_multiplier = get_singlenode_strategy(num_cpus=num_cpus)

    if num_gpus > 0:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            # According to TensorFlow's best practices for optimal model performance,
            # max out the L2 fetch granularity to 128 bytes when using NVIDIA GPUs
            _libcudart = ctypes.CDLL("libcudart.so")
            # Set device limit on the current device
            # cudaLimitMaxL2FetchGranularity = 0x05
            pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
            _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
            _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
            if pValue.contents.value != 128:
                logging.warning(
                    "Could not set L2 cache hint for GPU. Likely the GPU is too old for this optimization"
                    + ", you may experience slower training performance."
                )

    outdir = ""
    experiment = None
    if not horovod_enabled or hvd.rank() == 0:
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
        shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference
        experiment = create_comet_experiment(comet_exp_name, comet_offline=comet_offline, outdir=outdir)

    if experiment:
        experiment.set_name(outdir)
        experiment.log_code("mlpf/tfmodel/model.py")
        experiment.log_code("mlpf/tfmodel/utils.py")
        experiment.log_code(config_file_path)

    if jobid is not None:
        with open(f"{outdir}/{jobid}.txt", "w") as f:
            f.write(f"{jobid}\n")

    ds_train, ds_test, ds_val = get_train_test_val_datasets(config, num_batches_multiplier, ntrain, ntest, horovod_enabled)

    if config["dataset"]["enable_tfds_caching"]:
        ds_train.tensorflow_dataset = ds_train.tensorflow_dataset.cache()
        ds_test.tensorflow_dataset = ds_test.tensorflow_dataset.cache()

    ds_train.tensorflow_dataset = ds_train.tensorflow_dataset.prefetch(tf.data.AUTOTUNE)
    ds_test.tensorflow_dataset = ds_test.tensorflow_dataset.prefetch(tf.data.AUTOTUNE)

    if config["dataset"]["enable_tfds_caching"]:
        logging.info("ensuring dataset cache is hot")
        for ielem, elem in enumerate(ds_train.tensorflow_dataset):
            if ielem < 5:
                msk = elem[0][..., 0].numpy() != 0
                print("features {}".format(np.sum(msk)))
                for ifeat in range(elem[0].shape[-1]):
                    feat = elem[0][..., ifeat].numpy()[msk]
                    print(ifeat, feat.min(), feat.max(), np.mean(feat), np.std(feat))
                print("targets")
                print(elem[1])
                print("weights")
                print(elem[2])
        for elem in ds_test.tensorflow_dataset:
            pass

    epochs = config["setup"]["num_epochs"]
    total_steps = ds_train.num_steps() * epochs
    logging.info("num_train_steps: {}".format(ds_train.num_steps()))
    logging.info("num_test_steps: {}".format(ds_test.num_steps()))
    logging.info("epochs: {}, total_train_steps: {}".format(epochs, total_steps))

    if experiment:
        experiment.log_parameter("num_train_steps", ds_train.num_steps())
        experiment.log_parameter("num_test_steps", ds_test.num_steps())
        experiment.log_parameter("num_val_steps", ds_val.num_steps())

    if horovod_enabled:
        model, optim_callbacks, initial_epoch = model_scope(
            config,
            total_steps,
            weights=weights,
            horovod_enabled=horovod_enabled,
        )

        callbacks = prepare_callbacks(
            config,
            outdir,
            ds_val,
            comet_experiment=experiment,
            horovod_enabled=horovod_enabled,
            benchmark_dir=benchmark_dir,
            num_train_steps=ds_train.num_steps(),
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            train_samples=ds_train.num_samples,
        )

        # Horovod: broadcast initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or restored
        # from a checkpoint.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # For some reason, this hangs at the end of the epoch
        # callbacks.append(hvd.callbacks.MetricAverageCallback())

        callbacks.append(optim_callbacks)

        # this may crash if the other ranks can't find this file...
        if config["setup"]["use_normalizer"]:
            normalizer_cache = "{}/normalizations.npz".format(config["cache"])
            if not os.path.isfile(normalizer_cache):
                logging.error(f"Could not find normalizer cache in {normalizer_cache}" + "run once without horovod to create cache")
                return

            cache = np.load(normalizer_cache, allow_pickle=True)
            model.normalizer.mean = tf.convert_to_tensor(cache["mean"])
            model.normalizer.variance = tf.convert_to_tensor(cache["variance"])

        model.fit(
            ds_train.tensorflow_dataset.repeat(),
            validation_data=ds_test.tensorflow_dataset.repeat(),
            epochs=config["setup"]["num_epochs"],
            callbacks=callbacks,
            steps_per_epoch=ds_train.num_steps(),
            validation_steps=ds_test.num_steps(),
            initial_epoch=initial_epoch,
            verbose=1,
        )
    else:
        with strategy.scope():
            model, optim_callbacks, initial_epoch = model_scope(config, total_steps, weights=weights, horovod_enabled=False)

            callbacks = prepare_callbacks(
                config,
                outdir,
                ds_val,
                comet_experiment=experiment,
                horovod_enabled=horovod_enabled,
                benchmark_dir=benchmark_dir,
                num_train_steps=ds_train.num_steps(),
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                train_samples=ds_train.num_samples,
            )

            callbacks.append(optim_callbacks)

            if config["setup"]["use_normalizer"]:
                normalizer_cache = "{}/normalizations.npz".format(config["cache"])
                if not os.path.isfile(normalizer_cache):
                    logging.info(f"Could not find normalizer cache in {normalizer_cache}, recreating")
                    model.normalizer.adapt(
                        ds_train.tensorflow_dataset.prefetch(tf.data.AUTOTUNE).map(lambda X, y, w: X[:, :, 1:], num_parallel_calls=tf.data.AUTOTUNE)
                    )
                    print(model.normalizer.mean)
                    print(model.normalizer.variance)
                    np.savez(
                        normalizer_cache,
                        mean=model.normalizer.mean.numpy(),
                        variance=model.normalizer.variance.numpy(),
                    )

                cache = np.load(normalizer_cache, allow_pickle=True)
                model.normalizer.mean = tf.convert_to_tensor(cache["mean"])
                model.normalizer.variance = tf.convert_to_tensor(cache["variance"])

            model.fit(
                ds_train.tensorflow_dataset.repeat(),
                validation_data=ds_test.tensorflow_dataset.repeat(),
                epochs=config["setup"]["num_epochs"],
                callbacks=callbacks,
                steps_per_epoch=ds_train.num_steps(),
                validation_steps=ds_test.num_steps(),
                initial_epoch=initial_epoch,
                verbose=1,
            )


@main.command()
@click.help_option("-h", "--help")
@click.option(
    "--train-dir",
    required=True,
    help="directory containing a completed training",
    type=click.Path(),
)
@click.option("--config", help="configuration file", type=click.Path())
@click.option(
    "--weights",
    default=None,
    help="trained weights to load",
    type=click.Path(),
)
@click.option("--customize", help="customization function", type=str, default=None)
@click.option("--nevents", help="maximum number of events", type=int, default=-1)
def evaluate(config, train_dir, weights, customize, nevents):
    """Evaluate the trained model in train_dir"""
    if config is None:
        config = Path(train_dir) / "config.yaml"
        assert config.exists(), "Could not find config file in train_dir, please provide one with -c <path/to/config>"
    config, _ = parse_config(config, weights=weights)

    if customize:
        config = customization_functions[customize](config)

    # disable small graph optimization for onnx export (tf.cond is not well supported by ONNX export)
    if "small_graph_opt" in config["setup"]:
        config["setup"]["small_graph_opt"] = False

    if not weights:
        weights = get_best_checkpoint(train_dir)
        logging.info("Loading best weights that could be found from {}".format(weights))

    model, _, initial_epoch = model_scope(config, 1, weights=weights)

    if config["setup"]["use_normalizer"]:
        print("before loading")
        print(model.normalizer.mean)
        print(model.normalizer.variance)

        normalizer_cache_path = "{}/normalizations.npz".format(config["cache"])
        cache = np.load(normalizer_cache_path, allow_pickle=True)
        model.normalizer.mean = tf.convert_to_tensor(cache["mean"])
        model.normalizer.variance = tf.convert_to_tensor(cache["variance"])
        print("after loading")
        print(model.normalizer.mean)
        print(model.normalizer.variance)

    for dsname in config["evaluation_datasets"]:
        val_ds = config["evaluation_datasets"][dsname]
        ds_test = mlpf_dataset_from_config(
            dsname,
            config,
            "test",
            nevents if nevents >= 0 else val_ds["num_events"],
        )
        ds_test_tfds = ds_test.tensorflow_dataset.padded_batch(val_ds["batch_size"])
        eval_dir = str(Path(train_dir) / "evaluation" / "epoch_{}".format(initial_epoch) / dsname)
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        eval_model(model, ds_test_tfds, config, eval_dir)

    freeze_model(model, config, train_dir)  # export to ONNX


@main.command()
@click.help_option("-h", "--help")
@click.option(
    "--train-dir",
    default=None,
    help="directory containing a completed training",
    type=click.Path(),
)
@click.option("--config", help="configuration file", type=click.Path())
@click.option(
    "--weights",
    default=None,
    help="trained weights to load",
    type=click.Path(),
)
@click.option("--bs", help="batch size to use for inference", type=int, default=1)
@click.option("--customize", help="customization function", type=str, default=None)
@click.option("--nevents", help="maximum number of events", type=int, default=-1)
@click.option("--verbose", help="verbose output", type=int, default=0)
@click.option("--num-runs", help="how many times to run the inference", type=int, default=2)
@click.option("-o", "--output", help="write summary of results to file", type=Path)
@click.option("--cpus", help="CPU threads", type=int, default=None)
def infer(config, train_dir, weights, bs, customize, nevents, verbose, num_runs, output, cpus):
    import json

    strategy, num_gpus, num_batches_multiplier = get_singlenode_strategy(num_cpus=cpus)  # sets TF ENV variables to use num_cpus
    assert num_gpus < 2, "Multi-GPU inference is not supported"

    if output:
        assert num_runs > 1, "If writing summary results to file, num_runs must be >1"

    if train_dir is None:
        assert (config is not None) and (weights is not None), "Please provide a config and weight file when not giving train_dir"

    if config is None:
        config = Path(train_dir) / "config.yaml"
        assert config.exists(), "Could not find config file in train_dir, please provide one with -c <path/to/config>"
    config, _ = parse_config(config, weights=weights)

    if customize:
        config = customization_functions[customize](config)

    # disable small graph optimization for onnx export (tf.cond is not well supported by ONNX export)
    if "small_graph_opt" in config["setup"]:
        config["setup"]["small_graph_opt"] = False

    if not weights:
        weights = get_best_checkpoint(train_dir)
        logging.info("Loading best weights that could be found from {}".format(weights))

    model, _, initial_epoch = model_scope(config, 1, weights=weights)

    if config["setup"]["use_normalizer"]:
        print("before loading")
        print("model.normalizer.mean:", model.normalizer.mean)
        print("model.normalizer.variance:", model.normalizer.variance)

        normalizer_cache_path = "{}/normalizations.npz".format(config["cache"])
        cache = np.load(normalizer_cache_path, allow_pickle=True)
        model.normalizer.mean = tf.convert_to_tensor(cache["mean"])
        model.normalizer.variance = tf.convert_to_tensor(cache["variance"])
        print("after loading")
        print("model.normalizer.mean:", model.normalizer.mean)
        print("model.normalizer.variance:", model.normalizer.variance)

    num_events = nevents if nevents >= 0 else config["validation_num_events"]
    ds_val = mlpf_dataset_from_config(
        config["validation_dataset"],
        config,
        "test",
        num_events,
    )
    tfds_dataset = ds_val.tensorflow_dataset.padded_batch(bs)

    times = []
    predict_workers = bs  # 1 worker per sample in the batch
    # TODO: don't hardcode maximum allowed workers
    if predict_workers > 112:
        predict_workers = 112  # ensure workers is not more than available cpu threads
    for i in range(num_runs):
        # Using model.predict(tf_dataset) doesn't work because pfnetdense is not hashable due to the use of slicing in
        # __call__. Hence, we have to loop through the dataset.
        print("\nRun {}/{}".format(i + 1, num_runs))
        print("####### Inference using model.predict(sample) ############")
        start_time = tf.timestamp().numpy()
        for elem in tqdm.tqdm(tfds_dataset, desc="Model inference"):
            _ = model.predict(
                elem["X"],
                verbose=verbose,
                workers=predict_workers,
                use_multiprocessing=(predict_workers > 1),
            )
        stop_time = tf.timestamp().numpy()
        total_time = stop_time - start_time
        times.append(total_time)
        print("Total number of events used: {:d}".format(num_events))
        print("Batch size: {:d}".format(bs))
        print("Total inference time: {:.2f}s".format(total_time))
        print("##########################################################")

    if num_runs > 1:
        # Summarizing results

        # event throughput [1/s]
        #   - ignore batch padding
        throughput_per_run = num_events / np.array(times)

        # mean throughput
        #   - ignore first epoch (lazy graph construction)
        mean_throughput = round(np.mean(throughput_per_run[1:]), 4)
        print("mean_throughput:", mean_throughput)

        # mean epoch time
        #   - ignore first epoch (lazy graph construction)
        mean_run_time = round(np.mean(times[1:]), 4)
        # batch_size_total = bs * (num_gpus or num_cpus)
        print("mean_run_time:", mean_run_time)

        data = {
            "results": [
                {
                    "wl-scores": {
                        "mean_throughput": mean_throughput,
                        "mean_run_time": mean_run_time,
                    },
                    "wl-stats": {
                        "num_runs": len(times),
                        "run_times": np.round(times, 4),
                        "total_inference_time": round(sum(times[1:]), 4),
                        "GPU": num_gpus,
                        "CPU": cpus or -1,
                        # "train_set_size": self.train_set_size,
                        # "batch_size_per_device": self.batch_size_per_gpu,
                        # "batch_size_total": batch_size_total,
                        "batch_size": bs,
                        "steps_per_run": num_events // bs,
                        "events_per_run": num_events,
                        "throughput_per_run": list(np.round(throughput_per_run, 4)),
                    },
                }
            ],
        }

        if output:
            result_path = output.resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)

            if result_path.is_file():
                with result_path.open("r", encoding="utf-8") as f:
                    old_data = json.load(f)
            else:
                old_data = None

            with result_path.open("w", encoding="utf-8") as f:
                if old_data:
                    data = {"results": old_data["results"] + data["results"]}
                json.dump(data, f, ensure_ascii=False, indent=4, cls=NpEncoder)
                f.write("\n")
            print("Saved result to {}".format(result_path))


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option(
    "-o",
    "--outdir",
    help="output directory",
    type=click.Path(),
    default=".",
)
@click.option(
    "-n",
    "--figname",
    help="name of saved figure",
    type=click.Path(),
    default="lr_finder.jpg",
)
@click.option(
    "-l",
    "--logscale",
    help="use log scale on y-axis in figure",
    default=False,
    is_flag=True,
)
def find_lr(config, outdir, figname, logscale):
    """Run the Learning Rate Finder to produce a batch loss vs. LR plot from
    which an appropriate LR-range can be determined"""
    config, _ = parse_config(config)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, num_gpus, num_batches_multiplier = get_singlenode_strategy()

    ds_train = get_datasets(
        config["train_test_datasets"],
        config,
        num_batches_multiplier,
        "train",
    )

    with strategy.scope():
        model, _, initial_epoch = model_scope(config, 1)
        max_steps = 200
        lr_finder = LRFinder(max_steps=max_steps)
        callbacks = [lr_finder]

        model.fit(
            ds_train.tensorflow_dataset.repeat(),
            epochs=max_steps,
            callbacks=callbacks,
            steps_per_epoch=1,
        )

        lr_finder.plot(save_dir=outdir, figname=figname, log_scale=logscale)


@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", help="training directory", type=click.Path())
@click.option(
    "-d",
    "--dry_run",
    help="do not delete anything",
    is_flag=True,
    default=False,
)
def delete_all_but_best_ckpt(train_dir, dry_run):
    """Delete all checkpoint weights in <train_dir>/weights/ except the one with lowest loss in its filename."""
    delete_all_but_best_checkpoint(train_dir, dry_run)


@main.command()
@click.help_option("-h", "--help")
@click.option(
    "-c",
    "--config",
    help="configuration file",
    type=click.Path(),
    required=True,
)
@click.option("-o", "--outdir", help="output dir", type=click.Path(), required=True)
@click.option(
    "--ntrain",
    default=None,
    help="override the number of training events",
    type=int,
)
@click.option(
    "--ntest",
    default=None,
    help="override the number of testing events",
    type=int,
)
@click.option(
    "-r",
    "--recreate",
    help="overwrite old hypertune results",
    is_flag=True,
    default=False,
)
@click.option("--num-cpus", help="number of CPU threads to use", type=int, default=1)
def hypertune(config, outdir, ntrain, ntest, recreate, num_cpus):
    config_file_path = config
    config, _ = parse_config(config, ntrain=ntrain, ntest=ntest)

    # Override number of epochs with max_epochs from Hyperband config if specified
    if config["hypertune"]["algorithm"] == "hyperband":
        config["setup"]["num_epochs"] = config["hypertune"]["hyperband"]["max_epochs"]

    strategy, num_gpus, num_batches_multiplier = get_singlenode_strategy(num_cpus=num_cpus)

    ds_train, ds_test, ds_val = get_train_test_val_datasets(config, num_batches_multiplier, ntrain, ntest)

    model_builder, optim_callbacks = hypertuning.get_model_builder(config, ds_train.num_steps())

    callbacks = prepare_callbacks(
        config,
        outdir,
        ds_val,
    )

    for cb in optim_callbacks:
        callbacks.append(cb)
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=20, monitor="val_loss"))

    tuner = get_tuner(config["hypertune"], model_builder, outdir, recreate, strategy)
    tuner.search_space_summary()

    from tensorflow.keras.callbacks import TensorBoard

    tuner.search(
        ds_train.tensorflow_dataset.repeat(),
        epochs=config["setup"]["num_epochs"],
        validation_data=ds_test.tensorflow_dataset.repeat(),
        steps_per_epoch=ds_train.num_steps(),
        validation_steps=ds_test.num_steps(),
        callbacks=[TensorBoard(log_dir=outdir + "/logs")],
    )
    logging.info("Hyperparameter search complete.")
    shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    tuner.results_summary()
    for trial in tuner.oracle.get_best_trials(num_trials=10):
        print(trial.hyperparameters.values, trial.score)


#
# Raytune part
#


def raytune_build_model_and_train(
    config,
    checkpoint_dir=None,
    full_config=None,
    ntrain=None,
    ntest=None,
    name=None,
    seeds=False,
    comet_online=False,
    comet_exp_name="particleflow-raytune",
    nepochs=None,
    num_cpus=None,
):
    from collections import Counter

    from ray import tune
    from ray.tune.integration.keras import TuneReportCheckpointCallback
    from raytune.search_space import set_raytune_search_parameters

    if seeds:
        # Set seeds for reproducibility
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    config_file_path = full_config
    full_config, config_file_stem = parse_config(full_config, nepochs=nepochs)

    if config is not None:
        full_config = set_raytune_search_parameters(search_space=config, config=full_config)

    experiment = create_comet_experiment(comet_exp_name, comet_offline=(not comet_online), outdir=tune.get_trial_dir())

    if experiment:
        experiment.set_name(tune.get_trial_dir())
        experiment.log_code("mlpf/tfmodel/model.py")
        experiment.log_code("mlpf/tfmodel/utils.py")
        experiment.log_code(config_file_path)

    strategy, num_gpus, num_batches_multiplier = get_singlenode_strategy(num_cpus=num_cpus)
    ds_train, ds_test, ds_val = get_train_test_val_datasets(full_config, num_batches_multiplier, ntrain, ntest)

    logging.info("num_train_steps", ds_train.num_steps())
    logging.info("num_test_steps", ds_test.num_steps())
    total_steps = ds_train.num_steps() * full_config["setup"]["num_epochs"]
    logging.info("total_steps", total_steps)

    with strategy.scope():
        weights = get_latest_checkpoint(Path(checkpoint_dir).parent) if (checkpoint_dir is not None) else None
        model, optim_callbacks, initial_epoch = model_scope(full_config, total_steps, weights=weights)

    callbacks = prepare_callbacks(
        full_config,
        tune.get_trial_dir(),
        ds_val,
        comet_experiment=experiment,
        horovod_enabled=False,
        is_hpo_run=True,
    )

    callbacks.append(optim_callbacks)

    # if metrics=None all Keras logs should be logged to the Tune logs
    tune_report_checkpoint_callback = TuneReportCheckpointCallback(
        metrics=[
            # "adam_beta_1",
            "charge_loss",
            # "cls_acc_unweighted",
            "cls_loss",
            "cos_phi_loss",
            "energy_loss",
            "eta_loss",
            "learning_rate",
            "loss",
            "pt_loss",
            "sin_phi_loss",
            "val_charge_loss",
            # "val_cls_acc_unweighted",
            # "val_cls_acc_weighted",
            "val_cls_loss",
            "val_cos_phi_loss",
            "val_energy_loss",
            "val_eta_loss",
            "val_loss",
            "val_pt_loss",
            "val_sin_phi_loss",
            "val_jet_wd",
            "val_jet_iqr",
            "val_jet_med",
            "val_met_wd",
            "val_met_iqr",
            "val_met_med",
        ],
    )

    # To make TuneReportCheckpointCallback continue the numbering of checkpoints correctly
    if weights is not None:
        latest_saved_checkpoint_number = int(Path(weights).name.split("-")[1])
        logging.info("setting TuneReportCheckpointCallback epoch number to {}".format(latest_saved_checkpoint_number))
        tune_report_checkpoint_callback._checkpoint._counter = Counter()
        tune_report_checkpoint_callback._checkpoint._counter["epoch_end"] = latest_saved_checkpoint_number
        tune_report_checkpoint_callback._checkpoint._cp_count = latest_saved_checkpoint_number
    callbacks.append(tune_report_checkpoint_callback)

    try:
        model.fit(
            ds_train.tensorflow_dataset.repeat(),
            validation_data=ds_test.tensorflow_dataset.repeat(),
            epochs=full_config["setup"]["num_epochs"],
            callbacks=callbacks,
            steps_per_epoch=ds_train.num_steps(),
            validation_steps=ds_test.num_steps(),
            initial_epoch=initial_epoch,
        )
    except tf.errors.ResourceExhaustedError:
        logging.warning("Resource exhausted, skipping this hyperparameter configuration.")
        skiplog_file_path = Path(full_config["raytune"]["local_dir"]) / name / "skipped_configurations.txt"
        lines = ["{}: {}\n".format(item[0], item[1]) for item in config.items()]

        with open(skiplog_file_path, "a") as f:
            f.write("#" * 80 + "\n")
            for line in lines:
                f.write(line)
                logging.warning(line[:-1])
            f.write("#" * 80 + "\n\n")


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-n", "--name", help="experiment name", type=str, default="test_exp")
@click.option("-l", "--local", help="run locally", is_flag=True)
@click.option("--cpus", help="number of cpus per worker", type=int, default=1)
@click.option("--gpus", help="number of gpus per worker", type=int, default=0)
@click.option("--tune_result_dir", help="Tune result dir", type=str, default=None)
@click.option("-r", "--resume", help="resume run from local_dir", is_flag=True)
@click.option(
    "--ntrain",
    default=None,
    help="override the number of training steps",
    type=int,
)
@click.option(
    "--ntest",
    default=None,
    help="override the number of testing steps",
    type=int,
)
@click.option(
    "--nepochs",
    default=None,
    help="override the number of training epochs",
    type=int,
)
@click.option("-s", "--seeds", help="set the random seeds", is_flag=True)
@click.option("--comet-online", help="use comet-ml online logging", is_flag=True)
@click.option(
    "--comet-exp-name",
    help="comet experiment name",
    type=str,
    default="particleflow-raytune",
)
def raytune(
    config,
    name,
    local,
    cpus,
    gpus,
    tune_result_dir,
    resume,
    ntrain,
    ntest,
    nepochs,
    seeds,
    comet_online,
    comet_exp_name,
):
    import ray
    from ray import tune
    from ray.tune.logger import TBXLoggerCallback
    from raytune.search_space import raytune_num_samples, search_space
    from raytune.utils import get_raytune_schedule, get_raytune_search_alg

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"  # don't crash if a metric is missing

    if seeds:
        # Set seeds for reproducibility
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    cfg = load_config(config)
    config_file_path = config

    if tune_result_dir is not None:
        os.environ["TUNE_RESULT_DIR"] = tune_result_dir
    else:
        if isinstance(cfg["raytune"]["local_dir"], type(None)):
            raise TypeError("Please specify a local_dir in the raytune section of the config file.")
        trd = cfg["raytune"]["local_dir"] + "/tune_result_dir"
        os.environ["TUNE_RESULT_DIR"] = trd

    expdir = Path(cfg["raytune"]["local_dir"]) / name
    expdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "mlpf/raytune/search_space.py",
        str(Path(cfg["raytune"]["local_dir"]) / name / "search_space.py"),
    )  # Copy the config file to the train dir for later reference
    shutil.copy(
        config_file_path,
        str(Path(cfg["raytune"]["local_dir"]) / name / "config.yaml"),
    )  # Copy the config file to the train dir for later reference

    if not local:
        ray.init(address="auto")

    sched = get_raytune_schedule(cfg["raytune"])
    search_alg = get_raytune_search_alg(cfg["raytune"], seeds)

    sync_config = tune.SyncConfig(sync_to_driver=False)

    start = datetime.now()
    analysis = tune.run(
        partial(
            raytune_build_model_and_train,
            full_config=config_file_path,
            ntrain=ntrain,
            ntest=ntest,
            name=name,
            seeds=seeds,
            comet_online=comet_online,
            comet_exp_name=comet_exp_name,
            nepochs=nepochs,
            num_cpus=cpus,
        ),
        config=search_space,
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        name=name,
        scheduler=sched,
        search_alg=search_alg,
        num_samples=raytune_num_samples,
        local_dir=cfg["raytune"]["local_dir"],
        callbacks=[TBXLoggerCallback()],
        log_to_file=True,
        resume=resume,
        max_failures=2,
        sync_config=sync_config,
        # stop=tune.stopper.MaximumIterationStopper(cfg["setup"]["num_epochs"]),
    )
    end = datetime.now()
    logging.info("Total time of tune.run(...): {}".format(end - start))

    logging.info(
        "Best hyperparameters found according to {} were: ".format(cfg["raytune"]["default_metric"]),
        analysis.get_best_config(
            cfg["raytune"]["default_metric"],
            cfg["raytune"]["default_mode"],
        ),
    )

    skip = 20
    if skip > cfg["setup"]["num_epochs"]:
        skip = 0
    analysis.default_metric = cfg["raytune"]["default_metric"]
    analysis.default_mode = cfg["raytune"]["default_mode"]
    plot_ray_analysis(analysis, save=True, skip=skip)
    topk_summary_plot_v2(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)
    summarize_top_k(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)

    best_params = analysis.get_best_config(cfg["raytune"]["default_metric"], cfg["raytune"]["default_mode"])
    with open(
        Path(analysis.get_best_logdir()).parent / "best_parameters.txt",
        "a",
    ) as best_params_file:
        best_params_file.write("Best hyperparameters according to {}\n".format(cfg["raytune"]["default_metric"]))
        for key, val in best_params.items():
            best_params_file.write(("{}: {}\n".format(key, val)))

    with open(Path(analysis.get_best_logdir()).parent / "time.txt", "a") as timefile:
        timefile.write(str(end - start) + "\n")

    num_skipped = count_skipped_configurations(analysis.get_best_logdir())
    logging.info("Number of skipped configurations: {}".format(num_skipped))


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
def count_skipped(exp_dir):
    num_skipped = count_skipped_configurations(exp_dir)
    logging.info("Number of skipped configurations: {}".format(num_skipped))


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
@click.option("-s", "--save", help="save plots in trial dirs", is_flag=True)
@click.option(
    "-k",
    "--skip",
    help="skip first values to avoid large losses at start of training",
    type=int,
)
@click.option("--metric", help="experiment dir", type=str, default="val_loss")
@click.option("--mode", help="experiment dir", type=str, default="min")
def raytune_analysis(exp_dir, save, skip, mode, metric):
    from ray.tune import ExperimentAnalysis

    experiment_analysis = ExperimentAnalysis(exp_dir, default_metric=metric, default_mode=mode)
    plot_ray_analysis(experiment_analysis, save=save, skip=skip)
    analyze_ray_experiment(exp_dir, default_metric=metric, default_mode=mode)


@main.command()
@click.option("-c", "--config", help="configuration file", type=click.Path())
def test_datasets(config):
    from scipy.sparse import coo_matrix

    config_file_path = config
    config, config_file_stem = parse_config(config)
    logging.info(f"loaded config file: {config_file_path}")

    histograms = {}

    dataset_sizes = {"train": {}, "test": {}}
    for dataset_name in config["datasets"]:
        print(dataset_name)

        for split in ["train", "test"]:
            ds = mlpf_dataset_from_config(dataset_name, config, split)
            dataset = dataset_name + "_" + split
            print(dataset, ds.num_steps(), ds.num_samples)
            dataset_sizes[split][dataset_name] = ds.num_samples

            continue
            confusion_matrix_Xelem_to_ygen = np.zeros(
                (
                    config["dataset"]["num_input_classes"],
                    config["dataset"]["num_output_classes"],
                ),
                dtype=np.int64,
            )

            histograms[dataset] = {}
            histograms[dataset]["gen_energy"] = bh.Histogram(bh.axis.Regular(100, 0, 5000))
            histograms[dataset]["gen_energy_log"] = bh.Histogram(bh.axis.Regular(100, -1, 5))
            histograms[dataset]["cand_energy"] = bh.Histogram(bh.axis.Regular(100, 0, 5000))
            histograms[dataset]["cand_energy_log"] = bh.Histogram(bh.axis.Regular(100, -1, 5))

            histograms[dataset]["gen_eta_energy"] = bh.Histogram(bh.axis.Regular(100, -6, 6))
            histograms[dataset]["cand_eta_energy"] = bh.Histogram(bh.axis.Regular(100, -6, 6))

            histograms[dataset]["gen_pt"] = bh.Histogram(bh.axis.Regular(100, 0, 5000))
            histograms[dataset]["gen_pt_log"] = bh.Histogram(bh.axis.Regular(100, -1, 5))
            histograms[dataset]["cand_pt"] = bh.Histogram(bh.axis.Regular(100, 0, 5000))
            histograms[dataset]["cand_pt_log"] = bh.Histogram(bh.axis.Regular(100, -1, 5))

            histograms[dataset]["sum_gen_cand_energy"] = bh.Histogram(
                bh.axis.Regular(100, 0, 100000),
                bh.axis.Regular(100, 0, 100000),
            )
            histograms[dataset]["sum_gen_cand_energy_log"] = bh.Histogram(bh.axis.Regular(100, 2, 6), bh.axis.Regular(100, 2, 6))

            histograms[dataset]["sum_gen_cand_pt"] = bh.Histogram(
                bh.axis.Regular(100, 0, 100000),
                bh.axis.Regular(100, 0, 100000),
            )
            histograms[dataset]["sum_gen_cand_pt_log"] = bh.Histogram(bh.axis.Regular(100, 2, 6), bh.axis.Regular(100, 2, 6))

            histograms[dataset]["confusion_matrix_Xelem_to_ygen"] = confusion_matrix_Xelem_to_ygen

            for elem in tqdm.tqdm(ds.tensorflow_dataset, total=ds.num_steps()):
                X = elem["X"].numpy()
                ygen = elem["ygen"].numpy()
                ycand = elem["ycand"].numpy()
                # print(X.shape, ygen.shape, ycand.shape)

                # check that all elements in the event have a nonzero type
                assert np.sum(X[:, 0] == 0) == 0
                assert X.shape[0] == ygen.shape[0]
                assert X.shape[0] == ycand.shape[0]
                # assert X.shape[1] == config["dataset"]["num_input_features"]
                # assert ygen.shape[1] == config["dataset"]["num_output_features"] + 1
                # assert ycand.shape[1] == config["dataset"]["num_output_features"] + 1

                histograms[dataset]["confusion_matrix_Xelem_to_ygen"] += coo_matrix(
                    (
                        np.ones(len(X), dtype=np.int64),
                        (
                            np.array(X[:, 0], np.int32),
                            np.array(ygen[:, 0], np.int32),
                        ),
                    ),
                    shape=(
                        config["dataset"]["num_input_classes"],
                        config["dataset"]["num_output_classes"],
                    ),
                ).todense()

                vals_ygen = ygen[ygen[:, 0] != 0]
                vals_ygen = unpack_target(
                    vals_ygen,
                    config["dataset"]["num_output_classes"],
                    config,
                )
                # assert np.all(vals_ygen["energy"] > 0)
                # assert np.all(vals_ygen["pt"] > 0)
                # assert not np.any(np.isinf(ygen))
                # assert not np.any(np.isnan(ygen))

                histograms[dataset]["gen_energy"].fill(vals_ygen["energy"][:, 0])
                histograms[dataset]["gen_energy_log"].fill(np.log10(vals_ygen["energy"][:, 0]))
                histograms[dataset]["gen_pt"].fill(vals_ygen["pt"][:, 0])
                histograms[dataset]["gen_pt_log"].fill(np.log10(vals_ygen["pt"][:, 0]))
                histograms[dataset]["gen_eta_energy"].fill(
                    vals_ygen["eta"][:, 0],
                    weight=vals_ygen["energy"][:, 0],
                )

                vals_ycand = ycand[ycand[:, 0] != 0]
                vals_ycand = unpack_target(
                    vals_ycand,
                    config["dataset"]["num_output_classes"],
                    config,
                )
                # assert(np.all(vals_ycand["energy"]>0))
                # assert(np.all(vals_ycand["pt"]>0))
                # assert not np.any(np.isinf(ycand))
                # assert not np.any(np.isnan(ycand))

                histograms[dataset]["cand_energy"].fill(vals_ycand["energy"][:, 0])
                histograms[dataset]["cand_energy_log"].fill(np.log10(vals_ycand["energy"][:, 0]))
                histograms[dataset]["cand_pt"].fill(vals_ycand["pt"][:, 0])
                histograms[dataset]["cand_pt_log"].fill(np.log10(vals_ycand["pt"][:, 0]))
                histograms[dataset]["cand_eta_energy"].fill(
                    vals_ycand["eta"][:, 0],
                    weight=vals_ycand["energy"][:, 0],
                )

                histograms[dataset]["sum_gen_cand_energy"].fill(
                    np.sum(vals_ygen["energy"]),
                    np.sum(vals_ycand["energy"]),
                )
                histograms[dataset]["sum_gen_cand_energy_log"].fill(
                    np.log10(np.sum(vals_ygen["energy"])),
                    np.log10(np.sum(vals_ycand["energy"])),
                )
                histograms[dataset]["sum_gen_cand_pt"].fill(np.sum(vals_ygen["pt"]), np.sum(vals_ycand["pt"]))
                histograms[dataset]["sum_gen_cand_pt_log"].fill(
                    np.log10(np.sum(vals_ygen["pt"])),
                    np.log10(np.sum(vals_ycand["pt"])),
                )

            print(confusion_matrix_Xelem_to_ygen)

    for dsname, dsval in sorted(dataset_sizes["train"].items(), reverse=True, key=lambda x: x[1]):
        print("{}: {:.1E} {:.1E}".format(dsname, dsval, dataset_sizes["test"][dsname]))

    with open("datasets.pkl", "wb") as fi:
        pickle.dump(histograms, fi)


@main.command()
@click.help_option("-h", "--help")
@click.option(
    "--train-dir",
    required=True,
    help="directory containing a completed training",
    type=click.Path(),
)
@click.option(
    "--max-files",
    required=False,
    help="maximum number of files per dataset to load",
    type=int,
    default=None,
)
def plots(train_dir, max_files):
    import mplhep
    from plotting.plot_utils import (
        compute_met_and_ratio,
        format_dataset_name,
        load_eval_data,
        plot_jet_ratio,
        plot_met,
        plot_met_ratio,
        plot_num_elements,
        plot_particles,
        plot_sum_energy,
        load_loss_history,
        loss_plot,
        plot_jet_response_binned,
        plot_jet_response_binned_separate,
        plot_jet_response_binned_eta,
        plot_met_response_binned,
        get_class_names,
        plot_rocs,
        plot_particle_multiplicity,
        compute_3dmomentum_and_ratio,
        plot_3dmomentum_ratio,
        plot_3dmomentum_response_binned,
    )

    mplhep.set_style(mplhep.styles.CMS)

    eval_dir = Path(train_dir) / "evaluation"

    history = load_loss_history(str(Path(train_dir) / "history/history_*.json"))
    for loss in [
        "loss",
        "cls_loss",
        "pt_loss",
        "energy_loss",
        "eta_loss",
        "sin_phi_loss",
        "cos_phi_loss",
        "charge_loss",
    ]:
        loss_plot(
            history[loss].values,
            history["val_" + loss].values,
            loss + ".png",
            margin=0.5,
            smoothing=True,
            cp_dir=Path(train_dir),
            title=loss,
        )

    for epoch_dir in sorted(os.listdir(str(eval_dir))):
        eval_epoch_dir = eval_dir / epoch_dir
        for dataset in sorted(os.listdir(str(eval_epoch_dir))):
            class_names = get_class_names(dataset)

            _title = format_dataset_name(dataset)
            dataset_dir = eval_epoch_dir / dataset
            print(dataset_dir)
            cp_dir = dataset_dir / "plots"
            if not os.path.isdir(str(cp_dir)):
                os.makedirs(str(cp_dir))
            yvals, X, _ = load_eval_data(str(dataset_dir / "*.parquet"), max_files)

            plot_num_elements(X, cp_dir=cp_dir, title=_title)
            plot_sum_energy(yvals, class_names, cp_dir=cp_dir, title=_title)
            plot_particle_multiplicity(X, yvals, class_names, cp_dir=cp_dir, title=_title)
            plot_rocs(yvals, class_names, cp_dir=cp_dir, title=_title)

            plot_jet_ratio(
                yvals,
                cp_dir=cp_dir,
                title=_title,
                bins=np.linspace(0, 5, 100),
                logy=True,
            )
            plot_jet_ratio(
                yvals,
                cp_dir=cp_dir,
                title=_title,
                bins=np.linspace(0.5, 1.5, 100),
                logy=False,
                file_modifier="_bins_0p5_1p5",
            )

            met_data = compute_met_and_ratio(yvals)
            plot_met(met_data, cp_dir=cp_dir, title=_title)
            plot_met_ratio(
                met_data,
                cp_dir=cp_dir,
                title=_title,
                bins=np.linspace(0, 20, 100),
                logy=True,
            )
            plot_met_ratio(
                met_data,
                cp_dir=cp_dir,
                title=_title,
                bins=np.linspace(0, 2, 100),
                logy=False,
                file_modifier="_bins_0_2",
            )
            plot_met_ratio(
                met_data,
                cp_dir=cp_dir,
                title=_title,
                bins=np.linspace(0, 5, 100),
                logy=False,
                file_modifier="_bins_0_5",
            )

            plot_particles(yvals, cp_dir=cp_dir, title=_title)

            plot_jet_response_binned(yvals, cp_dir=cp_dir, title=_title)
            plot_jet_response_binned_eta(yvals, cp_dir=cp_dir, title=_title)
            plot_jet_response_binned_separate(yvals, cp_dir=cp_dir, title=_title)
            plot_met_response_binned(met_data, cp_dir=cp_dir, title=_title)

            mom_data = compute_3dmomentum_and_ratio(yvals)
            plot_3dmomentum_ratio(mom_data, cp_dir=cp_dir, title=_title, bins=np.linspace(0, 20, 100), logy=True)
            plot_3dmomentum_ratio(mom_data, cp_dir=cp_dir, title=_title, bins=np.linspace(0, 2, 100), logy=True, file_modifier="_bins_0_2")
            plot_3dmomentum_ratio(mom_data, cp_dir=cp_dir, title=_title, bins=np.linspace(0, 5, 100), logy=True, file_modifier="_bins_0_5")
            plot_3dmomentum_response_binned(mom_data, cp_dir=cp_dir, title=_title)


if __name__ == "__main__":
    main()
