from .model import PFNet, Transformer, DummyNet, PFNetDense

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
import kerastuner as kt
from argparse import Namespace
import time
import json
import random
import platform

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
    def __init__(self, outpath, X, y, dataset_transform, num_output_classes):
        super(CustomCallback, self).__init__()
        self.X = X
        self.y = y
        self.dataset_transform = dataset_transform
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

    def on_epoch_end(self, epoch, logs=None):

        with open("{}/history_{}.json".format(self.outpath, epoch), "w") as fi:
            json.dump(logs, fi)

        ypred = self.model(self.X, training=False)
        #ypred["cls"] = np.clip(ypred["cls"], 0.5, 1.0)
        
        ypred_id = np.argmax(ypred["cls"], axis=-1)

        ibatch = 0
       
        msk = self.X[:, :, 0] != 0
        # cm = sklearn.metrics.confusion_matrix(
        #     self.y[msk][:, 0].astype(np.int64).flatten(),
        #     ypred_id[msk].flatten(), labels=list(range(self.num_output_classes))
        # )
        # figure = plot_confusion_matrix(cm)
        # plt.savefig("{}/cm_{}.pdf".format(self.outpath, epoch), bbox_inches="tight")
        # plt.close("all")

        cm = sklearn.metrics.confusion_matrix(
            self.y[msk][:, 0].astype(np.int64).flatten(),
            ypred_id[msk].flatten(), labels=list(range(self.num_output_classes)), normalize="true"
        )
        figure = plot_confusion_matrix(cm)

        acc = sklearn.metrics.accuracy_score(
            self.y[msk][:, 0].astype(np.int64).flatten(),
            ypred_id[msk].flatten()
        )
        balanced_acc = sklearn.metrics.balanced_accuracy_score(
            self.y[msk][:, 0].astype(np.int64).flatten(),
            ypred_id[msk].flatten()
        )
        plt.title("acc={:.3f} bacc={:.3f}".format(acc, balanced_acc))
        plt.savefig("{}/cm_normed_{}.pdf".format(self.outpath, epoch), bbox_inches="tight")
        plt.close("all")

        # for icls in range(self.num_output_classes):
        #     fig = plt.figure(figsize=(4,4))
        #     msk = self.y[:, :, 0] == icls
        #     msk = msk.flatten()
        #     b = np.linspace(0,1,21)
        #     ids = ypred["cls"][:, :, icls].numpy().flatten()
        #     plt.hist(ids[msk], bins=b, density=True, histtype="step", lw=2)
        #     plt.hist(ids[~msk], bins=b, density=True, histtype="step", lw=2)
        #     plt.savefig("{}/cls{}_{}.pdf".format(self.outpath, icls, epoch), bbox_inches="tight")
        # for icls in range(self.num_output_classes):
        #     n_pred = np.sum(self.y[:, :, 0]==icls, axis=1)
        #     n_true = np.sum(ypred_id==icls, axis=1)
        #     figure = plot_num_particle(n_pred, n_true, icls)
        #     plt.savefig("{}/num_cls{}_{}.pdf".format(self.outpath, icls, epoch), bbox_inches="tight")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*5, 5))

        plt.axes(ax1)
        msk = self.X[ibatch, :, 0] != 0
        eta = self.X[ibatch][msk][:, 2]
        phi = self.X[ibatch][msk][:, 3]
        energy = self.X[ibatch][msk][:, 4]
        typ = self.X[ibatch][msk][:, 0]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in typ], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        plt.axes(ax3)
        #Plot the predicted particles
        msk = ypred_id[ibatch] != 0
        eta = ypred["eta"][ibatch][msk]
        sphi = ypred["sin_phi"][ibatch][msk]
        cphi = ypred["cos_phi"][ibatch][msk]
        phi = np.arctan2(sphi, cphi)
        energy = ypred["energy"][ibatch][msk]
        pdgid = ypred_id[ibatch][msk]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in pdgid], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        # Xconcat = np.concatenate([self.X[ibatch], ypred["cls"][ibatch]], axis=-1)
        # np.savez(self.outpath + "/event_{}.npz".format(epoch), Xconcat[Xconcat[:, 0]!=0])

        #Plot the target particles
        plt.axes(ax2)
        y = self.dataset_transform(self.X, self.y, None)[1]
        y_id = np.argmax(y["cls"], axis=-1)
        msk = y_id[ibatch] != 0
        eta = y["eta"][ibatch][msk]
        sphi = y["sin_phi"][ibatch][msk]
        cphi = y["cos_phi"][ibatch][msk]
        phi = np.arctan2(sphi, cphi)
        energy = y["energy"][ibatch][msk]
        pdgid = y_id[ibatch][msk]
        plt.scatter(eta, phi, marker="o", s=energy, c=[self.color_map[p] for p in pdgid], alpha=0.5, linewidths=0)
        plt.xlim(-8,8)
        plt.ylim(-4,4)

        plt.savefig("{}/event_{}.pdf".format(self.outpath, epoch), bbox_inches="tight")
        plt.close("all")

        np.savez("{}/pred_{}.npz".format(self.outpath, epoch), X=self.X, ytrue=self.y, **ypred)

def prepare_callbacks(model, outdir, X_val, y_val, dataset_transform, num_output_classes):
    callbacks = []
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=outdir, histogram_freq=1, write_graph=False, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,90),
        profile_batch=0,
    )
    tb.set_model(model)
    callbacks += [tb]

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir + "/weights-{epoch:02d}-{val_loss:.6f}.hdf5",
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model)
    callbacks += [cp_callback]

    cb = CustomCallback(outdir, X_val, y_val, dataset_transform, num_output_classes)
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

def make_weight_function(config):
    def weight_func(X,y,w):

        w_signal_only = tf.where(y[:, 0]==0, 0.0, 1.0)
        w_signal_only *= tf.cast(X[:, 0]!=0, tf.float32)

        w_none = tf.ones_like(w)
        w_none *= tf.cast(X[:, 0]!=0, tf.float32)

        w_invsqrt = tf.cast(tf.shape(w)[-1], tf.float32)/tf.sqrt(w)
        w_invsqrt *= tf.cast(X[:, 0]!=0, tf.float32)

        weight_d = {
            "none": w_none,
            "signal_only": w_signal_only,
            "inverse_sqrt": w_invsqrt
        }

        ret_w = {}
        for loss_component, weight_type in config["sample_weights"].items():
            ret_w[loss_component] = weight_d[weight_type]

        return X,y,ret_w
    return weight_func

def scale_outputs(X,y,w):
    ynew = y-out_m
    ynew = ynew/out_s
    return X, ynew, w

def targets_multi_output(num_output_classes):
    def func(X, y, w):
        return X, {
            "cls": tf.one_hot(tf.cast(y[:, :, 0], tf.int32), num_output_classes), 
            "charge": y[:, :, 1:2],
            "pt": y[:, :, 2:3],
            "eta": y[:, :, 3:4],
            "sin_phi": y[:, :, 4:5],
            "cos_phi": y[:, :, 5:6],
            "energy": y[:, :, 6:7],
        }, w
    return func

def make_model(config, dtype):
    model = config['parameters']['model']
    if model == 'gnn':
        return make_gnn(config, dtype)
    elif model == 'transformer':
        return make_transformer(config, dtype)
    elif model == 'dense':
        return make_dense(config, dtype)
    elif model == 'gnn_dense':
        return make_gnn_dense(config, dtype)
    raise KeyError("Unknown model type {}".format(model))

def make_gnn(config, dtype):
    activation = getattr(tf.nn, config['parameters']['activation'])

    parameters = [
        'bin_size',
        'num_convs_id',
        'num_convs_reg',
        'num_hidden_id_enc',
        'num_hidden_id_dec',
        'num_hidden_reg_enc',
        'num_hidden_reg_dec',
        'num_neighbors',
        'hidden_dim_id',
        'hidden_dim_reg',
        'dist_mult',
        'distance_dim',
        'dropout',
        'skip_connection'
    ]
    kwargs = {par: config['parameters'][par] for par in parameters}

    model = PFNet(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        num_momentum_outputs=config["dataset"]["num_momentum_outputs"],
        activation=activation,
        **kwargs
    )

    return model

def make_gnn_dense(config, dtype):

    parameters = [
        "layernorm",
        "hidden_dim",
        "bin_size",
        "clip_value_low",
        "num_conv",
        "num_gsl",
        "normalize_degrees",
        "distance_dim",
        "dropout",
        "separate_momentum",
        "input_encoding",
        "debug"
    ]

    kwargs = {par: config['parameters'][par] for par in parameters}

    model = PFNetDense(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        num_momentum_outputs=config["dataset"]["num_momentum_outputs"],
        **kwargs
    )

    return model

def make_transformer(config, dtype):
    parameters = [
        'num_layers', 'd_model', 'num_heads', 'dff', 'support', 'dropout'
    ]
    kwargs = {par: config['parameters'][par] for par in parameters}

    model = Transformer(
        multi_output=config["setup"]["multi_output"],
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        num_momentum_outputs=config["dataset"]["num_momentum_outputs"],
        dtype=dtype,
        **kwargs
    )
    return model

def make_dense(config, dtype):
    model = DummyNet(
        num_input_classes=config["dataset"]["num_input_classes"],
        num_output_classes=config["dataset"]["num_output_classes"],
        num_momentum_outputs=config["dataset"]["num_momentum_outputs"],
    )
    return model

def eval_model(X, ygen, ycand, model, config, outdir, global_batch_size):
    import scipy
    for ibatch in range(X.shape[0]//global_batch_size):
        nb1 = ibatch*global_batch_size
        nb2 = (ibatch+1)*global_batch_size

        y_pred = model.predict(X[nb1:nb2], batch_size=global_batch_size)
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

def main(args, yaml_path, config):
    #tf.debugging.enable_check_numerics()

    #Switch off multi-output for the evaluation for backwards compatibility
    multi_output = True
    if args.action == "eval":
        multi_output = False

    tf.config.run_functions_eagerly(config['tensorflow']['eager'])

    from tfmodel.data import Dataset
    cds = config["dataset"]

    dataset_def = Dataset(
        num_input_features=int(cds["num_input_features"]),
        num_output_features=int(cds["num_output_features"]),
        padded_num_elem_size=int(cds["padded_num_elem_size"]),
        raw_path=cds.get("raw_path", None),
        raw_files=cds.get("raw_files", None),
        processed_path=cds["processed_path"],
        validation_file_path=cds["validation_file_path"],
        schema=cds["schema"]
    )

    if args.action == "data":
        dataset_def.process(
            config["dataset"]["num_files_per_chunk"]
        )
        return

    global_batch_size = config['setup']['batch_size']
    config['setup']['multi_output'] = multi_output

    model_name = os.path.splitext(os.path.basename(yaml_path))[0] + "-" + str(uuid.uuid4())[:8] + "." + platform.node()
    print("model_name=", model_name)

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(dataset_def.processed_path))

    random.shuffle(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    num_events = 0
    for i in dataset:
        num_events += 1
    print("dataset loaded, len={}".format(num_events))

    n_train = config['setup']['num_events_train']
    n_test = config['setup']['num_events_test']

    if args.ntrain:
        n_train = args.ntrain
    if args.ntest:
        n_test = args.ntest

    n_epochs = config['setup']['num_epochs']
    weight_func = make_weight_function(config)
    assert(n_train + n_test <= num_events)

    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        {
            "cls": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "charge": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "energy": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "pt": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "eta": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "sin_phi": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "cos_phi": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
        }
    )

    ds_train = dataset.take(n_train).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)

    dataset_transform = None
    if multi_output:
        dataset_transform = targets_multi_output(config['dataset']['num_output_classes'])
        ds_train = ds_train.map(dataset_transform)
        ds_test = ds_test.map(dataset_transform)

    ds_train_r = ds_train.repeat(n_epochs)
    ds_test_r = ds_test.repeat(n_epochs)

    #small test dataset used in the callback for making monitoring plots
    #X_test = np.concatenate(list(ds_test.take(100).map(lambda x,y,w: x).as_numpy_iterator()))
    #y_test = np.concatenate(list(ds_test.take(100).map(lambda x,y,w: tf.concat(y, axis=-1)).as_numpy_iterator()))

    weights = config['setup']['weights']
    if args.weights:
        weights = args.weights

    if args.recreate or (weights is None):
        outdir = 'experiments/{}'.format(model_name)
        if os.path.isdir(outdir):
            print("Output directory exists: {}".format(outdir), file=sys.stderr)
            sys.exit(1)
    else:
        outdir = os.path.dirname(weights)

    try:
        gpus = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
        num_gpus = len(gpus)
        print("num_gpus=", num_gpus)
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            global_batch_size = num_gpus * global_batch_size
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        print("fallback to CPU", e)
        strategy = tf.distribute.OneDeviceStrategy("cpu")
        num_gpus = 0

    actual_lr = global_batch_size*float(config['setup']['lr'])
    
    Xs = []
    ygens = []
    ycands = []
    #for faster loading        
    if args.action == "train":
        val_filelist = dataset_def.val_filelist[:1]
    else:
        val_filelist = dataset_def.val_filelist
        if config['setup']['num_val_files']>0:
            val_filelist = val_filelist[:config['setup']['num_val_files']]

    for fi in val_filelist:
        X, ygen, ycand = dataset_def.prepare_data(fi)

        Xs.append(np.concatenate(X))
        ygens.append(np.concatenate(ygen))
        ycands.append(np.concatenate(ycand))

    assert(len(Xs) > 0)
    X_val = np.concatenate(Xs)
    ygen_val = np.concatenate(ygens)
    ycand_val = np.concatenate(ycands)

    with strategy.scope():
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            actual_lr,
            decay_steps=10000,
            decay_rate=0.99,
            staircase=True
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        if config['setup']['dtype'] == 'float16':
            model_dtype = tf.dtypes.float16
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        if args.action=="train" or args.action=="eval":
            model = make_model(config, model_dtype)

            #Evaluate model once to build the layers
            print(X_val.shape)
            model(tf.cast(X_val[:1], model_dtype))

            initial_epoch = 0
            if weights:
                #need to load the weights in the same trainable configuration as the model was set up
                configure_model_weights(model, config["setup"].get("weights_config", "all"))
                model.load_weights(weights, by_name=True)
                initial_epoch = int(weights.split("/")[-1].split("-")[1])
            model(tf.cast(X_val[:1], model_dtype))

            if config["setup"]["trainable"] == "classification":
                config["dataset"]["pt_loss_coef"] = 0.0
                config["dataset"]["eta_loss_coef"] = 0.0
                config["dataset"]["sin_phi_loss_coef"] = 0.0
                config["dataset"]["cos_phi_loss_coef"] = 0.0
                config["dataset"]["energy_loss_coef"] = 0.0
            elif config["setup"]["trainable"] == "regression":
                config["dataset"]["classification_loss_coef"] = 0.0
                config["dataset"]["charge_loss_coef"] = 0.0

            #now set the desirable layers as trainable for the optimization
            configure_model_weights(model, config["setup"]["trainable"])
            model(tf.cast(X_val[:1], model_dtype))

            if config["setup"]["classification_loss_type"] == "categorical_cross_entropy":
                cls_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            elif config["setup"]["classification_loss_type"] == "sigmoid_focal_crossentropy":
                cls_loss = make_focal_loss(config)
            else:
                raise KeyError("Unknown classification loss type: {}".format(config["setup"]["classification_loss_type"]))
            
            model.compile(
                loss={
                    "cls": cls_loss,
                    "charge": getattr(tf.keras.losses, config["dataset"].get("charge_loss", "MeanSquaredError"))(),
                    "pt": getattr(tf.keras.losses, config["dataset"].get("pt_loss", "MeanSquaredError"))(),
                    "eta": getattr(tf.keras.losses, config["dataset"].get("eta_loss", "MeanSquaredError"))(),
                    "sin_phi": getattr(tf.keras.losses, config["dataset"].get("sin_phi_loss", "MeanSquaredError"))(),
                    "cos_phi": getattr(tf.keras.losses, config["dataset"].get("cos_phi_loss", "MeanSquaredError"))(),
                    "energy": getattr(tf.keras.losses, config["dataset"].get("energy_loss", "MeanSquaredError"))(),
                },
                optimizer=opt,
                sample_weight_mode='temporal',
                loss_weights={
                    "cls": config["dataset"]["classification_loss_coef"],
                    "charge": config["dataset"]["charge_loss_coef"],
                    "pt": config["dataset"]["pt_loss_coef"],
                    "eta": config["dataset"]["eta_loss_coef"],
                    "sin_phi": config["dataset"]["sin_phi_loss_coef"],
                    "cos_phi": config["dataset"]["cos_phi_loss_coef"],
                    "energy": config["dataset"]["energy_loss_coef"],
                },
                metrics={
                    "cls": [
                        FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                        FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                    ] + [
                        SingleClassRecall(
                            icls,
                            name="rec_cls{}".format(icls),
                            dtype=tf.float64) for icls in range(config["dataset"]["num_output_classes"])
                    ]
                }
            )
            model.summary()
            
            if args.action=="train":
                #file_writer_cm = tf.summary.create_file_writer(outdir + '/val_extra')
                callbacks = prepare_callbacks(
                    model, outdir, X_val[:config['setup']['batch_size']], ycand_val[:config['setup']['batch_size']],
                    dataset_transform, config["dataset"]["num_output_classes"]
                )
                callbacks.append(LearningRateLoggingCallback())

                fit_result = model.fit(
                    ds_train_r, validation_data=ds_test_r, epochs=initial_epoch+n_epochs, callbacks=callbacks,
                    steps_per_epoch=n_train//global_batch_size, validation_steps=n_test//global_batch_size,
                    initial_epoch=initial_epoch
                )
                with open("{}/history.json".format(outdir), "w") as fi:
                    json.dump(fit_result.history, fi)
                model.save(outdir + "/model_full", save_format="tf")
            
            if args.action=="eval":
                eval_model(X_val, ygen_val, ycand_val, model, config, outdir, global_batch_size)
                freeze_model(model, config, outdir)

        if args.action=="time":
            synthetic_timing_data = []
            for iteration in range(config["timing"]["num_iter"]):
                numev = config["timing"]["num_ev"]
                for evsize in [128*10, 128*20, 128*30, 128*40, 128*50, 128*60, 128*70, 128*80, 128*90, 128*100]:
                    for batch_size in [1,2,3,4]:
                        x = np.random.randn(batch_size, evsize, config["dataset"]["num_input_features"]).astype(np.float32)

                        model = make_model(config, model_dtype)
                        model(x)

                        if weights:
                            model.load_weights(weights)

                        t0 = time.time()
                        for i in range(numev//batch_size):
                            model(x)
                        t1 = time.time()
                        dt = t1 - t0

                        time_per_event = 1000.0*(dt / numev)
                        synthetic_timing_data.append(
                                [{"iteration": iteration, "batch_size": batch_size, "event_size": evsize, "time_per_event": time_per_event}])
                        print("Synthetic random data: batch_size={} event_size={}, time={:.2f} ms/ev".format(batch_size, evsize, time_per_event))
            with open("{}/synthetic_timing.json".format(outdir), "w") as fi:
                json.dump(synthetic_timing_data, fi)
