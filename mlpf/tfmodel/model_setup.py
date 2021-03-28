from tfmodel.model import PFNet, Transformer, DummyNet
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

class PFNetLoss:
    def __init__(self, num_input_classes, num_output_classes, classification_loss_coef=1.0, charge_loss_coef=1e-3, momentum_loss_coef=1.0, momentum_loss_coefs=[1.0, 1.0, 1.0]):
        self.num_input_classes = num_input_classes
        self.num_output_classes = num_output_classes
        self.momentum_loss_coef = momentum_loss_coef
        self.momentum_loss_coefs = tf.constant(momentum_loss_coefs)
        self.charge_loss_coef = charge_loss_coef
        self.classification_loss_coef = classification_loss_coef
        self.gamma = 10.0

    def mse_unreduced(self, true, pred):
        return tf.math.pow(true-pred,2)

    def separate_prediction(self, y_pred):
        N = self.num_output_classes
        pred_id_logits = y_pred[:, :, :N]
        pred_charge = y_pred[:, :, N:N+1]
        pred_momentum = y_pred[:, :, N+1:]
        return pred_id_logits, pred_charge, pred_momentum

    def separate_truth(self, y_true):
        true_id = tf.cast(y_true[:, :, :1], tf.int32)
        true_charge = y_true[:, :, 1:2]
        true_momentum = y_true[:, :, 2:]
        return true_id, true_charge, true_momentum

    def loss_components(self, y_true, y_pred):
        pred_id_logits, pred_charge, pred_momentum = self.separate_prediction(y_pred)
        pred_id = tf.cast(tf.argmax(pred_id_logits, axis=-1), tf.int32)
        true_id, true_charge, true_momentum = self.separate_truth(y_true)
        true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=self.num_output_classes)

        #l1 = tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_logits)*self.classification_loss_coef
        l1 = tfa.losses.sigmoid_focal_crossentropy(tf.squeeze(true_id_onehot, [2]), pred_id_logits, from_logits=False, gamma=self.gamma)*self.classification_loss_coef
        l2 = self.mse_unreduced(true_momentum, pred_momentum) * self.momentum_loss_coef * self.momentum_loss_coefs
        l2s = tf.reduce_sum(l2, axis=-1)

        l3 = self.charge_loss_coef*self.mse_unreduced(true_charge, pred_charge)[:, :, 0]

        return l1, l2s, l3, l2

    def my_loss_full(self, y_true, y_pred):
        l1, l2, l3, _ = self.loss_components(y_true, y_pred)
        loss = l1 + l2 + l3

        return loss

    def my_loss_cls(self, y_true, y_pred):
        l1, l2, l3, _ = self.loss_components(y_true, y_pred)
        loss = l1

        return loss

    def my_loss_reg(self, y_true, y_pred):
        l1, l2, l3, _ = self.loss_components(y_true, y_pred)
        loss = l3

        return loss

def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Reconstructed PID (normed to gen)")
    plt.xlabel("MLPF PID")
    plt.ylabel("Gen PID")
    plt.colorbar()
    plt.tight_layout()
    return fig

def plot_regression(val_x, val_y, var_name, rng):
    fig = plt.figure(figsize=(5,5))
    plt.hist2d(
        val_x,
        val_y,
        bins=(rng, rng),
        cmap="Blues",
        #norm=matplotlib.colors.LogNorm()
    );
    plt.xlabel("Gen {}".format(var_name))
    plt.ylabel("MLPF {}".format(var_name))
    return fig

def plot_multiplicity(num_pred, num_true):
    fig = plt.figure(figsize=(5,5))
    xs = np.arange(len(num_pred))
    plt.bar(xs, num_true, alpha=0.8)
    plt.bar(xs, num_pred, alpha=0.8)
    plt.xticks(xs)
    return fig

def plot_num_particle(num_pred, num_true, pid):
    fig = plt.figure(figsize=(5,5))
    plt.scatter(num_true, num_pred)
    plt.title("particle id {}".format(pid))
    plt.xlabel("num true")
    plt.ylabel("num pred")
    a = min(np.min(num_true), np.min(num_pred))
    b = max(np.max(num_true), np.max(num_pred))
    plt.xlim(a, b)
    plt.ylim(a, b)
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

def plot_distributions(val_x, val_y, var_name, rng):
    fig = plt.figure(figsize=(5,5))
    plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="gen");
    plt.hist(val_y, bins=rng, density=True, histtype="step", lw=2, label="MLPF");
    plt.xlabel(var_name)
    plt.legend(loc="best", frameon=False)
    plt.ylim(0,1.5)
    return fig

def plot_particles(y_pred, y_true, pid=1):
    #Ground truth vs model prediction particles
    fig = plt.figure(figsize=(10,10))

    ev = y_true[0, :]
    msk = ev[:, 0] == pid
    plt.scatter(ev[msk, 3], np.arctan2(ev[msk, 4], ev[msk, 5]), s=2*ev[msk, 2], marker="o", alpha=0.5)

    ev = y_pred[0, :]
    msk = ev[:, 0] == pid
    plt.scatter(ev[msk, 3], np.arctan2(ev[msk, 4], ev[msk, 5]), s=2*ev[msk, 2], marker="s", alpha=0.5)

    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.xlim(-5,5)
    plt.ylim(-4,4)

    return fig

class ConfusionMatrixValidation:
    def __init__(self, X_test, y_test, loss_cls, outdir, model, num_input_classes, num_output_classes, file_writer_cm):
        self.X_test = X_test
        self.y_test = y_test
        self.loss_cls = loss_cls
        self.outdir = outdir
        self.model = model
        self.num_input_classes = num_input_classes
        self.num_output_classes = num_output_classes
        self.file_writer_cm = file_writer_cm

    def log_confusion_matrix(self, epoch, logs):
      
        outdir = self.outdir
        model = self.model
        X_test = self.X_test
        y_test = self.y_test

        test_pred = model.predict(X_test, batch_size=5)

        if isinstance(test_pred, tuple):
            test_pred = tf.concat(list(test_pred), axis=-1)

        l1, l2, l3, l2_r = self.loss_cls.loss_components(y_test, test_pred)

        logs["epoch"] = int(epoch)
        logs["l1"] = float(tf.reduce_mean(l1).numpy())
        logs["l2"] = float(tf.reduce_mean(l2).numpy())
        logs["l2_split"] = [float(x) for x in tf.reduce_mean(l2_r, axis=[0,1])]
        logs["l3"] = float(tf.reduce_mean(l3).numpy())

        with open("{}/logs_{}.json".format(outdir, epoch), "w") as fi:
            json.dump(logs, fi)

        test_pred_id = np.argmax(test_pred[:, :, :self.num_output_classes], axis=-1)
        
        counts_pred = np.unique(test_pred_id, return_counts=True)

        test_pred = np.concatenate([np.expand_dims(test_pred_id, axis=-1), test_pred[:, :, self.num_output_classes:]], axis=-1)

        cm = sklearn.metrics.confusion_matrix(
            y_test[:, :, 0].astype(np.int64).flatten(),
            test_pred[:, :, 0].flatten(), labels=list(range(self.num_output_classes)))
        cm_normed = sklearn.metrics.confusion_matrix(
            y_test[:, :, 0].astype(np.int64).flatten(),
            test_pred[:, :, 0].flatten(), labels=list(range(self.num_output_classes)), normalize="true")

        num_pred = np.sum(cm, axis=0)
        num_true = np.sum(cm, axis=1)

        figure = plot_confusion_matrix(cm)
        cm_image = plot_to_image(figure)

        figure = plot_confusion_matrix(cm_normed)
        cm_image_normed = plot_to_image(figure)

        msk = (test_pred[:, :, 0]!=0) & (y_test[:, :, 0]!=0)

        ch_true = y_test[msk, 1].flatten()
        ch_pred = test_pred[msk, 1].flatten()

        figure = plot_regression(ch_true, ch_pred, "charge", np.linspace(-2, 2, 100))
        ch_image = plot_to_image(figure)

        figure = plot_multiplicity(num_pred, num_true)
        n_image = plot_to_image(figure)

        images_mult = []
        for icls in range(self.num_output_classes):
            n_pred = np.sum(test_pred[:, :, 0]==icls, axis=1)
            n_true = np.sum(y_test[:, :, 0]==icls, axis=1)
            figure = plot_num_particle(n_pred, n_true, icls)
            images_mult.append(plot_to_image(figure))

        images = {}
        for ireg in range(l2_r.shape[-1]):
            reg_true = y_test[msk, 2+ireg].flatten()
            reg_pred = test_pred[msk, 2+ireg].flatten()

            figure = plot_regression(reg_true, reg_pred, "reg {}".format(ireg), np.linspace(np.mean(reg_true) - 3*np.std(reg_true), np.mean(reg_true) + 3*np.std(reg_true), 100))
            images[ireg] = plot_to_image(figure)

        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
            tf.summary.image("Confusion Matrix Normed", cm_image_normed, step=epoch)
            tf.summary.image("Confusion Matrix Normed", cm_image_normed, step=epoch)
            tf.summary.image("charge regression", ch_image, step=epoch)
            tf.summary.image("particle multiplicity", n_image, step=epoch)

            for icls, img in enumerate(images_mult):
                tf.summary.image("npart {}".format(icls), img, step=epoch)

            for ireg in images.keys():
                tf.summary.image("regression {}".format(ireg), images[ireg], step=epoch)

            tf.summary.scalar("loss_cls", tf.reduce_mean(l1), step=epoch)
            for i in range(l2_r.shape[-1]):
                tf.summary.scalar("loss_reg_{}".format(i), tf.reduce_mean(l2_r[:, :, i]), step=epoch)

            for i in range(cm_normed.shape[0]):
                tf.summary.scalar("acc_cls_{}".format(i), cm_normed[i, i], step=epoch)
                
            tf.summary.scalar("loss_chg", tf.reduce_mean(l3), step=epoch)

def prepare_callbacks(X_test, y_test, loss_cls, model, outdir, num_input_classes, num_output_classes, file_writer_cm):
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

    cmv = ConfusionMatrixValidation(
        X_test, y_test, loss_cls,
        outdir=outdir, model=model,
        num_input_classes=num_input_classes,
        num_output_classes=num_output_classes,
        file_writer_cm = file_writer_cm
    )

    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=cmv.log_confusion_matrix)
    callbacks += [cm_callback]

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

def compute_weights_invsqrt(X, y, w):
    wn = 1.0/tf.sqrt(w)
    wn *= tf.cast(X[:, 0]!=0, tf.float32)
    #wn /= tf.reduce_sum(wn)
    return X, y, wn

def compute_weights_none(X, y, w):
    wn = tf.ones_like(w)
    wn *= tf.cast(X[:, 0]!=0, tf.float32)
    return X, y, wn

weight_functions = {
    "inverse_sqrt": compute_weights_invsqrt,
    "none": compute_weights_none,
}

def scale_outputs(X,y,w):
    ynew = y-out_m
    ynew = ynew/out_s
    return X, ynew, w

def targets_multi_output(X,y,w):
    return X, (y[:, :, 0:1],  y[:, :, 1:2], y[:, :, 2:]), w

def make_model(config, dtype):
    model = config['parameters']['model']
    if model == 'gnn':
        return make_gnn(config, dtype)
    elif model == 'transformer':
        return make_transformer(config, dtype)
    elif model == 'dense':
        return make_dense(config, dtype)
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
    y_pred = model.predict(X, batch_size=global_batch_size)
    y_pred_raw_ids = y_pred[:, :, :config["dataset"]["num_output_classes"]]
    
    #softmax score must be over a threshold 0.6 to call it a particle (prefer low fake rate to high efficiency)
    y_pred_id_sm = scipy.special.softmax(y_pred_raw_ids, axis=-1)
    y_pred_id_sm[y_pred_id_sm < 0.] = 0.0

    msk = np.ones(y_pred_id_sm.shape, dtype=np.bool)

    #Use thresholds for charged and neutral hadrons based on matching the DelphesPF fake rate
    msk[y_pred_id_sm[:, :, 1] < 0.8, 1] = 0
    msk[y_pred_id_sm[:, :, 2] < 0.025, 2] = 0
    y_pred_id_sm = y_pred_id_sm*msk

    y_pred_id = np.argmax(y_pred_id_sm, axis=-1)

    y_pred_id = np.concatenate([np.expand_dims(y_pred_id, axis=-1), y_pred[:, :, config["dataset"]["num_output_classes"]:]], axis=-1)
    np_outfile = "{}/pred.npz".format(outdir)
    print("saving output to {}".format(np_outfile))
    np.savez(np_outfile, X=X, ygen=ygen, ycand=ycand, ypred=y_pred_id, ypred_raw=y_pred[:, :, :config["dataset"]["num_output_classes"]])

def freeze_model(model, config, outdir):
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

def main(args, yaml_path, config):

    #Switch off multi-output for the evaluation for backwards compatibility
    if args.action == "eval":
        config['setup']['multi_output'] = False

    tf.config.run_functions_eagerly(config['tensorflow']['eager'])

    from tfmodel.data import Dataset
    cds = config["dataset"]

    dataset_def = Dataset(
        num_input_features=int(cds["num_input_features"]),
        num_output_features=int(cds["num_output_features"]),
        padded_num_elem_size=int(cds["padded_num_elem_size"]),
        raw_path=cds["raw_path"],
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

    model_name = os.path.splitext(os.path.basename(yaml_path))[0] + "-" + str(uuid.uuid4())[:8]
    print("model_name=", model_name)

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(dataset_def.processed_path))

    dataset = tf.data.TFRecordDataset(tfr_files).map(dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    num_events = 0
    for i in dataset:
        num_events += 1
    print("dataset loaded, len={}".format(num_events))

    n_train = config['setup']['num_events_train']
    n_test = config['setup']['num_events_test']
    n_epochs = config['setup']['num_epochs']
    weight_func = weight_functions[config['setup']['sample_weights']]
    assert(n_train + n_test <= num_events)

    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, ])
    )

    ds_train = dataset.take(n_train).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)

    if config['setup']['multi_output']:
        ds_train = ds_train.map(targets_multi_output)
        ds_test = ds_test.map(targets_multi_output)

    #small test dataset used in the callback for making monitoring plots
    X_test = ds_test.take(100).map(lambda x,y,w: x)
    y_test = np.concatenate(list(ds_test.take(100).map(lambda x,y,w: tf.concat(y, axis=-1)).as_numpy_iterator()))

    ds_train_r = ds_train.repeat(n_epochs)
    ds_test_r = ds_test.repeat(n_epochs)

    weights = config['setup']['weights']
    if args.weights:
        weights = args.weights
    if weights is None:
        outdir = 'experiments/{}'.format(model_name)
        if os.path.isdir(outdir):
            print("Output directory exists: {}".format(outdir), file=sys.stderr)
            sys.exit(1)
    else:
        outdir = os.path.dirname(weights)

    try:
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
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
        dataset_def.val_filelist = dataset_def.val_filelist[:1]

    for fi in dataset_def.val_filelist:
        print(fi)
        X, ygen, ycand = dataset_def.prepare_data(fi)

        Xs.append(np.concatenate(X))
        ygens.append(np.concatenate(ygen))
        ycands.append(np.concatenate(ycand))

    X_val = np.concatenate(Xs)
    ygen_val = np.concatenate(ygens)
    ycand_val = np.concatenate(ycands)

    with strategy.scope():
        if config['setup']['dtype'] == 'float16':
            if config['setup']['multi_output']:
                raise Exception("float16 and multi_output are not supported at the same time")

            model_dtype = tf.dtypes.float16
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

            opt = mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(learning_rate=actual_lr),
                loss_scale="dynamic"
            )
        else:
            model_dtype = tf.dtypes.float32
            opt = tf.keras.optimizers.Adam(learning_rate=actual_lr)

            if config['setup']['multi_output']:
                from tfmodel.PCGrad_tf import PCGrad
                opt = PCGrad(tf.compat.v1.train.AdamOptimizer(actual_lr))

        if args.action=="train" or args.action=="eval":
            model = make_model(config, model_dtype)

            loss_cls = PFNetLoss(
                classification_loss_coef=config["dataset"]["classification_loss_coef"],
                charge_loss_coef=config["dataset"]["charge_loss_coef"],
                momentum_loss_coef=config["dataset"]["momentum_loss_coef"],
                num_input_classes=config["dataset"]["num_input_classes"],
                num_output_classes=config["dataset"]["num_output_classes"],
                momentum_loss_coefs=config["dataset"]["momentum_loss_coefs"]
            )

            loss_fn = loss_cls.my_loss_full
            if config["setup"]["trainable"] == "cls":
                model.set_trainable_classification()
                loss_fn = loss_cls.my_loss_cls
            elif config["setup"]["trainable"] == "reg":
                model.set_trainable_regression()
                loss_fn = loss_cls.my_loss_reg

            #we use the "temporal" mode to have per-particle weights
            if config['setup']['multi_output']:
                model.compile(
                    loss=(
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.losses.MeanSquaredError(),
                        tf.keras.losses.MeanSquaredError()
                    ),
                    optimizer=opt,
                    sample_weight_mode='temporal',
                    loss_weights=[
                        config["dataset"]["classification_loss_coef"],
                        config["dataset"]["charge_loss_coef"],
                        config["dataset"]["momentum_loss_coef"]
                    ]
                )
            else:
                model.compile(
                    loss=loss_cls.my_loss_full,
                    optimizer=opt,
                    sample_weight_mode='temporal',
                )

            #Evaluate model once to build the layers
            model(tf.cast(X_val[:1], model_dtype))
            model.summary()

            initial_epoch = 0
            if weights:
                model.load_weights(weights)
                initial_epoch = int(weights.split("/")[-1].split("-")[1])

            if args.action=="train":
                file_writer_cm = tf.summary.create_file_writer(outdir + '/val_extra')
                callbacks = prepare_callbacks(
                    X_test, y_test,
                    loss_cls,
                    model, outdir,
                    config["dataset"]["num_input_classes"], config["dataset"]["num_output_classes"],
                    file_writer_cm
                )

                model.fit(
                    ds_train_r, validation_data=ds_test_r, epochs=initial_epoch+n_epochs, callbacks=callbacks,
                    steps_per_epoch=n_train//global_batch_size, validation_steps=n_test//global_batch_size,
                    initial_epoch=initial_epoch
                )

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
