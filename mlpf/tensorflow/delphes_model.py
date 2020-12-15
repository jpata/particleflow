from tf_model import PFNet, Transformer, DummyNet
import tensorflow as tf
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
import glob
import PCGrad_tf
import io
import os
import yaml
import uuid
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import kerastuner as kt
from argparse import Namespace

num_input_classes = 2
num_output_classes = 6
mult_classification_loss = 100.0
mult_charge_loss = 1.0
mult_energy_loss = 1e-4
mult_phi_loss = 10.0
mult_eta_loss = 1.0
mult_pt_loss = 1.0
mult_total_loss = 1e3
datapath = "out/pythia8_ttbar/tfr/*.tfrecords"
pkl_path = "out/pythia8_ttbar/tev14_pythia8_ttbar_000_0.pkl"

def mse_unreduced(true, pred):
    return tf.math.pow(true-pred,2)

def separate_prediction(y_pred):
    N = num_output_classes
    pred_id_logits = y_pred[:, :, :N]
    pred_charge = y_pred[:, :, N:N+1]
    pred_momentum = y_pred[:, :, N+1:]
    return pred_id_logits, pred_charge, pred_momentum

def separate_truth(y_true):
    true_id = tf.cast(y_true[:, :, :1], tf.int32)
    true_charge = y_true[:, :, 1:2]
    true_momentum = y_true[:, :, 2:]
    return true_id, true_charge, true_momentum

def accuracy(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    is_true = true_id[:, :, 0]!=0
    is_same = true_id[:, :, 0] == pred_id

    acc = tf.reduce_sum(tf.cast(is_true&is_same, tf.int32)) / tf.reduce_sum(tf.cast(is_true, tf.int32))
    return tf.cast(acc, tf.float32)

def energy_resolution(y_true, y_pred):
    pred_id_onehot, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_onehot, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)

    msk = true_id[:, :, 0]!=0
    return tf.reduce_mean(mse_unreduced(true_momentum[msk][:, -1], pred_momentum[msk][:, -1]))

def loss_components(y_true, y_pred):
    pred_id_logits, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_logits, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)
    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=num_output_classes)

    l1 = mult_classification_loss*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_logits)
  
    #msk_good = tf.cast(true_id[:, :, 0] == pred_id, tf.float32)
    l2_0 = mult_pt_loss*mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mult_eta_loss*mse_unreduced(true_momentum[:, :, 1], pred_momentum[:, :, 1])
    l2_2 = mult_phi_loss*mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])
    l2_3 = mult_phi_loss*mse_unreduced(true_momentum[:, :, 3], pred_momentum[:, :, 3])
    l2_4 = mult_energy_loss*mse_unreduced(true_momentum[:, :, 4], pred_momentum[:, :, 4])

    l2 = (l2_0 + l2_1 + l2_2 + l2_3 + l2_4)

    l3 = mult_charge_loss*mse_unreduced(true_charge, pred_charge)[:, :, 0]

    return l1, l2, l3, (l2_0, l2_1, l2_2, l2_3, l2_4)

def my_loss_full(y_true, y_pred):

    l1, l2, l3, _ = loss_components(y_true, y_pred)
    loss = l1 + l2 + l3

    return mult_total_loss*loss

def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Reconstructed PID (normed to gen)")
    plt.xlabel("MLPF PID")
    plt.ylabel("Gen PID")
    plt.xticks(range(6), ["none", "ch.had", "n.had", "g", "el", "mu"]);
    plt.yticks(range(6), ["none", "ch.had", "n.had", "g", "el", "mu"]);
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

def log_confusion_matrix(epoch, logs):
   
    # if epoch==0 or epoch%5!=0:
    #     return

    test_pred = model.predict(X_test, batch_size=5)
    test_pred_p_charge = test_pred[:, :, num_output_classes:] 

    l1, l2, l3, (l2_0, l2_1, l2_2, l2_3, l2_4) = loss_components(y_test, test_pred)

    test_pred_id = np.argmax(test_pred[:, :, :num_output_classes], axis=-1)
    test_pred = np.concatenate([np.expand_dims(test_pred_id, axis=-1), test_pred[:, :, num_output_classes:]], axis=-1)

    cm = sklearn.metrics.confusion_matrix(
        y_test[:, :, 0].astype(np.int64).flatten(),
        test_pred[:, :, 0].flatten(), labels=list(range(num_output_classes)))
    cm_normed = sklearn.metrics.confusion_matrix(
        y_test[:, :, 0].astype(np.int64).flatten(),
        test_pred[:, :, 0].flatten(), labels=list(range(num_output_classes)), normalize="true")

    figure = plot_confusion_matrix(cm)
    cm_image = plot_to_image(figure)

    figure = plot_confusion_matrix(cm_normed)
    cm_image_normed = plot_to_image(figure)

    msk = (test_pred[:, :, 0]!=0) & (y_test[:, :, 0]!=0)

    ch_true = y_test[msk, 1].flatten()
    ch_pred = test_pred[msk, 1].flatten()

    pt_true = y_test[msk, 2].flatten()
    pt_pred = test_pred[msk, 2].flatten()

    e_true = y_test[msk, 6].flatten()
    e_pred = test_pred[msk, 6].flatten()

    eta_true = y_test[msk, 3].flatten()
    eta_pred = test_pred[msk, 3].flatten()

    sphi_true = y_test[msk, 4].flatten()
    sphi_pred = test_pred[msk, 4].flatten()

    cphi_true = y_test[msk, 5].flatten()
    cphi_pred = test_pred[msk, 5].flatten()

    figure = plot_regression(ch_true, ch_pred, "charge", np.linspace(-2, 2, 100))
    ch_image = plot_to_image(figure)

    figure = plot_regression(pt_true, pt_pred, "pt", np.linspace(0, 5, 100))
    pt_image = plot_to_image(figure)

    figure = plot_distributions(pt_true, pt_pred, "pt", np.linspace(0, 5, 100))
    pt_distr_image = plot_to_image(figure)

    figure = plot_regression(e_true, e_pred, "E", np.linspace(-1, 5, 100))
    e_image = plot_to_image(figure)

    figure = plot_distributions(e_true, e_pred, "E", np.linspace(-1, 5, 100))
    e_distr_image = plot_to_image(figure)

    figure = plot_regression(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100))
    eta_image = plot_to_image(figure)

    figure = plot_distributions(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100))
    eta_distr_image = plot_to_image(figure)

    figure = plot_regression(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100))
    sphi_image = plot_to_image(figure)

    figure = plot_distributions(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100))
    sphi_distr_image = plot_to_image(figure)

    figure = plot_regression(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100))
    cphi_image = plot_to_image(figure)

    figure = plot_distributions(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100))
    cphi_distr_image = plot_to_image(figure)

    figure = plot_particles(test_pred, y_test, 1)
    pid_image_1 = plot_to_image(figure)

    figure = plot_particles(test_pred, y_test, 2)
    pid_image_2 = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        tf.summary.image("Confusion Matrix Normed", cm_image_normed, step=epoch)
        tf.summary.image("charge regression", ch_image, step=epoch)
        tf.summary.image("pT regression", pt_image, step=epoch)
        tf.summary.image("pT distibution", pt_distr_image, step=epoch)
        tf.summary.image("E regression", e_image, step=epoch)
        tf.summary.image("E distribution", e_distr_image, step=epoch)
        tf.summary.image("eta regression", eta_image, step=epoch)
        tf.summary.image("eta distribution", eta_distr_image, step=epoch)
        tf.summary.image("sin phi regression", sphi_image, step=epoch)
        tf.summary.image("sin phi distribution", sphi_distr_image, step=epoch)
        tf.summary.image("cos phi regression", cphi_image, step=epoch)
        tf.summary.image("cos phi distribution", cphi_distr_image, step=epoch)

        tf.summary.image("charged hadron particles", pid_image_1, step=epoch)
        tf.summary.image("neutral hadron particles", pid_image_2, step=epoch)

        #tf.summary.histogram("dm_values", dm.values, step=epoch)
        tf.summary.scalar("loss_cls", tf.reduce_mean(l1), step=epoch)
        tf.summary.scalar("loss_reg_pt", tf.reduce_mean(l2_0), step=epoch)
        tf.summary.scalar("loss_reg_eta", tf.reduce_mean(l2_1), step=epoch)
        tf.summary.scalar("loss_reg_sphi", tf.reduce_mean(l2_2), step=epoch)
        tf.summary.scalar("loss_reg_cphi", tf.reduce_mean(l2_3), step=epoch)
        tf.summary.scalar("loss_reg_e", tf.reduce_mean(l2_4), step=epoch)
        tf.summary.scalar("loss_reg_charge", tf.reduce_mean(l3), step=epoch)

        tf.summary.scalar("ch_pred_mean", tf.reduce_mean(ch_pred), step=epoch)
        tf.summary.scalar("pt_pred_mean", tf.reduce_mean(pt_pred), step=epoch)
        tf.summary.scalar("e_pred_mean", tf.reduce_mean(e_pred), step=epoch)
        tf.summary.scalar("eta_pred_mean", tf.reduce_mean(eta_pred), step=epoch)
        tf.summary.scalar("sphi_pred_mean", tf.reduce_mean(sphi_pred), step=epoch)
        tf.summary.scalar("cphi_pred_mean", tf.reduce_mean(cphi_pred), step=epoch)

        tf.summary.scalar("ch_pred_std", tf.math.reduce_std(ch_pred), step=epoch)
        tf.summary.scalar("pt_pred_std", tf.math.reduce_std(pt_pred), step=epoch)
        tf.summary.scalar("e_pred_std", tf.math.reduce_std(e_pred), step=epoch)
        tf.summary.scalar("eta_pred_std", tf.math.reduce_std(eta_pred), step=epoch)
        tf.summary.scalar("sphi_pred_std", tf.math.reduce_std(sphi_pred), step=epoch)
        tf.summary.scalar("cphi_pred_std", tf.math.reduce_std(cphi_pred), step=epoch)

def prepare_callbacks(model, outdir):
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
        filepath=outdir + "/weights.{epoch:02d}-{val_loss:.6f}.hdf5",
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model)
    callbacks += [cp_callback]

    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
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

def compute_weights_inverse(X, y, w):
    wn = 1.0/tf.sqrt(w)
    #wn *= tf.cast(X[:, 0]!=0, tf.float32)
    wn /= tf.reduce_sum(wn)
    return X, y, wn

def scale_outputs(X,y,w):
    ynew = y-out_m
    ynew = ynew/out_s
    return X, ynew, w

def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f)
    return config

def make_model(config, dtype):
    model = config['parameters']['model']
    if model == 'gnn':
        return make_gnn(config, dtype)
    elif model == 'transformer':
        return make_transformer(config, dtype)
    elif model == 'dense':
        return make_dense(config, dtype)
    raise KeyError("Unknown model type {}".format(model))

# DelphesPF does muon identification based on generator information
# we need to encode gen-level muon information to the inputs
# to not disadvantage the MLPF algo unfairly
def encode_track_muon(X,y,w):
    msk_mu = tf.expand_dims(tf.cast(y[:, :, 0] == 5, tf.float32), axis=-1)
    X = tf.concat([X, msk_mu], axis=-1)
    return X,y,w

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
        'dropout'
    ]
    kwargs = {par: config['parameters'][par] for par in parameters}

    model = PFNet(
        num_input_classes=num_input_classes,
        num_output_classes=num_output_classes,
        num_momentum_outputs=5,
        activation=activation,
        **kwargs
    )

    return model

def make_transformer(config, dtype):
    parameters = [
        'num_layers', 'd_model', 'num_heads', 'dff', 'support'
    ]
    kwargs = {par: config['parameters'][par] for par in parameters}

    model = Transformer(
        num_input_classes=num_input_classes,
        num_output_classes=num_output_classes,
        num_momentum_outputs=5,
        dtype=dtype,
        **kwargs
    )
    return model

def make_dense(config, dtype):
    model = DummyNet(
        num_input_classes=num_input_classes,
        num_output_classes=num_output_classes,
        num_momentum_outputs=5,
    )
    return model


def model_builder_gnn(hp):
    args = Namespace()
    args.hidden_dim_id = hp.Choice('hidden_dim_id', values = [16, 32, 64, 128, 256])
    args.hidden_dim_reg = hp.Choice('hidden_dim_reg', values = [16, 32, 64, 128, 256])
    args.num_hidden_id_enc = hp.Choice('hidden_dim_id_enc', values = [0, 1, 2, 3])
    args.num_hidden_id_dec = hp.Choice('hidden_dim_id_dec', values = [0, 1, 2, 3])
    args.num_hidden_reg_enc = hp.Choice('hidden_dim_reg_enc', values = [0, 1, 2, 3])
    args.num_hidden_reg_dec = hp.Choice('hidden_dim_reg_dec', values = [0, 1, 2, 3])
    args.num_convs_id = hp.Choice('num_convs_id', values = [1, 2, 3, 4])
    args.num_convs_reg = hp.Choice('num_convs_reg', values = [1, 2, 3, 4])
    args.distance_dim = hp.Choice('distance_dim', values = [16, 32, 64, 128, 256])
    args.num_neighbors = hp.Choice('num_neighbors', [2, 4, 8, 16, 32])
    args.dropout = hp.Choice('dropout', values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args.bin_size = hp.Choice('bin_size', values = [32, 64, 128, 256])
    args.dist_mult = hp.Choice('dist_mult', values = [0.1, 1.0, 10.0])
    args.lr = hp.Choice('lr', values = [1e-5, 1e-4, 1e-3])

    model = PFNet(
        num_hidden_id_enc=args.num_hidden_id_enc,
        num_hidden_id_dec=args.num_hidden_id_dec,
        hidden_dim_id=args.hidden_dim_id,
        num_hidden_reg_enc=args.num_hidden_reg_enc,
        num_hidden_reg_dec=args.num_hidden_reg_dec,
        hidden_dim_reg=args.hidden_dim_reg,
        num_convs_id=args.num_convs_id,
        num_convs_reg=args.num_convs_reg,
        distance_dim=args.distance_dim,
        #convlayer=convlayer,
        dropout=args.dropout,
        bin_size=args.bin_size,
        num_neighbors=args.num_neighbors,
        dist_mult=args.dist_mult,
        num_input_classes=num_input_classes,
        num_output_classes=num_output_classes,
        num_momentum_outputs=5,
        activation=tf.nn.elu,
    )

    print(args)

    loss_fn = my_loss_full
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(optimizer=opt, loss=loss_fn, sample_weight_mode="temporal")

    from delphes_data import _parse_tfr_element, padded_num_elem_size, num_inputs, num_outputs
    model(np.random.randn(1, padded_num_elem_size, num_inputs + 1).astype(np.float32))

    model.summary()
    return model

if __name__ == "__main__":

    yaml_path = sys.argv[1]
    model_name = os.path.splitext(os.path.basename(yaml_path))[0] + "-" + str(uuid.uuid4())[:8]
    print("model_name=", model_name)

    config = load_config(yaml_path)
    do_training = config['setup']['train']
    weights = config['setup']['weights']

    tf.config.run_functions_eagerly(config['tensorflow']['eager'])

    from delphes_data import _parse_tfr_element, padded_num_elem_size, num_inputs, num_outputs
    tfr_files = glob.glob(datapath)
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(datapath))
        
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("dataset loaded")

    num_events = 0
    for i in dataset:
        num_events += 1

    global_batch_size = config['setup']['batch_size']
    n_train = config['setup']['num_events_train']
    n_test = config['setup']['num_events_test']
    n_epochs = config['setup']['num_epochs']
    assert(n_train + n_test <= num_events)

    ps = (tf.TensorShape([padded_num_elem_size, num_inputs]), tf.TensorShape([padded_num_elem_size, num_outputs]), tf.TensorShape([padded_num_elem_size, ]))
    ds_train = dataset.take(n_train).map(compute_weights_inverse).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).map(compute_weights_inverse).padded_batch(global_batch_size, padded_shapes=ps)

    #small test dataset used in the callback for making monitoring plots
    X_test = ds_test.take(100).map(lambda x,y,w: x)
    y_test = np.concatenate(list(ds_test.take(100).map(lambda x,y,w: tf.concat(y, axis=-1)).as_numpy_iterator()))

    ds_train_r = ds_train.repeat(n_epochs)
    ds_test_r = ds_test.repeat(n_epochs)

    if do_training:
        outdir = 'experiments/{}'.format(model_name)
        if os.path.isdir(outdir):
            print("Output directory exists: {}".format(outdir), file=sys.stderr)
            sys.exit(1)
        file_writer_cm = tf.summary.create_file_writer(outdir + '/val_extra')
    else:
        outdir = os.path.dirname(weights)

    try:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        print("num_gpus=", num_gpus)
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            global_batch_size = num_gpus * global_batch_size
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        print("fallback to CPU")
        strategy = tf.distribute.OneDeviceStrategy("cpu")
        num_gpus = 0
    actual_lr = global_batch_size*min(1,num_gpus)*float(config['setup']['lr'])


    # tuner = kt.Hyperband(
    #     model_builder_gnn,
    #     objective = 'val_loss', 
    #     max_epochs = 10,
    #     factor = 3,
    #     hyperband_iterations = 3,
    #     directory = './kerastuner_out',
    #     project_name = 'mlpf',
    #     max_model_size = 5000000
    # )

    # tuner.search(
    #    ds_train_r,
    #    validation_data=ds_test_r,
    #    steps_per_epoch=n_train/global_batch_size,
    #    validation_steps=n_test/global_batch_size,
    # )
    # tuner.results_summary()
    # for trial in tuner.oracle.get_best_trials(num_trials=10):
    #     print(trial.hyperparameters.values, trial.score)

    with strategy.scope():
        if config['setup']['dtype'] == 'float16':
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

        model = make_model(config, model_dtype)
        print("model created")

        #we use the "temporal" mode to have per-particle weights
        model.compile(
            loss=my_loss_full,
            optimizer=opt,
            sample_weight_mode='temporal'
        )

        for X,y,w in ds_test:
            model(X)
            break
        model.summary()

        if weights:
            model.load_weights(weights)

        if do_training:
            callbacks = prepare_callbacks(model, outdir)
            #callbacks = []
            model.fit(
                ds_train_r, validation_data=ds_test_r, epochs=n_epochs, callbacks=callbacks,
                steps_per_epoch=n_train/global_batch_size, validation_steps=n_test/global_batch_size
            )

            model.save(outdir + "/model_full", save_format="tf")
        
        from delphes_data import prepare_data
        X, ygen, ycand = prepare_data(pkl_path)

        X = np.concatenate(X)
        ygen = np.concatenate(ygen)
        ycand = np.concatenate(ycand)

        y_pred = model.predict(X, batch_size=global_batch_size)
        y_pred_id = np.argmax(y_pred[:, :, :num_output_classes], axis=-1)
        y_pred_id = np.concatenate([np.expand_dims(y_pred_id, axis=-1), y_pred[:, :, num_output_classes:]], axis=-1)

        np.savez("{}/pred.npz".format(outdir), ygen=ygen, ycand=ycand, ypred=y_pred_id, ypred_raw=y_pred[:, :, :num_output_classes])
