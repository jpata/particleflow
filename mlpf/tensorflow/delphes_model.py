from tf_model import PFNet
import tensorflow as tf
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
import glob

num_input_classes = 2
num_output_classes = 6
mult_classification_loss = 1e3
mult_charge_loss = 1.0
mult_energy_loss = 10.0
mult_phi_loss = 10.0
mult_eta_loss = 10.0
mult_pt_loss = 10.0
mult_total_loss = 1e3

padded_num_elem_size = 128*40

def prepare_data(fname):
    data = pickle.load(open(fname, "rb"))

    #make all inputs and outputs the same size with padding
    Xs = []
    ys = []
    for i in range(len(data["X"])):
        X = np.array(data["X"][i][:padded_num_elem_size], np.float32)
        X = np.pad(X, [(0, padded_num_elem_size - X.shape[0]), (0,0)])
        y = np.array(data["ygen"][i][:padded_num_elem_size], np.float32)
        y = np.pad(y, [(0, padded_num_elem_size - y.shape[0]), (0,0)])

        X = np.expand_dims(X, 0)
        y = np.expand_dims(y, 0)
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y

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
    return tf.reduce_mean(mse_unreduced(true_momentum[msk][:, 3], pred_momentum[msk][:, 3]))

def my_loss_full(y_true, y_pred):
    pred_id_logits, pred_charge, pred_momentum = separate_prediction(y_pred)
    pred_id = tf.cast(tf.argmax(pred_id_logits, axis=-1), tf.int32)
    true_id, true_charge, true_momentum = separate_truth(y_true)
    true_id_onehot = tf.one_hot(tf.cast(true_id, tf.int32), depth=num_output_classes)
    
    l1 = mult_classification_loss*tf.nn.softmax_cross_entropy_with_logits(true_id_onehot, pred_id_logits)
  
    l2_0 = mult_pt_loss*mse_unreduced(true_momentum[:, :, 0], pred_momentum[:, :, 0])
    l2_1 = mult_eta_loss*mse_unreduced(true_momentum[:, :, 1], pred_momentum[:, :, 1])
    l2_2 = mult_phi_loss*mse_unreduced(true_momentum[:, :, 2], pred_momentum[:, :, 2])
    l2_3 = mult_energy_loss*mse_unreduced(true_momentum[:, :, 3], pred_momentum[:, :, 3])

    l2 = (l2_0 + l2_1 + l2_2 + l2_3)

    l3 = mult_charge_loss*mse_unreduced(true_charge, pred_charge)[:, :, 0]
    loss = l1 + l2 + l3

    return mult_total_loss*loss

def prepare_callbacks(model, outdir):
    callbacks = []
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=outdir, histogram_freq=1, write_graph=False, write_images=False,
        update_freq='epoch',
        #profile_batch=(10,40),
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

def compute_weights(y):
    weights = np.ones((y.shape[0], y.shape[1]), dtype=np.float32)
    uniqs, counts = np.unique(y[:, :, 0], return_counts=True)

    #weight is inversely proportional to target particle PID frequency
    for val, c in zip(uniqs, counts):
        weights[y[:, :, 0] == val] = 1.0 / c

    return weights

if __name__ == "__main__":
    infiles = list(sorted(glob.glob("out/pythia8_ttbar/tev14_pythia8_ttbar_000_*.pkl")))[:5]

    Xs = []
    ys = []
    for infile in infiles:
        X, y = prepare_data(infile)
        print(infile, X.shape)
        Xs.append(X)
        ys.append(y)
    
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    w = compute_weights(y)

    model = PFNet(
    	num_input_classes=num_input_classes,
    	num_output_classes=num_output_classes,
    	num_momentum_outputs=4, #(pT, eta, phi, E)
    	bin_size=128,
    	num_convs_id=2,
    	num_convs_reg=2,
    	num_hidden_reg_dec=2,
    	num_hidden_id_dec=2,
        num_neighbors=8,
    )

    outdir = get_rundir('experiments')
    if os.path.isdir(outdir):
        print("Output directory exists: {}".format(outdir), file=sys.stderr)
        sys.exit(1)
    
    callbacks = prepare_callbacks(model, outdir)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=0)

    #call the model once, to make sure it runs
    model(X_train[:5])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #we use the "temporal" mode to have per-particle weights
    model.compile(loss=my_loss_full, optimizer=opt, metrics=[accuracy, energy_resolution], sample_weight_mode='temporal')

    #model.load_weights("experiments/run_05/weights.500-594286.750000.hdf5")
    
    model.fit(X_train, y_train, sample_weight=w_train, validation_data=(X_test, y_test, w_test), epochs=100, batch_size=10, callbacks=callbacks)

    y_pred = model.predict(X, batch_size=10)

    np.savez("{}/pred.npz".format(outdir), y_pred=y_pred)
