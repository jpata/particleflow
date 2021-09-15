import tensorflow as tf
import yaml
import numpy as np
import glob
import random

from tqdm import tqdm
from tfmodel.model_setup import make_model, targets_multi_output, CustomCallback
from tfmodel.utils import get_heptfds_dataset
from tfmodel.data import Dataset

#A deep sets conditional discriminator
def make_disc_model(config, reco_features):
    input_elems = tf.keras.layers.Input(shape=(config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))
    input_reco = tf.keras.layers.Input(shape=(config["dataset"]["padded_num_elem_size"], reco_features))

    nhidden = 256
    #process the input elements
    da1 = tf.keras.layers.Dense(nhidden, activation="elu")(input_elems)
    da2 = tf.keras.layers.Dense(nhidden, activation="elu")(da1)
    da3 = tf.keras.layers.Dense(nhidden, activation="elu")(da2)

    #process the target reco particles
    db1 = tf.keras.layers.Dense(nhidden, activation="elu")(input_reco)
    db2 = tf.keras.layers.Dense(nhidden, activation="elu")(db1)
    db3 = tf.keras.layers.Dense(nhidden, activation="elu")(db2)

    #concatenate the input element and reco target 
    c = tf.keras.layers.Concatenate()([da3, db3])

    #process the (element, target) pairs using a feedforward net
    dc1 = tf.keras.layers.Dense(nhidden, activation="elu")(c)
    dc2 = tf.keras.layers.Dense(nhidden/2, activation="elu")(dc1)

    #sum across the encoded (element, target) pairs in the event to create an event encoding
    msk = tf.keras.layers.Lambda(lambda x: tf.cast(x[:, :, 0:1]!=0, tf.float32))(input_elems)
    sc = tf.keras.layers.Lambda(lambda args: tf.reduce_sum(args[0]*args[1], axis=-2))([dc2, msk])

    #classify the embedded event as real (true target) or fake (MLPF reconstructed)
    c1 = tf.keras.layers.Dense(nhidden/2, activation="elu")(sc)
    c2 = tf.keras.layers.Dense(nhidden/4, activation="elu")(c1)
    c3 = tf.keras.layers.Dense(nhidden/8, activation="elu")(c2)

    #classification output logits
    c4 = tf.keras.layers.Dense(1, activation="linear")(c3)
    model_disc = tf.keras.models.Model(inputs=[input_elems, input_reco], outputs=[c4])
    return model_disc 

def concat_pf(args):
    ypred, X = args
    msk_X = tf.expand_dims(tf.cast(X[:, :, 0]!=0, tf.float32), axis=-1)
    return tf.concat([
        tf.keras.activations.softmax(ypred["cls"]*100.0, axis=-1)*msk_X,
        ypred["charge"]*msk_X,
        ypred["pt"]*msk_X,
        ypred["eta"]*msk_X,
        ypred["sin_phi"]*msk_X,
        ypred["cos_phi"]*msk_X,
        ypred["energy"]*msk_X
        ], axis=-1)

def main(config):
    tf.config.run_functions_eagerly(False)
    config['setup']['multi_output'] = True
    model_pf = make_model(config, tf.float32)

    tb = tf.keras.callbacks.TensorBoard(
        log_dir="logs", histogram_freq=1, write_graph=False, write_images=False,
        update_freq='epoch',
        profile_batch=0,
    )
    tb.set_model(model_pf)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="logs/weights-{epoch:02d}.hdf5",
        save_weights_only=True,
        verbose=0
    )
    cp_callback.set_model(model_pf)

    x = np.random.randn(1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"])
    ypred = concat_pf([model_pf(x), x])
    model_pf.load_weights("experiments/cms_20210909_132136_111774.gpu0.local/weights/weights-100-1.280379.hdf5", by_name=True)
    #model_pf.load_weights("./logs/weights-02.hdf5", by_name=True)

    model_disc = make_disc_model(config, ypred.shape[-1])

    num_gpus = 1
    ds = "cms_pf_ttbar"
    batch_size = 4
    config["datasets"][ds]["batch_per_gpu"] = batch_size
    ds_train, ds_info = get_heptfds_dataset(ds, config, num_gpus, "train", 128)
    ds_test, _ = get_heptfds_dataset(ds, config, num_gpus, "test", 128)
    ds_val, _ = get_heptfds_dataset(ds, config, num_gpus, "test", 128)

    cb = CustomCallback(
        "logs",
        ds_val,
        ds_info, plot_freq=10)

    cb.set_model(model_pf)

    input_elems = tf.keras.layers.Input(
        shape=(config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]),
        batch_size=2*batch_size,
        name="input_detector_elements"
    )
    input_reco = tf.keras.layers.Input(
        shape=(config["dataset"]["padded_num_elem_size"], ypred.shape[-1]), name="input_reco_particles")
    pf_out = tf.keras.layers.Lambda(concat_pf)([model_pf(input_elems), input_elems])
    disc_out1 = model_disc([input_elems, pf_out])
    disc_out2 = model_disc([input_elems, input_reco])
    m1 = tf.keras.models.Model(inputs=[input_elems], outputs=[disc_out1], name="model_mlpf_disc")
    m2 = tf.keras.models.Model(inputs=[input_elems, input_reco], outputs=[disc_out2], name="model_reco_disc")

    def loss(x,y):
        return tf.keras.losses.binary_crossentropy(x,y, from_logits=True)

    #The MLPF reconstruction model (generator) is optimized to confuse the discriminator
    optimizer1 = tf.keras.optimizers.Adam(lr=1e-5)
    model_disc.trainable = False
    m1.compile(loss=loss, optimizer=optimizer1)
    m1.summary()

    #The discriminator model (adversarial) is optimized to distinguish between the true target and MLPF-reconstructed events
    optimizer2 = tf.keras.optimizers.Adam(lr=1e-5)
    model_pf.trainable = False
    model_disc.trainable = True
    m2.compile(loss=loss, optimizer=optimizer2)
    m2.summary()

    epochs = 1000

    ibatch = 0
    for epoch in range(epochs):
        loss_tot1 = 0.0
        loss_tot2 = 0.0
        loss_tot1_test = 0.0
        loss_tot2_test = 0.0


        for step, (xb, yb, wb) in tqdm(enumerate(ds_train), desc="Training"):

            msk_x = tf.cast(xb[:, :, 0:1]!=0, tf.float32)

            yp = concat_pf([model_pf(xb, training=True), xb])
            yb = concat_pf([yb, xb])

            yb = yb*msk_x

            #Train the discriminative (adversarial) model
            #true target particles have a classification target of 1, MLPF reconstructed a target of 0
            mlpf_train_inputs = tf.concat([xb, xb], axis=0)
            # mlpf_train_inputs = mlpf_train_inputs + tf.random.normal(mlpf_train_inputs.shape, stddev=0.0001)

            mlpf_train_outputs = tf.concat([yb, yp], axis=0)
            mlpf_train_disc_targets = tf.concat([batch_size*[0.99], batch_size*[0.01]], axis=0)
            loss2 = m2.train_on_batch([mlpf_train_inputs, mlpf_train_outputs], mlpf_train_disc_targets)

            #Train the MLPF reconstruction (generative) model with an inverted target
            disc_train_disc_targets = tf.concat([batch_size*[1.0]], axis=0)
            loss1 = m1.train_on_batch(xb, disc_train_disc_targets)

            loss_tot1 += loss1
            loss_tot2 += loss2
            ibatch += 1

        import boost_histogram as bh
        import mplhep
        import matplotlib.pyplot as plt

        preds_0 = []
        preds_1 = []

        for step, (xb, yb, wb) in tqdm(enumerate(ds_test), desc="Testing"):
            msk_x = tf.cast(xb[:, :, 0:1]!=0, tf.float32)

            yp = concat_pf([model_pf(xb, training=False), xb])
            yb = concat_pf([yb, xb])

            yb = yb*msk_x

            #Train the discriminative (adversarial) model
            #true target particles have a classification target of 1, MLPF reconstructed a target of 0
            mlpf_train_inputs = tf.concat([xb, xb], axis=0)
            mlpf_train_outputs = tf.concat([yb, yp], axis=0)
            mlpf_train_disc_targets = tf.concat([batch_size*[0.99], batch_size*[0.01]], axis=0)
            loss2 = m2.test_on_batch([mlpf_train_inputs, mlpf_train_outputs], mlpf_train_disc_targets)

            #Train the MLPF reconstruction (generative) model with an inverted target
            disc_train_disc_targets = tf.concat([batch_size*[1.0]], axis=0)
            loss1 = m1.test_on_batch(xb, disc_train_disc_targets)

            p = m2.predict_on_batch([mlpf_train_inputs, mlpf_train_outputs])
            preds_0 += list(p[mlpf_train_disc_targets<0.5, 0])
            preds_1 += list(p[mlpf_train_disc_targets>=0.5, 0])

            loss_tot1_test += loss1
            loss_tot2_test += loss2

        print("Epoch {}, l1={:.5E}/{:.5E}, l2={:.5E}/{:.5E}".format(epoch, loss_tot1, loss_tot1_test, loss_tot2, loss_tot2_test))

        #Draw histograms of the discriminator outputs for monitoring
        minval = np.min(preds_0 + preds_1)
        maxval = np.max(preds_0 + preds_1)
        h0 = bh.Histogram(bh.axis.Regular(50, minval, maxval))
        h1 = bh.Histogram(bh.axis.Regular(50, minval, maxval))
        h0.fill(preds_0)
        h1.fill(preds_1)

        fig = plt.figure(figsize=(4,4))
        mplhep.histplot(h0, label="MLPF")
        mplhep.histplot(h1, label="Target")
        plt.xlabel("Adversarial classification output")
        plt.legend(loc="best", frameon=False)
        plt.savefig("logs/disc_{}.pdf".format(epoch), bbox_inches="tight")
        plt.close("all")

        tb.on_epoch_end(epoch, {
            "loss1": loss_tot1,
            "loss2": loss_tot2,
            "val_loss1": loss_tot1_test,
            "val_loss2": loss_tot2_test,
            "val_mean_p0": np.mean(preds_0),
            "val_std_p0": np.std(preds_0),
            "val_mean_p1": np.mean(preds_1),
            "val_std_p1": np.std(preds_1),
        })

        cp_callback.on_epoch_end(epoch)
        cb.on_epoch_end(epoch)

if __name__ == "__main__":
    config = yaml.load(open("parameters/cms.yaml"))
    main(config)