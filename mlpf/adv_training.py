import tensorflow as tf
import yaml
import numpy as np
import glob
import random

from tqdm import tqdm
from tfmodel.model_setup import make_model, targets_multi_output, CustomCallback
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
    model_pf.load_weights("./experiments/cms_20210902_172254_670759.gpu0.local/weights/weights-50-45.108299.hdf5")

    model_disc = make_disc_model(config, ypred.shape[-1])

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
    Xs = []
    ycands = []
    for fi in dataset_def.val_filelist[:5]:
        X, ygen, ycand = dataset_def.prepare_data(fi)
        Xs.append(np.concatenate(X))
        ycands.append(np.concatenate(ycand))

    X_val = np.concatenate(Xs)
    ycand_val = np.concatenate(ycands)

    dataset_transform = targets_multi_output(config['dataset']['num_output_classes'])
    cb = CustomCallback(dataset_def, "logs", X_val, ycand_val, dataset_transform, config['dataset']['num_output_classes'], plot_freq=1)
    cb.set_model(model_pf)

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    random.shuffle(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, ])
    )

    n_train = 1000
    n_test = 1000
    batch_size = 4

    ds_train = dataset.take(n_train).padded_batch(batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).padded_batch(batch_size, padded_shapes=ps)

    n_train = 0
    for elem in ds_train:
        n_train += 1
    n_test = 0
    for elem in ds_test:
        n_test += 1

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
    optimizer1 = tf.keras.optimizers.Adam(lr=0.000014)
    model_pf.trainable = True
    model_disc.trainable = False
    m1.compile(loss=loss, optimizer=optimizer1)
    m1.summary()

    #The discriminator model (adversarial) is optimized to distinguish between the true target and MLPF-reconstructed events
    optimizer2 = tf.keras.optimizers.Adam(lr=0.00001)
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


        for step, (xb, yb, wb) in tqdm(enumerate(ds_train), desc="Training", total=n_train):

            msk_x = tf.cast(xb[:, :, 0:1]!=0, tf.float32)

            yp = concat_pf([model_pf(xb, training=True), xb])

            yid = tf.one_hot(tf.cast(yb[:, :, 0], tf.int32), cds["num_output_classes"])
            yb = tf.concat([yid, yb[:, :, 1:]], axis=-1)
            yb = yb*msk_x

            #Train the discriminative (adversarial) model
            #true target particles have a classification target of 1, MLPF reconstructed a target of 0
            mlpf_train_inputs = tf.concat([xb, xb], axis=0)
            # mlpf_train_inputs = mlpf_train_inputs + tf.random.normal(mlpf_train_inputs.shape, stddev=0.0001)

            mlpf_train_outputs = tf.concat([yb, yp], axis=0)
            mlpf_train_disc_targets = tf.concat([batch_size*[0.9], batch_size*[0.1]], axis=0)
            loss2 = m2.train_on_batch([mlpf_train_inputs, mlpf_train_outputs], mlpf_train_disc_targets)

            #Train the MLPF reconstruction (generative) model with an inverted target
            disc_train_disc_targets = tf.concat([batch_size*[0.9]], axis=0)
            loss1 = m1.train_on_batch(xb, disc_train_disc_targets)

            loss_tot1 += loss1
            loss_tot2 += loss2
            ibatch += 1

        import boost_histogram as bh
        import mplhep
        import matplotlib.pyplot as plt

        preds_0 = []
        preds_1 = []

        for step, (xb, yb, wb) in tqdm(enumerate(ds_test), desc="Testing", total=n_test):
            msk_x = tf.cast(xb[:, :, 0:1]!=0, tf.float32)

            yp = concat_pf([model_pf(xb, training=False), xb])

            yid = tf.one_hot(tf.cast(yb[:, :, 0], tf.int32), cds["num_output_classes"])
            yb = tf.concat([yid, yb[:, :, 1:]], axis=-1)
            yb = yb*msk_x

            #Train the discriminative (adversarial) model
            #true target particles have a classification target of 1, MLPF reconstructed a target of 0
            mlpf_train_inputs = tf.concat([xb, xb], axis=0)
            mlpf_train_outputs = tf.concat([yb, yp], axis=0)
            mlpf_train_disc_targets = tf.concat([batch_size*[0.9], batch_size*[0.1]], axis=0)
            loss2 = m2.test_on_batch([mlpf_train_inputs, mlpf_train_outputs], mlpf_train_disc_targets)

            #Train the MLPF reconstruction (generative) model with an inverted target
            disc_train_disc_targets = tf.concat([batch_size*[0.9]], axis=0)
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