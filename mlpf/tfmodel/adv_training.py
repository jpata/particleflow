import tensorflow as tf
import yaml
import numpy as np
import glob
import random

from model_setup import make_model, targets_multi_output, CustomCallback
from data import Dataset

def make_disc_model(config, reco_features):
    input_elems = tf.keras.layers.Input(shape=(config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))
    input_reco = tf.keras.layers.Input(shape=(config["dataset"]["padded_num_elem_size"], reco_features))
    da1 = tf.keras.layers.Dense(128, activation="elu")(input_elems)
    da2 = tf.keras.layers.Dense(128, activation="elu")(da1)
    da3 = tf.keras.layers.Dense(128, activation="elu")(da2)
    db1 = tf.keras.layers.Dense(128, activation="elu")(input_reco)
    db2 = tf.keras.layers.Dense(128, activation="elu")(db1)
    db3 = tf.keras.layers.Dense(128, activation="elu")(db2)
    c = tf.keras.layers.Concatenate()([da3, db3])
    sc = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2))(c)
    c1 = tf.keras.layers.Dense(128, activation="elu")(sc)
    c2 = tf.keras.layers.Dense(128, activation="elu")(c1)
    c3 = tf.keras.layers.Dense(1, activation="linear")(c2)
    model_disc = tf.keras.models.Model(inputs=[input_elems, input_reco], outputs=[c3])
    return model_disc 

def concat_pf(ypred):
    return tf.concat([ypred["cls"], ypred["charge"], ypred["pt"], ypred["eta"], ypred["sin_phi"], ypred["cos_phi"], ypred["energy"]], axis=-1)

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
    ypred = concat_pf(model_pf(x))
    model_pf.load_weights("./experiments2/cms-gnn-dense-lite-e0108f63.gpu0.local/weights-232-41.542252.hdf5")

    model_disc = make_disc_model(config, ypred.shape[-1])
    model_disc.summary()

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
    cb = CustomCallback("logs", X_val, ycand_val, dataset_transform, config['dataset']['num_output_classes'])
    cb.set_model(model_pf)

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    random.shuffle(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, ])
    )

    n_train = 10
    n_test = 10
    batch_size = 2

    ds_train = dataset.take(n_train).padded_batch(batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).padded_batch(batch_size, padded_shapes=ps)

    input_elems = tf.keras.layers.Input(
        shape=(config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]),
        batch_size=2*batch_size
    )
    input_reco = tf.keras.layers.Input(shape=(config["dataset"]["padded_num_elem_size"], ypred.shape[-1]))
    pf_out = tf.keras.layers.Lambda(concat_pf)(model_pf(input_elems))
    disc_out1 = model_disc([input_elems, pf_out])
    disc_out2 = model_disc([input_elems, input_reco])
    m1 = tf.keras.models.Model(inputs=[input_elems], outputs=[disc_out1])
    m2 = tf.keras.models.Model(inputs=[input_elems, input_reco], outputs=[disc_out2])

    optimizer1 = tf.keras.optimizers.Adam(lr=1e-6)
    m1.compile(loss=lambda x,y: -tf.keras.losses.binary_crossentropy(x,y, from_logits=True), optimizer=optimizer1)
    optimizer2 = tf.keras.optimizers.Adam(lr=1e-6)
    m2.compile(loss=lambda x,y: tf.keras.losses.binary_crossentropy(x,y, from_logits=True), optimizer=optimizer2)

    epochs = 100
    warmup_epochs = 0
    for epoch in range(epochs):

        loss_tot1 = 0.0
        loss_tot2 = 0.0
        loss_tot1_test = 0.0
        loss_tot2_test = 0.0

        for step, (xb, yb, _) in enumerate(ds_train):

            yp = concat_pf(model_pf(xb))
            xb = tf.concat([xb, xb], axis=0)
            yid = tf.one_hot(tf.cast(yb[:, :, 0], tf.int32), cds["num_output_classes"])
            yb = tf.concat([yid, yb[:, :, 1:]], axis=-1)
            yb = tf.concat([yb, yp], axis=0)
            yt = tf.concat([batch_size*[1], batch_size*[0]], axis=0)

            model_pf.trainable = False
            model_disc.trainable = True
            loss_tot2 += m2.train_on_batch([xb, yb], yt)
            #print(m2([xb, yb]))

            if epoch >= warmup_epochs:
                model_pf.trainable = True
                model_disc.trainable = False
                loss_tot1 -= m1.train_on_batch(xb, yt)

        for step, (xb, yb, _) in enumerate(ds_test):
            yp = concat_pf(model_pf(xb))
            xb = tf.concat([xb, xb], axis=0)
            yid = tf.one_hot(tf.cast(yb[:, :, 0], tf.int32), cds["num_output_classes"])
            yb = tf.concat([yid, yb[:, :, 1:]], axis=-1)
            yb = tf.concat([yb, yp], axis=0)
            yt = tf.concat([batch_size*[1], batch_size*[0]], axis=0)

            loss_tot2_test += m2.test_on_batch([xb, yb], yt)
            if epoch >= warmup_epochs:
                loss_tot1_test -= m1.test_on_batch(xb, yt)
        print("Epoch {}, l1={:.5E}/{:.5E}, l2={:.5E}/{:.5E}".format(epoch, loss_tot1, loss_tot1_test, loss_tot2, loss_tot2_test))
        tb.on_epoch_end(epoch, {
            "loss1": loss_tot1, "loss2": loss_tot2,
            "val_loss1": loss_tot1_test, "val_loss2": loss_tot2_test,
        })

        if epoch >= warmup_epochs:
            cp_callback.on_epoch_end(epoch)
        cb.on_epoch_end(epoch)

if __name__ == "__main__":
    config = yaml.load(open("parameters/cms-gnn-dense-lite.yaml"))
    main(config)