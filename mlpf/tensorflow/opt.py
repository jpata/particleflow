import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tf_model import load_dataset_ttbar, my_loss_cls, num_max_elems, weight_schemes, PFNet
from tf_model import cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution
from argparse import Namespace
import kerastuner as kt

args = Namespace()
args.datapath = "data/TTbar_14TeV_TuneCUETP8M1_cfi"
args.ntrain = 500
args.ntest = 100
args.weights = "inverse"
args.convlayer = "ghconv"
args.batch_size = 10
args.nepochs = 100
args.target = "cand"
args.lr = 0.001
args.outdir = "testout"
#train_model(args)

def model_builder(hp):
    args.hidden_dim = hp.Choice('hidden_dim', values = [16, 32, 64, 128, 256])
    args.distance_dim = hp.Choice('distance_dim', values = [16, 32, 64, 128, 256])
    args.attention_layer_cutoff = hp.Float('attention_layer_cutoff', 0.01, 1.0, sampling="log")
    args.dropout = hp.Choice('dropout', values = [0.1,0.2,0.3,0.4,0.5])
    args.nbins = hp.Choice('nbins', values = [10,20,50])

    model = PFNet(
        hidden_dim=args.hidden_dim,
        distance_dim=args.distance_dim,
        convlayer=args.convlayer,
        dropout=args.dropout,
        batch_size=args.batch_size,
        nbins=args.nbins,
        attention_layer_cutoff=args.attention_layer_cutoff,
    )

    loss_fn = my_loss_cls
    model.gnn_reg.trainable = False
    model.layer_momentum.trainable = False
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    print(args)

    model.compile(optimizer=opt, loss=loss_fn,
        metrics=[cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution],
        sample_weight_mode="temporal")
    return model

    # callbacks = []
    # tb = tf.keras.callbacks.TensorBoard(
    #     log_dir=args.outdir, histogram_freq=0, write_graph=False, write_images=False, profile_batch=0,
    #     update_freq='epoch',
    # )
    # tb.set_model(model)
    # callbacks += [tb]

    # callbacks += [hp.KerasCallback(logdir, hparams)]
    # ret = model.fit(ds_train_r,
    #     validation_data=ds_test_r, epochs=args.nepochs,
    #     steps_per_epoch=args.ntrain/global_batch_size, validation_steps=args.ntest/global_batch_size,
    #     verbose=True,
    #     callbacks=callbacks
    # )

count = 0
class MyTuner(kt.Hyperband):
    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        global count
        vals = trial.hyperparameters.get_config()['values']
        with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
            score = trial.score
            tf.summary.scalar("val_loss", trial.score, step=count)
            tf.summary.scalar("hidden_dim", vals["hidden_dim"], step=count)
            tf.summary.scalar("distance_dim", vals["distance_dim"], step=count)
            tf.summary.scalar("attention_layer_cutoff", vals["attention_layer_cutoff"], step=count)
            tf.summary.scalar("dropout", vals["dropout"], step=count)
            tf.summary.scalar("nbins", vals["nbins"], step=count)
        count += 1


global_batch_size = args.batch_size
dataset = load_dataset_ttbar(args.datapath, args.target)
ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps)
ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps)

ds_train_r = ds_train.repeat()
ds_test_r = ds_test.repeat()

tuner = MyTuner(
	model_builder,
    objective = 'val_loss', 
    max_epochs = args.nepochs,
    factor = 3,
    hyperband_iterations = 1,
    directory = 'my_dir',
    project_name = 'intro_to_kt')

tuner.search(ds_train_r, validation_data=ds_test_r,
	steps_per_epoch=args.ntrain/args.batch_size,
	validation_steps=args.ntest/args.batch_size,
	callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

# with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#   hp.hparams_config(
#     hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
#     metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
#   )
