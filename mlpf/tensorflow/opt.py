from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tf_model import load_dataset_ttbar, my_loss_full, num_max_elems, weight_schemes, PFNet
from tf_model import cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution
from argparse import Namespace
import kerastuner as kt

args = Namespace()
args.datapath = "./data/TTbar_14TeV_TuneCUETP8M1_cfi"
args.ntrain = 10000
args.ntest = 1000
args.weights = "inverse"
args.convlayer = "ghconv"
args.batch_size = 1
args.nepochs = 20
args.target = "cand"
args.lr = 0.0001
args.outdir = "testout"

def model_builder(hp):
    args.hidden_dim_id = hp.Choice('hidden_dim_id', values = [16, 32, 64, 128, 256])
    args.hidden_dim_reg = hp.Choice('hidden_dim_reg', values = [16, 32, 64, 128, 256])
    args.num_hidden_id_enc = hp.Choice('hidden_dim_id_enc', values = [0, 1, 2, 3])
    args.num_hidden_id_dec = hp.Choice('hidden_dim_id_dec', values = [0, 1, 2, 3])
    args.num_hidden_reg_enc = hp.Choice('hidden_dim_reg_enc', values = [0, 1, 2, 3])
    args.num_hidden_reg_dec = hp.Choice('hidden_dim_reg_dec', values = [0, 1, 2, 3])
    args.num_convs_id = hp.Choice('num_convs_id', values = [1, 2, 3, 4])
    args.num_convs_reg = hp.Choice('num_convs_reg', values = [1, 2, 3, 4])
    args.distance_dim = hp.Choice('distance_dim', values = [16, 32, 64, 128, 256])
    args.num_neighbors = hp.Choice('num_neighbors', [2, 3, 4, 5, 6, 7, 8, 9, 10])
    args.dropout = hp.Choice('dropout', values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args.bin_size = hp.Choice('bin_size', values = [100, 200, 500, 1000])
    args.dist_mult = hp.Choice('dist_mult', values = [0.1, 1.0, 10.0])
    args.cosine_dist = hp.Choice('cosine_dist', values = [True, False])

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
        convlayer=args.convlayer,
        dropout=args.dropout,
        bin_size=args.bin_size,
        num_neighbors=args.num_neighbors,
        dist_mult=args.dist_mult,
        cosine_dist=args.cosine_dist
    )
    loss_fn = my_loss_full
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    print(args)

    model.compile(optimizer=opt, loss=loss_fn,
        metrics=[cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution],
        sample_weight_mode="temporal")
    return model

if __name__ == "__main__":
    global_batch_size = args.batch_size
    dataset = load_dataset_ttbar(args.datapath, args.target)

    ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
    ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps)
    
    ds_train_r = ds_train.repeat()
    ds_test_r = ds_test.repeat()
    
    tuner = kt.Hyperband(
        model_builder,
        objective = 'val_loss', 
        max_epochs = args.nepochs,
        factor = 3,
        hyperband_iterations = 3,
        directory = '/scratch/joosep/kerastuner_out',
        project_name = 'mlpf')
    
    tuner.search(
        ds_train_r,
        validation_data=ds_test_r,
        steps_per_epoch=args.ntrain/args.batch_size,
        validation_steps=args.ntest/args.batch_size,
        #callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
    )
    #tuner.results_summary()
    #for trial in tuner.oracle.get_best_trials(num_trials=10):
    #    print(trial.hyperparameters.values, trial.score)
