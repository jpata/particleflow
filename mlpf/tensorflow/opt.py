from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tf_model import load_dataset_ttbar, my_loss_cls, num_max_elems, weight_schemes, PFNet
from tf_model import cls_130, cls_211, cls_22, energy_resolution, eta_resolution, phi_resolution
from argparse import Namespace
import kerastuner as kt

args = Namespace()
args.datapath = "./data/TTbar_14TeV_TuneCUETP8M1_cfi"
args.ntrain = 1000
args.ntest = 200
args.weights = "inverse"
args.convlayer = "ghconv"
args.batch_size = 5
args.nepochs = 10
args.target = "cand"
args.lr = 0.005
args.outdir = "testout"
args.num_conv_reg = 1
args.hidden_dim_reg = 256

def model_builder(hp):
    args.hidden_dim_id = hp.Choice('hidden_dim_id', values = [16, 32, 64, 128, 256])
    args.num_convs_id = hp.Choice('num_convs_id', values = [1,2,3])
    args.distance_dim = hp.Choice('distance_dim', values = [16, 32, 64, 128, 256])
    args.num_neighbors = hp.Choice('num_neighbors', [2,3,4,5,6])
    args.dropout = hp.Choice('dropout', values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args.nbins = hp.Choice('nbins', values = [10, 20, 50])

    model = PFNet(
        hidden_dim_id=args.hidden_dim_id,
        hidden_dim_reg=args.hidden_dim_reg,
        num_convs_id=args.num_convs_id,
        num_convs_reg=args.num_convs_reg,
        distance_dim=args.distance_dim,
        convlayer=args.convlayer,
        dropout=args.dropout,
        batch_size=args.batch_size,
        nbins=args.nbins,
        num_neighbors=args.num_neighbors
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

if __name__ == "__main__":
    global_batch_size = args.batch_size
    dataset = load_dataset_ttbar(args.datapath, args.target)

    ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
    ds_train = dataset.take(args.ntrain).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = dataset.skip(args.ntrain).take(args.ntest).map(weight_schemes[args.weights]).padded_batch(global_batch_size, padded_shapes=ps).cache().prefetch(tf.data.experimental.AUTOTUNE)
    
    ds_train_r = ds_train.repeat()
    ds_test_r = ds_test.repeat()
    
    tuner = kt.Hyperband(
        model_builder,
        objective = 'val_loss', 
        max_epochs = args.nepochs,
        factor = 3,
        hyperband_iterations = 3,
        directory = 'my_dir',
        project_name = 'intro_to_kt')
    
    tuner.search(
        ds_train_r,
        validation_data=ds_test_r,
        steps_per_epoch=args.ntrain/args.batch_size,
        validation_steps=args.ntest/args.batch_size,
    )
