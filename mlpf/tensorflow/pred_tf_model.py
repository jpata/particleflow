import os
import time
import glob
import numpy as np
import json

from tf_model import parse_args

def get_X(X,y,w):
    return X

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PFNet", help="type of model to train", choices=["PFNet"])
    parser.add_argument("--weights", type=str, default=None, help="model weights to load")
    parser.add_argument("--hidden-dim-id", type=int, default=256, help="hidden dimension")
    parser.add_argument("--hidden-dim-reg", type=int, default=256, help="hidden dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="number of events in training batch")
    parser.add_argument("--num-convs-id", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--num-convs-reg", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--num-hidden-id-enc", type=int, default=2, help="number of encoder layers for multiclass")
    parser.add_argument("--num-hidden-id-dec", type=int, default=2, help="number of decoder layers for multiclass")
    parser.add_argument("--num-hidden-reg-enc", type=int, default=2, help="number of encoder layers for regression")
    parser.add_argument("--num-hidden-reg-dec", type=int, default=2, help="number of decoder layers for regression")
    parser.add_argument("--num-neighbors", type=int, default=5, help="number of knn neighbors")
    parser.add_argument("--distance-dim", type=int, default=256, help="distance dimension")
    parser.add_argument("--bin-size", type=int, default=100, help="number of points per LSH bin")
    parser.add_argument("--dist-mult", type=float, default=1.0, help="Exponential multiplier")
    parser.add_argument("--num-conv", type=int, default=1, help="number of convolution layers (powers)")
    parser.add_argument("--attention-layer-cutoff", type=float, default=0.2, help="Sparsify attention matrix by masking values below this threshold")
    parser.add_argument("--nthreads", type=int, default=-1, help="number of threads to use")
    parser.add_argument("--ntrain", type=int, default=80, help="number of training events")
    parser.add_argument("--ntest", type=int, default=20, help="number of testing events")
    parser.add_argument("--gpu", action="store_true", help="use GPU")
    parser.add_argument("--convlayer", type=str, default="sgconv", choices=["sgconv", "ghconv"], help="Type of graph convolutional layer")
    parser.add_argument("--datapath", type=str, help="Input data path", required=True)
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        import setGPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental_run_functions_eagerly(False)

    from tf_model import num_max_elems

    tf.gfile = tf.io.gfile
    from tf_model import PFNet, prepare_df
    from tf_data import _parse_tfr_element
    tfr_files = glob.glob("{}/tfr/{}/*.tfrecords".format(args.datapath, args.target))
    assert(len(tfr_files)>0)
    tf.config.optimizer.set_jit(False)

    if args.nthreads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.nthreads)
        tf.config.threading.set_intra_op_parallelism_threads(args.nthreads)
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    nev = args.ntest
    ps = (tf.TensorShape([num_max_elems, 15]), tf.TensorShape([num_max_elems, 5]), tf.TensorShape([num_max_elems, ]))
    dataset = tf.data.TFRecordDataset(tfr_files).map(
        _parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE).skip(args.ntrain).take(nev).padded_batch(args.batch_size, padded_shapes=ps)
    dataset_X = dataset.map(get_X)

    base_model = PFNet(
        hidden_dim_id=args.hidden_dim_id,
        hidden_dim_reg=args.hidden_dim_reg,
        num_convs_id=args.num_convs_id,
        num_convs_reg=args.num_convs_reg,
        num_hidden_id_enc=args.num_hidden_id_enc,
        num_hidden_id_dec=args.num_hidden_id_dec,
        num_hidden_reg_enc=args.num_hidden_reg_enc,
        num_hidden_reg_dec=args.num_hidden_reg_dec,
        distance_dim=args.distance_dim,
        convlayer=args.convlayer,
        dropout=0.0,
        bin_size=args.bin_size,
        num_neighbors=args.num_neighbors,
        dist_mult=args.dist_mult
    )
    model = base_model.create_model(num_max_elems, training=False)

    #load the weights
    model.load_weights(args.weights)
    model_dir = os.path.dirname(args.weights)

    #prepare the dataframe
    prepare_df(model, dataset, model_dir, args.target, save_raw=False)

    print("now timing")
    neval = 0
    t0 = time.time()
    for X in dataset_X:
        ret = model(X)
        print(".", end="")
        neval += 1
    print()
    t1 = time.time()
    time_per_dsrow = (t1-t0)/neval
    time_per_event = time_per_dsrow/args.batch_size
    print("prediction time per event: {:.2f} ms".format(1000.0*time_per_event))

    synthetic_timing_data = []
    for iteration in range(3):
        numev = 500
        for evsize in [1000, 5000, 10000, 20000]:
            for batch_size in [1, 2, 4, 10, 20]:
                t0 = time.time()
                for i in range(numev//batch_size):
                    x = np.random.randn(batch_size, evsize, 15)
                    model(x)
                t1 = time.time()
                dt = t1 - t0
                time_per_event = 1000.0*(dt / numev)
                synthetic_timing_data.append(
                        [{"iteration": iteration, "batch_size": batch_size, "event_size": evsize, "time_per_event": time_per_event}])
                print("Synthetic random data: batch_size={} event_size={}, time={:.2f} ms/ev".format(batch_size, evsize, time_per_event))

    with open("{}/synthetic_timing.json".format(model_dir), "w") as fi:
        json.dump(synthetic_timing_data, fi)

    #https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
    # Get frozen ConcreteFunction
    full_model = tf.function(lambda x: base_model(x, training=False))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec((None, None, 15), tf.float32))
    from tensorflow.python.framework import convert_to_constants
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    print(full_model.graph.inputs)
    print(full_model.graph.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="{}/model_frozen".format(model_dir),
                      name="frozen_graph.pb",
                      as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="{}/model_frozen".format(model_dir),
                      name="frozen_graph.pbtxt",
                      as_text=True)
    #model.save('model', overwrite=True, include_optimizer=False)
