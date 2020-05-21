import os
import time
import glob
import numpy as np

def get_X(X,y,w):
    return X

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="model weights to load")
    parser.add_argument("--nhidden", type=int, default=256, help="hidden dimension")
    parser.add_argument("--distance-dim", type=int, default=256, help="distance dimension")
    parser.add_argument("--num-conv", type=int, default=1, help="number of convolution layers")
    parser.add_argument("--nthreads", type=int, default=-1, help="number of threads to use")
    parser.add_argument("--ntrain", type=int, default=80, help="number of training events")
    parser.add_argument("--ntest", type=int, default=20, help="number of testing events")
    parser.add_argument("--gpu", action="store_true", help="use GPU")
    parser.add_argument("--convlayer", type=str, default="sgconv", choices=["sgconv", "ghconv"], help="Type of graph convolutional layer")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        import setGPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    from tf_model import PFNet, prepare_df
    from tf_data import _parse_tfr_element
    tfr_files = glob.glob("data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr/cand/*.tfrecords")
    #tf.config.optimizer.set_jit(True)

    if args.nthreads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.nthreads)
        tf.config.threading.set_intra_op_parallelism_threads(args.nthreads)
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    nev = args.ntest
    ps = (tf.TensorShape([None, 15]), tf.TensorShape([None, 5]), tf.TensorShape([None, ]))
    batch_size = 1
    dataset = tf.data.TFRecordDataset(tfr_files).map(
        _parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE).skip(args.ntrain).take(nev).padded_batch(batch_size, padded_shapes=ps)
    dataset_X = dataset.map(get_X)

    model = PFNet(hidden_dim=args.nhidden, distance_dim=args.distance_dim, num_conv=args.num_conv, convlayer=args.convlayer)

    num_particles = []
    for X,y,w in dataset:
        num_particles += [len(y)]

    #ensure model is compiled   
    for X in dataset_X:
        print(X.shape)
        model(X)
        break

    #load the weights
    model.load_weights(args.weights)

    #prepare the dataframe
    prepare_df(0, model, dataset, ".", save_raw=True)

    print("now timing")
    t0 = time.time()
    for X in dataset_X:
        model.predict_on_batch(X)
    print()
    t1 = time.time()
    print("prediction time per event: {:.2f} ms".format(1000.0*(t1-t0)/nev))
    print("prediction time per particle: {:.2f} us".format(1000000.0*(t1-t0)/np.median(num_particles)))

    #https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec((None, None, model.inputs[0].shape[-1]), model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    from tensorflow.python.framework import convert_to_constants
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    print(full_model.graph.inputs)
    print(full_model.graph.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./model",
                      name="frozen_graph.pb",
                      as_text=False)

    #signatures = {
    #    'serving_default': model.call,
    #}
    #model.save('model', save_format='tf', overwrite=True, include_optimizer=False)
