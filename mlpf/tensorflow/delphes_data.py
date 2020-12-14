import numpy as np
import glob
import multiprocessing
import os
import pickle

import tensorflow as tf

padded_num_elem_size = 128*40
num_inputs = 10
num_outputs = 7

def prepare_data(fname):
    data = pickle.load(open(fname, "rb"))

    #make all inputs and outputs the same size with padding
    Xs = []
    ygens = []
    ycands = []
    for i in range(len(data["X"])):
        X = np.array(data["X"][i][:padded_num_elem_size], np.float32)
        X = np.pad(X, [(0, padded_num_elem_size - X.shape[0]), (0,0)])

        ygen = np.array(data["ygen"][i][:padded_num_elem_size], np.float32)
        ygen = np.pad(ygen, [(0, padded_num_elem_size - ygen.shape[0]), (0,0)])

        ycand = np.array(data["ycand"][i][:padded_num_elem_size], np.float32)
        ycand = np.pad(ycand, [(0, padded_num_elem_size - ycand.shape[0]), (0,0)])

        X = np.expand_dims(X, 0)
        ygen = np.expand_dims(ygen, 0)
        ycand = np.expand_dims(ycand, 0)
        
        Xs.append(X)
        ygens.append(ygen)
        ycands.append(ycand)

    X = [np.concatenate(Xs)]
    ygen = [np.concatenate(ygens)]
    ycand = [np.concatenate(ycands)]
    return X, ygen, ycand

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True, help="Input data path")
    parser.add_argument("--num-files-per-tfr", type=int, default=10, help="Number of pickle files to merge to one TFRecord file")
    args = parser.parse_args()
    return args

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _parse_tfr_element(element):
    parse_dic = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'w': tf.io.FixedLenFeature([], tf.string),
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    X = example_message['X']
    arr_X = tf.io.parse_tensor(X, out_type=tf.float32)
    y = example_message['y']
    arr_y = tf.io.parse_tensor(y, out_type=tf.float32)
    w = example_message['w']
    arr_w = tf.io.parse_tensor(w, out_type=tf.float32)
    
    #https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-577325475
    arr_X.set_shape(tf.TensorShape((None, num_inputs)))
    arr_y.set_shape(tf.TensorShape((None, num_outputs)))
    arr_w.set_shape(tf.TensorShape((None, )))
    #inds = tf.stack([arr_dm_row, arr_dm_col], axis=-1)
    #dm_sparse = tf.SparseTensor(values=arr_dm_data, indices=inds, dense_shape=[tf.shape(arr_X)[0], tf.shape(arr_X)[0]])

    return arr_X, arr_y, arr_w

def serialize_X_y_w(writer, X, y, w):
    feature = {
        'X': _bytes_feature(tf.io.serialize_tensor(X)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
        'w': _bytes_feature(tf.io.serialize_tensor(w)),
    }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(sample.SerializeToString())

def serialize_chunk(args):
    path, files, ichunk = args
    out_filename = os.path.join(path, "chunk_{}.tfrecords".format(ichunk))
    writer = tf.io.TFRecordWriter(out_filename)
    Xs = []
    ys = []
    ws = []
    dms = []

    for fi in files:
        X, y, _ = prepare_data(fi)

        Xs += X
        ys += y

    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)
    assert(Xs.shape[2] == num_inputs)
    assert(Xs.shape[1] == padded_num_elem_size)
    assert(ys.shape[2] == num_outputs)
    assert(ys.shape[1] == padded_num_elem_size)

    #set weights for each sample to be equal to the number of samples of this type
    #in the training script, this can be used to compute either inverse or class-balanced weights
    uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
    for i in range(len(ys)):
        w = np.ones(len(ys[i]), dtype=np.float32)
        for uv, uc in zip(uniq_vals, uniq_counts):
            w[ys[i][:, 0]==uv] = uc
        ws += [w]
    import pdb;pdb.set_trace()

    for X, y, w in zip(Xs, ys, ws):
        serialize_X_y_w(writer, X, y, w)

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    tf.config.experimental_run_functions_eagerly(True)

    datapath = args.datapath

    filelist = sorted(glob.glob("{}/*.pkl".format(datapath)))
    print("found {} files".format(len(filelist)))
    #means, stds = extract_means_stds(filelist)
    outpath = "{}/tfr".format(datapath)

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    pars = []
    for ichunk, files in enumerate(chunks(filelist, args.num_files_per_tfr)):
        pars += [(outpath, files, ichunk)]
    #serialize_chunk(pars[0])
    pool = multiprocessing.Pool(20)
    pool.map(serialize_chunk, pars)
    #list(map(serialize_chunk, pars))


    #Load and test the dataset 
    tfr_dataset = tf.data.TFRecordDataset(glob.glob(outpath + "/*.tfrecords"))
    dataset = tfr_dataset.map(_parse_tfr_element)
    num_ev = 0
    num_particles = 0
    for X, y, w in dataset:
        num_ev += 1
        num_particles += len(X)
        
    print("Created TFRecords dataset in {} with {} events, {} particles".format(
        datapath, num_ev, num_particles))
