import tensorflow as tf
import numpy as np
import glob
import os
import pickle
import bz2
import re

from numpy.lib.recfunctions import append_fields

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


#https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_X_y_w(writer, X, y, w):
    feature = {
        'X': bytes_feature(tf.io.serialize_tensor(X)),
        'y': bytes_feature(tf.io.serialize_tensor(y)),
        'w': bytes_feature(tf.io.serialize_tensor(w)),
    }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(sample.SerializeToString())

class Dataset:
    def __init__(self, **kwargs):
        self.num_input_features = kwargs.get("num_input_features")
        self.num_output_features = kwargs.get("num_output_features")
        self.padded_num_elem_size = kwargs.get("padded_num_elem_size")
        self.raw_path = kwargs.get("raw_path")
        self.processed_path = kwargs.get("processed_path")
        self.target_particles = kwargs.get("target_particles")
        self.raw_filelist = kwargs.get("raw_files")

        self.validation_file_path = kwargs.get("validation_file_path")

        if self.raw_filelist is None:
            self.raw_filelist = sorted(glob.glob(self.raw_path))

        self.val_filelist = sorted(glob.glob(self.validation_file_path))
        print("raw files: {}".format(len(self.raw_filelist)))
        print("val files: {}".format(len(self.val_filelist)))

        self.schema = kwargs.get("schema")
        if self.schema == "delphes":
            self.prepare_data = self.prepare_data_delphes
        elif self.schema == "cms":
            self.prepare_data = self.prepare_data_cms

#       NONE = 0,
#       TRACK = 1,
#       PS1 = 2,
#       PS2 = 3,
#       ECAL = 4,
#       HCAL = 5,
#       GSF = 6,
#       BREM = 7,
#       HFEM = 8,
#       HFHAD = 9,
#       SC = 10,
#       HO = 11,
        self.elem_labels_cms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        #ch.had, n.had, HFEM, HFHAD, gamma, ele, mu
        self.class_labels_cms = [0, 211, 130, 1, 2, 22, 11, 13]

    def prepare_data_cms(self, fn):
        Xs = []
        ygens = []
        ycands = []

        data = pickle.load(open(fn, "rb"), encoding='iso-8859-1')
        for event in data:
            Xelem = event["Xelem"]
            ygen = event["ygen"]
            ycand = event["ycand"]

            #remove PS from inputs, they don't seem to be very useful
            msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3)

            Xelem = Xelem[~msk_ps]
            ygen = ygen[~msk_ps]
            ycand = ycand[~msk_ps]

            Xelem = append_fields(Xelem, "typ_idx", np.array([self.elem_labels_cms.index(int(i)) for i in Xelem["typ"]], dtype=np.float32))
            ygen = append_fields(ygen, "typ_idx", np.array([self.class_labels_cms.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32))
            ycand = append_fields(ycand, "typ_idx", np.array([self.class_labels_cms.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32))
        
            Xelem_flat = np.stack([Xelem[k].view(np.float32).data for k in [
                'typ_idx',
                'pt', 'eta', 'phi', 'e',
                'layer', 'depth', 'charge', 'trajpoint',
                'eta_ecal', 'phi_ecal', 'eta_hcal', 'phi_hcal',
                'muon_dt_hits', 'muon_csc_hits']], axis=-1
            )
            ygen_flat = np.stack([ygen[k].view(np.float32).data for k in [
                'typ_idx', 'charge',
                'pt', 'eta', 'sin_phi', 'cos_phi', 'e',
                ]], axis=-1
            )
            ycand_flat = np.stack([ycand[k].view(np.float32).data for k in [
                'typ_idx', 'charge',
                'pt', 'eta', 'sin_phi', 'cos_phi', 'e',
                ]], axis=-1
            )

            #take care of outliers
            Xelem_flat[np.isnan(Xelem_flat)] = 0
            Xelem_flat[np.abs(Xelem_flat) > 1e4] = 0
            ygen_flat[np.isnan(ygen_flat)] = 0
            ygen_flat[np.abs(ygen_flat) > 1e4] = 0
            ycand_flat[np.isnan(ycand_flat)] = 0
            ycand_flat[np.abs(ycand_flat) > 1e4] = 0

            X  = Xelem_flat[:self.padded_num_elem_size]
            X = np.pad(X, [(0, self.padded_num_elem_size - X.shape[0]), (0,0)])

            ygen = ygen_flat[:self.padded_num_elem_size]
            ygen = np.pad(ygen, [(0, self.padded_num_elem_size - ygen.shape[0]), (0,0)])

            ycand = ycand_flat[:self.padded_num_elem_size]
            ycand = np.pad(ycand, [(0, self.padded_num_elem_size - ycand.shape[0]), (0,0)])

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

    def prepare_data_delphes(self, fname):

        if fname.endswith(".pkl"):
            data = pickle.load(open(fname, "rb"))
        elif fname.endswith(".pkl.bz2"):
            data = pickle.load(bz2.BZ2File(fname, "rb"))
        else:
            raise Exception("Unknown file: {}".format(fname))

        #make all inputs and outputs the same size with padding
        Xs = []
        ygens = []
        ycands = []
        for i in range(len(data["X"])):
            X = np.array(data["X"][i][:self.padded_num_elem_size], np.float32)
            X = np.pad(X, [(0, self.padded_num_elem_size - X.shape[0]), (0,0)])

            ygen = np.array(data["ygen"][i][:self.padded_num_elem_size], np.float32)
            ygen = np.pad(ygen, [(0, self.padded_num_elem_size - ygen.shape[0]), (0,0)])

            ycand = np.array(data["ycand"][i][:self.padded_num_elem_size], np.float32)
            ycand = np.pad(ycand, [(0, self.padded_num_elem_size - ycand.shape[0]), (0,0)])

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

    def parse_tfr_element(self, element):
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
        arr_X.set_shape(tf.TensorShape((None, self.num_input_features)))
        arr_y.set_shape(tf.TensorShape((None, self.num_output_features)))
        arr_w.set_shape(tf.TensorShape((None, )))
        #inds = tf.stack([arr_dm_row, arr_dm_col], axis=-1)
        #dm_sparse = tf.SparseTensor(values=arr_dm_data, indices=inds, dense_shape=[tf.shape(arr_X)[0], tf.shape(arr_X)[0]])

        return arr_X, arr_y, arr_w

    def serialize_chunk(self, path, files, ichunk):
        out_filename = os.path.join(path, "chunk_{}.tfrecords".format(ichunk))
        
        if os.path.isfile(out_filename):
            raise Exception("Output file {} exists, please delete it if you wish to recreate it".format(out_filename))

        writer = tf.io.TFRecordWriter(out_filename)

        Xs = []
        ys = []
        ws = []

        for fi in files:
            X, ygen, ycand = self.prepare_data(fi)

            Xs += X

            if self.target_particles == "cand":
                ys += ycand
            else:
                ys += ygen

        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys)

        #set weights for each sample to be equal to the number of samples of this type
        #in the training script, this can be used to compute either inverse or class-balanced weights
        uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
        for i in range(len(ys)):
            w = np.ones(len(ys[i]), dtype=np.float32)
            for uv, uc in zip(uniq_vals, uniq_counts):
                w[ys[i][:, 0]==uv] = uc
            ws += [w]

        for X, y, w in zip(Xs, ys, ws):
            serialize_X_y_w(writer, X, y, w)
        
        print("Created {}".format(out_filename))
        writer.close()


    def process(self, num_files_per_tfr):

        processed_path = os.path.dirname(self.processed_path)

        if not os.path.isdir(processed_path):
            os.makedirs(processed_path)

        for ichunk, files in enumerate(chunks(self.raw_filelist, num_files_per_tfr)):
            print(files)
            self.serialize_chunk(processed_path, files, ichunk)
