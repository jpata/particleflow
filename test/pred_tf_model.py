import setGPU
import tensorflow as tf
from tf_model import PFNet, prepare_df
from tf_data import _parse_tfr_element
import time
import glob

def get_X(X,y,w):
    return X

if __name__ == "__main__":
    tfr_files = glob.glob("data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr/cand/*.tfrecords")

    #tf.config.threading.set_inter_op_parallelism_threads(1)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.set_visible_devices([], 'GPU')

    nev = 100
    dataset = tf.data.TFRecordDataset(tfr_files).map(_parse_tfr_element, num_parallel_calls=1).skip(1000).take(nev)
    dataset_X = dataset.map(get_X)

    model = PFNet(hidden_dim=512, distance_dim=512, num_conv=2)

    #ensure model is compiled   
    for X in dataset_X:
        model(X)
        break

    #load the weights
    model.load_weights("experiments/run_02/weights.09-0.038057.hdf5")
    
    prepare_df(0, model, dataset, ".")

    print("now timing")
    t0 = time.time()
    model.predict(dataset_X, verbose=True)
    print()
    t1 = time.time()
    print("prediction time per event: {:.2f} ms".format(1000.0*(t1-t0)/nev))
