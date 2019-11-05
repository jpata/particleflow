try:
    import setGPU
except:
    print("Could not import setGPU, Nvidia device not found")

import numpy as np
import glob
import matplotlib.pyplot as plt
import numba
from collections import Counter
import math
import sklearn
import sklearn.metrics
import sklearn.ensemble
import scipy.sparse
import keras
import json

@numba.njit
def fill_target_matrix(mat, blids):
    for i in range(len(blids)):
        for j in range(i+1, len(blids)):
            if blids[i] == blids[j]:
                mat[i,j] = 1

@numba.njit
def encode_triu(i,j,n):
    k = (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
    return int(k)

@numba.njit
def decode_triu(k,n):
    i = n - 2 - math.floor(math.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return int(i), int(j)

@numba.njit
def fill_elem_pairs(elem_pairs_X, elem_pairs_y, elems, dm, target_matrix, skip_dm_0):
    n = 0
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            if n >= elem_pairs_X.shape[0]:
                break
            if dm[i,j] > 0 or skip_dm_0==False:
                elem_pairs_X[n, 0] = elems[i, 0]
                elem_pairs_X[n, 1] = elems[i, 1]
                elem_pairs_X[n, 2] = elems[j, 0]
                elem_pairs_X[n, 3] = elems[j, 1]
                elem_pairs_X[n, 4] = dm[i,j]
                elem_pairs_y[n, 0] = int(target_matrix[i,j])
                n += 1
    return n


#Given an event file, creates a list of all the elements pairs that have a non-infinite distance as per PFAlgo
#Will produce the X vector with [n_elem_pairs, 3], where the columns are (elem1_type, elem2_type, dist)
#and an y vector (classification target) with [n_elem_pairs, 1], where the value is 0 or 1, depending
#on if the elements are in the same miniblock according to PFAlgo
def load_element_pairs(fn):

    #Load the elements
    fi = open(fn, "rb")
    data = np.load(fi)
    els = data["elements"]
    els_blid = data["element_block_id"]

    #Load the distance matrix
    fi = open(fn.replace("ev", "dist"), "rb")
    dm = scipy.sparse.load_npz(fi).todense()
    
    #Create the matrix of elements that are connected according to the miniblock id
    target_matrix = np.zeros((len(els_blid), len(els_blid)), dtype=np.int32)
    fill_target_matrix(target_matrix, els_blid)

    #Fill the element pairs
    elem_pairs_X = np.zeros((20000,5), dtype=np.float32)
    elem_pairs_y = np.zeros((20000,1), dtype=np.float32)
    n = fill_elem_pairs(elem_pairs_X, elem_pairs_y, els, dm, target_matrix, True)

    elem_pairs_X = elem_pairs_X[:n]
    elem_pairs_y = elem_pairs_y[:n]
    
    return elem_pairs_X, elem_pairs_y

if __name__ == "__main__":
    all_elem_pairs_X = []
    all_elem_pairs_y = []
    
    for i in range(1,6):
        for j in range(500):
            fn = "data/TTbar/191009_155100/step3_AOD_{0}_ev{1}.npz".format(i, j)
            print("Loading {0}".format(fn))
            elem_pairs_X, elem_pairs_y = load_element_pairs(fn)
            all_elem_pairs_X += [elem_pairs_X]
            all_elem_pairs_y += [elem_pairs_y]
        
    elem_pairs_X = np.vstack(all_elem_pairs_X)
    elem_pairs_y = np.vstack(all_elem_pairs_y)
    
    shuf = np.random.permutation(range(len(elem_pairs_X)))
    elem_pairs_X = elem_pairs_X[shuf]
    elem_pairs_y = elem_pairs_y[shuf]

    weights = np.zeros(len(elem_pairs_y))
    ns = np.sum(elem_pairs_y[:, 0]==1)
    nb = np.sum(elem_pairs_y[:, 0]==0)
    weights[elem_pairs_y[:, 0]==1] = 1.0/ns
    weights[elem_pairs_y[:, 0]==0] = 1.0/nb

    nunit = 256
    dropout = 0.2
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(nunit, input_shape=(elem_pairs_X.shape[1], )))
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nunit))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nunit))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nunit))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nunit))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nunit))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.advanced_activations.LeakyReLU())
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    
    model.compile(loss="binary_crossentropy", optimizer=opt)
    model.summary()

    ntrain = int(0.8*len(elem_pairs_X))
    ret = model.fit(
        elem_pairs_X[:ntrain], elem_pairs_y[:ntrain, 0], sample_weight=weights[:ntrain],
        validation_data=(elem_pairs_X[ntrain:], elem_pairs_y[ntrain:, 0], weights[ntrain:]),
        batch_size=10000, epochs=100
    )

    pp = model.predict(elem_pairs_X, batch_size=10000)
    confusion = sklearn.metrics.confusion_matrix(elem_pairs_y[ntrain:, 0], pp[ntrain:]>0.5)

    print(confusion)

    training_info = {
        "loss": ret.history["loss"],
        "val_loss": ret.history["val_loss"]
    }

    with open("clustering.json", "w") as fi:
        json.dump(training_info, fi)
    model.save("clustering.h5")
