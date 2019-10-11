import setGPU
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
import pickle

def get_unique_X_y(X, Xbl, y, ybl, blsize=3, maxn=3):
    uniqs = np.unique(Xbl)

    Xs = []
    ys = []
    for bl in uniqs:
        subX = X[Xbl==bl]
        suby = y[ybl==bl][:maxn]
        
        if subX.shape[0] > blsize:
            continue

        subX = np.pad(subX, ((0, blsize - subX.shape[0]), (0,0)), mode="constant")
        suby = np.pad(suby, ((0, maxn - suby.shape[0]), (0,0)), mode="constant")

        Xs += [subX]
        ys += [suby]

    return Xs, ys

if __name__ == "__main__":
    all_Xs = []
    all_ys = []
    
    for fi in glob.glob("data/TTbar/step3_AOD_1_ev*.npz"):
        fi = open(fi, "rb")
        data = np.load(fi)
        
        Xs, ys = get_unique_X_y(data["elements"], data["element_block_id"], data["candidates"], data["candidate_block_id"])
    
        all_Xs += [Xs]
        all_ys += [ys]
        
    all_Xs = np.vstack(all_Xs)
    all_ys = np.vstack(all_ys)
    
    shuf = np.random.permutation(range(len(all_Xs)))
    all_Xs = all_Xs[shuf]
    all_ys = all_ys[shuf]

    all_Xs_types = all_Xs[:, :, 0]
    all_Xs_kin = all_Xs[:, :, 1:]
    
    all_ys_types = all_ys[:, :, 0]
    all_ys_kin = all_ys[:, :, 1:]
    
    all_Xs_kin = np.copy(all_Xs_kin.reshape(all_Xs_kin.shape[0], all_Xs_kin.shape[1]*all_Xs_kin.shape[2]))
    all_ys_kin = np.copy(all_ys_kin.reshape(all_ys_kin.shape[0], all_ys_kin.shape[1]*all_ys_kin.shape[2]))

    scaler_X = sklearn.preprocessing.StandardScaler().fit(all_Xs_kin)
    scaler_y = sklearn.preprocessing.StandardScaler().fit(all_ys_kin)
    
    print("scaler_X", scaler_X.get_params())
    print("scaler_y", scaler_y.get_params())

    cluster_types = np.unique(all_Xs_types.flatten())
    pdgid_types = np.unique(all_ys_types.flatten())
    enc_X = sklearn.preprocessing.OneHotEncoder(categories=3*[cluster_types], sparse=False)
    enc_y = sklearn.preprocessing.OneHotEncoder(categories=3*[pdgid_types], sparse=False)
    enc_X.fit(all_Xs_types)
    enc_y.fit(all_ys_types)
   
    with open("preprocessing.pkl", "wb") as fi:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y, "enc_X": enc_X, "enc_y": enc_y}, fi)
 
    trf = enc_X.transform(all_Xs_types)
    X = np.hstack([trf, scaler_X.transform(all_Xs_kin)])
    
    trf = enc_y.transform(all_ys_types)
    y = np.hstack([trf, scaler_y.transform(all_ys_kin)])
    
    num_onehot_y = trf.shape[1]

    model = keras.models.Sequential()

    nunit = 128
    dropout = 0.2
    
    model.add(keras.layers.Dense(nunit, input_shape=(X.shape[1], )))
    
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
    model.add(keras.layers.Dense(y.shape[1]))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    
    model.compile(loss="mse", optimizer=opt)
    model.summary()

    ntrain = int(0.8*len(all_Xs))
    ret = model.fit(
        X[:ntrain], y[:ntrain],
        validation_data=(X[ntrain:], y[ntrain:]),
        batch_size=1000, epochs=100
    )
    model.save("regression.h5")

    pp = model.predict(X, batch_size=10000)
    pred_types = enc_y.inverse_transform(pp[:, :num_onehot_y])
  
    ncands_true = (all_ys_types!=0).sum(axis=1)
    ncands = (pred_types!=0).sum(axis=1)

    msk_test = np.zeros(len(X), dtype=np.bool)
    msk_test[ntrain:] = 1
    
    msk_1true = np.zeros(len(X), dtype=np.bool)
    msk_1true[ncands_true==1] = 1
    
    msk_1pred = np.zeros(len(X), dtype=np.bool)
    msk_1pred[ncands==1] = 1
    
    msk_2true = np.zeros(len(X), dtype=np.bool)
    msk_2true[ncands_true==2] = 1
    
    msk_3true = np.zeros(len(X), dtype=np.bool)
    msk_3true[ncands_true==3] = 1

    cmatrix_ncands = sklearn.metrics.confusion_matrix(ncands_true[msk_test], ncands[msk_test], labels=[0,1,2,3])
    print(cmatrix_ncands)

    labels = np.unique(all_ys_types)
    mat = sklearn.metrics.confusion_matrix(all_ys_types[msk_test & msk_1true, 0], pp_candids[msk_test & msk_1true, 0], labels=labels)
    print(mat) 
