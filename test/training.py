#!/usr/bin/env python3

import uproot
import numpy as np
import glob
import os
import datetime
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
import sklearn.feature_extraction
import skimage

def normalize(arr):
    m = arr==0
    am = np.ma.array(arr, mask=m)
    
    mean = am.mean()
    am = am - mean
    std = am.std()
    am = am / std
    
    return am.data

def to_image(iev, Xs_cluster, Xs_track, ys_cand, image_bins):
    bins = [np.linspace(-5, 5, image_bins + 1), np.linspace(-5, 5, image_bins + 1)]
    h_cluster = np.histogram2d(
        Xs_cluster[iev][:, 1],
        Xs_cluster[iev][:, 2],
        weights=Xs_cluster[iev][:, 0],
        bins=bins
    )

    inner_valid = (Xs_track[iev][:, 1] != 0) & (Xs_track[iev][:, 2] != 0)
    h_track_inner = np.histogram2d(
        Xs_track[iev][inner_valid, 1],
        Xs_track[iev][inner_valid, 2],
        weights=Xs_track[iev][inner_valid, 0],
        bins=bins
    )
    outer_valid = (Xs_track[iev][:, 3] != 0) & (Xs_track[iev][:, 4] != 0)
    h_track_outer = np.histogram2d(
        Xs_track[iev][outer_valid, 3],
        Xs_track[iev][outer_valid, 4],
        weights=Xs_track[iev][outer_valid, 0],
        bins=bins
    )
    center_valid = (Xs_track[iev][:, 5] != 0) & (Xs_track[iev][:, 6] != 0)
    h_track_center = np.histogram2d(
        Xs_track[iev][center_valid, 5],
        Xs_track[iev][center_valid, 6],
        weights=Xs_track[iev][center_valid, 0],
        bins=bins
    )
    
    h_cand = np.histogram2d(
        ys_cand[iev][:, 1],
        ys_cand[iev][:, 2],
        weights=ys_cand[iev][:, 0],
        bins=bins
    )
    
    h_input = np.stack([h_cluster[0], h_track_inner[0], h_track_outer[0], h_track_center[0]], axis=-1)
    
    return h_input, h_cand[0]

def to_patches(img_in, img_out):
    channels = 3
    patch_size = 8
    patch_step = 6
    patch_minpt = 1

    patches_in = skimage.util.view_as_windows(img_in, (patch_size, patch_size, channels), patch_step)
    patches_in = patches_in.reshape((patches_in.shape[0]*patches_in.shape[1], patch_size, patch_size, channels))
    
    patches_out = skimage.util.view_as_windows(img_out, (patch_size, patch_size), patch_step)
    patches_out = patches_out.reshape((patches_out.shape[0]*patches_out.shape[1], patch_size, patch_size))

    selpatch = patches_out.sum(axis=(1,2)) > patch_minpt

    return patches_in[selpatch], patches_out[selpatch]

def build_dense(image_size, image_channels):
    
    def layer(din, n_units, do_dropout=True, do_bn=True):
        d = Dense(n_units)(din)
        d = LeakyReLU(alpha=0.2)(d)
        if do_dropout:
            d = Dropout(0.2)(d)
        if do_bn:
            d = BatchNormalization()(d)
        return d
     
    inp = Input(shape=(image_size, image_size, image_channels))
    d = Flatten()(inp)
    d = layer(d, 256, do_bn=False)
    d = layer(d, 128)
    d = layer(d, 64)
    d = layer(d, 32)
    d = layer(d, 16)
    d = layer(d, 32)
    d = layer(d, 64)
    d = layer(d, 128)
    d = layer(d, 256)
    out = Dense(image_size*image_size, activation="relu")(d)
    out = Reshape((image_size, image_size, 1))(out)
    m = Model(inp, out)
    m.summary()
    return m

def build_dense_flat(clusters_shape, tracks_shape, out_shape):
    print(clusters_shape, tracks_shape, out_shape) 
    def layer(din, n_units, do_dropout=True, do_bn=True):
        d = Dense(n_units)(din)
        d = LeakyReLU(alpha=0.2)(d)
        if do_dropout:
            d = Dropout(0.4)(d)
        if do_bn:
            d = BatchNormalization()(d)
        return d
     
    inp1 = Input(shape=clusters_shape)
    d1 = Flatten()(inp1)
    inp2 = Input(shape=tracks_shape)
    d2 = Flatten()(inp2)
    d = Concatenate()([d1, d2])
    d = layer(d, 256, do_dropout=True, do_bn=True)
    d = layer(d, 256, do_dropout=True, do_bn=True)
    d = layer(d, 256, do_dropout=True, do_bn=True)
    d = layer(d, 256, do_dropout=True, do_bn=True)
    d = layer(d, 256, do_dropout=True, do_bn=False)
    out = Dense(out_shape[0]*out_shape[1], activation="linear")(d)
    out = Reshape(out_shape)(out)
    m = Model([inp1, inp2], out)
    m.summary()
    return m

def build_generator(img_shape, gf):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.2):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='relu')(u7)

    model = Model(d0, output_img)
    model.summary()
    return model

def build_discriminator(img_shape, img_shape_out, df):

    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape_out)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

#https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
class Pix2Pix():
    def __init__(self, output_dir, image_bins, input_channels, output_channels):

        self.output_dir = output_dir

        # Input shape
        self.img_rows = image_bins
        self.img_cols = image_bins
        self.channels = input_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape_out = (self.img_rows, self.img_cols, output_channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 16

        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self.img_shape, self.img_shape_out, self.df)
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator(self.img_shape, self.gf)

        # Input images and their conditioning images
        img_in = Input(shape=self.img_shape)
        img_out = Input(shape=self.img_shape_out)

        # By conditioning on B generate a fake version of A
        fake_out = self.generator(img_in)

        ## For the combined model we will only train the generator
        self.discriminator.trainable = False

        ## Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_in, img_out])

        self.combined = Model(inputs=[img_in, img_out], outputs=[valid, fake_out])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def train(self, data_images_in, data_images_out, epochs=100, batch_size=1, sample_interval=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        losses_d = []
        losses_g = [] 

        for epoch in range(epochs):
            for batch_i, (imgs_in, imgs_out) in enumerate(myGenerator(batch_size, data_images_in, data_images_out)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_out = self.generator.predict(imgs_in)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_in, imgs_out], valid)
                d_loss_fake = self.discriminator.train_on_batch([imgs_in, fake_out], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_in, imgs_out], [valid, imgs_out])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))

            losses_d += [d_loss]
            losses_g += [g_loss]

            with open("{0}/losses.json".format(self.output_dir), "w") as fi:
                json.dump({"loss_discriminator": losses_d, "loss_generator": losses_g}, fi, indent=2)

            if epoch % sample_interval == 0:
                model.generator.save('{0}/model_g_{1}.h5'.format(self.output_dir, epoch))
                model.discriminator.save('{0}/model_d_{1}.h5'.format(self.output_dir, epoch))
                pred = model.generator.predict(data_images_in[:10])
                with open("{0}/pred_{1}.npz".format(self.output_dir, epoch), "wb") as fi:
                    np.savez(fi, pred=pred)

        return losses_d, losses_g

class RegressionModel:
    def __init__(self, output_dir, image_bins, input_channels, output_channels):
        self.output_dir = output_dir
        self.img_rows = image_bins
        self.img_cols = image_bins
        self.channels = input_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape_out = (self.img_rows, self.img_cols, output_channels)
        self.gf = 16
        
        self.generator = build_generator(self.img_shape, self.gf)
        optimizer = Adam(0.0001, 0.5)
        self.generator.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

    def train(self, data_images_in, data_images_out, data_images_in_val, data_images_out_val, epochs=100, batch_size=1, sample_interval=1):

        start_time = datetime.datetime.now()

        losses = []
        val_losses = []

        for epoch in range(epochs):
            for batch_i, (imgs_in, imgs_out) in enumerate(myGenerator(batch_size, data_images_in, data_images_out)):
                loss = self.generator.train_on_batch(imgs_in, imgs_out)
            losses += [float(loss[0])]

            elapsed_time = datetime.datetime.now() - start_time
            if epoch % sample_interval == 0:
                self.generator.save('{0}/model_g_{1}.h5'.format(self.output_dir, epoch))
                pred = self.generator.predict(data_images_in[:10])
                with open("{0}/pred_{1}.npz".format(self.output_dir, epoch), "wb") as fi:
                    np.savez(fi, pred=pred)

            val_loss = self.generator.evaluate(data_images_in_val, data_images_out_val, batch_size=100, verbose=False)
            val_losses += [float(val_loss[0])]

            with open("{0}/losses.json".format(self.output_dir), "w") as fi:
                json.dump({"loss_training": losses, "loss_testing": val_losses}, fi, indent=2)

            print("[Epoch %d/%d] [loss: %f] [val loss: %f]  time: %s" % (epoch, epochs,
                loss[0], val_loss[0],
                elapsed_time))

        return losses

def myGenerator(batch_size, data_images_in, data_images_out):
    ibatch = 0
    nbatches = data_images_in.shape[0] / batch_size
    while True:
        img_A = data_images_in[ibatch*batch_size:(ibatch+1)*batch_size]
        img_B = data_images_out[ibatch*batch_size:(ibatch+1)*batch_size]
        yield img_A, img_B
        ibatch += 1
        if ibatch == nbatches:
            break

# https://github.com/master/nima/blob/master/nima.py#L58
#def tril_indices(n, k=0):
#    """Return the indices for the lower-triangle of an (n, m) array.
#    Works similarly to `np.tril_indices`
#    Args:
#      n: the row dimension of the arrays for which the returned indices will
#        be valid.
#      k: optional diagonal offset (see `np.tril` for details).
#    Returns:
#      inds: The indices for the triangle. The returned tuple contains two arrays,
#        each with the indices along one dimension of the array.
#    """
#    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
#    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
#    mask = (m1 - m2) >= -k
#    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
#    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
#    return ix1, ix2
#
#
#def ecdf(p):
#    """Estimate the cumulative distribution function.
#    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
#    function with jump 1/n at each observation (possibly with multiple jumps
#    at one place if there are ties).
#    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
#    observations less or equal to t, i.e.,
#    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
#    Args:
#      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
#        Classes are assumed to be ordered.
#    Returns:
#      A 2-D `Tensor` of estimated ECDFs.
#    """
#    n = p.get_shape().as_list()[1]
#    indices = tril_indices(n)
#    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
#    ones = tf.ones([n * (n + 1) / 2])
#    triang = tf.scatter_nd(indices, ones, [n, n])
#    return tf.matmul(p, triang)
#
#
#def emd_loss(p, p_hat, r=2, scope=None):
#    """Compute the Earth Mover's Distance loss.
#    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
#    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
#    arXiv:1611.05916 (2016).
#    Args:
#      p: a 2-D `Tensor` of the ground truth probability mass functions.
#      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
#      r: a constant for the r-norm.
#      scope: optional name scope.
#    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
#    \sum^{N}_{i=1} p_hat_i
#    Returns:
#      A 0-D `Tensor` of r-normed EMD loss.
#    """
#    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
#        ecdf_p = ecdf(p)
#        ecdf_p_hat = ecdf(p_hat)
#        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
#        emd = tf.pow(emd, 1 / r)
#        return tf.reduce_mean(emd)

def load_data(filename_pattern, image_bins, maxclusters, maxtracks, maxcands):
    print("loading data from ROOT files: {0}".format(filename_pattern))
    Xs_cluster = []
    Xs_track = []
    ys_cand = []
        
    for fn in glob.glob(filename_pattern):
        try:
            fi = uproot.open(fn)
            tree = fi.get("pftree")
        except Exception as e:
            print("Could not open file {0}".format(fn))
            continue
        data = tree.arrays(tree.keys())
        data = {str(k, 'ascii'): v for k, v in data.items()}
        for iev in range(len(tree)):
            pt = data["pfcands_pt"][iev]
            eta = data["pfcands_eta"][iev]
            phi = data["pfcands_phi"][iev]
            charge = data["pfcands_charge"][iev]
    
            Xs_cluster += [np.stack([
                data["clusters_energy"][iev][:maxclusters],
                data["clusters_eta"][iev][:maxclusters],
                data["clusters_phi"][iev][:maxclusters],
                ], axis=1)
            ]
            Xs_track += [np.stack([
                np.abs(1.0/data["tracks_qoverp"][iev][:maxtracks]),
                data["tracks_inner_eta"][iev][:maxtracks],
                data["tracks_inner_phi"][iev][:maxtracks],
                data["tracks_outer_eta"][iev][:maxtracks],
                data["tracks_outer_phi"][iev][:maxtracks],
                data["tracks_eta"][iev][:maxtracks],
                data["tracks_phi"][iev][:maxtracks],
                ], axis=1)
            ]
            ys_cand += [np.stack([
                pt[:maxcands],
                eta[:maxcands],
                phi[:maxcands],
                charge[:maxcands]
                ], axis=1)
            ]
    print("Loaded {0} events".format(len(Xs_cluster)))

    return Xs_cluster, Xs_track, ys_cand    

def create_images(Xs_cluster, Xs_track, ys_cand, image_bins):
    #convert lists of particles to images 
    data_images_in = []
    data_images_out = []
    print("Creating images")
    for i in range(len(Xs_cluster)):
        h_in, h_out = to_image(i, Xs_cluster, Xs_track, ys_cand, image_bins)
        h_out = h_out.reshape((h_out.shape[0], h_out.shape[1], 1))
        #p_in, p_out = to_patches(h_in, h_out)
        data_images_in += [h_in]
        data_images_out += [h_out]
        if i%10 == 0:
            print("converted {0}/{1}".format(i, len(Xs_cluster)))
    print("Generated data for {0} images".format(len(data_images_in)))

    data_images_in = np.stack(data_images_in, axis=0)
    data_images_out = np.stack(data_images_out, axis=0)
    print("Stacked patches {0}".format(len(data_images_in)))
    shuf = np.random.permutation(range(len(data_images_in)))
    return data_images_in[shuf], data_images_out[shuf]

def zeropad(Xs_cluster, Xs_track, ys_cand, maxclusters, maxtracks, maxcands):
    #zero pad 
    for i in range(len(Xs_cluster)):
        Xs_cluster[i] = np.pad(Xs_cluster[i], [(0, maxclusters - Xs_cluster[i].shape[0]), (0,0)], mode='constant')
    
    for i in range(len(Xs_track)):
        Xs_track[i] = np.pad(Xs_track[i], [(0, maxtracks - Xs_track[i].shape[0]), (0,0)], mode='constant')
    
    for i in range(len(ys_cand)):
        ys_cand[i] = np.pad(ys_cand[i], [(0,maxcands - ys_cand[i].shape[0]), (0,0)], mode='constant')

    Xs_cluster = np.stack(Xs_cluster, axis=0)
    Xs_track = np.stack(Xs_track, axis=0)
    ys_cand = np.stack(ys_cand, axis=0)

    #Xs_cluster, m1, s1 = normalize_and_reshape(Xs_cluster)
    #Xs_track, m2, s2 = normalize_and_reshape(Xs_track)
    #ys_cand, m3, s3 = normalize_and_reshape(ys_cand)

    #Xs_cluster = Xs_cluster.reshape(Xs_cluster.shape[0], maxclusters, 4, 1)
    #Xs_track = Xs_track.reshape(Xs_track.shape[0], maxtracks, 5, 1)
    #ys_cand = ys_cand.reshape(ys_cand.shape[0], maxcands, 3)
    return Xs_cluster, Xs_track, ys_cand

def normalize_and_reshape(arr):
    arr = arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))
    m = arr.mean(axis=0)
    arr -= m
    s = arr.std(axis=0)
    arr /= s
    return arr, m, s

def unique_folder():
    mydir = os.path.join(os.getcwd(), "out", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    return mydir

if __name__ == "__main__":
    import setGPU
    import keras.backend as K
    import tensorflow as tf
    import keras
    from keras.datasets import mnist
    from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
    from keras.layers import BatchNormalization, Activation, ZeroPadding2D
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.convolutional import UpSampling2D, Conv2D
    from keras.models import Sequential, Model
    from keras.optimizers import Adam
    mydir = unique_folder()

#    maxclusters = -1
#    maxtracks = -1
#    maxcands = -1
#
#    image_bins = 256
#    input_channels = 4 #(calo cluster, inner track, outer track, central track)
#    output_channels = 1 #(candidates)
#   
#    data_images_in = [] 
#    data_images_out = []
#    filelist = [
#        "./data/TTbar/step3_AOD_10.npz",
#        "./data/TTbar/step3_AOD_11.npz",
#        "./data/TTbar/step3_AOD_12.npz",
#        "./data/TTbar/step3_AOD_13.npz"
#    ] 
#    for cache_filename in filelist: 
#        with open(cache_filename, "rb") as fi:
#            data = np.load(fi)
#            data_images_in += [data["data_images_in"]] 
#            data_images_out += [data["data_images_out"]]
#    data_images_in = np.vstack(data_images_in)
#    data_images_out = np.vstack(data_images_out)
# 
#    ntrain = int(0.8*len(data_images_in))
#    #model = build_dense(image_bins, input_channels)
#    #model.compile(loss="mse", optimizer=Adam(0.0001))
#    #model.fit(
#    #    data_images_in[:ntrain],
#    #    data_images_out[:ntrain],
#    #    validation_data=(data_images_in[ntrain:], data_images_out[ntrain:]),
#    #    batch_size=10, epochs=100)
#    #model.save("{0}/model.h5".format(mydir))
#
#    #model = Pix2Pix(mydir, image_bins, input_channels, output_channels)
#    #model.train(
#    #    data_images_in[:ntrain], data_images_out[:ntrain],
#    #    epochs=50, batch_size=50)
#    
#    model = RegressionModel(mydir, image_bins, input_channels, output_channels)
#    model.train(
#        data_images_in[:ntrain], data_images_out[:ntrain],
#        data_images_in[ntrain:], data_images_out[ntrain:],
#        epochs=50, batch_size=50)
   
    Xs_cluster = [] 
    Xs_track = [] 
    ys_cand = [] 
    filelist_train = [
        "./data/TTbar/step3_AOD_10_flat.npz",
        "./data/TTbar/step3_AOD_11_flat.npz",
        "./data/TTbar/step3_AOD_12_flat.npz",
        "./data/TTbar/step3_AOD_13_flat.npz",
        "./data/TTbar/step3_AOD_14_flat.npz",
        "./data/TTbar/step3_AOD_15_flat.npz",
        "./data/TTbar/step3_AOD_16_flat.npz",
        "./data/TTbar/step3_AOD_17_flat.npz",
        "./data/TTbar/step3_AOD_18_flat.npz",
        "./data/TTbar/step3_AOD_4_flat.npz",
        "./data/TTbar/step3_AOD_5_flat.npz",
        "./data/TTbar/step3_AOD_7_flat.npz",
        "./data/TTbar/step3_AOD_8_flat.npz",
        "./data/TTbar/step3_AOD_9_flat.npz",
    ]
    filelist_val = [
        "./data/TTbar/step3_AOD_1_flat.npz",
        "./data/TTbar/step3_AOD_2_flat.npz",
        "./data/TTbar/step3_AOD_3_flat.npz",
    ]
    
    for cache_filename in filelist_train: 
        with open(cache_filename, "rb") as fi:
            data = np.load(fi)
            Xs_cluster += [data["Xs_cluster"]] 
            Xs_track += [data["Xs_track"]]
            ys_cand += [data["ys_cand"]]
    Xs_cluster = np.vstack(Xs_cluster)
    Xs_track = np.vstack(Xs_track)
    ys_cand = np.vstack(ys_cand)
    Xs_track[np.isinf(Xs_track)] = 0.0
    ntrain = int(0.8*len(Xs_cluster))
 
    model = build_dense_flat(Xs_cluster[0].shape, Xs_track[0].shape, ys_cand[0].shape)
    model.compile(loss="mse", optimizer=Adam(0.0001))
    print(model.summary())
    model.fit(
        [Xs_cluster[:ntrain], Xs_track[:ntrain]], ys_cand[:ntrain],
        validation_data=[[Xs_cluster[ntrain:], Xs_track[ntrain:]], ys_cand[ntrain:]],
        epochs=50, batch_size=10)
    model.save("{0}/model_flat.h5".format(mydir))
