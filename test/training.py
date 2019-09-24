import setGPU
import uproot
import numpy as np
import keras
import glob
import keras.backend as K
import tensorflow as tf
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

maxclusters = -1
maxtracks = -1
maxcands = -1
image_bins = 256

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

def to_image(iev, Xs_cluster, Xs_track, ys_cand):
    bins = [np.linspace(-5, 5, image_bins + 1), np.linspace(-5, 5, image_bins + 1)]
    h_cluster = np.histogram2d(
        Xs_cluster[iev][:, 1],
        Xs_cluster[iev][:, 2],
        weights=Xs_cluster[iev][:, 0],
        bins=bins
    )
    h_track_inner = np.histogram2d(
        Xs_track[iev][:, 1],
        Xs_track[iev][:, 2],
        weights=Xs_track[iev][:, 0],
        bins=bins
    )
    h_track_outer = np.histogram2d(
        Xs_track[iev][:, 3],
        Xs_track[iev][:, 4],
        weights=Xs_track[iev][:, 0],
        bins=bins
    )
    
    h_cand = np.histogram2d(
        ys_cand[iev][:, 1],
        ys_cand[iev][:, 2],
        weights=ys_cand[iev][:, 0],
        bins=bins
    )
    
    h_input = np.stack([h_cluster[0], h_track_inner[0], h_track_outer[0]], axis=-1)
    return h_input, h_cand[0]

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = image_bins
        self.img_cols = image_bins
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape_out = (self.img_rows, self.img_cols, 1)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        ## Input images and their conditioning images
        #img_in = Input(shape=self.img_shape)
        #img_out = Input(shape=self.img_shape_out)

        ## By conditioning on B generate a fake version of A
        #fake_out = self.generator(img_in)

        ## For the combined model we will only train the generator
        #self.discriminator.trainable = False

        ## Discriminators determines validity of translated images / condition pairs
        #valid = self.discriminator([img_in, img_out])

        #self.combined = Model(inputs=[img_in, img_out], outputs=[valid, fake_out])
        #self.combined.compile(loss=['mse', 'mae'],
        #                      loss_weights=[1, 100],
        #                      optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        #d7 = conv2d(d6, self.gf*8)

        # Upsampling
        #u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(d6, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='relu')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape_out)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, data_images_in, data_images_out, epochs, batch_size=1, sample_interval=1):

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
            if epoch % sample_interval == 0:
                model.generator.save('model_g_{0}.h5'.format(epoch))
                model.discriminator.save('model_d_{0}.h5'.format(epoch))
                pred = model.generator.predict(data_images_in[:10])
                with open("pred_{0}.npz".format(epoch), "wb") as fi:
                    np.savez(fi, pred=pred)

        return losses_d, losses_g

class EMDModel:
    def __init__(self):
        self.base_model = Pix2Pix()
        optimizer = Adam(0.0002, 0.5)
        self.base_model.generator.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

    def train(self, data_images_in, data_images_out, data_images_in_val, data_images_out_val, epochs, batch_size=1, sample_interval=1):

        start_time = datetime.datetime.now()

        losses = []

        for epoch in range(epochs):
            for batch_i, (imgs_in, imgs_out) in enumerate(myGenerator(batch_size, data_images_in, data_images_out)):
                loss = self.base_model.generator.train_on_batch(imgs_in, imgs_out)
                losses += [loss]

            elapsed_time = datetime.datetime.now() - start_time
            if epoch % sample_interval == 0:
                self.base_model.generator.save('model_g_{0}.h5'.format(epoch))
                pred = self.base_model.generator.predict(data_images_in[:10])
                with open("pred_{0}.npz".format(epoch), "wb") as fi:
                    np.savez(fi, pred=pred)

            val_loss = self.base_model.generator.evaluate(data_images_in_val, data_images_out_val, batch_size=100, verbose=False)

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

def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.
    Works similarly to `np.tril_indices`
    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).
    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2


def ecdf(p):
    """Estimate the cumulative distribution function.
    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).
    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,
    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.
    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)


def emd_loss(p, p_hat, r=2, scope=None):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        ecdf_p = ecdf(p)
        ecdf_p_hat = ecdf(p_hat)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
        emd = tf.pow(emd, 1 / r)
        return tf.reduce_mean(emd)

def load_data(filename_pattern):
    print("loading data")
    Xs_cluster = []
    Xs_track = []
    ys_cand = []
        
    for fn in glob.glob(filename_pattern):
        fi = uproot.open(fn)
        tree = fi.get("pftree")
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
                1.0/data["tracks_qoverp"][iev][:maxtracks],
                data["tracks_inner_eta"][iev][:maxtracks],
                data["tracks_inner_phi"][iev][:maxtracks],
                data["tracks_outer_eta"][iev][:maxtracks],
                data["tracks_outer_phi"][iev][:maxtracks],
                ], axis=1)
            ]
            ys_cand += [np.stack([
                pt[:maxcands],
                eta[:maxcands],
                phi[:maxcands],
                charge[:maxcands]
                ], axis=1)
            ]

    #zero pad 
    #for i in range(len(Xs_cluster)):
    #    Xs_cluster[i] = np.pad(Xs_cluster[i], [(0, maxclusters - Xs_cluster[i].shape[0]), (0,0)], mode='constant')
    #
    #for i in range(len(Xs_track)):
    #    Xs_track[i] = np.pad(Xs_track[i], [(0, maxtracks - Xs_track[i].shape[0]), (0,0)], mode='constant')
    #
    #for i in range(len(ys_cand)):
    #    ys_cand[i] = np.pad(ys_cand[i], [(0,maxcands - ys_cand[i].shape[0]), (0,0)], mode='constant')

    #Xs_cluster = np.stack(Xs_cluster, axis=0)
    #Xs_track = np.stack(Xs_track, axis=0)
    #ys_cand = np.stack(ys_cand, axis=0)

    #Xs_cluster, m1, s1 = normalize_and_reshape(Xs_cluster)
    #Xs_track, m2, s2 = normalize_and_reshape(Xs_track)
    #ys_cand, m3, s3 = normalize_and_reshape(ys_cand)

    #Xs_cluster = Xs_cluster.reshape(Xs_cluster.shape[0], maxclusters, 4, 1)
    #Xs_track = Xs_track.reshape(Xs_track.shape[0], maxtracks, 5, 1)
    #ys_cand = ys_cand.reshape(ys_cand.shape[0], maxcands, 3)
    
    data_images_in = []
    data_images_out = []
    for i in range(len(Xs_cluster)):
        h_in, h_out = to_image(i, Xs_cluster, Xs_track, ys_cand)
        data_images_in += [h_in]
        data_images_out += [h_out]

    data_images_in = np.stack(data_images_in, axis=0)
    data_images_out = np.stack(data_images_out, axis=0)
    data_images_out = data_images_out.reshape((data_images_out.shape[0], data_images_out.shape[1], data_images_out.shape[2], 1))

    return data_images_in, data_images_out

def normalize_and_reshape(arr):
    arr = arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))
    m = arr.mean(axis=0)
    arr -= m
    s = arr.std(axis=0)
    arr /= s
    return arr, m, s

if __name__ == "__main__":
    #data_images_in, data_images_out = load_data("out_*.root")
    #with open("data.npz", "wb") as fi:
    #    np.savez(fi, data_images_in=data_images_in, data_images_out=data_images_out)

    model = EMDModel()
    with open("data.npz", "rb") as fi:
        data = np.load(fi)
        data_images_in = data["data_images_in"] 
        data_images_out = data["data_images_out"] 
   
        model.train(
            data_images_in[:8000], data_images_out[:8000],
            data_images_in[8000:], data_images_out[8000:],
            epochs=50, batch_size=50)
