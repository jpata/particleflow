# This file contains the generic MLPF model definitions
# PFNet: the GNN-based model with graph building based on LSH+kNN
# Transformer: the transformer-based model using fast attention
# DummyNet: simple elementwise feed forward network for cross-checking

import tensorflow as tf

from .fast_attention import Attention, SelfAttention

import numpy as np
from numpy.lib.recfunctions import append_fields

regularizer_weight = 0.0

def split_indices_to_bins(cmul, nbins, bin_size):
    bin_idx = tf.argmax(cmul, axis=-1)
    bins_split = tf.reshape(tf.argsort(bin_idx), (nbins, bin_size))
    return bins_split

def split_indices_to_bins_batch(cmul, nbins, bin_size, msk):
    bin_idx = tf.argmax(cmul, axis=-1) + tf.cast(tf.where(~msk, nbins-1, 0), tf.int64)
    bins_split = tf.reshape(tf.argsort(bin_idx), (tf.shape(cmul)[0], nbins, bin_size))
    return bins_split


def pairwise_gaussian_dist(A, B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)

    # na as a row and nb as a column vectors
    na = tf.expand_dims(na, -1)
    nb = tf.expand_dims(nb, -2)

    # return pairwise euclidean difference matrix
    # note that this matrix multiplication can go out of range for float16 in case the absolute values of A and B are large
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 1e-6))
    return D

def pairwise_learnable_dist(A, B, ffn):
    shp = tf.shape(A)

    #stack node feature vectors of src[i], dst[j] into a matrix res[i,j] = (src[i], dst[j])
    a, b, c, d = tf.meshgrid(tf.range(shp[0]), tf.range(shp[1]), tf.range(shp[2]), tf.range(shp[2]), indexing="ij")
    inds1 = tf.stack([a,b,c], axis=-1)
    inds2 = tf.stack([a,b,d], axis=-1)
    res = tf.concat([
        tf.gather_nd(A, inds1),
        tf.gather_nd(B, inds2)], axis=-1
    ) #(batch, bin, elem, elem, feat)

    #run a feedforward net on (src, dst) -> 1
    res_transformed = ffn(res)

    return res_transformed

def pairwise_sigmoid_dist(A, B):
    return tf.nn.sigmoid(tf.matmul(A, tf.transpose(B, perm=[0,2,1])))

"""
sp_a: (nbatch, nelem, nelem) sparse distance matrices
b: (nbatch, nelem, ncol) dense per-element feature matrices
"""
def sparse_dense_matmult_batch(sp_a, b):

    dtype = b.dtype
    b = tf.cast(b, tf.float32)

    num_batches = tf.shape(b)[0]

    def map_function(x):
        i, dense_slice = x[0], x[1]
        num_points = tf.shape(b)[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            tf.cast(sp_a, tf.float32), [i, 0, 0], [1, num_points, num_points]),
            [num_points, num_points])
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, num_batches, delta=1, dtype=tf.int64), b)
    ret = tf.map_fn(map_function, elems, fn_output_signature=tf.TensorSpec((None, None), b.dtype), back_prop=True)
    return tf.cast(ret, dtype) 

@tf.function
def reverse_lsh(bins_split, points_binned_enc):
    # batch_dim = points_binned_enc.shape[0]
    # n_points = points_binned_enc.shape[1]*points_binned_enc.shape[2]
    # n_features = points_binned_enc.shape[-1]
    
    shp = tf.shape(points_binned_enc)
    batch_dim = shp[0]
    n_points = shp[1]*shp[2]
    n_features = shp[-1]

    bins_split_flat = tf.reshape(bins_split, (batch_dim, n_points))
    points_binned_enc_flat = tf.reshape(points_binned_enc, (batch_dim, n_points, n_features))
    
    batch_inds = tf.reshape(tf.repeat(tf.range(batch_dim), n_points), (batch_dim, n_points))
    bins_split_flat_batch = tf.stack([batch_inds, bins_split_flat], axis=-1)

    ret = tf.scatter_nd(
        bins_split_flat_batch,
        points_binned_enc_flat,
        shape=(batch_dim, n_points, n_features)
    )
        
    return ret

class InputEncoding(tf.keras.layers.Layer):
    def __init__(self, num_input_classes):
        super(InputEncoding, self).__init__()
        self.num_input_classes = num_input_classes

    """
        X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data
    """        
    @tf.function
    def call(self, X):

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)

        #X[:, :, 1:] - all the other non-categorical features
        Xprop = X[:, :, 1:]
        return tf.concat([Xid, Xprop], axis=-1)

"""
For the CMS dataset, precompute additional features:
- log of pt and energy
- sinh, cosh of eta
- sin, cos of phi angles
- scale layer and depth values (small integers) to a larger dynamic range
"""
class InputEncodingCMS(tf.keras.layers.Layer):
    def __init__(self, num_input_classes):
        super(InputEncodingCMS, self).__init__()
        self.num_input_classes = num_input_classes

    """
        X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data
    """        
    @tf.function
    def call(self, X):

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)
        #Xpt = tf.expand_dims(tf.math.log1p(X[:, :, 1]), axis=-1)
        Xpt = tf.expand_dims(tf.math.log(X[:, :, 1] + 1.0), axis=-1)
        Xeta1 = tf.expand_dims(tf.sinh(X[:, :, 2]), axis=-1)
        Xeta2 = tf.expand_dims(tf.cosh(X[:, :, 2]), axis=-1)
        Xphi1 = tf.expand_dims(tf.sin(X[:, :, 3]), axis=-1)
        Xphi2 = tf.expand_dims(tf.cos(X[:, :, 3]), axis=-1)
        #Xe = tf.expand_dims(tf.math.log1p(X[:, :, 4]), axis=-1)
        Xe = tf.expand_dims(tf.math.log(X[:, :, 4]+1.0), axis=-1)
        Xlayer = tf.expand_dims(X[:, :, 5]*10.0, axis=-1)
        Xdepth = tf.expand_dims(X[:, :, 6]*10.0, axis=-1)

        Xphi_ecal1 = tf.expand_dims(tf.sin(X[:, :, 10]), axis=-1)
        Xphi_ecal2 = tf.expand_dims(tf.cos(X[:, :, 10]), axis=-1)
        Xphi_hcal1 = tf.expand_dims(tf.sin(X[:, :, 12]), axis=-1)
        Xphi_hcal2 = tf.expand_dims(tf.cos(X[:, :, 12]), axis=-1)

        return tf.concat([
            Xid, Xpt,
            Xeta1, Xeta2,
            Xphi1, Xphi2,
            Xe, Xlayer, Xdepth,
            Xphi_ecal1, Xphi_ecal2, Xphi_hcal1, Xphi_hcal2,
            X], axis=-1
        )

#https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConvDense(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.output_dim = kwargs.pop("output_dim")
        self.normalize_degrees = kwargs.pop("normalize_degrees", True)

        super(GHConvDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.hidden_dim = input_shape[0][-1]
        self.nelem = input_shape[0][-2]
        self.W_t = self.add_weight(shape=(self.hidden_dim, self.output_dim), name="w_t", initializer="random_normal", trainable=True, regularizer=tf.keras.regularizers.L1(regularizer_weight))
        self.b_t = self.add_weight(shape=(self.output_dim,), name="b_t", initializer="random_normal", trainable=True, regularizer=tf.keras.regularizers.L1(regularizer_weight))
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.output_dim), name="w_h", initializer="random_normal", trainable=True, regularizer=tf.keras.regularizers.L1(regularizer_weight))
        self.theta = self.add_weight(shape=(self.hidden_dim, self.output_dim), name="theta", initializer="random_normal", trainable=True, regularizer=tf.keras.regularizers.L1(regularizer_weight))
 
    #@tf.function
    def call(self, inputs):
        x, adj, msk = inputs
        #compute the normalization of the adjacency matrix
        if self.normalize_degrees:
            in_degrees = tf.clip_by_value(tf.reduce_sum(tf.abs(adj), axis=-1), 0, 1000)

            #add epsilon to prevent numerical issues from 1/sqrt(x)
            norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)*msk

        f_hom = tf.linalg.matmul(x*msk, self.theta)*msk
        if self.normalize_degrees:
            f_hom = tf.linalg.matmul(adj, f_hom*norm)*norm
        else:
            f_hom = tf.linalg.matmul(adj, f_hom)

        f_het = tf.linalg.matmul(x*msk, self.W_h)
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)

        out = gate*f_hom + (1.0-gate)*f_het
        return self.activation(out)*msk

class MPNNNodeFunction(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):

        self.output_dim = kwargs.pop("output_dim")
        self.hidden_dim = kwargs.pop("hidden_dim")
        self.num_layers = kwargs.pop("num_layers")
        self.activation = kwargs.pop("activation")

        self.ffn = point_wise_feed_forward_network(self.output_dim, self.hidden_dim, num_layers=self.num_layers, activation=self.activation)
        super(MPNNNodeFunction, self).__init__(*args, **kwargs)

    def call(self, inputs):
        x, adj, msk = inputs
        avg_message = tf.reduce_mean(adj, axis=-2)
        x2 = tf.concat([x, avg_message], axis=-1)*msk
        return self.ffn(x2)

def point_wise_feed_forward_network(d_model, dff, num_layers=1, activation='elu', dtype=tf.dtypes.float32, name=None):
    bias_regularizer =  tf.keras.regularizers.L1(regularizer_weight)
    kernel_regularizer = tf.keras.regularizers.L1(regularizer_weight)
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation=activation, bias_regularizer=bias_regularizer, kernel_regularizer=kernel_regularizer) for i in range(num_layers)] +
        [tf.keras.layers.Dense(d_model, dtype=dtype)],
        name=name
    )

def get_conv_layer(config_dict):
    config_dict = config_dict.copy()
    class_name = config_dict.pop("type")
    classes = {
        "MPNNNodeFunction": MPNNNodeFunction,
        "GHConvDense": GHConvDense
    }
    conv_cls = classes[class_name]

    return conv_cls(**config_dict)

class GraphBuilderDense(tf.keras.layers.Layer):
    def __init__(self, clip_value_low=0.0, distance_dim=128, max_num_bins=200, bin_size=128, dist_mult=0.1, **kwargs):
        self.dist_mult = dist_mult
        self.distance_dim = distance_dim
        self.max_num_bins = max_num_bins
        self.bin_size = bin_size
        self.clip_value_low = clip_value_low

        self.kernel = kwargs.pop("kernel")

        if self.kernel == "learnable":
            self.ffn_dist = point_wise_feed_forward_network(32, 32, num_layers=2, activation="elu")
        elif self.kernel == "gaussian":
            pass

        super(GraphBuilderDense, self).__init__(**kwargs)


    def build(self, input_shape):
        #(n_batch, n_points, n_features)
    
        #generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = self.add_weight(
            shape=(self.distance_dim, self.max_num_bins//2), initializer="random_normal",
            trainable=False, name="lsh_projections"
        )
        
    def call(self, x_dist, x_features, msk):
        msk_f = tf.expand_dims(tf.cast(msk, x_dist.dtype), -1)
        n_batches = tf.shape(x_dist)[0]
        n_points = tf.shape(x_dist)[1]
        n_features = tf.shape(x_dist)[2]

        #compute the number of LSH bins to divide the input points into on the fly
        #n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)

        #put each input item into a bin defined by the argmax output across the LSH embedding
        mul = tf.linalg.matmul(x_dist, self.codebook_random_rotations[:, :n_bins//2])
        cmul = tf.concat([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk)
        x_dist_binned = tf.gather(x_dist, bins_split, batch_dims=1)
        x_features_binned = tf.gather(x_features, bins_split, batch_dims=1)
        msk_f_binned = tf.gather(msk_f, bins_split, batch_dims=1)

        if self.kernel == "learnable":
            dm = pairwise_learnable_dist(x_dist_binned, x_dist_binned, self.ffn_dist)
            dm = tf.keras.activations.elu(dm)
        elif self.kernel == "gaussian":
            dm = pairwise_gaussian_dist(x_dist_binned, x_dist_binned)
            dm = tf.exp(-self.dist_mult*dm)
            dm = tf.clip_by_value(dm, self.clip_value_low, 1)

        #multiply the distance matrix row-wise and column-wise by the mask
        dm = tf.einsum("abijk,abi->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))
        dm = tf.einsum("abijk,abj->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))

        return bins_split, x_features_binned, dm, msk_f_binned

class CombinedGraphLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
    
        self.max_num_bins = kwargs.pop("max_num_bins")
        self.bin_size = kwargs.pop("bin_size")
        self.dist_mult = kwargs.pop("dist_mult")

        self.distance_dim = kwargs.pop("distance_dim")
        self.output_dim = kwargs.pop("output_dim")
        
        self.do_layernorm = kwargs.pop("layernorm")
        self.clip_value_low = kwargs.pop("clip_value_low")
        self.num_conv = kwargs.pop("num_conv")
        self.normalize_degrees = kwargs.pop("normalize_degrees")
        self.dropout = kwargs.pop("dropout")
        self.kernel = kwargs.pop("kernel")
        self.conv_config = kwargs.pop("conv_config")

        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

        self.ffn_dist = point_wise_feed_forward_network(self.distance_dim, self.distance_dim)
        self.dist = GraphBuilderDense(clip_value_low=self.clip_value_low, distance_dim=self.distance_dim, max_num_bins=self.max_num_bins , bin_size=self.bin_size, dist_mult=self.dist_mult, kernel=self.kernel)
        self.convs = [
            get_conv_layer(self.conv_config) for iconv in range(self.num_conv)
        ]
        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        super(CombinedGraphLayer, self).__init__(*args, **kwargs)

    def call(self, x, msk, training):

        if self.do_layernorm:
            x = self.layernorm(x)

        x_dist = self.ffn_dist(x)
        bins_split, x_binned, dm, msk_binned = self.dist(x_dist, x, msk)
        for conv in self.convs:
            x_binned = conv((x_binned, dm, msk_binned))
            if self.dropout_layer:
                x_binned = self.dropout_layer(x_binned, training)

        x_enc = reverse_lsh(bins_split, x_binned)

        return {"enc": x_enc, "dist": x_dist, "bins": bins_split, "dm": dm}

class PFNetDense(tf.keras.Model):
    def __init__(self,
            multi_output=False,
            num_input_classes=8,
            num_output_classes=3,
            num_momentum_outputs=3,
            max_num_bins=200,
            bin_size=320,
            dist_mult=0.1,
            distance_dim=128,
            hidden_dim=256,
            layernorm=False,
            clip_value_low=0.0,
            activation=tf.keras.activations.elu,
            num_conv=2,
            num_gsl=1,
            normalize_degrees=False,
            dropout=0.0,
            separate_momentum=True,
            input_encoding="cms",
            focal_loss_from_logits=False,
            graph_kernel="gaussian",
            skip_connection=False,
            regression_use_classification=True,
            conv_config={"type": "GHConvDense", "activation": "elu", "output_dim": 128, "normalize_degrees": True},
            debug=False
        ):
        super(PFNetDense, self).__init__()

        self.multi_output = multi_output
        self.num_momentum_outputs = num_momentum_outputs
        self.activation = activation
        self.separate_momentum = separate_momentum
        self.focal_loss_from_logits = focal_loss_from_logits
        self.debug = debug

        self.skip_connection = skip_connection
        self.regression_use_classification = regression_use_classification

        self.num_conv = num_conv
        self.num_gsl = num_gsl

        if input_encoding == "cms":
            self.enc = InputEncodingCMS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        dff = hidden_dim
        self.ffn_enc_id = point_wise_feed_forward_network(dff, dff, activation=activation, name="ffn_enc_id")
        self.ffn_enc_reg = point_wise_feed_forward_network(dff, dff, activation=activation, name="ffn_enc_reg")

        self.momentum_mult = self.add_weight(shape=(num_momentum_outputs, ), initializer=tf.keras.initializers.Ones(), name="momentum_multiplication")

        kwargs_cg = {
            "output_dim": dff,
            "max_num_bins": max_num_bins,
            "bin_size": bin_size,
            "dist_mult": dist_mult,
            "distance_dim": distance_dim,
            "layernorm": layernorm,
            "clip_value_low": clip_value_low,
            "num_conv": num_conv,
            "normalize_degrees": normalize_degrees,
            "dropout": dropout,
            "kernel": graph_kernel,
            "conv_config": conv_config
        }
        self.cg_id = [CombinedGraphLayer(**kwargs_cg) for i in range(num_gsl)]
        self.cg_reg = [CombinedGraphLayer(**kwargs_cg) for i in range(num_gsl)]

        self.ffn_id = point_wise_feed_forward_network(num_output_classes, dff, name="ffn_cls", dtype=tf.dtypes.float32, num_layers=4, activation=activation)
        self.ffn_charge = point_wise_feed_forward_network(1, dff, name="ffn_charge", dtype=tf.dtypes.float32, num_layers=2, activation=activation)
        
        if self.separate_momentum:
            self.ffn_momentum = [
                point_wise_feed_forward_network(
                    1, dff, name="ffn_momentum{}".format(imomentum),
                    dtype=tf.dtypes.float32, num_layers=4, activation=activation
                ) for imomentum in range(num_momentum_outputs)
            ]
        else:
            self.ffn_momentum = point_wise_feed_forward_network(num_momentum_outputs, dff, name="ffn_momentum", dtype=tf.dtypes.float32, num_layers=4, activation=activation)

    def call(self, inputs, training=False):
        X = inputs

        #mask padded elements
        msk = X[:, :, 0] != 0
        msk_input = tf.expand_dims(tf.cast(msk, tf.float32), -1)

        enc = self.enc(X)
        enc_id = self.activation(self.ffn_enc_id(enc))
        encs_id = []

        debugging_data = {}

        #encode the elements for classification (id)
        for cg in self.cg_id:
            enc_id_all = cg(enc_id, msk, training)
            enc_id = enc_id_all["enc"]
            if self.debug:
                debugging_data[cg.name] = enc_id_all
            encs_id.append(enc_id)

        #encode the elements for regression
        enc_reg = self.activation(self.ffn_enc_reg(enc))
        encs_reg = []
        for cg in self.cg_reg:
            enc_reg_all = cg(enc_reg, msk, training)
            enc_reg = enc_reg_all["enc"]
            if self.debug:
                debugging_data[cg.name] = enc_reg_all
            encs_reg.append(enc_reg)

        dec_input_cls = []
        if self.skip_connection:
            dec_input_cls.append(enc)
        dec_input_cls += encs_id

        graph_sum = tf.reduce_sum(encs_id[-1], axis=-2)/tf.cast(tf.shape(X)[1], X.dtype)
        graph_sum = tf.tile(tf.expand_dims(graph_sum, 1), [1, tf.shape(X)[1], 1])
        dec_input_cls.append(graph_sum)

        dec_output_id = tf.concat(dec_input_cls, axis=-1)*msk_input
        if self.debug:
            debugging_data["dec_output_id"] = dec_output_id

        out_id_logits = self.ffn_id(dec_output_id)*msk_input

        if self.focal_loss_from_logits:
            out_id_softmax = out_id_logits
        else:
            out_id_softmax = tf.clip_by_value(tf.nn.softmax(out_id_logits), 0, 1)

        out_charge = self.ffn_charge(dec_output_id)*msk_input

        dec_input_reg = []
        if self.skip_connection:
            dec_input_reg.append(enc)
        if self.regression_use_classification:
            dec_input_reg.append(tf.cast(out_id_logits, X.dtype))
        dec_input_reg += encs_reg

        graph_sum = tf.reduce_sum(encs_reg[-1], axis=-2)/tf.cast(tf.shape(X)[1], X.dtype)
        graph_sum = tf.tile(tf.expand_dims(graph_sum, 1), [1, tf.shape(X)[1], 1])
        dec_input_reg.append(graph_sum)

        dec_output_reg = tf.concat(dec_input_reg, axis=-1)*msk_input
        if self.debug:
            debugging_data["dec_output_reg"] = dec_output_reg

        if self.separate_momentum:
            pred_momentum = [ffn(dec_output_reg) for ffn in self.ffn_momentum]
            pred_momentum = tf.concat(pred_momentum, axis=-1)*msk_input
        else:
            pred_momentum = self.ffn_momentum(dec_output_reg)*msk_input

        pred_momentum = self.momentum_mult*pred_momentum

        out_charge = tf.clip_by_value(out_charge, -2, 2)

        ret = {
            "cls": out_id_softmax,
            "charge": out_charge,
            "pt": tf.exp(tf.clip_by_value(pred_momentum[:, :, 0:1], -6, 8)),
            "eta": pred_momentum[:, :, 1:2],
            "sin_phi": pred_momentum[:, :, 2:3],
            "cos_phi": pred_momentum[:, :, 3:4],
            "energy": tf.exp(tf.clip_by_value(pred_momentum[:, :, 4:5], -6, 8)),
        }
        if self.debug:
            for k in debugging_data.keys():
                ret[k] = debugging_data[k]

        if self.multi_output:
            return ret
        else:
            return tf.concat([ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1)

    def set_trainable_classification(self):
        self.trainable = True
        for layer in self.layers:
            layer.trainable = True

        self.ffn_enc_reg.trainable = False
        for cg in self.cg_reg:
            cg.trainable = False
        self.ffn_momentum.trainable = False

    def set_trainable_regression(self):
        self.trainable = True
        for layer in self.layers:
            layer.trainable = True

        self.ffn_enc_id.trainable = False
        for cg in self.cg_id:
            cg.trainable = False
        self.ffn_id.trainable = False
        self.ffn_charge.trainable = False

    def set_trainable_named(self, layer_names):
        self.trainable = True

        for layer in self.layers:
            layer.trainable = False

        for layer in layer_names:
            self.get_layer(layer).trainable = True

class DummyNet(tf.keras.Model):
    def __init__(self,
                num_input_classes=8,
                num_output_classes=3,
                num_momentum_outputs=3):
        super(DummyNet, self).__init__()

        self.num_momentum_outputs = num_momentum_outputs

        self.enc = InputEncoding(num_input_classes)

        self.ffn_id = point_wise_feed_forward_network(num_output_classes, 256)
        self.ffn_charge = point_wise_feed_forward_network(1, 256)
        self.ffn_momentum = point_wise_feed_forward_network(num_momentum_outputs, 256)

    def call(self, inputs, training):
        X = inputs
        msk_input = tf.expand_dims(tf.cast(X[:, :, 0] != 0, tf.float32), -1)

        enc = self.enc(X)

        out_id_logits = self.ffn_id(enc)
        out_charge = self.ffn_charge(enc)*msk_input

        dec_output_reg = tf.concat([enc, out_id_logits], axis=-1)
        pred_momentum = self.ffn_momentum(dec_output_reg)*msk_input

        ret = tf.concat([out_id_logits, out_charge, pred_momentum], axis=-1)

        return ret
