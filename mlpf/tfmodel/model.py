# This file contains the generic MLPF model definitions
# PFNet: the GNN-based model with graph building based on LSH+kNN
# Transformer: the transformer-based model using fast attention
# DummyNet: simple elementwise feed forward network for cross-checking

import tensorflow as tf

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

class GHConvDense(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.activation = getattr(tf.keras.activations, kwargs.pop("activation"))
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

        adj = tf.squeeze(adj)
        
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

class NodeMessageLearnable(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):

        self.output_dim = kwargs.pop("output_dim")
        self.hidden_dim = kwargs.pop("hidden_dim")
        self.num_layers = kwargs.pop("num_layers")
        self.activation = getattr(tf.keras.activations, kwargs.pop("activation"))
        self.aggregation_direction = kwargs.pop("aggregation_direction")

        if self.aggregation_direction == "dst":
            self.agg_dim = -2
        elif self.aggregation_direction == "src":
            self.agg_dim = -3

        self.ffn = point_wise_feed_forward_network(self.output_dim, self.hidden_dim, num_layers=self.num_layers, activation=self.activation, name=kwargs.get("name")+"_ffn")
        super(NodeMessageLearnable, self).__init__(*args, **kwargs)

    def call(self, inputs):
        x, adj, msk = inputs
        avg_message = tf.reduce_mean(adj, axis=self.agg_dim)
        max_message = tf.reduce_max(adj, axis=self.agg_dim)
        x2 = tf.concat([x, avg_message, max_message], axis=-1)*msk
        return self.activation(self.ffn(x2))

def point_wise_feed_forward_network(d_model, dff, name, num_layers=1, activation='elu', dtype=tf.dtypes.float32, dim_decrease=False, dropout=0.0):

    if regularizer_weight > 0:
        bias_regularizer =  tf.keras.regularizers.L1(regularizer_weight)
        kernel_regularizer = tf.keras.regularizers.L1(regularizer_weight)
    else:
        bias_regularizer = None
        kernel_regularizer = None

    layers = []
    for ilayer in range(num_layers):
        _name = name + "_dense_{}".format(ilayer)

        layers.append(tf.keras.layers.Dense(
            dff, activation=activation, bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer, name=_name))

        if dropout>0.0:
            layers.append(tf.keras.layers.Dropout(dropout))

        if dim_decrease:
            dff = dff // 2

    layers.append(tf.keras.layers.Dense(d_model, dtype=dtype, name="{}_dense_{}".format(name, ilayer+1)))
    return tf.keras.Sequential(layers, name=name)

def get_message_layer(config_dict, name):
    config_dict = config_dict.copy()
    class_name = config_dict.pop("type")
    classes = {
        "NodeMessageLearnable": NodeMessageLearnable,
        "GHConvDense": GHConvDense
    }
    conv_cls = classes[class_name]

    return conv_cls(name=name, **config_dict)

class NodePairGaussianKernel(tf.keras.layers.Layer):
    def __init__(self, clip_value_low=0.0, dist_mult=0.1, **kwargs):
        self.clip_value_low = clip_value_low
        self.dist_mult = dist_mult
        super(NodePairGaussianKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, 1) message matrix
    """
    def call(self, x_msg_binned):
        dm = tf.expand_dims(pairwise_gaussian_dist(x_msg_binned, x_msg_binned), axis=-1)
        dm = tf.exp(-self.dist_mult*dm)
        dm = tf.clip_by_value(dm, self.clip_value_low, 1)
        return dm

class NodePairTrainableKernel(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, hidden_dim=32, num_layers=2, activation="elu", **kwargs):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = getattr(tf.keras.activations, activation)

        self.ffn_kernel = point_wise_feed_forward_network(
            self.output_dim,
            self.hidden_dim,
            kwargs.get("name") + "_" + "ffn",
            num_layers=self.num_layers,
            activation=self.activation
        )

        super(NodePairTrainableKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, output_dim) message matrix
    """
    def call(self, x_msg_binned):
        dm = pairwise_learnable_dist(x_msg_binned, x_msg_binned, self.ffn_kernel)
        dm = self.activation(dm)
        return dm

def build_kernel_from_conf(kernel_dict, name):
    kernel_dict = kernel_dict.copy()

    cls_type = kernel_dict.pop("type")
    clss = {
        "NodePairGaussianKernel": NodePairGaussianKernel,
        "NodePairTrainableKernel": NodePairTrainableKernel
    }

    return clss[cls_type](name=name, **kernel_dict)

class MessageBuildingLayerLSH(tf.keras.layers.Layer):
    def __init__(self, distance_dim=128, max_num_bins=200, bin_size=128, kernel=NodePairGaussianKernel(), **kwargs):
        self.distance_dim = distance_dim
        self.max_num_bins = max_num_bins
        self.bin_size = bin_size
        self.kernel = kernel

        super(MessageBuildingLayerLSH, self).__init__(**kwargs)

    def build(self, input_shape):
        #(n_batch, n_points, n_features)
    
        #generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = self.add_weight(
            shape=(self.distance_dim, self.max_num_bins//2), initializer="random_normal",
            trainable=False, name="lsh_projections"
        )
    
    """
    x_msg: (n_batch, n_points, n_msg_features)
    x_node: (n_batch, n_points, n_node_features)
    """
    def call(self, x_msg, x_node, msk):
        msk_f = tf.expand_dims(tf.cast(msk, x_msg.dtype), -1)

        shp = tf.shape(x_msg)
        n_batches = shp[0]
        n_points = shp[1]
        n_message_features = shp[2]

        #compute the number of LSH bins to divide the input points into on the fly
        #n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)

        #put each input item into a bin defined by the argmax output across the LSH embedding
        mul = tf.linalg.matmul(x_msg, self.codebook_random_rotations[:, :n_bins//2])
        cmul = tf.concat([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk)
        x_msg_binned = tf.gather(x_msg, bins_split, batch_dims=1)
        x_features_binned = tf.gather(x_node, bins_split, batch_dims=1)
        msk_f_binned = tf.gather(msk_f, bins_split, batch_dims=1)

        #Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg_binned)

        #remove the masked points row-wise and column-wise
        dm = tf.einsum("abijk,abi->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))
        dm = tf.einsum("abijk,abj->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))

        return bins_split, x_features_binned, dm, msk_f_binned


class OutputDecoding(tf.keras.Model):
    def __init__(self, activation, hidden_dim, regression_use_classification, num_output_classes, schema, dropout, **kwargs):
        super(OutputDecoding, self).__init__(**kwargs)

        self.regression_use_classification = regression_use_classification
        self.schema = schema
        self.dropout = dropout

        self.ffn_id = point_wise_feed_forward_network(
            num_output_classes, hidden_dim*4,
            "ffn_cls",
            dtype=tf.dtypes.float32,
            num_layers=4,
            activation=activation,
            dim_decrease=True,
            dropout=dropout
        )
        self.ffn_charge = point_wise_feed_forward_network(
            1, hidden_dim,
            "ffn_charge",
            dtype=tf.dtypes.float32,
            num_layers=2,
            activation=activation,
            dim_decrease=True,
            dropout=dropout
        )
        
        self.ffn_pt = point_wise_feed_forward_network(
            4, hidden_dim, "ffn_pt",
            dtype=tf.dtypes.float32, num_layers=3, activation=activation, dim_decrease=True,
            dropout=dropout
        )

        self.ffn_eta = point_wise_feed_forward_network(
            2, hidden_dim, "ffn_eta",
            dtype=tf.dtypes.float32, num_layers=3, activation=activation, dim_decrease=True,
            dropout=dropout
        )

        self.ffn_phi = point_wise_feed_forward_network(
            4, hidden_dim, "ffn_phi",
            dtype=tf.dtypes.float32, num_layers=3, activation=activation, dim_decrease=True,
            dropout=dropout
        )

        self.ffn_energy = point_wise_feed_forward_network(
            4, hidden_dim*4, "ffn_energy",
            dtype=tf.dtypes.float32, num_layers=4, activation=activation, dim_decrease=True,
            dropout=dropout
        )

    """
    X_input: (n_batch, n_elements, n_input_features)
    X_encoded_id: (n_batch, n_elements, n_encoded_features)
    X_encoded_reg: (n_batch, n_elements, n_encoded_features)
    msk_input: (n_batch, n_elements) boolean mask
    """
    def call(self, args, training=False):

        X_input, X_encoded, msk_input = args

        out_id_logits = self.ffn_id(X_encoded, training)*msk_input
        out_id_softmax = tf.clip_by_value(tf.nn.softmax(out_id_logits), 0, 1)
        out_charge = self.ffn_charge(X_encoded, training)*msk_input

        #orig_pt = X_input[:, :, 1:2]
        orig_eta = X_input[:, :, 2:3]

        #FIXME: schema 
        if self.schema == "cms":
            orig_sin_phi = tf.math.sin(X_input[:, :, 3:4])
            orig_cos_phi = tf.math.cos(X_input[:, :, 3:4])
            orig_energy = X_input[:, :, 4:5]
        elif self.schema == "delphes":
            orig_sin_phi = X_input[:, :, 3:4]
            orig_cos_phi = X_input[:, :, 4:5]
            orig_energy = X_input[:, :, 5:6]

        if self.regression_use_classification:
            X_encoded = tf.concat([X_encoded, tf.stop_gradient(out_id_softmax)], axis=-1)

        pred_eta_corr = self.ffn_eta(X_encoded, training)*msk_input
        pred_phi_corr = self.ffn_phi(X_encoded, training)*msk_input

        eta_sigmoid = tf.keras.activations.sigmoid(pred_eta_corr[:, :, 0:1])
        pred_eta = orig_eta*eta_sigmoid + (1.0 - eta_sigmoid)*pred_eta_corr[:, :, 1:2]

        sin_phi_sigmoid = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 0:1])
        cos_phi_sigmoid = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 2:3])
        pred_sin_phi = orig_sin_phi*sin_phi_sigmoid + (1.0 - sin_phi_sigmoid)*pred_phi_corr[:, :, 1:2]
        pred_cos_phi = orig_cos_phi*cos_phi_sigmoid + (1.0 - cos_phi_sigmoid)*pred_phi_corr[:, :, 3:4]

        X_encoded = tf.concat([X_encoded, tf.stop_gradient(pred_eta)], axis=-1)
        pred_energy_corr = self.ffn_energy(X_encoded, training)*msk_input
        pred_pt_corr = self.ffn_pt(X_encoded, training)*msk_input

        energy_sigmoid1 = tf.keras.activations.sigmoid(pred_energy_corr[:, :, 0:1])
        energy_sigmoid2 = tf.keras.activations.sigmoid(pred_energy_corr[:, :, 1:2])
        pred_energy = orig_energy*(1.0 + energy_sigmoid1*pred_energy_corr[:, :, 2:3]) + energy_sigmoid2*pred_energy_corr[:, :, 3:4]
        
        orig_pt = tf.stop_gradient(pred_energy - tf.math.log(tf.math.cosh(tf.clip_by_value(pred_eta, -8, 8))))
        pt_sigmoid1 = tf.keras.activations.sigmoid(pred_pt_corr[:, :, 0:1])
        pt_sigmoid2 = tf.keras.activations.sigmoid(pred_pt_corr[:, :, 1:2])
        pred_pt = orig_pt*(1.0 + pt_sigmoid1*pred_pt_corr[:, :, 2:3]) + pt_sigmoid2*pred_pt_corr[:, :, 3:4]

        ret = {
            "cls": out_id_softmax,
            "charge": out_charge*msk_input,
            "pt": pred_pt*msk_input,
            "eta": pred_eta*msk_input,
            "sin_phi": pred_sin_phi*msk_input,
            "cos_phi": pred_cos_phi*msk_input,
            "energy": pred_energy*msk_input,
        }

        return ret

    def set_trainable_named(self, layer_names):
        self.trainable = True

        for layer in self.layers:
            layer.trainable = False

        for layer in layer_names:
            self.get_layer(layer).trainable = True

class CombinedGraphLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
    
        self.max_num_bins = kwargs.pop("max_num_bins")
        self.bin_size = kwargs.pop("bin_size")
        self.distance_dim = kwargs.pop("distance_dim")
        self.do_layernorm = kwargs.pop("layernorm")
        self.num_node_messages = kwargs.pop("num_node_messages")
        self.dropout = kwargs.pop("dropout")
        self.kernel = kwargs.pop("kernel")
        self.node_message = kwargs.pop("node_message")
        self.hidden_dim = kwargs.pop("hidden_dim")

        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

        self.ffn_dist = point_wise_feed_forward_network(
            self.distance_dim,
            self.hidden_dim,
            kwargs.get("name") + "_ffn_dist",
            num_layers=2, activation="elu",
            dropout=self.dropout
        )
        self.message_building_layer = MessageBuildingLayerLSH(
            distance_dim=self.distance_dim,
            max_num_bins=self.max_num_bins,
            bin_size=self.bin_size,
            kernel=build_kernel_from_conf(self.kernel, kwargs.get("name")+"_kernel")
        )
        self.message_passing_layers = [
            get_message_layer(self.node_message, "{}_msg_{}".format(kwargs.get("name"), iconv)) for iconv in range(self.num_node_messages)
        ]
        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        super(CombinedGraphLayer, self).__init__(*args, **kwargs)

    def call(self, x, msk, training):

        if self.do_layernorm:
            x = self.layernorm(x)

        #compute node features for graph building
        x_dist = self.ffn_dist(x)

        #compute the element-to-element messages / distance matrix / graph structure
        bins_split, x_binned, dm, msk_binned = self.message_building_layer(x_dist, x, msk)

        #run the node update with message passing
        for msg in self.message_passing_layers:
            x_binned = msg((x_binned, dm, msk_binned))
            if self.dropout_layer:
                x_binned = self.dropout_layer(x_binned, training)

        x_enc = reverse_lsh(bins_split, x_binned)

        return {"enc": x_enc, "dist": x_dist, "bins": bins_split, "dm": dm}

class PFNetDense(tf.keras.Model):
    def __init__(self,
            multi_output=False,
            num_input_classes=8,
            num_output_classes=3,
            max_num_bins=200,
            bin_size=320,
            distance_dim=128,
            hidden_dim=256,
            layernorm=False,
            activation=tf.keras.activations.elu,
            num_node_messages=2,
            num_graph_layers=1,
            dropout=0.0,
            input_encoding="cms",
            focal_loss_from_logits=False,
            graph_kernel={"type": "NodePairGaussianKernel"},
            skip_connection=True,
            regression_use_classification=True,
            node_message={"type": "GHConvDense", "activation": "elu", "output_dim": 128, "normalize_degrees": True},
            debug=False,
            schema="cms"
        ):
        super(PFNetDense, self).__init__()

        self.multi_output = multi_output
        self.activation = activation
        self.focal_loss_from_logits = focal_loss_from_logits
        self.debug = debug
        self.separate_graph_layers = False

        self.skip_connection = skip_connection

        self.num_node_messages = num_node_messages
        self.num_graph_layers = num_graph_layers

        if input_encoding == "cms":
            self.enc = InputEncodingCMS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        kwargs_cg = {
            "max_num_bins": max_num_bins,
            "bin_size": bin_size,
            "distance_dim": distance_dim,
            "layernorm": layernorm,
            "num_node_messages": num_node_messages,
            "dropout": dropout,
            "kernel": graph_kernel,
            "node_message": node_message,
            "hidden_dim": hidden_dim
        }

        self.cg = [CombinedGraphLayer(name="cg_{}".format(i), **kwargs_cg) for i in range(num_graph_layers)]

        self.output_dec = OutputDecoding(self.activation, hidden_dim, regression_use_classification, num_output_classes, schema, dropout)

    def call(self, inputs, training=False):
        X = inputs
        debugging_data = {}

        #mask padded elements
        msk = X[:, :, 0] != 0
        msk_input = tf.expand_dims(tf.cast(msk, tf.float32), -1)

        #encode the elements for classification (id)
        enc = self.enc(X)

        enc_cg = enc
        encs = []
        for cg in self.cg:
            enc_all = cg(enc_cg, msk, training)
            enc_cg = enc_all["enc"]
            if self.debug:
                debugging_data[cg.name] = enc_all
            encs.append(enc_cg)

        dec_input = []
        if self.skip_connection:
            dec_input.append(enc)
        dec_input += encs
        dec_output = tf.concat(dec_input, axis=-1)*msk_input
        if self.debug:
            debugging_data["dec_output"] = dec_output

        ret = self.output_dec([X, dec_output, msk_input], training)

        if self.debug:
            for k in debugging_data.keys():
                ret[k] = debugging_data[k]

        if self.multi_output:
            return ret
        else:
            return tf.concat([ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1)

    def set_trainable_named(self, layer_names):
        self.trainable = True

        for layer in self.layers:
            layer.trainable = False

        self.output_dec.set_trainable_named(layer_names)

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
