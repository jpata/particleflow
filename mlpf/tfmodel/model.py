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

def pairwise_learnable_dist(A, B, ffn, training=False):
    shp = tf.shape(A)

    #stack node feature vectors of src[i], dst[j] into a matrix res[i,j] = (src[i], dst[j])
    mg = tf.meshgrid(tf.range(shp[0]), tf.range(shp[1]), tf.range(shp[2]), tf.range(shp[2]), indexing="ij")
    inds1 = tf.stack([mg[0],mg[1],mg[2]], axis=-1)
    inds2 = tf.stack([mg[0],mg[1],mg[3]], axis=-1)
    res = tf.concat([
        tf.gather_nd(A, inds1),
        tf.gather_nd(B, inds2)], axis=-1
    ) #(batch, bin, elem, elem, feat)

    #run a feedforward net on (src, dst) -> 1
    res_transformed = ffn(res, training=training)

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

        log_energy = tf.expand_dims(tf.math.log(X[:, :, 4]+1.0), axis=-1)

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)
        #Xpt = tf.expand_dims(tf.math.log1p(X[:, :, 1]), axis=-1)
        Xpt = tf.expand_dims(tf.math.log(X[:, :, 1] + 1.0), axis=-1)

        Xpt_0p5 = tf.math.sqrt(Xpt)
        Xpt_2 = tf.math.pow(Xpt, 2)

        Xeta1 = tf.clip_by_value(tf.expand_dims(tf.sinh(X[:, :, 2]), axis=-1), -10, 10)
        Xeta2 = tf.clip_by_value(tf.expand_dims(tf.cosh(X[:, :, 2]), axis=-1), -10, 10)
        Xabs_eta = tf.expand_dims(tf.math.abs(X[:, :, 2]), axis=-1)
        Xphi1 = tf.expand_dims(tf.sin(X[:, :, 3]), axis=-1)
        Xphi2 = tf.expand_dims(tf.cos(X[:, :, 3]), axis=-1)

        #Xe = tf.expand_dims(tf.math.log1p(X[:, :, 4]), axis=-1)
        Xe = log_energy
        Xe_0p5 = tf.math.sqrt(log_energy)
        Xe_2 = tf.math.pow(log_energy, 2)

        Xe_transverse = log_energy - tf.math.log(Xeta2)

        Xlayer = tf.expand_dims(X[:, :, 5]*10.0, axis=-1)
        Xdepth = tf.expand_dims(X[:, :, 6]*10.0, axis=-1)

        Xphi_ecal1 = tf.expand_dims(tf.sin(X[:, :, 10]), axis=-1)
        Xphi_ecal2 = tf.expand_dims(tf.cos(X[:, :, 10]), axis=-1)
        Xphi_hcal1 = tf.expand_dims(tf.sin(X[:, :, 12]), axis=-1)
        Xphi_hcal2 = tf.expand_dims(tf.cos(X[:, :, 12]), axis=-1)

        return tf.concat([
            Xid,
            Xpt, Xpt_0p5, Xpt_2,
            Xeta1, Xeta2,
            Xabs_eta,
            Xphi1, Xphi2,
            Xe, Xe_0p5, Xe_2,
            Xe_transverse,
            Xlayer, Xdepth,
            Xphi_ecal1, Xphi_ecal2,
            Xphi_hcal1, Xphi_hcal2,
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

        self.ffn = point_wise_feed_forward_network(
            self.output_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            name=kwargs.get("name")+"_ffn"
        )
        super(NodeMessageLearnable, self).__init__(*args, **kwargs)

    def call(self, inputs):
        x, adj, msk = inputs

        #collect incoming messages
        avg_message_dst = tf.reduce_mean(adj, axis=-2)
        max_message_dst = tf.reduce_max(adj, axis=-2)
        min_message_dst = tf.reduce_min(adj, axis=-2)

        #collect outgoing messages
        avg_message_src = tf.reduce_mean(adj, axis=-3)
        max_message_src = tf.reduce_max(adj, axis=-3)
        min_message_src = tf.reduce_min(adj, axis=-3)

        #node update
        x2 = tf.concat([x, avg_message_dst, max_message_dst, min_message_dst, avg_message_src, max_message_src, min_message_src], axis=-1)
        return self.activation(self.ffn(x2))*msk

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
    def call(self, x_msg_binned, msk, training=False):
        dm = tf.expand_dims(pairwise_gaussian_dist(x_msg_binned*msk, x_msg_binned*msk), axis=-1)
        dm = tf.exp(-self.dist_mult*dm)
        dm = tf.clip_by_value(dm, self.clip_value_low, 1)
        return dm

class NodePairTrainableKernel(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, hidden_dim=32, num_layers=2, activation="elu", **kwargs):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = getattr(tf.keras.activations, activation)

        self.ffn_node = point_wise_feed_forward_network(
            8,
            self.hidden_dim,
            kwargs.get("name") + "_" + "node",
            num_layers=self.num_layers,
            activation=self.activation
        )

        self.pair_kernel = point_wise_feed_forward_network(
            self.output_dim,
            self.hidden_dim,
            kwargs.get("name") + "_" + "pair_kernel",
            num_layers=self.num_layers,
            activation=self.activation
        )

        super(NodePairTrainableKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, output_dim) message matrix
    """
    def call(self, x_msg_binned, msk, training=False):

        node_proj = self.activation(self.ffn_node(x_msg_binned))*msk

        dm = pairwise_learnable_dist(node_proj, node_proj, self.pair_kernel, training=training)
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
    def call(self, x_msg, x_node, msk, training=False):
        msk_f = tf.expand_dims(tf.cast(msk, x_msg.dtype), -1)

        shp = tf.shape(x_msg)
        n_batches = shp[0]
        n_points = shp[1]
        n_message_features = shp[2]

        #compute the number of LSH bins to divide the input points into on the fly
        #n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)

        #put each input item into a bin defined by the argmax output across the LSH embedding
        #FIXME: this needs n_bins to be at least 2 to work correctly!
        mul = tf.linalg.matmul(x_msg, self.codebook_random_rotations[:, :n_bins//2])
        cmul = tf.concat([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk)
        x_msg_binned = tf.gather(x_msg, bins_split, batch_dims=1)
        x_features_binned = tf.gather(x_node, bins_split, batch_dims=1)
        msk_f_binned = tf.gather(msk_f, bins_split, batch_dims=1)

        #Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg_binned, msk_f_binned, training=training)

        #remove the masked points row-wise and column-wise
        dm = tf.einsum("abijk,abi->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))
        dm = tf.einsum("abijk,abj->abijk", dm, tf.squeeze(msk_f_binned, axis=-1))

        return bins_split, x_features_binned, dm, msk_f_binned

class MessageBuildingLayerFull(tf.keras.layers.Layer):
    def __init__(self, distance_dim=128, kernel=NodePairGaussianKernel(), **kwargs):
        self.distance_dim = distance_dim
        self.kernel = kernel

        super(MessageBuildingLayerFull, self).__init__(**kwargs)
    
    """
    x_msg: (n_batch, n_points, n_msg_features)
    """
    def call(self, x_msg, msk, training=False):
        msk_f = tf.expand_dims(tf.cast(msk, x_msg.dtype), -1)

        shp = tf.shape(x_msg)
        n_batches = shp[0]
        n_points = shp[1]
        n_message_features = shp[2]

        #Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg, training=training)

        #remove the masked points row-wise and column-wise
        dm = tf.einsum("bijk,bi->bijk", dm, tf.squeeze(msk_f, axis=-1))
        dm = tf.einsum("bijk,bj->bijk", dm, tf.squeeze(msk_f, axis=-1))

        return dm

class OutputDecoding(tf.keras.Model):
    def __init__(self,
        activation="elu",
        regression_use_classification=True,
        num_output_classes=8,
        schema="cms",
        dropout=0.0,

        pt_skip_gate=True,
        eta_skip_gate=True,
        phi_skip_gate=True,
        energy_skip_gate=True,

        id_dim_decrease=True,
        charge_dim_decrease=True,
        pt_dim_decrease=False,
        eta_dim_decrease=False,
        phi_dim_decrease=False,
        energy_dim_decrease=False,

        id_hidden_dim=128,
        charge_hidden_dim=128,
        pt_hidden_dim=128,
        eta_hidden_dim=128,
        phi_hidden_dim=128,
        energy_hidden_dim=128,

        id_num_layers=4,
        charge_num_layers=2,
        pt_num_layers=3,
        eta_num_layers=3,
        phi_num_layers=3,
        energy_num_layers=3,

        layernorm=False,
        mask_reg_cls0=True,
        energy_multimodal=True,
        **kwargs):

        super(OutputDecoding, self).__init__(**kwargs)

        self.regression_use_classification = regression_use_classification
        self.schema = schema
        self.dropout = dropout

        self.pt_skip_gate = pt_skip_gate
        self.eta_skip_gate = eta_skip_gate
        self.phi_skip_gate = phi_skip_gate

        self.mask_reg_cls0 = mask_reg_cls0

        self.energy_multimodal = energy_multimodal

        self.do_layernorm = layernorm
        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, name="output_layernorm")

        self.ffn_id = point_wise_feed_forward_network(
            num_output_classes, id_hidden_dim,
            "ffn_cls",
            dtype=tf.dtypes.float32,
            num_layers=id_num_layers,
            activation=activation,
            dim_decrease=id_dim_decrease,
            dropout=dropout
        )
        self.ffn_charge = point_wise_feed_forward_network(
            1, charge_hidden_dim,
            "ffn_charge",
            dtype=tf.dtypes.float32,
            num_layers=charge_num_layers,
            activation=activation,
            dim_decrease=charge_dim_decrease,
            dropout=dropout
        )
        
        self.ffn_pt = point_wise_feed_forward_network(
            2, pt_hidden_dim, "ffn_pt",
            dtype=tf.dtypes.float32, num_layers=pt_num_layers, activation=activation, dim_decrease=pt_dim_decrease,
            dropout=dropout
        )

        self.ffn_eta = point_wise_feed_forward_network(
            2, eta_hidden_dim, "ffn_eta",
            dtype=tf.dtypes.float32, num_layers=eta_num_layers, activation=activation, dim_decrease=eta_dim_decrease,
            dropout=dropout
        )

        self.ffn_phi = point_wise_feed_forward_network(
            4, phi_hidden_dim, "ffn_phi",
            dtype=tf.dtypes.float32, num_layers=phi_num_layers, activation=activation, dim_decrease=phi_dim_decrease,
            dropout=dropout
        )

        self.ffn_energy = point_wise_feed_forward_network(
            num_output_classes if self.energy_multimodal else 1, energy_hidden_dim, "ffn_energy",
            dtype=tf.dtypes.float32, num_layers=energy_num_layers, activation=activation, dim_decrease=energy_dim_decrease,
            dropout=dropout)

    """
    X_input: (n_batch, n_elements, n_input_features) raw node input features
    X_encoded: (n_batch, n_elements, n_encoded_features) encoded/transformed node features
    msk_input: (n_batch, n_elements) boolean mask of active nodes
    """
    def call(self, args, training=False):

        X_input, X_encoded, X_encoded_energy, msk_input = args

        if self.do_layernorm:
            X_encoded = self.layernorm(X_encoded)

        out_id_logits = self.ffn_id(X_encoded, training=training)*msk_input

        out_id_softmax = tf.nn.softmax(out_id_logits, axis=-1)
        out_id_hard_softmax = tf.stop_gradient(tf.nn.softmax(100*out_id_logits, axis=-1))
        out_charge = self.ffn_charge(X_encoded, training=training)*msk_input

        orig_eta = X_input[:, :, 2:3]

        #FIXME: better schema propagation 
        #skip connection from raw input values
        if self.schema == "cms":
            orig_sin_phi = tf.math.sin(X_input[:, :, 3:4])*msk_input
            orig_cos_phi = tf.math.cos(X_input[:, :, 3:4])*msk_input
            orig_energy = X_input[:, :, 4:5]*msk_input
        elif self.schema == "delphes":
            orig_sin_phi = X_input[:, :, 3:4]*msk_input
            orig_cos_phi = X_input[:, :, 4:5]*msk_input
            orig_energy = X_input[:, :, 5:6]*msk_input

        if self.regression_use_classification:
            X_encoded = tf.concat([X_encoded, tf.stop_gradient(out_id_logits)], axis=-1)

        pred_eta_corr = self.ffn_eta(X_encoded, training=training)*msk_input
        pred_phi_corr = self.ffn_phi(X_encoded, training=training)*msk_input

        if self.eta_skip_gate:
            eta_gate = tf.keras.activations.sigmoid(pred_eta_corr[:, :, 0:1])
            pred_eta = orig_eta + pred_eta_corr[:, :, 1:2]
        else:
            pred_eta = orig_eta*pred_eta_corr[:, :, 0:1] + pred_eta_corr[:, :, 1:2]
        
        if self.phi_skip_gate:
            sin_phi_gate = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 0:1])
            cos_phi_gate = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 2:3])
            pred_sin_phi = orig_sin_phi + pred_phi_corr[:, :, 1:2]
            pred_cos_phi = orig_cos_phi + pred_phi_corr[:, :, 3:4]
        else:
            pred_sin_phi = orig_sin_phi*pred_phi_corr[:, :, 0:1] + pred_phi_corr[:, :, 1:2]
            pred_cos_phi = orig_cos_phi*pred_phi_corr[:, :, 2:3] + pred_phi_corr[:, :, 3:4]

        X_encoded_energy = tf.concat([X_encoded, X_encoded_energy], axis=-1)
        if self.regression_use_classification:
            X_encoded_energy = tf.concat([X_encoded_energy, tf.stop_gradient(out_id_logits)], axis=-1)

        pred_energy_corr = self.ffn_energy(X_encoded_energy, training=training)*msk_input

        if self.energy_multimodal:
            pred_energy = tf.reduce_sum(out_id_hard_softmax*pred_energy_corr, axis=-1, keepdims=True)
        else:
            pred_energy = pred_energy_corr

        # classwise_energy_corr = self.ffn_energy_classwise(tf.concat([orig_energy, out_id_logits], axis=-1), training=training)
        # pred_energy += classwise_energy_corr[:, :, 0:1]
        # pred_energy *= classwise_energy_corr[:, :, 1:2]

        # pred_energy = tf.math.exp(tf.clip_by_value(pred_energy, -3, 8))

        #prediction is pred_log_energy=log(energy + 1.0), energy=exp(pred_log_energy) - 1.0
        #pred_energy = tf.math.exp(tf.clip_by_value(pred_log_energy, -6, 6)) - 1.0

        #compute pt=E/cosh(eta)
        orig_pt = tf.stop_gradient(pred_energy/tf.math.cosh(tf.clip_by_value(pred_eta, -8, 8)))
        #orig_log_pt = tf.math.log(orig_pt + 1.0)

        pred_pt_corr = self.ffn_pt(X_encoded_energy, training=training)*msk_input
        if self.pt_skip_gate:
            pt_gate = tf.keras.activations.sigmoid(pred_pt_corr[:, :, 0:1])
            pred_pt = orig_pt + pt_gate*pred_pt_corr[:, :, 1:2]
        else:
            pred_pt = orig_pt*pred_pt_corr[:, :, 0:1] + pred_pt_corr[:, :, 1:2]
        
        if self.mask_reg_cls0:
            msk_output = tf.expand_dims(tf.cast(tf.argmax(out_id_hard_softmax, axis=-1)!=0, tf.float32), axis=-1)
            out_charge = out_charge*msk_output
            pred_pt = pred_pt*msk_output
            pred_eta = pred_eta*msk_output
            pred_sin_phi = pred_sin_phi*msk_output
            pred_cos_phi = pred_cos_phi*msk_output
            pred_energy = pred_energy*msk_output

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

    def set_trainable_regression(self):
        self.ffn_id.trainable = False
        self.ffn_charge.trainable = False
        self.ffn_phi.trainable = False
        self.ffn_eta.trainable = False
        self.ffn_pt.trainable = False
        self.ffn_energy.trainable = True

    def set_trainable_classification(self):
        self.ffn_id.trainable = True
        self.ffn_charge.trainable = True
        self.ffn_phi.trainable = False
        self.ffn_eta.trainable = False
        self.ffn_pt.trainable = False
        self.ffn_energy.trainable = False

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
        self.ffn_dist_hidden_dim = kwargs.pop("ffn_dist_hidden_dim")
        self.do_lsh = kwargs.pop("do_lsh", True)
        self.ffn_dist_num_layers = kwargs.pop("ffn_dist_num_layers", 2)
        self.activation = getattr(tf.keras.activations, kwargs.pop("activation"))
        self.dist_activation = getattr(tf.keras.activations, kwargs.pop("dist_activation", "linear"))

        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6, name=kwargs.get("name")+"_layernorm")

        #self.gaussian_noise = tf.keras.layers.GaussianNoise(0.01)
        self.ffn_dist = point_wise_feed_forward_network(
            self.distance_dim,
            self.ffn_dist_hidden_dim,
            kwargs.get("name") + "_ffn_dist",
            num_layers=self.ffn_dist_num_layers, activation=self.activation,
            dropout=self.dropout
        )
        if self.do_lsh:
            self.message_building_layer = MessageBuildingLayerLSH(
                distance_dim=self.distance_dim,
                max_num_bins=self.max_num_bins,
                bin_size=self.bin_size,
                kernel=build_kernel_from_conf(self.kernel, kwargs.get("name")+"_kernel")
            )
        else:
            self.message_building_layer = MessageBuildingLayerFull(
                distance_dim=self.distance_dim,
                kernel=build_kernel_from_conf(self.kernel, kwargs.get("name")+"_kernel")
            )

        self.message_passing_layers = [
            get_message_layer(self.node_message, "{}_msg_{}".format(kwargs.get("name"), iconv)) for iconv in range(self.num_node_messages)
        ]
        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        super(CombinedGraphLayer, self).__init__(*args, **kwargs)

    def call(self, x, msk, training=False):

        if self.do_layernorm:
            x = self.layernorm(x, training=training)

        #compute node features for graph building
        x_dist = self.dist_activation(self.ffn_dist(x, training=training))

        #x_dist = self.gaussian_noise(x_dist, training=training)

        #compute the element-to-element messages / distance matrix / graph structure
        if self.do_lsh:
            bins_split, x, dm, msk_f = self.message_building_layer(x_dist, x, msk)
        else:
            dm = self.message_building_layer(x_dist, msk)
            msk_f = tf.expand_dims(tf.cast(msk, x.dtype), axis=-1)
            bins_split = None

        #run the node update with message passing
        for msg in self.message_passing_layers:
            x = msg((x, dm, msk_f))

            #x = self.gaussian_noise(x, training=training)

            if self.dropout_layer:
                x = self.dropout_layer(x, training=training)

        #undo the binning according to the element-to-bin indices
        if self.do_lsh:
            x = reverse_lsh(bins_split, x)

        return {"enc": x, "dist": x_dist, "bins": bins_split, "dm": dm}

class PFNetDense(tf.keras.Model):
    def __init__(self,
            do_node_encoding=False,
            node_encoding_hidden_dim=128,
            dropout=0.0,
            activation="gelu",
            multi_output=False,
            num_input_classes=8,
            num_output_classes=3,
            num_graph_layers_common=1,
            num_graph_layers_energy=1,
            input_encoding="cms",
            skip_connection=True,
            graph_kernel={},
            combined_graph_layer={},
            node_message={},
            output_decoding={},
            debug=False,
            schema="cms",
            node_update_mode="concat"
        ):
        super(PFNetDense, self).__init__()

        self.multi_output = multi_output
        self.debug = debug

        self.skip_connection = skip_connection
        
        self.do_node_encoding = do_node_encoding
        self.node_encoding_hidden_dim = node_encoding_hidden_dim
        self.dropout = dropout
        self.node_update_mode = node_update_mode
        self.activation = getattr(tf.keras.activations, activation)

        if self.do_node_encoding:
            self.node_encoding = point_wise_feed_forward_network(
                combined_graph_layer["node_message"]["output_dim"],
                self.node_encoding_hidden_dim,
                "node_encoding",
                num_layers=1,
                activation=self.activation,
                dropout=self.dropout
            )

        if input_encoding == "cms":
            self.enc = InputEncodingCMS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        self.cg = [CombinedGraphLayer(name="cg_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_common)]
        self.cg_energy = [CombinedGraphLayer(name="cg_energy_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_energy)]

        output_decoding["schema"] = schema
        output_decoding["num_output_classes"] = num_output_classes
        self.output_dec = OutputDecoding(**output_decoding)

    def call(self, inputs, training=False):
        X = inputs
        debugging_data = {}

        #mask padded elements
        msk = X[:, :, 0] != 0
        msk_input = tf.expand_dims(tf.cast(msk, tf.float32), -1)

        #encode the elements for classification (id)
        X_enc = self.enc(X)

        encs = []
        if self.skip_connection:
            encs.append(X_enc)

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_ffn = self.activation(self.node_encoding(X_enc_cg, training=training))
            X_enc_cg = X_enc_ffn

        for cg in self.cg:
            enc_all = cg(X_enc_cg, msk, training=training)

            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all
        
        if self.node_update_mode == "concat":
            dec_output = tf.concat(encs, axis=-1)*msk_input
        elif self.node_update_mode == "additive":
            dec_output = X_enc_cg

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_cg = X_enc_ffn

        encs_energy = []
        for cg in self.cg_energy:
            enc_all = cg(X_enc_cg, msk, training=training)
            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs_energy.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all
            encs_energy.append(X_enc_cg)

        if self.node_update_mode == "concat":
            dec_output_energy = tf.concat(encs_energy, axis=-1)*msk_input
        elif self.node_update_mode == "additive":
            dec_output_energy = X_enc_cg

        if self.debug:
            debugging_data["dec_output"] = dec_output
            debugging_data["dec_output_energy"] = dec_output_energy

        ret = self.output_dec([X, dec_output, dec_output_energy, msk_input], training=training)

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

    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x, y, sample_weights = data

    #     if not hasattr(self, "step"):
    #         self.step = 0

    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass

    #         #regression losses computed only for correctly classified particles
    #         pred_cls = tf.argmax(y_pred["cls"], axis=-1)
    #         true_cls = tf.argmax(y["cls"], axis=-1)
    #         #msk_loss = tf.cast((pred_cls==true_cls) & (true_cls!=0), tf.float32)
    #         #sample_weights["energy"] *= msk_loss
    #         #sample_weights["pt"] *= msk_loss
    #         #sample_weights["eta"] *= msk_loss
    #         #sample_weights["sin_phi"] *= msk_loss
    #         #sample_weights["cos_phi"] *= msk_loss

    #         for icls in [3, ]:
    #             msk1 = (true_cls==icls)
    #             msk2 = (pred_cls==icls)
    #             import matplotlib
    #             import matplotlib.pyplot as plt

    #             plt.figure(figsize=(4,4))
    #             minval = np.min(y["energy"][msk1].numpy().flatten())
    #             maxval = np.max(y["energy"][msk1].numpy().flatten())
    #             plt.scatter(
    #                 y["energy"][msk1&msk2].numpy().flatten(),
    #                 y_pred["energy"][msk1&msk2].numpy().flatten(),
    #                 marker=".", alpha=0.5
    #             )
    #             plt.xlabel("true")
    #             plt.ylabel("pred")
    #             plt.plot([minval,maxval], [minval,maxval], color="black", ls="--", lw=1.0)
    #             plt.savefig("train_cls{}_{}.png".format(icls, self.step), bbox_inches="tight")
    #             plt.close("all")

    #         loss = self.compiled_loss(y, y_pred, sample_weights, regularization_losses=self.losses)

    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value

    #     self.step += 1
    #     return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     # Unpack the data
    #     x, y, sample_weights = data
    #     # Compute predictions
    #     y_pred = self(x, training=False)

    #     pred_cls = tf.argmax(y_pred["cls"], axis=-1)
    #     true_cls = tf.argmax(y["cls"], axis=-1)

    #     for icls in [3, ]:
    #         msk1 = (true_cls==icls)
    #         msk2 = (pred_cls==icls)
    #         import matplotlib
    #         import matplotlib.pyplot as plt

    #         plt.figure(figsize=(4,4))
    #         minval = np.min(y["energy"][msk1].numpy().flatten())
    #         maxval = np.max(y["energy"][msk1].numpy().flatten())
    #         plt.scatter(
    #             y["energy"][msk1&msk2].numpy().flatten(),
    #             y_pred["energy"][msk1&msk2].numpy().flatten(),
    #             marker=".", alpha=0.5
    #         )
    #         plt.xlabel("true")
    #         plt.ylabel("pred")
    #         plt.plot([minval,maxval], [minval,maxval], color="black", ls="--", lw=1.0)
    #         plt.savefig("test_cls{}_{}.png".format(icls, self.step), bbox_inches="tight")
    #         plt.close("all")


    #     # Updates the metrics tracking the loss
    #     self.compiled_loss(y, y_pred, sample_weights, regularization_losses=self.losses)
    #     # Update the metrics.
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).

    #     self.step += 1
    #     return {m.name: m.result() for m in self.metrics}

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
