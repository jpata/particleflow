# This file contains the generic MLPF model definitions
# PFNet: the GNN-based model with graph building based on LSH+kNN
# Transformer: the transformer-based model using fast attention
# DummyNet: simple elementwise feed forward network for cross-checking

import tensorflow as tf

from tfmodel.fast_attention import Attention, SelfAttention

import numpy as np
from numpy.lib.recfunctions import append_fields

def split_indices_to_bins(cmul, nbins, bin_size):
    bin_idx = tf.argmax(cmul, axis=-1)
    bins_split = tf.reshape(tf.argsort(bin_idx), (nbins, bin_size))
    return bins_split

def pairwise_gaussian_dist(A, B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)

    # na as a row and nb as a column vectors
    na = tf.expand_dims(na, -1)
    nb = tf.expand_dims(nb, -2)

    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 1e-6))
    return D

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

#https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConv(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.pop("activation")

        super(GHConv, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.hidden_dim = input_shape[0][-1]
        self.nelem = input_shape[0][-2]
        self.W_t = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_t", initializer="random_normal", trainable=True)
        self.b_t = self.add_weight(shape=(self.hidden_dim,), name="b_t", initializer="random_normal", trainable=True)
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_h", initializer="random_normal", trainable=True)
        self.theta = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="theta", initializer="random_normal", trainable=True)
 
    #@tf.function
    def call(self, inputs):
        x, adj = inputs

        #compute the normalization of the adjacency matrix
        in_degrees = tf.sparse.reduce_sum(tf.abs(adj), axis=-1)
        in_degrees = tf.reshape(in_degrees, (tf.shape(x)[0], tf.shape(x)[1]))

        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)

        f_hom = tf.linalg.matmul(x, self.theta)
        f_hom = sparse_dense_matmult_batch(adj, f_hom*norm)*norm

        f_het = tf.linalg.matmul(x, self.W_h)
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)

        out = gate*f_hom + (1-gate)*f_het
        return self.activation(out)

class SGConv(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.k = kwargs.pop("k")
        super(SGConv, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):
        hidden_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(hidden_dim, hidden_dim), name="w", initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(hidden_dim,), name="b", initializer="random_normal", trainable=True)

    #@tf.function
    def call(self, inputs):
        x, adj = inputs
        #compute the normalization of the adjacency matrix
        in_degrees = tf.sparse.reduce_sum(tf.abs(adj), axis=-1)

        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)

        support = tf.linalg.matmul(x, self.W)
     
        #k-th power of the normalized adjacency matrix is nearly equivalent to k consecutive GCN layers
        #adj_k = tf.pow(adj, self.k)
        out = sparse_dense_matmult_batch(adj, support*norm)*norm

        return self.activation(out + self.b)

def point_wise_feed_forward_network(d_model, dff, num_layers=1, activation='elu', dtype=tf.dtypes.float32):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation=activation) for i in range(num_layers)] +
        [tf.keras.layers.Dense(d_model, dtype=dtype)]
    )

class SparseHashedNNDistance(tf.keras.layers.Layer):
    def __init__(self, distance_dim=128, max_num_bins=200, bin_size=500, num_neighbors=5, dist_mult=0.1, **kwargs):
        super(SparseHashedNNDistance, self).__init__(**kwargs)
        self.num_neighbors = tf.constant(num_neighbors)
        self.dist_mult = dist_mult
        self.distance_dim = distance_dim

        #generate the codebook for LSH hashing at model instantiation for up to this many bins
        #set this to a high-enough value at model generation to take into account the largest possible input 
        self.max_num_bins = tf.constant(max_num_bins)

        #each bin will receive this many input elements, in total we can accept max_num_bins*bin_size input elements
        #in each bin, we will do a dense top_k evaluation
        self.bin_size = bin_size
        self.layer_encoding = point_wise_feed_forward_network(distance_dim, 128)
        self.layer_edge = point_wise_feed_forward_network(1, 128)

    def build(self, input_shape):
        #(n_batch, n_points, n_features)

        #generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = self.add_weight(
            shape=(self.distance_dim, self.max_num_bins//2), initializer="random_normal", trainable=False, name="lsh_projections"
        )

    @tf.function
    def call(self, inputs, training=True):

        #(n_batch, n_points, n_features)
        point_embedding = self.layer_encoding(inputs)

        n_batches = tf.shape(point_embedding)[0]
        n_points = tf.shape(point_embedding)[1]
        #points_neighbors = n_points * self.num_neighbors

        #cannot concat sparse tensors directly as that incorrectly destroys the gradient, see
        #https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/python/ops/sparse_grad.py#L33
        def func(args):
            ibatch, points_batch = args[0], args[1]
            inds, vals = self.construct_sparse_dm_batch(points_batch)
            inds = tf.concat([tf.expand_dims(tf.cast(ibatch, tf.int64)*tf.ones(tf.shape(inds)[0], dtype=tf.int64), -1), inds], axis=-1)
            return inds, vals

        elems = (tf.range(0, n_batches, delta=1, dtype=tf.int64), point_embedding)
        ret = tf.map_fn(func, elems, fn_output_signature=(tf.TensorSpec((None, 3), tf.int64), tf.TensorSpec((None, ), inputs.dtype)), parallel_iterations=2, back_prop=True)

        # #now create a new SparseTensor that is a concatenation of the per-batch tensor indices and values
        shp = tf.shape(ret[0])
        dms = tf.SparseTensor(
            tf.reshape(ret[0], (shp[0]*shp[1], shp[2])),
            tf.reshape(ret[1], (shp[0]*shp[1],)),
            (n_batches, n_points, n_points)
        )

        dm = tf.sparse.reorder(dms)

        i1 = tf.transpose(tf.stack([dm.indices[:, 0], dm.indices[:, 1]]))
        i2 = tf.transpose(tf.stack([dm.indices[:, 0], dm.indices[:, 2]]))
        x1 = tf.gather_nd(inputs, i1)
        x2 = tf.gather_nd(inputs, i2)

        #run an edge net on (src node, dst node, edge)
        edge_vals = tf.nn.sigmoid(self.layer_edge(tf.concat([x1, x2, tf.expand_dims(dm.values, axis=-1)], axis=-1)))
        dm2 = tf.sparse.SparseTensor(indices=dm.indices, values=edge_vals[:, 0], dense_shape=dm.dense_shape)

        return dm2

    @tf.function
    def subpoints_to_sparse_matrix(self, subindices, subpoints):

        #find the distance matrix between the given points in all the LSH bins
        #dm = pairwise_gaussian_dist(subpoints, subpoints)
        #dm = tf.exp(-self.dist_mult*dm)

        dm = pairwise_sigmoid_dist(subpoints, subpoints) #(LSH_bins, points_per_bin, points_per_bin)

        dmshape = tf.shape(dm)
        nbins = dmshape[0]
        nelems = dmshape[1]

        #run KNN in the dense distance matrix, accumulate each index pair into a sparse distance matrix
        top_k = tf.nn.top_k(dm, k=self.num_neighbors)
        top_k_vals = tf.reshape(top_k.values, (nbins*nelems, self.num_neighbors))

        indices_gathered = tf.map_fn(
            lambda i: tf.gather_nd(subindices, top_k.indices[:, :, i:i+1], batch_dims=1),
            tf.range(self.num_neighbors, dtype=tf.int32)
        )
        indices_gathered = tf.transpose(indices_gathered, [1,2,0])

        def func(i):
           dst_ind = indices_gathered[:, :, i] #(nbins, nelems)
           dst_ind = tf.reshape(dst_ind, (nbins*nelems, ))
           src_ind = tf.reshape(tf.stack(subindices), (nbins*nelems, ))
           src_dst_inds = tf.cast(tf.transpose(tf.stack([src_ind, dst_ind])), dtype=tf.int64)
           return src_dst_inds, top_k_vals[:, i]

        ret = tf.map_fn(func, tf.range(0, self.num_neighbors, delta=1, dtype=tf.int32), fn_output_signature=(tf.int64, subpoints.dtype))
        
        shp = tf.shape(ret[0])
        inds = tf.reshape(ret[0], (shp[0]*shp[1], 2))
        vals = tf.reshape(ret[1], (shp[0]*shp[1],))
        return inds, vals

    def construct_sparse_dm_batch(self, points):
        #points: (n_points, n_features) input elements for graph construction
        n_points = tf.shape(points)[0]
        n_features = tf.shape(points)[1]

        #compute the number of LSH bins to divide the input points into on the fly
        #n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)

        #put each input item into a bin defined by the softmax output across the LSH embedding
        mul = tf.linalg.matmul(points, self.codebook_random_rotations[:, :n_bins//2])
        cmul = tf.concat([mul, -mul], axis=-1)

        #cmul is now an integer in [0..nbins) for each input point
        #bins_split: (n_bins, bin_size) of integer bin indices, which put each input point into a bin of size (n_points/n_bins)
        bins_split = split_indices_to_bins(cmul, n_bins, self.bin_size)

        #parts: (n_bins, bin_size, n_features), the input points divided up into bins
        parts = tf.gather(points, bins_split)

        #sparse_distance_matrix: (n_points, n_points) sparse distance matrix
        #where higher values (closer to 1) are associated with points that are closely related
        sparse_distance_matrix = self.subpoints_to_sparse_matrix(bins_split, parts)

        return sparse_distance_matrix

class EncoderDecoderGNN(tf.keras.layers.Layer):
    def __init__(self, encoders, decoders, dropout, activation, conv, **kwargs):
        super(EncoderDecoderGNN, self).__init__(**kwargs)
        name = kwargs.get("name")

        #assert(encoders[-1] == decoders[0])
        self.encoders = encoders
        self.decoders = decoders

        self.encoding_layers = []
        for ilayer, nunits in enumerate(encoders):
            self.encoding_layers.append(
                tf.keras.layers.Dense(nunits, activation=activation, name="encoding_{}_{}".format(name, ilayer)))
            if dropout > 0.0:
                self.encoding_layers.append(tf.keras.layers.Dropout(dropout))

        self.conv = conv

        self.decoding_layers = []
        for ilayer, nunits in enumerate(decoders):
            self.decoding_layers.append(
                tf.keras.layers.Dense(nunits, activation=activation, name="decoding_{}_{}".format(name, ilayer)))
            if dropout > 0.0:
                self.decoding_layers.append(tf.keras.layers.Dropout(dropout))

    @tf.function
    def call(self, inputs, distance_matrix, training=True):
        x = inputs

        for layer in self.encoding_layers:
            x = layer(x)

        for convlayer in self.conv:
            x = convlayer([x, distance_matrix])

        for layer in self.decoding_layers:
            x = layer(x)

        return x

class AddSparse(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddSparse, self).__init__(**kwargs)

    def call(self, matrices):
        ret = matrices[0]
        for mat in matrices[1:]:
            ret = tf.sparse.add(ret, mat)
        return ret

#Simple message passing based on a matrix multiplication
class PFNet(tf.keras.Model):
    def __init__(self,
        multi_output=False,
        num_input_classes=8,
        num_output_classes=3,
        num_momentum_outputs=3,
        activation=tf.nn.selu,
        hidden_dim_id=256,
        hidden_dim_reg=256,
        distance_dim=256,
        convlayer="ghconv",
        dropout=0.1,
        bin_size=10,
        num_convs_id=1,
        num_convs_reg=1,
        num_hidden_id_enc=1,
        num_hidden_id_dec=1,
        num_hidden_reg_enc=1,
        num_hidden_reg_dec=1,
        num_neighbors=5,
        dist_mult=0.1,
        skip_connection=False):

        super(PFNet, self).__init__()
        self.activation = activation
        self.num_dists = 1
        self.num_momentum_outputs = num_momentum_outputs
        self.skip_connection = skip_connection
        self.multi_output = multi_output

        encoding_id = []
        decoding_id = []
        encoding_reg = []
        decoding_reg = []

        #the encoder outputs and decoder inputs have to have the hidden dim (convlayer size)
        for ihidden in range(num_hidden_id_enc):
            encoding_id.append(hidden_dim_id)

        for ihidden in range(num_hidden_id_dec):
            decoding_id.append(hidden_dim_id)

        for ihidden in range(num_hidden_reg_enc):
            encoding_reg.append(hidden_dim_reg)

        for ihidden in range(num_hidden_reg_dec):
            decoding_reg.append(hidden_dim_reg)

        self.enc = InputEncoding(num_input_classes)
        #self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dist1 = SparseHashedNNDistance(distance_dim=distance_dim, bin_size=bin_size, num_neighbors=num_neighbors, dist_mult=dist_mult)
        self.gnn_dm = EncoderDecoderGNN([128, 128], [128, 128], dropout, activation, [GHConv(activation=activation, name="conv_dist0")], name="gnn_dist")

        self.dist2 = SparseHashedNNDistance(distance_dim=distance_dim, bin_size=bin_size, num_neighbors=num_neighbors, dist_mult=dist_mult)

        convs_id = []
        convs_reg = []
        if convlayer == "sgconv":
            for iconv in range(num_convs_id):
                convs_id.append(SGConv(k=1, activation=activation, name="conv_id{}".format(iconv)))
            for iconv in range(num_convs_reg):
                convs_reg.append(SGConv(k=1, activation=activation, name="conv_reg{}".format(iconv)))
        elif convlayer == "ghconv":
            for iconv in range(num_convs_id):
                convs_id.append(GHConv(activation=activation, name="conv_id{}".format(iconv)))
            for iconv in range(num_convs_reg):
                convs_reg.append(GHConv(activation=activation, name="conv_reg{}".format(iconv)))

        self.gnn_id = EncoderDecoderGNN(encoding_id, decoding_id, dropout, activation, convs_id, name="gnn_id")
        self.layer_id = point_wise_feed_forward_network(num_output_classes, hidden_dim_id, num_layers=3, activation=activation)
        self.layer_charge = point_wise_feed_forward_network(1, hidden_dim_id, num_layers=3, activation=activation)
        
        self.gnn_reg = EncoderDecoderGNN(encoding_reg, decoding_reg, dropout, activation, convs_reg, name="gnn_reg")
        self.layer_momentum = point_wise_feed_forward_network(num_momentum_outputs, hidden_dim_reg, num_layers=3, activation=activation)

    # def create_model(self, num_max_elems, num_input_features, training=True):
    #     inputs = tf.keras.Input(shape=(num_max_elems, num_input_features,))
    #     return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs, training), name="MLPFNet")

    def call(self, inputs, training=True):
        X = inputs
        msk_input = tf.expand_dims(tf.cast(X[:, :, 0] != 0, tf.dtypes.float32), -1)

        enc = self.enc(inputs)

        #create graph structure by predicting a sparse distance matrix
        dm1 = self.dist1(enc, training)

        #graph net to encode-decode the nodes
        x_dm = self.gnn_dm(enc, dm1, training)

        #create another graph structure from the encoded nodes
        dm2 = self.dist2(x_dm, training)

        #run graph net for multiclass id prediction
        x_id = self.gnn_id(enc, dm2, training)
        
        if self.skip_connection:
            to_decode = tf.concat([enc, x_id], axis=-1)
        else:
            to_decode = tf.concat([x_id], axis=-1)

        out_id_logits = self.layer_id(to_decode)*msk_input
        out_charge = self.layer_charge(to_decode)*msk_input

        #run graph net for regression output prediction, taking as an additonal input the ID predictions
        x_reg = self.gnn_reg(tf.concat([enc, tf.cast(out_id_logits, X.dtype)], axis=-1), dm2, training)

        if self.skip_connection:
            to_decode = tf.concat([enc, tf.cast(out_id_logits, X.dtype), x_reg], axis=-1)
        else:
            to_decode = tf.concat([tf.cast(out_id_logits, X.dtype), x_reg], axis=-1)

        pred_momentum = self.layer_momentum(to_decode)*msk_input

        if self.multi_output:
            return {"cls": tf.clip_by_value(tf.nn.sigmoid(out_id_logits), 0, 1), "charge": tf.clip_by_value(out_charge, -2, 2), "momentum": pred_momentum}
        else:
            return tf.concat([tf.clip_by_value(tf.nn.sigmoid(out_id_logits), 0, 1), tf.clip_by_value(out_charge, -2, 2), pred_momentum], axis=-1)

    def set_trainable_classification(self):
        for layer in self.layers:
            layer.trainable = False
        self.gnn_id.trainable = True
        self.layer_id.trainable = True

    def set_trainable_regression(self):
        for layer in self.layers:
            layer.trainable = False
        self.gnn_reg.trainable = True
        self.layer_momentum.trainable = True



#Transformer code from the TF example
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, support=8, dtype=tf.dtypes.float32):
        super(EncoderLayer, self).__init__()

        self.mha = SelfAttention(d_model, num_heads, rate, projection_matrix_type=True, nb_random_features=support)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        attn_output = self.mha(x, None, training=training)    # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)    # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)    # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)    # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, support=8, dtype=tf.dtypes.float32):
        super(DecoderLayer, self).__init__()

        self.mha1 = SelfAttention(d_model, num_heads, rate, projection_matrix_type=True, nb_random_features=support)
        self.mha2 = Attention(d_model, num_heads, rate, projection_matrix_type=True, nb_random_features=support)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, None, training=training)    # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, out1, None, training=training)    # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)    # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)    # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)    # (batch_size, target_seq_len, d_model)

        return out3

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, support=32, rate=0.1, dtype=tf.dtypes.float32):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, support=support, dtype=dtype) 
                                             for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        x = self.dropout(x, training=training)
        return x    # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, support=32, rate=0.1, dtype=tf.dtypes.float32):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, support=support, dtype=dtype) 
                                             for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training):

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training)

        x = self.dropout(x, training=training)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x



class Transformer(tf.keras.Model):
    def __init__(self,
                num_layers, d_model, num_heads, dff,
                dropout=0.1,
                support=32,
                num_input_classes=8,
                num_output_classes=3,
                num_momentum_outputs=3,
                dtype=tf.dtypes.float32,
                skip_connection=False,
                multi_output=False):
        super(Transformer, self).__init__()

        self.skip_connection = skip_connection
        self.multi_output = multi_output
        self.num_momentum_outputs = num_momentum_outputs

        self.enc = InputEncoding(num_input_classes)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn_embed_id = point_wise_feed_forward_network(d_model, dff)
        self.ffn_embed_reg = point_wise_feed_forward_network(d_model, dff)

        self.encoder_id = Encoder(num_layers, d_model, num_heads, dff, support, dropout, dtype)
        self.decoder_id = Decoder(num_layers, d_model, num_heads, dff, support, dropout, dtype)

        self.encoder_reg = Encoder(num_layers, d_model, num_heads, dff, support, dropout, dtype)
        self.decoder_reg = Decoder(num_layers, d_model, num_heads, dff, support, dropout, dtype)

        self.ffn_id = point_wise_feed_forward_network(num_output_classes, dff, dtype=tf.dtypes.float32)
        self.ffn_charge = point_wise_feed_forward_network(1, dff, dtype=tf.dtypes.float32)
        self.ffn_momentum = point_wise_feed_forward_network(num_momentum_outputs, dff, dtype=tf.dtypes.float32)

    def call(self, inputs, training):
        X = inputs
        msk_input = tf.expand_dims(tf.cast(X[:, :, 0] != 0, tf.float32), -1)

        enc = self.enc(X)
        enc = self.layernorm1(enc)

        enc_id = self.ffn_embed_id(enc)
        enc_reg = self.ffn_embed_reg(enc)

        enc_output_id = self.encoder_id(enc_id, training)
        enc_output_id = self.layernorm2(enc_output_id)
        dec_output_id = self.decoder_id(enc_id, enc_output_id, training)

        if self.skip_connection:
            dec_output_id = tf.concat([enc_id, dec_output_id], axis=-1)

        enc_output_reg = self.encoder_reg(enc_reg, training)
        enc_output_reg = self.layernorm3(enc_output_reg)
        dec_output_reg = self.decoder_reg(enc_reg, enc_output_reg, training)

        out_id_logits = self.ffn_id(dec_output_id)
        out_charge = self.ffn_charge(dec_output_id)*msk_input

        if self.skip_connection:
            dec_output_reg = tf.concat([enc_reg, tf.cast(out_id_logits, X.dtype), dec_output_reg], axis=-1)
        else:
            dec_output_reg = tf.concat([tf.cast(out_id_logits, X.dtype), dec_output_reg], axis=-1)
        pred_momentum = self.ffn_momentum(dec_output_reg)*msk_input

        if self.multi_output:
            return {"cls": tf.nn.sigmoid(out_id_logits), "charge": out_charge, "momentum": pred_momentum}
        else:
            return tf.concat([tf.nn.sigmoid(out_id_logits), out_charge, pred_momentum], axis=-1)



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
