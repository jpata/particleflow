import tensorflow as tf

# FIXME: this should be configurable
regularizer_weight = 0.0


def split_indices_to_bins(cmul, nbins, bin_size):
    bin_idx = tf.argmax(cmul, axis=-1)
    bins_split = tf.reshape(tf.argsort(bin_idx), (nbins, bin_size))
    return bins_split


def split_indices_to_bins_batch(cmul, nbins, bin_size, msk):
    bin_idx = tf.argmax(cmul, axis=-1) + tf.cast(tf.where(~msk, nbins - 1, 0), tf.int64)
    bins_split = tf.reshape(tf.argsort(bin_idx), (tf.shape(cmul)[0], nbins, bin_size))
    return bins_split


def pairwise_l2_dist(A, B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)

    # na as a row and nb as a column vectors
    na = tf.expand_dims(na, -1)
    nb = tf.expand_dims(nb, -2)

    # return pairwise euclidean difference matrix
    # note that this matrix multiplication can go out of range for float16 in case the absolute values of A and B are large
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 1e-6))
    return D


def pairwise_l1_dist(A, B):
    na = tf.expand_dims(A, -2)
    nb = tf.expand_dims(B, -3)
    D = tf.abs(tf.reduce_sum(na - nb, axis=-1))
    return D


def pairwise_learnable_dist(A, B, ffn, training=False):
    shp = tf.shape(A)

    # stack node feature vectors of src[i], dst[j] into a matrix res[i,j] = (src[i], dst[j])
    mg = tf.meshgrid(tf.range(shp[0]), tf.range(shp[1]), tf.range(shp[2]), tf.range(shp[2]), indexing="ij")
    inds1 = tf.stack([mg[0], mg[1], mg[2]], axis=-1)
    inds2 = tf.stack([mg[0], mg[1], mg[3]], axis=-1)
    res = tf.concat([tf.gather_nd(A, inds1), tf.gather_nd(B, inds2)], axis=-1)  # (batch, bin, elem, elem, feat)

    # run a feedforward net on ffn(src, dst) -> output_dim
    res_transformed = ffn(res, training=training)

    return res_transformed


def pairwise_sigmoid_dist(A, B):
    return tf.nn.sigmoid(tf.matmul(A, tf.transpose(B, perm=[0, 2, 1])))


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
        sparse_slice = tf.sparse.reshape(
            tf.sparse.slice(tf.cast(sp_a, tf.float32), [i, 0, 0], [1, num_points, num_points]), [num_points, num_points]
        )
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, num_batches, delta=1, dtype=tf.int64), b)
    ret = tf.map_fn(map_function, elems, fn_output_signature=tf.TensorSpec((None, None), b.dtype), back_prop=True)
    return tf.cast(ret, dtype)


@tf.function
def reverse_lsh(bins_split, points_binned_enc):
    tf.debugging.assert_shapes(
        [
            (bins_split, ("n_batch", "n_bins", "n_points_bin")),
            (points_binned_enc, ("n_batch", "n_bins", "n_points_bin", "num_features")),
        ]
    )

    shp = tf.shape(points_binned_enc)
    batch_dim = shp[0]
    n_points = shp[1] * shp[2]
    n_features = shp[-1]

    bins_split_flat = tf.reshape(bins_split, (batch_dim, n_points))
    points_binned_enc_flat = tf.reshape(points_binned_enc, (batch_dim, n_points, n_features))

    batch_inds = tf.reshape(tf.repeat(tf.range(batch_dim), n_points), (batch_dim, n_points))
    bins_split_flat_batch = tf.stack([batch_inds, bins_split_flat], axis=-1)

    ret = tf.scatter_nd(bins_split_flat_batch, points_binned_enc_flat, shape=(batch_dim, n_points, n_features))

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

        # X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)

        # X[:, :, 1:] - all the other non-categorical features
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

    def call(self, X):
        # X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)
        Xpt = tf.expand_dims(tf.math.log(X[:, :, 1] + 1.0), axis=-1)
        Xe = tf.expand_dims(tf.math.log(X[:, :, 4] + 1.0), axis=-1)

        Xpt_0p5 = tf.math.sqrt(Xpt)
        Xpt_2 = tf.math.pow(Xpt, 2)

        Xeta1 = tf.clip_by_value(tf.expand_dims(tf.sinh(X[:, :, 2]), axis=-1), -10, 10)
        Xeta2 = tf.clip_by_value(tf.expand_dims(tf.cosh(X[:, :, 2]), axis=-1), -10, 10)
        Xabs_eta = tf.expand_dims(tf.math.abs(X[:, :, 2]), axis=-1)
        Xphi1 = tf.expand_dims(tf.sin(X[:, :, 3]), axis=-1)
        Xphi2 = tf.expand_dims(tf.cos(X[:, :, 3]), axis=-1)

        Xe_0p5 = tf.math.sqrt(Xe)
        Xe_2 = tf.math.pow(Xe, 2)

        Xphi_ecal1 = tf.expand_dims(tf.sin(X[:, :, 10]), axis=-1)
        Xphi_ecal2 = tf.expand_dims(tf.cos(X[:, :, 10]), axis=-1)
        Xphi_hcal1 = tf.expand_dims(tf.sin(X[:, :, 12]), axis=-1)
        Xphi_hcal2 = tf.expand_dims(tf.cos(X[:, :, 12]), axis=-1)

        return tf.concat(
            [
                Xid,
                Xpt,
                Xpt_0p5,
                Xpt_2,
                Xeta1,
                Xeta2,
                Xabs_eta,
                Xphi1,
                Xphi2,
                Xe,
                Xe_0p5,
                Xe_2,
                Xphi_ecal1,
                Xphi_ecal2,
                Xphi_hcal1,
                Xphi_hcal2,
                X,
            ],
            axis=-1,
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
        self.W_t = self.add_weight(
            shape=(self.hidden_dim, self.output_dim),
            name="w_t",
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(regularizer_weight),
        )
        self.b_t = self.add_weight(
            shape=(self.output_dim,),
            name="b_t",
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(regularizer_weight),
        )
        self.W_h = self.add_weight(
            shape=(self.hidden_dim, self.output_dim),
            name="w_h",
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(regularizer_weight),
        )
        self.theta = self.add_weight(
            shape=(self.hidden_dim, self.output_dim),
            name="theta",
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(regularizer_weight),
        )

    def call(self, inputs):
        x, adj, msk = inputs
        # tf.print("GHConvDense.call:x", x.shape)
        # tf.print("GHConvDense.call:adj", adj.shape)
        # tf.print("GHConvDense.call:msk", msk.shape)

        # remove last dim from distance/adjacency matrix
        tf.debugging.assert_equal(tf.shape(adj)[-1], 1)
        adj = tf.squeeze(adj, axis=-1)

        # compute the normalization of the adjacency matrix
        if self.normalize_degrees:
            # in_degrees = tf.clip_by_value(tf.reduce_sum(tf.abs(adj), axis=-1), 0, 1000)
            in_degrees = tf.reduce_sum(tf.abs(adj), axis=-1)

            # add epsilon to prevent numerical issues from 1/sqrt(x)
            norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1) * msk

        f_hom = tf.linalg.matmul(x * msk, self.theta) * msk
        if self.normalize_degrees:
            f_hom = tf.linalg.matmul(adj, f_hom * norm) * norm
        else:
            f_hom = tf.linalg.matmul(adj, f_hom)

        f_het = tf.linalg.matmul(x * msk, self.W_h)
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)

        out = gate * f_hom + (1.0 - gate) * f_het
        tf.debugging.assert_shapes(
            [
                (x, ("n_batch", "n_bins", "n_points_bin", "num_features")),
                (adj, ("n_batch", "n_bins", "n_points_bin", "n_points_bin")),
                (msk, ("n_batch", "n_bins", "n_points_bin", 1)),
                (out, ("n_batch", "n_bins", "n_points_bin", self.output_dim)),
            ]
        )
        # tf.print("GHConvDense.call:out", out.shape)
        return self.activation(out) * msk


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
            name=kwargs.get("name") + "_ffn",
        )
        super(NodeMessageLearnable, self).__init__(*args, **kwargs)

    def call(self, inputs):
        x, adj, msk = inputs

        # collect incoming messages (batch, bins, elems, elems, msg_dim) -> (batch, bins, elems, msg_dim)
        max_message_dst = tf.reduce_max(adj, axis=-2)

        # collect outgoing messages (batch, bins, elems, elems, msg_dim) -> (batch, bins, elems, msg_dim)
        max_message_src = tf.reduce_max(adj, axis=-3)

        # node update (batch, bins, elems, elems, elem_dim + msg_dim + msg_dim)
        x2 = tf.concat([x, max_message_dst, max_message_src], axis=-1)
        return tf.cast(self.activation(self.ffn(x2)), x.dtype)


def point_wise_feed_forward_network(
    d_model, dff, name, num_layers=1, activation="elu", dtype=tf.dtypes.float32, dim_decrease=False, dropout=0.0
):

    if regularizer_weight > 0:
        bias_regularizer = tf.keras.regularizers.L1(regularizer_weight)
        kernel_regularizer = tf.keras.regularizers.L1(regularizer_weight)
    else:
        bias_regularizer = None
        kernel_regularizer = None

    layers = []
    for ilayer in range(num_layers):
        _name = name + "_dense_{}".format(ilayer)

        layers.append(
            tf.keras.layers.Dense(
                dff,
                activation=activation,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
                name=_name,
            )
        )

        if dropout > 0.0:
            layers.append(tf.keras.layers.Dropout(dropout))

        if dim_decrease:
            dff = dff // 2

    layers.append(tf.keras.layers.Dense(d_model, dtype=dtype, name="{}_dense_{}".format(name, ilayer + 1)))
    return tf.keras.Sequential(layers, name=name)


def get_message_layer(config_dict, name):
    config_dict = config_dict.copy()
    class_name = config_dict.pop("type")
    classes = {"NodeMessageLearnable": NodeMessageLearnable, "GHConvDense": GHConvDense}
    conv_cls = classes[class_name]

    return conv_cls(name=name, **config_dict)


class NodePairGaussianKernel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.clip_value_low = kwargs.pop("clip_value_low", 0.0)
        self.dist_mult = kwargs.pop("dist_mult", 0.1)

        dist_norm = kwargs.pop("dist_norm", "l2")
        if dist_norm == "l1":
            self.dist_norm = pairwise_l1_dist
        elif dist_norm == "l2":
            self.dist_norm = pairwise_l2_dist
        else:
            raise Exception("Unkown dist_norm: {}".format(dist_norm))
        super(NodePairGaussianKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, 1) message matrix
    """

    def call(self, x_msg_binned, msk, training=False):
        x = x_msg_binned * msk
        dm = tf.expand_dims(self.dist_norm(x, x), axis=-1)
        dm = tf.exp(-self.dist_mult * dm)
        dm = tf.clip_by_value(dm, self.clip_value_low, 1)
        return dm


class NodePairTrainableKernel(tf.keras.layers.Layer):
    def __init__(self, output_dim=4, hidden_dim_node=128, hidden_dim_pair=32, num_layers=1, activation="elu", **kwargs):
        self.output_dim = output_dim
        self.hidden_dim_node = hidden_dim_node
        self.hidden_dim_pair = hidden_dim_pair
        self.num_layers = num_layers
        self.activation = getattr(tf.keras.activations, activation)

        self.ffn_node = point_wise_feed_forward_network(
            self.output_dim,
            self.hidden_dim_node,
            kwargs.get("name") + "_" + "node",
            num_layers=self.num_layers,
            activation=self.activation,
        )

        self.pair_kernel = point_wise_feed_forward_network(
            self.output_dim,
            self.hidden_dim_pair,
            kwargs.get("name") + "_" + "pair_kernel",
            num_layers=self.num_layers,
            activation=self.activation,
        )

        super(NodePairTrainableKernel, self).__init__(**kwargs)

    """
    x_msg_binned: (n_batch, n_bins, n_points, n_msg_features)

    returns: (n_batch, n_bins, n_points, n_points, output_dim) message matrix
    """

    def call(self, x_msg_binned, msk, training=False):

        node_proj = self.activation(self.ffn_node(x_msg_binned))

        dm = tf.cast(pairwise_learnable_dist(node_proj, node_proj, self.pair_kernel, training=training), x_msg_binned.dtype)
        return dm


def build_kernel_from_conf(kernel_dict, name):
    kernel_dict = kernel_dict.copy()

    cls_type = kernel_dict.pop("type")
    clss = {"NodePairGaussianKernel": NodePairGaussianKernel, "NodePairTrainableKernel": NodePairTrainableKernel}

    return clss[cls_type](name=name, **kernel_dict)


class MessageBuildingLayerLSH(tf.keras.layers.Layer):
    def __init__(self, distance_dim=128, max_num_bins=200, bin_size=128, kernel=NodePairGaussianKernel(), **kwargs):
        self.distance_dim = distance_dim
        self.max_num_bins = max_num_bins
        self.bin_size = bin_size
        self.kernel = kernel

        super(MessageBuildingLayerLSH, self).__init__(**kwargs)

    def build(self, input_shape):
        # (n_batch, n_points, n_features)

        # generate the LSH codebook for random rotations (num_features, max_num_bins/2)
        self.codebook_random_rotations = self.add_weight(
            shape=(self.distance_dim, self.max_num_bins // 2),
            initializer="random_normal",
            trainable=False,
            name="lsh_projections",
        )

    """
    x_msg: (n_batch, n_points, n_msg_features)
    x_node: (n_batch, n_points, n_node_features)
    """

    def call(self, x_msg, x_node, msk, training=False):
        msk_f = tf.expand_dims(tf.cast(msk, x_msg.dtype), -1)

        tf.debugging.assert_shapes(
            [
                (x_msg, ("n_batch", "n_points", "n_msg_features")),
                (x_node, ("n_batch", "n_points", "n_node_features")),
                (msk_f, ("n_batch", "n_points", 1)),
            ]
        )

        shp = tf.shape(x_msg)
        n_points = shp[1]

        # compute the number of LSH bins to divide the input points into on the fly
        # n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)
        tf.debugging.assert_greater(
            n_bins, 0, "number of points (dim 1) must be greater than bin_size={}".format(self.bin_size)
        )
        tf.debugging.assert_equal(
            tf.math.floormod(n_points, self.bin_size),
            0,
            "number of points (dim 1) must be an integer multiple of bin_size={}".format(self.bin_size),
        )

        # put each input item into a bin defined by the argmax output across the LSH embedding
        mul = tf.linalg.matmul(x_msg, self.codebook_random_rotations[:, : tf.math.maximum(1, n_bins // 2)])
        cmul = tf.concat([mul, -mul], axis=-1)
        bins_split = split_indices_to_bins_batch(cmul, n_bins, self.bin_size, msk)
        x_msg_binned = tf.gather(x_msg, bins_split, batch_dims=1)
        x_features_binned = tf.gather(x_node, bins_split, batch_dims=1)
        msk_f_binned = tf.gather(msk_f, bins_split, batch_dims=1)

        tf.debugging.assert_equal(tf.shape(x_msg_binned)[1], n_bins)

        # Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg_binned, msk_f_binned, training=training)

        # remove the masked points row-wise and column-wise
        msk_f_binned_squeeze = tf.squeeze(msk_f_binned, axis=-1)
        shp_dm = tf.shape(dm)
        rshp_row = [shp_dm[0], shp_dm[1], shp_dm[2], 1, 1]
        rshp_col = [shp_dm[0], shp_dm[1], 1, shp_dm[3], 1]
        msk_row = tf.cast(tf.reshape(msk_f_binned_squeeze, rshp_row), dm.dtype)
        msk_col = tf.cast(tf.reshape(msk_f_binned_squeeze, rshp_col), dm.dtype)
        dm = tf.math.multiply(dm, msk_row)
        dm = tf.math.multiply(dm, msk_col)
        tf.debugging.assert_shapes(
            [
                (x_msg_binned, ("n_batch", "n_bins", "n_points_bin", "n_msg_features")),
                (x_features_binned, ("n_batch", "n_bins", "n_points_bin", "n_node_features")),
                (msk_f_binned, ("n_batch", "n_bins", "n_points_bin", 1)),
                (dm, ("n_batch", "n_bins", "n_points_bin", "n_points_bin", 1)),
            ]
        )

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

        # Run the node-to-node kernel (distance computation / graph building / attention)
        dm = self.kernel(x_msg, training=training)

        # remove the masked points row-wise and column-wise
        dm = tf.einsum("bijk,bi->bijk", dm, tf.squeeze(msk_f, axis=-1))
        dm = tf.einsum("bijk,bj->bijk", dm, tf.squeeze(msk_f, axis=-1))

        return dm


class OutputDecoding(tf.keras.Model):
    def __init__(
        self,
        activation="elu",
        regression_use_classification=True,
        num_output_classes=8,
        schema="cms",
        dropout=0.0,
        id_dim_decrease=True,
        charge_dim_decrease=True,
        pt_dim_decrease=False,
        eta_dim_decrease=False,
        phi_dim_decrease=False,
        energy_dim_decrease=False,
        pt_as_correction=True,
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
        event_set_output=False,
        met_output=False,
        cls_output_as_logits=False,
        **kwargs
    ):

        super(OutputDecoding, self).__init__(**kwargs)

        self.regression_use_classification = regression_use_classification
        self.schema = schema
        self.dropout = dropout

        self.mask_reg_cls0 = mask_reg_cls0
        self.pt_as_correction = pt_as_correction

        self.do_layernorm = layernorm
        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, name="output_layernorm")

        self.event_set_output = event_set_output
        self.met_output = met_output
        self.cls_output_as_logits = cls_output_as_logits

        self.ffn_id = point_wise_feed_forward_network(
            num_output_classes,
            id_hidden_dim,
            "ffn_cls",
            num_layers=id_num_layers,
            activation=activation,
            dim_decrease=id_dim_decrease,
            dropout=dropout,
        )
        self.ffn_charge = point_wise_feed_forward_network(
            1,
            charge_hidden_dim,
            "ffn_charge",
            num_layers=charge_num_layers,
            activation=activation,
            dim_decrease=charge_dim_decrease,
            dropout=dropout,
        )

        self.ffn_pt = point_wise_feed_forward_network(
            2,
            pt_hidden_dim,
            "ffn_pt",
            num_layers=pt_num_layers,
            activation=activation,
            dim_decrease=pt_dim_decrease,
            dropout=dropout,
        )

        self.ffn_eta = point_wise_feed_forward_network(
            1,
            eta_hidden_dim,
            "ffn_eta",
            num_layers=eta_num_layers,
            activation=activation,
            dim_decrease=eta_dim_decrease,
            dropout=dropout,
        )

        # sin_phi, cos_phi outputs
        self.ffn_phi = point_wise_feed_forward_network(
            2,
            phi_hidden_dim,
            "ffn_phi",
            num_layers=phi_num_layers,
            activation=activation,
            dim_decrease=phi_dim_decrease,
            dropout=dropout,
        )

        self.ffn_energy = point_wise_feed_forward_network(
            1,
            energy_hidden_dim,
            "ffn_energy",
            num_layers=energy_num_layers,
            activation=activation,
            dim_decrease=energy_dim_decrease,
            dropout=dropout,
        )

    """
    X_input: (n_batch, n_elements, n_input_features) raw node input features
    X_encoded: (n_batch, n_elements, n_encoded_features) encoded/transformed node features
    msk_input: (n_batch, n_elements) boolean mask of active nodes
    """

    def call(self, args, training=False):

        X_input, X_encoded, X_encoded_energy, msk_input = args

        if self.do_layernorm:
            X_encoded = self.layernorm(X_encoded)

        out_id_logits = self.ffn_id(X_encoded, training=training)
        in_dtype = X_encoded.dtype
        out_dtype = out_id_logits.dtype
        msk_input_outtype = tf.cast(msk_input, out_dtype)

        # mask the classification outputs for zero-padded inputs across batches
        out_id_logits = out_id_logits * msk_input_outtype

        if self.cls_output_as_logits:
            out_id_transformed = out_id_logits
        else:
            out_id_transformed = tf.nn.softmax(out_id_logits, axis=-1)

        out_charge = self.ffn_charge(X_encoded, training=training)
        out_charge = out_charge * msk_input_outtype

        orig_eta = tf.cast(X_input[:, :, 2:3], out_dtype)

        # FIXME: better schema propagation between hep_tfds
        # skip connection from raw input values
        if self.schema == "cms":
            orig_sin_phi = tf.cast(tf.math.sin(X_input[:, :, 3:4]) * msk_input, out_dtype)
            orig_cos_phi = tf.cast(tf.math.cos(X_input[:, :, 3:4]) * msk_input, out_dtype)
            orig_energy = tf.cast(X_input[:, :, 4:5] * msk_input, out_dtype)
            orig_pt = X_input[:, :, 1:2]
        elif self.schema == "delphes":
            orig_sin_phi = tf.cast(X_input[:, :, 3:4] * msk_input, out_dtype)
            orig_cos_phi = tf.cast(X_input[:, :, 4:5] * msk_input, out_dtype)
            orig_energy = tf.cast(X_input[:, :, 5:6] * msk_input, out_dtype)
            orig_pt = X_input[:, :, 1:2]

        if self.regression_use_classification:
            X_encoded = tf.concat([X_encoded, tf.cast(tf.stop_gradient(out_id_logits), in_dtype)], axis=-1)

        pred_eta_corr = self.ffn_eta(X_encoded, training=training)
        pred_eta_corr = pred_eta_corr * msk_input_outtype
        pred_phi_corr = self.ffn_phi(X_encoded, training=training)
        pred_phi_corr = pred_phi_corr * msk_input_outtype

        pred_eta = orig_eta + pred_eta_corr[:, :, 0:1]
        pred_sin_phi = orig_sin_phi + pred_phi_corr[:, :, 0:1]
        pred_cos_phi = orig_cos_phi + pred_phi_corr[:, :, 1:2]

        X_encoded_energy = tf.concat([X_encoded, X_encoded_energy], axis=-1)
        if self.regression_use_classification:
            X_encoded_energy = tf.concat([X_encoded_energy, tf.cast(tf.stop_gradient(out_id_logits), in_dtype)], axis=-1)

        pred_energy_corr = self.ffn_energy(X_encoded_energy, training=training)
        pred_energy_corr = pred_energy_corr * msk_input_outtype

        # In case of a multimodal prediction, weight the per-class energy predictions by the approximately one-hot vector
        pred_energy = orig_energy + pred_energy_corr
        pred_energy = tf.abs(pred_energy)

        pred_pt_corr = self.ffn_pt(X_encoded_energy, training=training) * msk_input_outtype
        if self.pt_as_correction:
            pred_pt = tf.cast(orig_pt, out_dtype) * pred_pt_corr[..., 0:1] + pred_pt_corr[..., 1:2]
        else:
            pred_pt = pred_pt_corr[..., 0:1]
        pred_pt = tf.abs(pred_pt)

        # mask the regression outputs for the nodes with a class prediction 0

        ret = {
            "cls": out_id_transformed,
            "charge": out_charge,
            "pt": pred_pt * msk_input_outtype,
            "eta": pred_eta * msk_input_outtype,
            "sin_phi": pred_sin_phi * msk_input_outtype,
            "cos_phi": pred_cos_phi * msk_input_outtype,
            "energy": pred_energy * msk_input_outtype,
        }

        if self.event_set_output:
            if self.mask_reg_cls0:
                softmax_cls = (1.0 - tf.nn.softmax(out_id_logits, axis=-1)[..., 0:1]) * msk_input_outtype
                pt_e_eta_phi = tf.concat(
                    [
                        pred_pt * msk_input_outtype * softmax_cls,
                        pred_energy * msk_input_outtype * softmax_cls,
                        pred_eta * msk_input_outtype * softmax_cls,
                        pred_sin_phi * msk_input_outtype * softmax_cls,
                        pred_cos_phi * msk_input_outtype * softmax_cls,
                    ],
                    axis=-1,
                )
            ret["pt_e_eta_phi"] = pt_e_eta_phi

        if self.met_output:
            px = pred_pt * pred_cos_phi * msk_input_outtype
            py = pred_pt * pred_sin_phi * msk_input_outtype
            met = tf.sqrt(tf.reduce_sum(px**2 + py**2, axis=-2))
            ret["met"] = met

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
            self.layernorm1 = tf.keras.layers.LayerNormalization(
                axis=-1, epsilon=1e-6, name=kwargs.get("name") + "_layernorm1"
            )

        # self.gaussian_noise = tf.keras.layers.GaussianNoise(0.01)
        self.ffn_dist = point_wise_feed_forward_network(
            self.distance_dim,
            self.ffn_dist_hidden_dim,
            kwargs.get("name") + "_ffn_dist",
            num_layers=self.ffn_dist_num_layers,
            activation=self.activation,
            dropout=self.dropout,
        )

        self.message_building_layer = MessageBuildingLayerLSH(
            distance_dim=self.distance_dim,
            max_num_bins=self.max_num_bins,
            bin_size=self.bin_size,
            kernel=build_kernel_from_conf(self.kernel, kwargs.get("name") + "_kernel"),
        )

        self.message_passing_layers = [
            get_message_layer(self.node_message, "{}_msg_{}".format(kwargs.get("name"), iconv))
            for iconv in range(self.num_node_messages)
        ]
        self.dropout_layer = None
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        super(CombinedGraphLayer, self).__init__(*args, **kwargs)

    def call(self, x, msk, training=False):

        if self.do_layernorm:
            x = self.layernorm1(x, training=training)

        # compute node features for graph building
        x_dist = self.dist_activation(self.ffn_dist(x, training=training))

        # compute the element-to-element messages / distance matrix / graph structure
        bins_split, x, dm, msk_f = self.message_building_layer(x_dist, x, msk)
        # tf.print("CombinedGraphLayer.call:bins_split", bins_split.shape)
        # tf.print("CombinedGraphLayer.call:x", x.shape)
        # tf.print("CombinedGraphLayer.call:dm", dm.shape)
        # tf.print("CombinedGraphLayer.call:msk_f", msk_f.shape)

        tf.debugging.assert_shapes(
            [
                (bins_split, ("n_batch", "n_bins", "n_points_bin")),
                (x, ("n_batch", "n_bins", "n_points_bin", "n_node_features")),
                (dm, ("n_batch", "n_bins", "n_points_bin", "n_points_bin", 1)),
                (msk_f, ("n_batch", "n_bins", "n_points_bin", 1)),
            ]
        )

        # run the node update with message passing
        for msg in self.message_passing_layers:
            x_out = msg((x, dm, msk_f))
            tf.debugging.assert_shapes(
                [
                    (x, ("n_batch", "n_bins", "n_points_bin", "feat_in")),
                    (x_out, ("n_batch", "n_bins", "n_points_bin", "feat_out")),
                ]
            )
            x = x_out
            if self.dropout_layer:
                x = self.dropout_layer(x, training=training)

        # undo the binning according to the element-to-bin indices
        x = reverse_lsh(bins_split, x)

        return {"enc": x, "dist": x_dist, "bins": bins_split, "dm": dm}


class PFNetDense(tf.keras.Model):
    def __init__(
        self,
        do_node_encoding=False,
        node_encoding_hidden_dim=128,
        dropout=0.0,
        activation="gelu",
        multi_output=False,
        num_input_classes=8,
        num_output_classes=3,
        num_graph_layers_id=1,
        num_graph_layers_reg=1,
        input_encoding="cms",
        skip_connection=True,
        graph_kernel={},
        combined_graph_layer={},
        node_message={},
        output_decoding={},
        debug=False,
        schema="cms",
        node_update_mode="concat",
        event_set_output=False,
        met_output=False,
        cls_output_as_logits=False,
        **kwargs
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
                dropout=self.dropout,
            )

        if input_encoding == "cms":
            self.enc = InputEncodingCMS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        self.cg_id = [
            CombinedGraphLayer(name="cg_id_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_id)
        ]
        self.cg_reg = [
            CombinedGraphLayer(name="cg_reg_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_reg)
        ]

        output_decoding["schema"] = schema
        output_decoding["num_output_classes"] = num_output_classes
        output_decoding["event_set_output"] = event_set_output
        output_decoding["met_output"] = met_output
        output_decoding["cls_output_as_logits"] = cls_output_as_logits
        self.output_dec = OutputDecoding(**output_decoding)

    def call(self, inputs, training=False):
        X = inputs
        debugging_data = {}

        # encode the elements for classification (id)
        X_enc = self.enc(X)

        # mask padded elements
        msk = X[:, :, 0] != 0
        msk_input = tf.expand_dims(tf.cast(msk, X_enc.dtype), -1)

        encs_id = []
        if self.skip_connection:
            encs_id.append(X_enc)

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_ffn = self.activation(self.node_encoding(X_enc_cg, training=training))
            X_enc_cg = X_enc_ffn

        for cg in self.cg_id:
            enc_all = cg(X_enc_cg, msk, training=training)

            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs_id.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all

        if self.node_update_mode == "concat":
            dec_output_id = tf.concat(encs_id, axis=-1) * msk_input
        elif self.node_update_mode == "additive":
            dec_output_id = X_enc_cg

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_cg = X_enc_ffn

        encs_reg = []
        if self.skip_connection:
            encs_reg.append(X_enc)

        for cg in self.cg_reg:
            enc_all = cg(X_enc_cg, msk, training=training)
            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs_reg.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all
            encs_reg.append(X_enc_cg)

        if self.node_update_mode == "concat":
            dec_output_reg = tf.concat(encs_reg, axis=-1) * msk_input
        elif self.node_update_mode == "additive":
            dec_output_reg = X_enc_cg

        if self.debug:
            debugging_data["dec_output_id"] = dec_output_id
            debugging_data["dec_output_reg"] = dec_output_reg

        ret = self.output_dec([X, dec_output_id, dec_output_reg, msk_input], training=training)

        if self.debug:
            for k in debugging_data.keys():
                ret[k] = debugging_data[k]

        if self.multi_output:
            return ret
        else:
            return tf.concat(
                [ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1
            )

    def set_trainable_named(self, layer_names):
        self.trainable = True

        for layer in self.layers:
            layer.trainable = False

        self.output_dec.set_trainable_named(layer_names)

    # Uncomment these if you want to explicitly debug the training loop
    # def train_step(self, data):
    #     import numpy as np
    #     x, y, sample_weights = data
    #     if not hasattr(self, "step"):
    #         self.step = 0

    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         loss = self.compiled_loss(y, y_pred, sample_weights, regularization_losses=self.losses)

    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     for tv, g in zip(trainable_vars, gradients):
    #         g = g.numpy()
    #         num_nan = np.sum(np.isnan(g))
    #         if num_nan>0:
    #             print(tv.name, num_nan, g.shape)

    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     self.compiled_metrics.update_state(y, y_pred)

    #     self.step += 1
    #     return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     # Unpack the data
    #     x, y, sample_weights = data
    #     # Compute predictions
    #     y_pred = self(x, training=False)

    #     pred_cls = tf.argmax(y_pred["cls"], axis=-1)
    #     true_cls = tf.argmax(y["cls"], axis=-1)

    #     # Updates the metrics tracking the loss
    #     self.compiled_loss(y, y_pred, sample_weights, regularization_losses=self.losses)
    #     # Update the metrics.
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).

    #     self.step += 1
    #     return {m.name: m.result() for m in self.metrics}


class KernelEncoder(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        from official.nlp.modeling.layers.kernel_attention import KernelAttention

        self.key_dim = kwargs.pop("key_dim")
        num_heads = 8

        self.attn = KernelAttention(
            feature_transform="elu", num_heads=num_heads, key_dim=self.key_dim, name=kwargs.get("name") + "_attention"
        )
        self.ffn = point_wise_feed_forward_network(
            self.key_dim, self.key_dim, kwargs.get("name") + "_ffn", num_layers=1, activation="elu"
        )
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, name=kwargs.get("name") + "_ln0")
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, name=kwargs.get("name") + "_ln1")
        super(KernelEncoder, self).__init__(*args, **kwargs)

    def call(self, args, training=False):
        Q, X, mask = args
        msk_input = tf.expand_dims(tf.cast(mask, tf.float32), -1)

        X = self.norm1(X)
        attn_output = self.attn(query=Q, value=X, key=X, training=training, attention_mask=mask) * msk_input
        out1 = self.norm2(X + attn_output)

        out2 = self.ffn(out1)

        return out2


class PFNetTransformer(tf.keras.Model):
    def __init__(
        self,
        num_input_classes=8,
        num_output_classes=3,
        input_encoding="cms",
        schema="cms",
        output_decoding={},
        multi_output=True,
        event_set_output=False,
        met_output=False,
    ):
        super(PFNetTransformer, self).__init__()

        self.multi_output = multi_output

        if input_encoding == "cms":
            self.enc = InputEncodingCMS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        key_dim = 128

        self.ffn = point_wise_feed_forward_network(key_dim, key_dim, "ffn", num_layers=1, activation="elu")

        self.encoders = []
        for i in range(4):
            self.encoders.append(KernelEncoder(key_dim=key_dim, name="enc{}".format(i)))

        self.decoders_cls = []
        for i in range(4):
            self.decoders_cls.append(KernelEncoder(key_dim=key_dim, name="dec-cls-{}".format(i)))

        self.decoders_reg = []
        for i in range(4):
            self.decoders_reg.append(KernelEncoder(key_dim=key_dim, name="dec-reg-{}".format(i)))

        output_decoding["schema"] = schema
        output_decoding["num_output_classes"] = num_output_classes
        output_decoding["event_set_output"] = event_set_output
        output_decoding["met_output"] = met_output
        self.output_dec = OutputDecoding(**output_decoding)

        self.Q_cls = self.add_weight(
            shape=(
                1,
                1,
                128,
            ),
            name="Q_cls",
            initializer="random_normal",
            trainable=True,
        )
        self.Q_reg = self.add_weight(
            shape=(
                1,
                1,
                128,
            ),
            name="Q_reg",
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs, training=False):
        X = inputs
        batch_size = tf.shape(X)[0]

        # mask padded elements
        msk = tf.cast(X[:, :, 0] != 0, tf.float32)
        msk_input = tf.expand_dims(tf.cast(msk, tf.float32), -1)

        X_enc = self.enc(X)
        X_enc = self.ffn(X_enc)

        for enc in self.encoders:
            X_enc = enc([X_enc, X_enc, msk], training=training) * msk_input

        X_cls = X_enc
        Q_cls = tf.repeat(
            self.Q_cls,
            repeats=[
                batch_size,
            ],
            axis=0,
        )
        for dec in self.decoders_cls:
            X_cls = dec([Q_cls, X_cls, msk], training=training) * msk_input

        X_reg = X_enc
        Q_reg = tf.repeat(
            self.Q_reg,
            repeats=[
                batch_size,
            ],
            axis=0,
        )
        for dec in self.decoders_reg:
            X_reg = dec([Q_reg, X_reg, msk], training=training) * msk_input

        ret = self.output_dec([X, X_cls, X_reg, msk_input], training=training)

        if self.multi_output:
            return ret
        else:
            return tf.concat(
                [ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1
            )
