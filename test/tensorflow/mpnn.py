import tensorflow as tf

class EdgeNetwork(tf.keras.Model):
    """EdgeNetwork is a choice for message function that allow vector valued edge features.

    M(h_v, h_w, e_{vw}) = A(e_{vw})h_w. where A is a neural network which maps the edge vector e_{vw} to
    a d x d matrix. where d is the dimension of node state vector.

    Here we have the simplest nn - relu(linear)
    """
    def __init__(self, state_dim, name='edgenetwork'):
        super(EdgeNetwork, self).__init__(name=name)
        self.state_dim = state_dim
        self.nn = tf.keras.layers.Dense(units=state_dim ** 2, activation=tf.nn.relu)

    def call(self, states, edges):
        """
        Input
        -----
            states:   bs x #nodes^2 x state_dim
            edges:    bs x #nodes^2 x #edge_features

        Output
        ------
            messages: bs x #nodes^2 x state_dim

        Map edge vectors to d x d matrices. Reshape both states and edges to do matrix mulltiplication.
        the matrix mutltiplication is doing dot products between:
            <embedded edge from node_i to node_j in graph k> and <state vector of node_j from graph k>
        The output message tensor represents:
        [
            [
                <message from node_i to node_j in graph_k>
                for i in range(n_nodes)
                    for j in range(n_nodes)
            ] for k in range(n_graph)
        ]
        """
        total_edges = tf.shape(edges)[1]
        state_dim = self.state_dim

        Ae_vw = self.nn(edges)                                           # bs x #nodes^2 x state_dim^2
        Ae_vw = tf.reshape(Ae_vw, [-1, state_dim, state_dim])            # bs * #nodes^2 x state_dim x state_dim
        states = tf.reshape(states, [-1, state_dim, 1])                  # bs * #nodes^2 x state_dim x 1
        messages = tf.matmul(Ae_vw, states)                       # bs * #nodes^2 x state_dim x 1
        messages = tf.reshape(messages, [-1, total_edges, state_dim])    # bs x #nodes^2 x state_dim
        return messages


class Aggregation(tf.keras.Model):
    def __init__(self, method='sum', axis=2, name='aggregation'): 
        assert method in ['sum', 'mean'], 'Unsupported aggregation method'
        super(Aggregation, self).__init__(name=name)
        self.method = method
        self.axis = axis

    def call(self, x, keepdims= False, *args , **kwargs): #KYR: added keepdims
        if self.method == 'sum':
            return tf.reduce_sum(x, self.axis, keepdims)
        else:
            return tf.reduce_mean(messages, self.axis)


class UpdateFunction(tf.keras.Model):
    """Node states update function via GRU.

    U_t = GRU(h_v^t, m_v^{t+1})

    The same update function is used at each time step t.
    """
    def __init__(self, state_dim, name='message_update_function'):
        super(UpdateFunction, self).__init__(name=name)
        self.state_dim = state_dim
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(units=state_dim)

    def call(self, states, messages):
        """
        Input
        -----
            states:   bs x #nodes x state_dim
            messages: bs x #nodes x state_dim
        """
        num_nodes = tf.shape(states)[1]
        state_dim = self.state_dim
        states = tf.reshape(states, [-1, 1, state_dim])
        messages = tf.reshape(messages, [-1, 1, state_dim])
        concat = self.concat([states, messages])
        updated_messages = self.GRU(concat)
        updated_messages = tf.reshape(updated_messages, [-1, num_nodes, state_dim])
        return updated_messages


class MessagePassing(tf.keras.Model):
    """
    > The message passing phrase runs for T time steps and is defined in terms of
        1. message function M_t
        2. vertex update function U_t
      during the message passing phase, hidden states h_v^t at each node in the graph are updated based on
      messages m_v^{t+1} according to:
        1. m_v^{t+1} = \Sigma_{w \in N(v)}{M_t(h_v^t, h_w^t, e_{vw})}
        2. h_v^{t+1} = U_t(h_v^t, m_v^{t+1})

    To generalize a bit, we can use other aggregation function instead of summation.
    """
    def __init__(self, state_dim, name='message_passing'):
        super(MessagePassing, self).__init__(self, name=name)
        self.state_dim = state_dim
        self.message_function = EdgeNetwork(state_dim=state_dim, name=name + '/message_func')
        self.message_aggregation = Aggregation(name=name + '/message_agg')
        self.update_function = UpdateFunction(state_dim=state_dim, name=name + '/state_update')

    def call(self, states, edges, masks, training=False):
        """
        Input
        -----
            nodes: bs x #nodes x state_dim
            edges: bs x #nodes^2 x #edge_features
            masks: bs x #nodes^2 x 1               binary matrix indicating whether edge exist or not

        """
        num_nodes = tf.shape(states)[1]
        state_dim = tf.shape(states)[2]
        masks = tf.reshape(masks, [-1, num_nodes ** 2, 1])
        states_j = tf.tile(states, [1, num_nodes, 1])
        messages = self.message_function(states_j, edges)
        masked_messages = tf.multiply(messages, masks)
        # reshape to batch, from_node, to_node, message
        masked_messages = tf.reshape(masked_messages, [-1, num_nodes, num_nodes, state_dim])
        aggregated_messages = self.message_aggregation(masked_messages)
        updated_messages = self.update_function(states, aggregated_messages)
        return updated_messages


class ReadoutEdge(tf.keras.Model):
    def __init__(self, hidden_sizes, num_outputs, name='readout_edges'):
        super(ReadoutEdge, self).__init__(name=name)
        self.concat = tf.keras.layers.Concatenate()
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(units=hidden_size, activation='relu', name=name + '/hidden_{}'.format(i))
        for i, hidden_size in enumerate(hidden_sizes)])
        self.last_linear = tf.keras.layers.Dense(units=num_outputs, name=name + '/last_linear')


    def call(self, states, edges, training=False):
        num_nodes = tf.shape(states)[1]
        state_dim = tf.shape(states)[2]
        states_i = tf.reshape(tf.tile(states, [1, 1, num_nodes]), [-1, num_nodes ** 2, state_dim])  #
        states_j = tf.tile(states, [1, num_nodes, 1])                                               # 
        concat = self.concat([states_i, edges, states_j])
        features = self.hidden_layers(concat)
        output = self.last_linear(features)
        return output


class ReadoutNodes(tf.keras.Model):
    def __init__(self, hidden_sizes, num_outputs, name='readout_nodes'):
        super(ReadoutNodes, self).__init__(name=name)
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(units=hidden_size, activation='relu', name=name + '/hidden_{}'.format(i))
        for i, hidden_size in enumerate(hidden_sizes)])
        self.last_linear = tf.keras.layers.Dense(units=num_outputs, name=name + '/last_linear')

    def call(self, states, training=False):
        features = self.hidden_layers(states)
        output = self.last_linear(features)
        return output


class ReadoutGraph(tf.keras.Model):
    def __init__(self, hidden_sizes, num_outputs, agg_function, name='readout_graph'):
        super(ReadoutGraph, self).__init__(name=name)
        self.agg_function = agg_function
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(units=hidden_size, activation='relu', name=name + '/hidden_{}'.format(i))
        for i, hidden_size in enumerate(hidden_sizes)])
        self.last_linear = tf.keras.layers.Dense(units=num_outputs, name=name + '/last_linear')


    def call(self, states, masks, training=False):
        num_nodes = tf.shape(states)[1]
        masks = tf.reshape(masks, [-1, num_nodes, 1])
        masked_states = tf.multiply(states, masks)
        graph_states = self.agg_function(masked_states, keepdims = True) #KEPT DIMENSIONS
        features = self.hidden_layers(graph_states)
        output = self.last_linear(features)
        return output


class MPNN(tf.keras.Model):
    """Implementation of Message Passing Neural Network.

    reference: https://arxiv.org/abs/1704.01212i
    """
    def __init__(self, hidden_sizes, num_outputs, state_dim, update_steps, name='mpnn'):
        super(MPNN, self).__init__(name=name)
        self.update_steps = int(update_steps)
        self.node_embedding = tf.keras.layers.Dense(units=state_dim, activation='relu')
        self.message_passing = MessagePassing(state_dim=state_dim)
        self.readout_func = ReadoutGraph(hidden_sizes, num_outputs, Aggregation('sum', 1))

    def call(self, nodes, edges, node_masks=None, edge_masks=None, training=False):
        states = self.node_embedding(nodes)
        for time_step in range(self.update_steps):
            states = self.message_passing(states, edges, edge_masks, training=training)
        readout = self.readout_func(states, node_masks, training=training)
        return readout


def _test_edgenetwork():
    """testcase for edgenetwork forward pass
    a batch of 32 graphs, each with 3 nodes, include self pointing eage - 9 edges per graph, each node has 5 features.
    """
    edges = tf.random.uniform((32, 9, 5))
    states = tf.tile(tf.random.uniform((32, 3, 3)), [1, 3, 1])
    m = EdgeNetwork(state_dim=3)
    o = m(states, edges)
    assert o.shape == (32, 9, 3)


def _test_message_update():
    states = tf.random.uniform((32, 9, 3))
    messages = tf.random.uniform((32, 9, 3))
    m = UpdateFunction(3)
    o = m(states, messages)
    assert o.shape == (32, 9, 3)


def _test_message_passing():
    """test case for message passing.

    a batch of 2 graphs, each has 2 nodes, each node has a state vector of size 2, each edge has 3 features.
    """
    states = tf.convert_to_tensor([[[1, 2], [2, 1]], [[3, 4], [4, 3]]], dtype='float')
    edges = tf.convert_to_tensor([
        [[0, 0, 0], [1, 2, 3], [3, 2, 1], [0, 0, 0]], [[0, 0, 0], [3, 4, 2], [2, 4, 3], [0, 0, 0]]
    ], dtype='float')
    masks = tf.expand_dims(tf.convert_to_tensor([[[0], [1], [1], [0]], [[0], [1], [1], [0]]], dtype='float'), axis=-1)
    m = MessagePassing(2)
    o = m(states, edges, masks)
    assert o.shape == (2, 2, 2)


def _test_edge_readout():
    states = tf.random.uniform((32, 3, 3))
    edges = tf.random.uniform((32, 9, 2))
    m = ReadoutEdge([3, 2], 1)
    o = m(states, edges)
    assert o.shape == (32, 9, 1)


def _test_node_readout():
    states = tf.random.uniform((32, 3, 3))
    m = ReadoutNodes([3, 2], 1)
    o = m(states)
    assert o.shape == (32, 3, 1)


def _test_graph_readout():
    states = tf.random.uniform((32, 3, 3))
    masks = tf.expand_dims(
        tf.convert_to_tensor([[1, 1, 0]] * 8 + [[1, 0, 1]] * 8 + [[0, 1, 1]] * 16, dtype='float'),
        axis=-1)
    agg_func = Aggregation(method='sum', axis=1)
    m = ReadoutGraph([3, 3], 1, agg_func)
    o = m(states, masks)
    assert o.shape == (32, 1)


def _test_mpnn():
    nodes = tf.random.uniform((32, 3, 3))
    edges = tf.random.uniform((32, 3 * 3, 2))
    node_masks = tf.expand_dims(
        tf.convert_to_tensor([[1, 1, 0]] * 8 + [[1, 0, 1]] * 10 + [[0, 1, 1]] * 14, dtype='float'),
        axis=-1)
    edge_masks = tf.expand_dims(
        tf.convert_to_tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0]] * 16 + [[0, 1, 1, 1, 0, 0, 1, 0, 0]] * 16, dtype='float'),
        axis=-1)
    m = MPNN([5, 5,], 1, 8, 3)
    o = m(nodes, edges, node_masks=node_masks, edge_masks=edge_masks)
    print("-----THIS WORKS ------")
    assert o.shape == (32, 1)

if __name__ == '__main__':
    _test_edgenetwork()
    _test_message_update()
    _test_message_passing()
    _test_edge_readout()
    _test_node_readout()
    _test_graph_readout()
    _test_mpnn()
