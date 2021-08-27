import tensorflow as tf
import sys
import numpy as np

bin_size = 320
num_features = 15

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v1.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph 

graph = load_graph(sys.argv[1])

with tf.compat.v1.Session(graph=graph) as sess:
    out = sess.run("Identity:0", feed_dict={"x:0": np.random.randn(1, 10*bin_size, num_features)})
    print(out)
