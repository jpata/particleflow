
class Dataset:
    def __init__(self, **kwargs):
        self.num_input_features = kwargs.get("num_input_features")
        self.num_output_features = kwargs.get("num_output_features")
        self.padded_num_elem_size = kwargs.get("padded_num_elem_size")
        self.schema = kwargs.get("schema")
