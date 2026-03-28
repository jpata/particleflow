import torch
from torch.utils.data import Dataset
from types import SimpleNamespace


class MockDictDataset(Dataset):
    """
    A flexible mock dataset that returns dictionaries of tensors.
    """

    def __init__(self, size=20, offset=0, keys=("X",), shapes=((1,),)):
        self.size = size
        self.offset = offset
        self.keys = keys
        self.shapes = shapes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ret = {}
        for key, shape in zip(self.keys, self.shapes):
            # Fill with index + offset for traceability
            val = torch.full(shape, float(idx + self.offset))
            ret[key] = val
        return ret


class MockDataset(Dataset):
    """
    A simple mock dataset that returns the index itself.
    """

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


class MockTFDS:
    """
    Mocks the structure of a TFDS builder/data_source for testing TFDSDataSource.
    """

    def __init__(self, data_list, name="cms_test"):
        self.data_list = data_list
        self.data_source = SimpleNamespace(__getitems__=lambda items: [data_list[i] for i in items])
        self.dataset_info = SimpleNamespace(name=name, features=SimpleNamespace(deserialize_example_np=lambda x, decoders: x), config_name="test")
        self.decoders = None

    def __len__(self):
        return len(self.data_list)
