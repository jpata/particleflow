import tensorflow_datasets as tfds
import heptfds

VERSION = "1.0.0"

train_dataset, train_dataset_info = tfds.load(
    "cms_pf:{}".format(VERSION),
    split="train",
    as_supervised=False,
    with_info=True,
    shuffle_files=True,
)
print("train_dataset_info", train_dataset_info)

test_dataset, test_dataset_info = tfds.load(
    "cms_pf:{}".format(VERSION),
    split="test",
    as_supervised=False,
    with_info=True,
    shuffle_files=True,
)
print("test_dataset_info", test_dataset_info)
