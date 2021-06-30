from tqdm import tqdm

import tensorflow_datasets as tfds
import heptfds

VERSION = "1.0.0"

dataset, dataset_info = tfds.load(
            "cms_pf:{}".format(VERSION),
            split="train",
            as_supervised=False,
            with_info=True,
            shuffle_files=True,
        )
print("dataset_info", dataset_info)

for item in dataset:
    print("X shape:", item["X"].shape)
    print("ygen shape:", item["ygen"].shape)
    print("ycand shape:", item["ycand"].shape)
    break
