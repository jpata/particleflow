from tqdm import tqdm
import argparse

import tensorflow_datasets as tfds
import heptfds

VERSION = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Dataset name")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory to store tfrecords.")
    args = parser.parse_args()
    return args


def main(args):
    dataset, dataset_info = tfds.load(
        "{}:{}".format(args.name, VERSION),
        split="train",
        as_supervised=False,
        with_info=True,
        shuffle_files=True,
        data_dir=args.data_dir,
    )
    print("dataset_info", dataset_info)

    for item in dataset:
        print("X shape:", item["X"].shape)
        print("ygen shape:", item["ygen"].shape)
        print("ycand shape:", item["ycand"].shape)
        break


if __name__ == "__main__":
    args = parse_args()
    main(args)
