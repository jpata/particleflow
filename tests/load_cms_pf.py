import tensorflow_datasets as tfds
import heptfds
import argparse
import os

VERSION = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--manual_dir", type=str, help="Original data directory (before processing to tfrecords).")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory to store tfrecords.")
    parser.add_argument(
        "-n",
        "--max_examples_per_split",
        type=int,
        default=None,
        help="When set, only generate the first X examples (default to 1), rather than the full dataset.",
    )
    args = parser.parse_args()
    return args

def main(args):
    download_config = tfds.download.DownloadConfig(manual_dir=args.manual_dir, max_examples_per_split=args.max_examples_per_split)

    train_dataset, train_dataset_info = tfds.load(
        "cms_pf:{}".format(VERSION),
        split="train",
        data_dir=args.data_dir,
        as_supervised=False,
        with_info=True,
        shuffle_files=True,
        download_and_prepare_kwargs={"download_config": download_config},
    )
    print("train_dataset_info:\n", train_dataset_info)

    test_dataset, test_dataset_info = tfds.load(
        "cms_pf:{}".format(VERSION),
        split="test",
        data_dir=args.data_dir,
        as_supervised=False,
        with_info=True,
        shuffle_files=True,
        download_and_prepare_kwargs={"download_config": download_config},
    )
    print("test_dataset_info:\n", test_dataset_info)


if __name__ == "__main__":
    args = parse_args()
    main(args)