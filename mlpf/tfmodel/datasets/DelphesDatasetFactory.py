from tfmodel.datasets import BaseDatasetFactory

import tensorflow_datasets as tfds


class DelphesDatasetFactory(BaseDatasetFactory):
    def get_dataset(self, dataset_name, dataset_dict, split, max_examples_per_split=None):
        download_config = tfds.download.DownloadConfig(
            manual_dir=dataset_dict["manual_dir"], max_examples_per_split=max_examples_per_split
        )
        dataset, dataset_info = tfds.load(
            "{}:{}".format(dataset_name, dataset_dict["version"]),
            split=split,
            as_supervised=False,
            data_dir=dataset_dict["data_dir"],
            with_info=True,
            shuffle_files=False,
            download_and_prepare_kwargs={"download_config": download_config},
        )

        return dataset, dataset_info
