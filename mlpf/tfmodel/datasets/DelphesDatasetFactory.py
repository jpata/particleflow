import tensorflow as tf
import tensorflow_datasets as tfds
import heptfds

from tfmodel.datasets import BaseDatasetFactory


class DelphesDatasetFactory(BaseDatasetFactory):
    def get_dataset(self, split, max_examples_per_split=None):
        download_config = tfds.download.DownloadConfig(
            manual_dir=None, max_examples_per_split=max_examples_per_split
        )
        dataset, dataset_info = tfds.load(
            "delphes_pf:{}".format(self.cfg["delphes_pf"]["version"]),
            split=split,
            as_supervised=False,
            data_dir=self.cfg["delphes_pf"]["data_dir"],
            with_info=True,
            shuffle_files=True,
            download_and_prepare_kwargs={"download_config": download_config},
        )
        print("INFO: sample_weights setting has no effect. Not yet implemented for DelphesDatasetFactory.")

        return dataset, dataset_info
