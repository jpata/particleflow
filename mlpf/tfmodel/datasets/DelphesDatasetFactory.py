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

        # weight_func = make_weight_function(self.cfg)
        # dataset = dataset.map(weight_func)
        print("INFO: sample_weights setting has no effect. Not yet implemented for DelphesDatasetFactory.")

        if self.cfg["setup"]["multi_output"]:
            dataset = dataset.map(self.get_map_to_supervised())

        return dataset, dataset_info
