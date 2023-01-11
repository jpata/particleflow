import multiprocessing
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import Dataset


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class PFGraphDataset(Dataset):
    """
    Initialize parameters of graph dataset
    Args:
        root (str): path
    """

    def __init__(self, root, data, transform=None, pre_transform=None):
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)
        self._processed_dir = Dataset.processed_dir.fget(self)
        self.data = data

    @property
    def raw_file_names(self):
        raw_list = glob(osp.join(self.raw_dir, "*"))
        print("PFGraphDataset nfiles={}".format(len(raw_list)))
        return sorted([raw_path.replace(self.raw_dir, ".") for raw_path in raw_list])

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list])

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process_single_file(self, raw_file_name):
        """
        Loads raw datafile information and generates PyG Data() objects and stores them in .pt format.

        Args
            raw_file_name: raw data file name.
        Returns
            batched_data: a list of Data() objects of the form
             cms ~ Data(x=[#, 41], ygen=[#, 6], ygen_id=[#, 9], ycand=[#, 6], ycand_id=[#, 9])
             delphes ~ Data(x=[#, 12], ygen=[#elem, 6], ygen_id=[#, 6], ycand=[#, 6], ycand_id=[#, 6])
        """

        if self.data == "cms":
            from cms.cms_utils import prepare_data_cms

            return prepare_data_cms(osp.join(self.raw_dir, raw_file_name))

        elif self.data == "delphes":
            from delphes.delphes_utils import prepare_data_delphes

            return prepare_data_delphes(osp.join(self.raw_dir, raw_file_name))

    def process_multiple_files(self, filenames, idx_file):
        datas = []
        for fn in filenames:
            x = self.process_single_file(fn)
            if x is None:
                continue
            datas.append(x)

        datas = sum(datas, [])
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx_file))
        torch.save(datas, p)
        print(f"saved file {p}")

    def process(self, num_files_to_batch):
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            self.process_multiple_files(fns, idx_file)
            idx_file += 1

    def process_parallel(self, num_files_to_batch, num_proc):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(num_proc)
        pool.map(process_func, pars)

    def get(self, idx):
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx))
        data = torch.load(p, map_location="cpu")
        return data

    def __getitem__(self, idx):
        return self.get(idx)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="'cms' or 'delphes'?")
    parser.add_argument("--dataset", type=str, required=True, help="Input data path")
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    """
    e.g. to run for cms
    python PFGraphDataset.py --data cms --dataset ../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi --processed_dir \
        ../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed --num-files-merge 1 --num-proc 1

    e.g. to run for delphes
    python3 PFGraphDataset.py --data delphes --dataset $sample --processed_dir $sample/processed \
        --num-files-merge 1 --num-proc 1

    """

    args = parse_args()

    pfgraphdataset = PFGraphDataset(root=args.dataset, data=args.data)

    if args.processed_dir:
        pfgraphdataset._processed_dir = args.processed_dir

    pfgraphdataset.process_parallel(args.num_files_merge, args.num_proc)
