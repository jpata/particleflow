try:
    from pyg.cms_utils import prepare_data_cms
except ImportError:
    from cms_utils import prepare_data_cms

import multiprocessing
import os.path as osp
import pickle
from glob import glob

import torch
from torch_geometric.data import Data, Dataset


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


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
        raw_list = glob(osp.join(self.raw_dir, "*.pkl"))
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
        Loads a list of 100 events from a pkl file and generates pytorch geometric Data() objects
        and stores them in .pt format.
        For cms data, each element is assumed to be a dict('Xelem', 'ygen', ycand')
        of numpy rec_arrays with the first element in ygen/ycand is the pid
        For delphes data, each element is assumed to be a dict('X', 'ygen', ycand')
        of numpy standard arrays with the first element in ygen/ycand is the pid

        Args
            raw_file_name: a pkl file
        Returns
            batched_data: a list of Data() objects of the form
             cms ~ Data(x=[#elem, 41], ygen=[#elem, 6], ygen_id=[#elem, 9], ycand=[#elem, 6], ycand_id=[#elem, 9])
             delphes ~ Data(x=[#elem, 12], ygen=[#elem, 6], ygen_id=[#elem, 6], ycand=[#elem, 6], ycand_id=[#elem, 6])
        """

        if self.data == "cms":
            return prepare_data_cms(osp.join(self.raw_dir, raw_file_name))

        elif self.data == "delphes":
            # load the data pkl file
            with open(osp.join(self.raw_dir, raw_file_name), "rb") as fi:
                data = pickle.load(fi, encoding="iso-8859-1")

            batched_data = []
            for i in range(len(data["X"])):
                # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
                d = Data(
                    x=torch.tensor(data["X"][i], dtype=torch.float),
                    ygen=torch.tensor(data["ygen"][i], dtype=torch.float)[:, 1:],
                    ygen_id=torch.tensor(data["ygen"][i], dtype=torch.float)[:, 0].long(),
                    ycand=torch.tensor(data["ycand"][i], dtype=torch.float)[:, 1:],
                    ycand_id=torch.tensor(data["ycand"][i], dtype=torch.float)[:, 0].long(),
                )

                batched_data.append(d)

        return batched_data

    def process_multiple_files(self, filenames, idx_file):
        datas = [self.process_single_file(fn) for fn in filenames]
        datas = sum(datas, [])
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx_file))
        print(p)
        torch.save(datas, p)

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
