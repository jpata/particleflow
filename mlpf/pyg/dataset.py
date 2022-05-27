import pandas
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data, Batch
from glob import glob

import pickle
import multiprocessing


def relabel_indices(pid_array):
    """
    relabels classes for convenient ML operations/training
    """
    pid_array[pid_array == 15] = 8  # taus for now
    pid_array[pid_array == 211] = 7
    pid_array[pid_array == 130] = 6
    pid_array[pid_array == 22] = 5
    pid_array[pid_array == 13] = 4
    pid_array[pid_array == 11] = 3
    return pid_array


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
        raw_list = glob(osp.join(self.raw_dir, '*.pkl'))
        print("PFGraphDataset nfiles={}".format(len(raw_list)))
        return sorted([l.replace(self.raw_dir, '.') for l in raw_list])

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, '*.pt'))
        return sorted([l.replace(self.processed_dir, '.') for l in proc_list])

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process_single_file(self, raw_file_name):
        """
        Loads a list of 100 events from a pkl file and generates pytorch geometric Data() objects and stores them in .pt format.
        For cms data, each element is assumed to be a dict('Xelem', 'ygen', ycand') of numpy rec_arrays with the first element in ygen/ycand is the pid
        For delphes data, each element is assumed to be a dict('X', 'ygen', ycand') of numpy standard arrays with the first element in ygen/ycand is the pid

        Args
            raw_file_name: a pkl file
        Returns
            batched_data: a list of Data() objects of the form
             cms ~ Data(x=[#elem, 41], ygen=[#elem, 6], ygen_id=[#elem, 9], ycand=[#elem, 6], ycand_id=[#elem, 9])
             delphes ~ Data(x=[#elem, 12], ygen=[#elem, 6], ygen_id=[#elem, 6], ycand=[#elem, 6], ycand_id=[#elem, 6])
        """

        # load the data pkl file
        with open(osp.join(self.raw_dir, raw_file_name), "rb") as fi:
            data = pickle.load(fi, encoding='iso-8859-1')

        batched_data = []
        if self.data == 'delphes':
            num_classes = 12
            for i in range(len(data['X'])):
                # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
                d = Data(
                    x=torch.tensor(data['X'][i], dtype=torch.float),
                    ygen=torch.tensor(data['ygen'][i], dtype=torch.float)[:, 1:],
                    ygen_id=one_hot_embedding(torch.tensor(data['ygen'][i], dtype=torch.float)[:, 0].long(), num_classes),
                    ycand=torch.tensor(data['ycand'][i], dtype=torch.float)[:, 1:],
                    ycand_id=one_hot_embedding(torch.tensor(data['ycand'][i], dtype=torch.float)[:, 0].long(), num_classes)
                )
                batched_data.append(d)
        elif self.data == 'cms':
            num_classes = 41
            for i in range(len(data)):
                Xelem = torch.tensor(pd.DataFrame(data[i]['Xelem']).to_numpy(), dtype=torch.float)
                ygen = torch.tensor(pd.DataFrame(data[i]['ygen']).to_numpy(), dtype=torch.float)[:, 1:]
                ygen_id = torch.tensor(pd.DataFrame(data[i]['ygen']).to_numpy(), dtype=torch.float)[:, 0].long()
                ycand = torch.tensor(pd.DataFrame(data[i]['ycand']).to_numpy(), dtype=torch.float)[:, 1:]
                ycand_id = torch.tensor(pd.DataFrame(data[i]['ycand']).to_numpy(), dtype=torch.float)[:, 0].long()

                ygen_id = one_hot_embedding(relabel_indices(ygen_id), num_classes)
                ycand_id = one_hot_embedding(relabel_indices(ycand_id), num_classes)

                # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
                d = Data(
                    x=Xelem,
                    ygen=ygen,
                    ygen_id=ygen_id,
                    ycand=ycand,
                    ycand_id=ycand_id
                )

                batched_data.append(d)
        return batched_data

    def process_multiple_files(self, filenames, idx_file):
        datas = [self.process_single_file(fn) for fn in filenames]
        datas = sum(datas, [])
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx_file))
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
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
        data = torch.load(p)
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
    python3 preprocess_data.py --data cms --dataset $sample --processed_dir $sample/processed --num-files-merge 1 --num-proc 1

    e.g. to run for delphes
    python3 preprocess_data.py --data delphes --dataset $sample --processed_dir $sample/processed --num-files-merge 1 --num-proc 1

    """

    args = parse_args()

    pfgraphdataset = PFGraphDataset(root=args.dataset, data=args.data)

    if args.processed_dir:
        pfgraphdataset._processed_dir = args.processed_dir

    pfgraphdataset.process_parallel(args.num_files_merge, args.num_proc)
