import numpy as np
import os
import os.path as osp
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data, Batch
import itertools
from glob import glob
import numba
from numpy.lib.recfunctions import append_fields

import pickle
import scipy
import scipy.sparse
import math
import multiprocessing

# assumes pkl files exist in /test_tmp_delphes/data/delphes_cfi/raw
# they are processed and saved as pt files in /test_tmp_delphes/data/delphes_cfi/processed
# PFGraphDataset -> returns for 1 event: Data(x=[5139, 12], ycand=[5139, 6], ycand_id=[5139, 6], ygen=[5139, 6], ygen_id=[5139, 6])

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
    def __init__(self, root, transform=None, pre_transform=None):
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)
        self._processed_dir = Dataset.processed_dir.fget(self)

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
        with open(osp.join(self.raw_dir, raw_file_name), "rb") as fi:
            data = pickle.load(fi, encoding='iso-8859-1')

        x=[]
        ygen=[]
        ycand=[]
        d=[]
        batch_data = []
        ygen_id=[]
        ycand_id=[]

        for i in range(len(data['X'])):
            x.append(torch.tensor(data['X'][i], dtype=torch.float))
            ygen.append(torch.tensor(data['ygen'][i], dtype=torch.float))
            ycand.append(torch.tensor(data['ycand'][i], dtype=torch.float))

            # one-hot encoding the first element in ygen & ycand (which is the PID) and store it in ygen_id & ycand_id
            ygen_id.append(ygen[i][:,0])
            ycand_id.append(ycand[i][:,0])

            ygen_id[i] = ygen_id[i].long()
            ycand_id[i] = ycand_id[i].long()

            ygen_id[i] = one_hot_embedding(ygen_id[i], 6)
            ycand_id[i] = one_hot_embedding(ycand_id[i], 6)

            # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
            d = Data(
                x=x[i],
                ygen=ygen[i][:,1:], ygen_id=ygen_id[i],
                ycand=ycand[i][:,1:], ycand_id=ycand_id[i]
            )

            batch_data.append(d)
        return batch_data

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")
    args = parser.parse_args()

    pfgraphdataset = PFGraphDataset(root=args.dataset)

    if args.processed_dir:
        pfgraphdataset._processed_dir = args.processed_dir

    pfgraphdataset.process_parallel(args.num_files_merge,args.num_proc)
    #pfgraphdataset.process(args.num_files_merge)
