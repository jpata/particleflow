import numpy as np
import torch
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# define a function that casts the dataset into a dataloader for efficient NN training
def from_data_to_loader(full_dataset, n_train, n_val, batch_size):

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    valid_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=n_train+n_val))

    # preprocessing the train_dataset in a good format for passing correct batches of events to the GNN
    train_dataset_batched=[]
    for i in range(len(train_dataset)):
        train_dataset_batched += train_dataset[i]
    train_dataset_batched = [[i] for i in train_dataset_batched]

    # preprocessing the valid_dataset in a good format for passing correct batches of events to the GNN
    valid_dataset_batched=[]
    for i in range(len(valid_dataset)):
        valid_dataset_batched += valid_dataset[i]
    valid_dataset_batched = [[i] for i in valid_dataset_batched]

    #hack for multi-gpu training
    if not multi_gpu:
        def collate(items):
            l = sum(items, [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum(items, [])
            return l

    train_loader = DataListLoader(train_dataset_batched, batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset_batched, batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate

    return train_loader, valid_loader
