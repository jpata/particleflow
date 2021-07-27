import numpy as np
import torch
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

# if not multigpu we have to pass batches that are stacked as "batch.type() = Batch" (not list) so that pytorch can access attributes like ygen_id through batch.ygen_id
# if multigpu we have to pass list of "Data" elements.. then behind the scene, pytorch DP will convert the list to appropriate Batches to fit on the gpus available so that batch.ygen_id works out of the box

# define a function that casts the ttbar dataset into a dataloader for efficient NN training
def data_to_loader_ttbar(full_dataset, n_train, n_valid, batch_size):

    # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    valid_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=n_train+n_valid))

    # preprocessing the train_dataset in a good format for passing correct batches of events to the GNN
    train_data=[]
    for i in range(len(train_dataset)):
        train_data = train_data + train_dataset[i]

    # preprocessing the valid_dataset in a good format for passing correct batches of events to the GNN
    valid_data=[]
    for i in range(len(valid_dataset)):
        valid_data = valid_data + valid_dataset[i]

    if not multi_gpu:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    else:
        #https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/data_parallel.html
        train_loader = DataListLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataListLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader

def data_to_loader_qcd(full_dataset, n_test, batch_size):

    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_test))

    # preprocessing the test_dataset in a good format for passing correct batches of events to the GNN
    test_data=[]
    for i in range(len(test_dataset)):
        test_data = test_data + test_dataset[i]

    if not multi_gpu:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    else:
        #https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/data_parallel.html
        test_loader = DataListLoader(test_data, batch_size=batch_size, shuffle=True)

    return test_loader

#----------------------------------------------------------------------------------------
