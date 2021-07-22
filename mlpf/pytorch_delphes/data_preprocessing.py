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
# from graph_data_delphes import PFGraphDataset, one_hot_embedding
# # the next part initializes some args values (to run the script not from terminal)
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
# 'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../test_tmp_delphes/data/pythia8_qcd',
# 'outpath': '../../test_tmp_delphes/experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 0.001, 'l3': 1, 'dropout': 0.5,
# 'radius': 0.1, 'convlayer': 'gravnet-knn', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
# 'input_encoding': 0, 'load': False, 'load_epoch': 0, 'load_model': 'PFNet7_cand_ntrain_3_nepochs_1', 'evaluate': True, 'evaluate_on_cpu': True})
#
# full_dataset = PFGraphDataset(args.dataset)
# full_dataset_qcd = PFGraphDataset(args.dataset_qcd)
#
# train_loader, valid_loader = data_to_loader_ttbar(full_dataset, args.n_train, args.n_valid, batch_size=args.batch_size)
# test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)
#
# for batch in train_loader:
#     break
#
# batch
# len(train_loader)
#
#
# # if multigpu: a "Batch" of size 3 is given by: [Data(x=[5k, 12], ycand=[5k, 6], ...) , Data(x=[5k, 12], ...), Data(x=[5k, 12], ...)]
# # then when we pass it to the model, DP takes care of converting it into batches like this (for 2 gpus):
# # Batch(batch=[2*5k], x=[2*5k, 12], ...)
# # Batch(batch=[5k], x=[5k, 12], ...)
#
# # if not multigpu: a "Batch" of size 2 is directly given by: Batch(batch=(2*5k), x=(2*5k,12), ...)
# # Note: batch is a column vector which maps each node to its respective graph in the batch:
# batch.batch
