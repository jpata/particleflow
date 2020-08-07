#!/use/bin/env python3


import torch
import torch_geometric
import pickle
import numpy as np

#Sorting data according to input_multiplicity


from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

#dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"

pkl_multiplicity= open("input_multiplicity.pkl","rb")
input_multiplicity = pickle.load(pkl_multiplicity)
input_multiplicity = input_multiplicity.tolist()
#print(input_multiplicity[49999])

pkl_data= open("50k_pkl.pkl","rb")
data_items = pickle.load(pkl_data)

#print(data_items[49999])

k3 = 0
k4 = 0
k5 = 0
k6 = 0
k7 = 0
k8 = 0
k9 = 0
k10 = 0
k11 = 0
k12 = 0




k = 0
for i in input_multiplicity:
    print(f'fail nr:{k}')
    j = input_multiplicity.index(i)
    k = k +1
    
    #bin 3k-4k
    if i in range(3000,4000):
        torch.save(data_items[j], "data/bin_3k/data_{}.pt".format(k3))
        k3 = k3 +1

    #bin 4k-5k
    if i in range(4000,5000):
        torch.save(data_items[j], "data/bin_4k/data_{}.pt".format(k4))
        k4= k4 +1

    #bin 5k-6k
    if i in range(5000,6000):
        torch.save(data_items[j], "data/bin_5k/data_{}.pt".format(k5))
        k5 = k5 +1

    #bin 6k-7k
    if i in range(6000,7000):
        torch.save(data_items[j], "data/bin_6k/data_{}.pt".format(k6))
        k6 = k6 + 1

    #bin 7k-8k
    if i in range(7000,8000):
        torch.save(data_items[j], "data/bin_7k/data_{}.pt".format(k7))
        k7 = k7 + 1
    #bin 8k-9k
    if i in range(8000,9000):
        torch.save(data_items[j], "data/bin_8k/data_{}.pt".format(k8))
        k8 = k8 + 1

    #bin 9k-10k
    if i in range(9000,10000):
        torch.save(data_items[j], "data/bin_9k/data_{}.pt".format(k9))
        k9 = k9 + 1

    #bin 10k-11k
    if i in range(10000,11000):
        torch.save(data_items[j], "data/bin_10k/data_{}.pt".format(k10))
        k10 = k10 + 1

    #bin 11k-12k
    if i in range(11000,12000):
        torch.save(data_items[j], "data/bin_11k/data_{}.pt".format(k11))
        k11 = k11 + 1

    #bin 12k-13k
    if i in range(12000,13000):
        torch.save(data_items[j], "data/bin_12k/data_{}.pt".format(k12))
        k12 = k12 + 1











#Goal: measure the evaluation cost of the MLPF model as a function of input multipplicity

            #task 1: plot the distribution of the input multiplicities across the events using numpy.histogram and matplotlib.histogram

            #task 2: save the `data` object using torch.save(data, "data/TTbar_14TeV_TuneCUETP8M1_cfi/bin_i/file_j.pt") to
            #subfolders based on the input multiplicity binning



                                                                              
